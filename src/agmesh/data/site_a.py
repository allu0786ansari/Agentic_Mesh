"""Site A (HAI 21.03): splits and the four-client partition (Proposal §4.1, §6.2).

Training data
    ``scheme: time`` (G0 decision 2): each client is one recording period, as set by
    ``period_assignment`` (``train3:first_half`` is the first half of ``train3`` by
    rows). ``scheme: iid``: training files are cut into ``block_hours`` blocks that
    are shuffled with a fixed seed and dealt to clients in equal numbers (X2).
    In both, the last ``normal_val_fraction`` of each client's data becomes normal
    validation, separated from training by a gap of ``gap_windows`` windows.

Test files
    Each test file is cut into one *event segment* per attack event, with boundaries
    halfway between neighbouring events and a one-window gap at each internal
    boundary. Segments are grouped by the processes the event targeted, and within
    each group about ``attack_val_fraction`` go to attack validation (largest-
    remainder rounding over the groups). Segments are dealt to clients round-robin,
    so each client gets attack segments in both validation and test.

Every output row keeps a ``run_id``: windows never cross a run, so no window spans
a split or a gap.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agmesh.common.config import Params
from agmesh.common.provenance import sha256_file
from agmesh.data import hai
from agmesh.data.events import runs
from agmesh.data.manifest import PARTITIONS, iso, write_manifest, write_table
from agmesh.data.windows import frame_window_starts, window_labels

RUN_ID = "run_id"


@dataclass(frozen=True)
class Segment:
    segment_id: str
    file: str
    start_row: int  # inclusive
    stop_row: int  # exclusive
    event_start_row: int
    event_stop_row: int
    processes: str


@dataclass(frozen=True)
class Assignment:
    segment: Segment
    partition: str  # "val_attack" or "test"
    client: str


# ----------------------------------------------------------------------- test segments


def event_segments(name: str, frame: pd.DataFrame, cfg: Params, gap_rows: int) -> list[Segment]:
    """One segment per attack event in a test file (see module docstring)."""
    label = frame[cfg.data.hai.label_column].to_numpy() > 0
    events = runs(label)
    if not events:
        return []
    n = len(frame)
    segments: list[Segment] = []
    lower = 0
    for i, (start, stop) in enumerate(events):
        if i + 1 < len(events):
            middle = (stop + events[i + 1][0]) // 2
            upper = middle - gap_rows // 2
            next_lower = upper + gap_rows
        else:
            upper, next_lower = n, n
        upper = max(upper, stop)
        segments.append(
            Segment(
                segment_id=f"{name}:e{i:02d}",
                file=name,
                start_row=min(lower, start),
                stop_row=upper,
                event_start_row=start,
                event_stop_row=stop,
                processes=hai.process_combination(frame, cfg.data.hai, start, stop),
            )
        )
        lower = next_lower
    return segments


def _validation_quota(groups: dict[str, list[Segment]], fraction: float) -> dict[str, int]:
    """Largest-remainder rounding so the total is round(fraction * all segments)."""
    total = round(fraction * sum(len(v) for v in groups.values()))
    exact = {key: fraction * len(items) for key, items in groups.items()}
    quota = {key: math.floor(value) for key, value in exact.items()}
    remainder = total - sum(quota.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - quota[key]), key))
    for key in order[: max(remainder, 0)]:
        quota[key] += 1
    return quota


def assign_segments(
    segments: list[Segment], clients: list[str], fraction: float, seed: int
) -> list[Assignment]:
    """Stratified validation/test split of event segments, dealt round-robin to clients."""
    groups: dict[str, list[Segment]] = defaultdict(list)
    for segment in sorted(segments, key=lambda s: s.segment_id):
        groups[segment.processes].append(segment)
    quota = _validation_quota(groups, fraction)
    rng = np.random.default_rng(seed)
    turn = {"val_attack": 0, "test": 0}
    assignments: list[Assignment] = []
    for key in sorted(groups):
        items = groups[key]
        order = rng.permutation(len(items))
        for rank, index in enumerate(order):
            partition = "val_attack" if rank < quota[key] else "test"
            client = clients[turn[partition] % len(clients)]
            turn[partition] += 1
            assignments.append(Assignment(items[int(index)], partition, client))
    return sorted(assignments, key=lambda a: a.segment.segment_id)


# --------------------------------------------------------------------- training splits


def _tail_split(
    frame: pd.DataFrame, run_id: str, fraction: float, gap_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one contiguous frame into (train, normal validation) with a gap between."""
    n = len(frame)
    cut = round(n * (1.0 - fraction))
    train = frame.iloc[:cut].assign(**{RUN_ID: run_id})
    val = frame.iloc[min(cut + gap_rows, n) :].assign(**{RUN_ID: f"{run_id}:val"})
    return train, val


def _period_rows(frames: dict[str, pd.DataFrame], period: str) -> tuple[str, pd.DataFrame]:
    name, _, half = period.partition(":")
    if name not in frames:
        raise ValueError(f"period_assignment names {name!r}, which is not a HAI training file")
    frame = frames[name]
    middle = len(frame) // 2
    if half == "first_half":
        return f"{name}:h1", frame.iloc[:middle]
    if half == "second_half":
        return f"{name}:h2", frame.iloc[middle:]
    return name, frame


def split_time(
    train: dict[str, pd.DataFrame], params: Params, gap_rows: int
) -> dict[str, dict[str, pd.DataFrame]]:
    site = params.data.partition.site_a
    assert site.period_assignment is not None  # guaranteed by the config validator
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for client in site.clients:
        run_id, rows = _period_rows(train, site.period_assignment[client])
        tr, val = _tail_split(rows, run_id, params.data.splits.normal_val_fraction, gap_rows)
        out[client] = {"train": tr, "val_normal": val}
    return out


def split_iid(
    train: dict[str, pd.DataFrame], params: Params, gap_rows: int
) -> dict[str, dict[str, pd.DataFrame]]:
    site = params.data.partition.site_a
    block = site.block_hours * 3600 * params.data.hai.sample_rate_hz
    blocks: list[tuple[str, pd.DataFrame]] = []
    for name in sorted(train):
        frame = train[name]
        for i, begin in enumerate(range(0, len(frame), block)):
            piece = frame.iloc[begin + gap_rows : begin + block]
            if len(piece) >= block // 2:
                blocks.append((f"{name}:b{i:02d}", piece))
    order = np.random.default_rng(params.data.splits.seed).permutation(len(blocks))
    per_client: dict[str, list[tuple[str, pd.DataFrame]]] = {c: [] for c in site.clients}
    usable = len(blocks) - len(blocks) % len(site.clients)
    for rank, index in enumerate(order[:usable]):
        per_client[site.clients[rank % len(site.clients)]].append(blocks[int(index)])
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for client, items in per_client.items():
        n_val = max(1, round(len(items) * params.data.splits.normal_val_fraction))
        tr = [f.assign(**{RUN_ID: rid}) for rid, f in items[:-n_val]]
        val = [f.assign(**{RUN_ID: f"{rid}:val"}) for rid, f in items[-n_val:]]
        out[client] = {"train": pd.concat(tr), "val_normal": pd.concat(val)}
    return out


# --------------------------------------------------------------------------- partition


def _summary(frame: pd.DataFrame, params: Params, sha: str) -> dict[str, Any]:
    win = params.data.windows.site_a
    starts = frame_window_starts(frame, win.length_s, win.train_stride_s)
    label = frame[params.data.hai.label_column].to_numpy()
    attack = window_labels(label, starts, win.length_s)
    return {
        "rows": len(frame),
        "runs": int(frame[RUN_ID].nunique()),
        "windows": int(starts.size),
        "attack_windows": int(attack.sum()),
        "attack_rows": int((label > 0).sum()),
        "start": iso(frame["time"].min()) if len(frame) else None,
        "end": iso(frame["time"].max()) if len(frame) else None,
        "sha256": sha,
    }


def partition(params: Params, hai_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Build every Site A partition file and return the manifest (also written)."""
    cfg = params.data.hai
    site = params.data.partition.site_a
    win = params.data.windows.site_a
    gap_rows = params.data.splits.gap_windows * win.length_s * cfg.sample_rate_hz
    frames, paths, report = hai.load(hai_dir, cfg)
    train = {k: v for k, v in frames.items() if k.startswith("train")}
    test = {k: v for k, v in frames.items() if k.startswith("test")}

    dropped_attack_rows = 0
    for name, frame in train.items():
        attack = frame[cfg.label_column].to_numpy() > 0
        dropped_attack_rows += int(attack.sum())
        train[name] = frame.loc[~attack].reset_index(drop=True)
    spec = hai.build_feature_spec(list(train.values()), cfg)

    if site.scheme == "time":
        clients = split_time(train, params, gap_rows)
    elif site.scheme == "iid":
        clients = split_iid(train, params, gap_rows)
    else:
        raise ValueError(
            "scheme 'regime' is not supported: HAI 21.03 has no separable operating "
            "regimes (ADR 0001, decision 2). Use 'time' or 'iid'."
        )

    segments = [
        s for name in sorted(test) for s in event_segments(name, test[name], params, gap_rows)
    ]
    assignments = assign_segments(
        segments, site.clients, params.data.splits.attack_val_fraction, params.data.splits.seed
    )
    by_client: dict[str, dict[str, list[pd.DataFrame]]] = {
        c: {"val_attack": [], "test": []} for c in site.clients
    }
    for item in assignments:
        seg = item.segment
        rows = test[seg.file].iloc[seg.start_row : seg.stop_row].assign(**{RUN_ID: seg.segment_id})
        by_client[item.client][item.partition].append(rows)

    columns = ["time", RUN_ID, *hai.feature_columns(next(iter(frames.values())), cfg)]
    columns += hai.label_columns(cfg)
    nodes: dict[str, Any] = {}
    for client in site.clients:
        tables = dict(clients[client])
        for part in ("val_attack", "test"):
            parts = by_client[client][part]
            tables[part] = pd.concat(parts) if parts else pd.DataFrame(columns=columns)
        summaries: dict[str, Any] = {}
        for part in PARTITIONS:
            table = tables[part][columns].copy()
            table[RUN_ID] = table[RUN_ID].astype(str)
            sha = write_table(table, out_dir / client / f"{part}.parquet")
            summaries[part] = _summary(table, params, sha)
        summaries["val_attack"]["events"] = sorted(
            a.segment.segment_id
            for a in assignments
            if a.client == client and a.partition == "val_attack"
        )
        summaries["test"]["events"] = sorted(
            a.segment.segment_id
            for a in assignments
            if a.client == client and a.partition == "test"
        )
        period = site.period_assignment.get(client) if site.period_assignment else None
        nodes[client] = {"period": period, "partitions": summaries}

    times = {name: frame["time"] for name, frame in frames.items()}
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "site": "site_a",
        "dataset": f"HAI {cfg.release}",
        "scheme": site.scheme,
        "config": {
            "hai": cfg.model_dump(mode="json"),
            "window": win.model_dump(mode="json"),
            "splits": params.data.splits.model_dump(mode="json"),
            "partition": site.model_dump(mode="json"),
        },
        "sources": {
            name: {
                "file": paths[name].name,
                "sha256": sha256_file(paths[name]),
                "rows": len(frames[name]),
                "start": iso(times[name].min()),
                "end": iso(times[name].max()),
            }
            for name in sorted(frames)
        },
        "load": {**asdict(report), "train_attack_rows_dropped": dropped_attack_rows},
        "features": spec.to_dict(),
        "events": {
            "total": len(segments),
            "val_attack": sum(a.partition == "val_attack" for a in assignments),
            "test": sum(a.partition == "test" for a in assignments),
            "segments": [
                {
                    **asdict(a.segment),
                    "partition": a.partition,
                    "client": a.client,
                    "event_start": iso(
                        test[a.segment.file]["time"].iloc[a.segment.event_start_row]
                    ),
                    "event_seconds": a.segment.event_stop_row - a.segment.event_start_row,
                }
                for a in assignments
            ],
        },
        "nodes": nodes,
    }
    write_manifest(manifest, out_dir)
    return manifest
