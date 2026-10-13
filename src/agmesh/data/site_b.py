"""Site B (CIC Modbus 2023): window features, labels and the three-node partition.

Stages
    :func:`prepare` parses every capture into ``interim/modbus/<split>/<device>/``
    (DVC stage ``prepare``). :func:`partition` turns those tables into 10-second
    window features per node and writes the partitions (DVC stage ``partition``).

Windows (Proposal §6.2)
    Windows lie on a fixed 10-second grid in epoch time, so windows on different
    nodes line up for fusion. A window is kept only if it lies fully inside one
    capture and does not touch an *idle* stretch, where the HMI sent no request to
    or from this node for more than ``idle_gap_s`` (MB-7).

Features (Proposal §6.6; MB-4, MB-9, MB-10, MB-14)
    Computed from the node's own side of the traffic, so the HMI (a client) and
    the IEDs (servers) share one schema. Sets of peers and addresses, the maximum
    requested quantity per function code and the slow-response threshold are
    learned per node from that node's *training* windows only.

Labels (G0 decision 4)
    A window is an attack window on a node if that node saw any IPv4 packet to or
    from the attacker's address in it. Attacker activity separated by more than
    ``episode_gap_s`` forms *episodes*. The first ``attack_val_episodes`` episodes
    of the attack day go to attack validation and the rest to test, with one gap
    window at the boundary. Benign days give training and normal validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from agmesh.common.config import Params
from agmesh.common.provenance import sha256_file
from agmesh.data.events import cluster_times
from agmesh.data.manifest import PARTITIONS, iso, write_manifest, write_table
from agmesh.data.modbus import READ_FCS, WRITE_FCS, capture_files, ip_to_int, parse_capture

SPLITS = ("benign", "attack")
INTERIM_MANIFEST = "manifest.json"
FEATURES = (
    "req_sent",
    "req_recv",
    "req_read",
    "req_write",
    "req_other",
    "resp_exception",
    "peers",
    "new_peers",
    "unseen_addr",
    "unseen_addr_write",
    "oversized_qty",
    "tcp_syn",
    "tcp_rst",
    "stacked_frames",
    "malformed_frames",
    "latency_median_ms",
    "slow_responses",
    "unanswered",
)
LATENCY_FEATURES = ("latency_median_ms", "slow_responses")
AUX_COLUMNS = ("window_start", "time", "hmi_requests", "attacker_packets", "label", "episode")


# ------------------------------------------------------------------------------ prepare


def prepare(params: Params, modbus_dir: Path, interim_dir: Path) -> dict[str, Any]:
    """Parse every capture of every configured device into interim parquet tables."""
    out_root = interim_dir / "modbus"
    attacker = params.data.modbus.attacker_ip
    captures: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in SPLITS:
        captures[split] = {}
        for device in params.data.modbus.devices:
            folder = modbus_dir / split / device
            if not folder.is_dir():
                raise FileNotFoundError(f"missing capture folder {folder}")
            entries: list[dict[str, Any]] = []
            for path in capture_files(folder):
                print(f"  Modbus {split}/{device}/{path.name}", flush=True)
                tables = parse_capture(path, attacker)
                target = out_root / split / device
                stem = path.stem
                for kind, frame in (
                    ("frames", tables.frames),
                    ("tcp", tables.tcp),
                    ("watched", tables.watched),
                ):
                    write_table(frame, target / f"{stem}.{kind}.parquet")
                entries.append(
                    {
                        "file": path.name,
                        "stem": stem,
                        "sha256": sha256_file(path),
                        **tables.span(),
                        "modbus_frames": len(tables.frames),
                        "attacker_packets": len(tables.watched),
                    }
                )
            captures[split][device] = entries
    manifest = {
        "schema_version": "1.0",
        "stage": "prepare",
        "attacker_ip": attacker,
        "devices": params.data.modbus.devices,
        "captures": captures,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / INTERIM_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


@dataclass
class DeviceData:
    frames: pd.DataFrame
    tcp: pd.DataFrame
    watched: npt.NDArray[np.float64]
    spans: list[tuple[float, float]]


def load_device(
    interim_dir: Path, split: str, device: str, entries: list[dict[str, Any]]
) -> DeviceData:
    folder = interim_dir / "modbus" / split / device
    frames, tcp, watched, spans = [], [], [], []
    for entry in entries:
        stem = entry["stem"]
        frames.append(pd.read_parquet(folder / f"{stem}.frames.parquet"))
        tcp.append(pd.read_parquet(folder / f"{stem}.tcp.parquet"))
        watched.append(pd.read_parquet(folder / f"{stem}.watched.parquet")["ts"].to_numpy())
        if entry["first_ts"] is not None:
            spans.append((float(entry["first_ts"]), float(entry["last_ts"])))
    return DeviceData(
        frames=pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        tcp=pd.concat(tcp, ignore_index=True) if tcp else pd.DataFrame(),
        watched=np.concatenate(watched) if watched else np.empty(0),
        spans=spans,
    )


# ------------------------------------------------------------------------------ windows


def grid_windows(spans: list[tuple[float, float]], length: int) -> npt.NDArray[np.int64]:
    """Window indices (start = index * length) lying fully inside one capture span."""
    pieces = []
    for first, last in spans:
        lo = int(np.ceil(first / length))
        hi = int(np.floor(last / length))  # window hi-1 ends at hi*length <= last
        if hi > lo:
            pieces.append(np.arange(lo, hi, dtype=np.int64))
    if not pieces:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(pieces))


def idle_intervals(
    activity: npt.NDArray[np.float64], spans: list[tuple[float, float]], max_gap: float
) -> list[tuple[float, float]]:
    """Stretches longer than ``max_gap`` with no HMI activity, including capture edges."""
    if not spans:
        return []
    first = min(s[0] for s in spans)
    last = max(s[1] for s in spans)
    points = np.sort(activity)
    if points.size == 0:
        return [(first, last)]
    bounds = np.concatenate([[first], points, [last]])
    gaps = np.diff(bounds)
    return [(float(bounds[i]), float(bounds[i + 1])) for i in np.flatnonzero(gaps > max_gap)]


def drop_idle(
    windows: npt.NDArray[np.int64], intervals: list[tuple[float, float]], length: int
) -> npt.NDArray[np.bool_]:
    """Mask of windows that do not touch any idle interval (open intervals)."""
    keep = np.ones(windows.size, dtype=bool)
    starts = windows * length
    for a, b in intervals:
        keep &= ~((starts < b) & (starts + length > a))
    return keep


# ----------------------------------------------------------------------------- features


@dataclass(frozen=True)
class Vocabulary:
    """What one node saw during its training windows (never leaves the node)."""

    peers: frozenset[int]
    addresses: frozenset[tuple[int, int, int]]  # (server, function code, address)
    max_qty: dict[int, int]  # function code -> largest requested quantity
    slow_latency_s: float  # training p99.9 response latency

    def summary(self) -> dict[str, Any]:
        return {
            "peers": len(self.peers),
            "addresses": len(self.addresses),
            "max_qty": {str(k): v for k, v in sorted(self.max_qty.items())},
            "slow_latency_ms": round(self.slow_latency_s * 1000, 3),
        }


def _node_frames(frames: pd.DataFrame, node: int) -> pd.DataFrame:
    mine = frames[(frames["src"] == node) | (frames["dst"] == node)].copy()
    mine["peer"] = np.where(mine["src"] == node, mine["dst"], mine["src"]).astype(np.uint32)
    mine["server"] = np.where(mine["is_request"], mine["dst"], mine["src"]).astype(np.uint32)
    return mine


def learn_vocabulary(frames: pd.DataFrame) -> Vocabulary:
    req = frames[frames["is_request"]]
    with_addr = req[req["addr"] >= 0]
    addresses = frozenset(
        zip(
            with_addr["server"].astype(int),
            with_addr["fc"].astype(int),
            with_addr["addr"].astype(int),
            strict=True,
        )
    )
    with_qty = req[req["qty"] >= 0]
    largest = with_qty.groupby("fc")["qty"].max()
    max_qty = {
        int(fc): int(q) for fc, q in zip(largest.index.to_numpy(), largest.to_numpy(), strict=True)
    }
    latency = req["latency_s"].dropna()
    slow = float(np.quantile(latency, 0.999)) if latency.size else float("inf")
    return Vocabulary(
        peers=frozenset(int(p) for p in frames["peer"].unique()),
        addresses=addresses,
        max_qty=max_qty,
        slow_latency_s=slow,
    )


def window_features(
    frames: pd.DataFrame,
    tcp: pd.DataFrame,
    windows: npt.NDArray[np.int64],
    node: int,
    vocab: Vocabulary,
    length: int,
) -> pd.DataFrame:
    """One row of :data:`FEATURES` per window index (frames already filtered to the node)."""
    out = pd.DataFrame(0.0, index=pd.Index(windows, name="w"), columns=list(FEATURES))
    if windows.size == 0:
        return out.astype(np.float32)
    f = frames.assign(w=np.floor(frames["ts"].to_numpy() / length).astype(np.int64))
    f = f[f["w"].isin(windows)]
    req = f[f["is_request"]]
    rsp = f[~f["is_request"]]

    def count(mask: pd.Series, frame: pd.DataFrame) -> pd.Series:
        counts: pd.Series = frame.loc[mask, "w"].value_counts()
        return counts

    sent = req["src"] == node
    cols: dict[str, pd.Series] = {
        "req_sent": count(sent, req),
        "req_recv": count(~sent, req),
        "req_read": count(req["fc"].isin(READ_FCS), req),
        "req_write": count(req["fc"].isin(WRITE_FCS), req),
        "req_other": count(~req["fc"].isin((*READ_FCS, *WRITE_FCS)), req),
        "resp_exception": count(rsp["fc"] >= 0x80, rsp),
        "peers": f.groupby("w")["peer"].nunique(),
        "stacked_frames": count((f["frame_index"] == 0) & (f["frames_in_packet"] > 1), f),
        "malformed_frames": count(f["len_mismatch"] | f["truncated"], f),
        "unanswered": count(req["latency_s"].isna(), req),
    }
    new_peer = ~f["peer"].isin(list(vocab.peers))
    cols["new_peers"] = f.loc[new_peer].groupby("w")["peer"].nunique()

    keys = list(
        zip(req["server"].astype(int), req["fc"].astype(int), req["addr"].astype(int), strict=True)
    )
    seen = np.fromiter((k in vocab.addresses for k in keys), dtype=bool, count=len(keys))
    unseen = pd.Series(~seen & (req["addr"].to_numpy() >= 0), index=req.index)
    cols["unseen_addr"] = count(unseen, req)
    cols["unseen_addr_write"] = count(unseen & req["fc"].isin(WRITE_FCS), req)
    limit_by_fc = np.full(256, -1, dtype=np.int64)
    for fc, qty in vocab.max_qty.items():
        limit_by_fc[fc] = qty
    limit = limit_by_fc[req["fc"].to_numpy(dtype=np.int64)]
    oversized = (req["qty"].to_numpy() >= 0) & (req["qty"].to_numpy() > limit)
    cols["oversized_qty"] = count(pd.Series(oversized, index=req.index), req)

    answered = req.dropna(subset=["latency_s"])
    cols["latency_median_ms"] = answered.groupby("w")["latency_s"].median() * 1000.0
    cols["slow_responses"] = count(answered["latency_s"] > vocab.slow_latency_s, answered)

    if not tcp.empty:
        t = tcp[(tcp["src"] == node) | (tcp["dst"] == node)]
        t = t.assign(w=np.floor(t["ts"].to_numpy() / length).astype(np.int64))
        t = t[t["w"].isin(windows)]
        cols["tcp_syn"] = t.loc[~t["rst"], "w"].value_counts()
        cols["tcp_rst"] = t.loc[t["rst"], "w"].value_counts()

    for name, series in cols.items():
        out[name] = series.reindex(out.index).fillna(0.0).to_numpy()
    return out.astype(np.float32)


def _per_window_count(
    times: npt.NDArray[np.float64], windows: npt.NDArray[np.int64], length: int
) -> npt.NDArray[np.int64]:
    if times.size == 0 or windows.size == 0:
        return np.zeros(windows.size, dtype=np.int64)
    idx, counts = np.unique(np.floor(times / length).astype(np.int64), return_counts=True)
    lookup = pd.Series(counts, index=idx)
    return lookup.reindex(windows).fillna(0).to_numpy(dtype=np.int64)


def _episode_of(
    windows: npt.NDArray[np.int64], episodes: list[tuple[float, float]], length: int
) -> npt.NDArray[np.int16]:
    result = np.full(windows.size, -1, dtype=np.int16)
    starts = windows * length
    for i, (a, b) in enumerate(episodes):
        result[(starts <= b) & (starts + length > a)] = i
    return result


# ---------------------------------------------------------------------------- partition


@dataclass
class NodeWindows:
    windows: npt.NDArray[np.int64]
    hmi_requests: npt.NDArray[np.int64]
    attacker: npt.NDArray[np.int64]
    dropped_idle: int
    dropped_idle_attack: int


def _node_windows(data: DeviceData, node: int, hmi: int, params: Params) -> NodeWindows:
    length = params.data.windows.site_b.length_s
    grid = grid_windows(data.spans, length)
    f = data.frames
    if f.empty:
        hmi_times = np.empty(0)
    else:
        hmi_req = f["is_request"] & (f["src"] == hmi) & ((f["dst"] == node) | (f["src"] == node))
        hmi_times = f.loc[hmi_req, "ts"].to_numpy()
    intervals = idle_intervals(hmi_times, data.spans, params.data.modbus.idle_gap_s)
    keep = drop_idle(grid, intervals, length)
    attacker_all = _per_window_count(data.watched, grid, length)
    kept = grid[keep]
    return NodeWindows(
        windows=kept,
        hmi_requests=_per_window_count(hmi_times, kept, length),
        attacker=attacker_all[keep],
        dropped_idle=int((~keep).sum()),
        dropped_idle_attack=int(((~keep) & (attacker_all > 0)).sum()),
    )


def _table(
    nw: NodeWindows,
    mask: npt.NDArray[np.bool_],
    features: pd.DataFrame,
    episodes: list[tuple[float, float]],
    length: int,
) -> pd.DataFrame:
    windows = nw.windows[mask]
    table = features.loc[windows].reset_index(drop=True)
    table.insert(0, "window_start", windows * length)
    table.insert(1, "time", pd.to_datetime(windows * length, unit="s", utc=True))
    table["hmi_requests"] = nw.hmi_requests[mask]
    table["attacker_packets"] = nw.attacker[mask]
    table["label"] = (nw.attacker[mask] > 0).astype(np.int8)
    table["episode"] = _episode_of(windows, episodes, length)
    return table


def _summary(table: pd.DataFrame, sha: str) -> dict[str, Any]:
    episodes = sorted(int(e) for e in table.loc[table["label"] > 0, "episode"].unique() if e >= 0)
    return {
        "windows": len(table),
        "attack_windows": int(table["label"].sum()),
        "episodes": episodes,
        "start": iso(int(table["window_start"].min())) if len(table) else None,
        "end": iso(int(table["window_start"].max())) if len(table) else None,
        "sha256": sha,
    }


def partition(params: Params, interim_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Build the Site B partitions from the ``prepare`` output; returns the manifest."""
    cfg = params.data.modbus
    length = params.data.windows.site_b.length_s
    gap = params.data.splits.gap_windows
    interim = json.loads((interim_dir / "modbus" / INTERIM_MANIFEST).read_text(encoding="utf-8"))
    hmi = ip_to_int(cfg.devices["scada-hmi"]) if "scada-hmi" in cfg.devices else -1

    attack_data = {
        device: load_device(interim_dir, "attack", device, interim["captures"]["attack"][device])
        for device in cfg.devices
    }
    attacker_times = np.concatenate([d.watched for d in attack_data.values()])
    episodes = cluster_times(attacker_times, cfg.episode_gap_s)
    if len(episodes) <= cfg.attack_val_episodes:
        raise ValueError(
            f"{len(episodes)} attacker episodes found; attack_val_episodes="
            f"{cfg.attack_val_episodes} would leave none for test"
        )
    k = cfg.attack_val_episodes
    boundary_t = (episodes[k - 1][1] + episodes[k][0]) / 2.0
    boundary_w = int(np.floor(boundary_t / length))

    nodes: dict[str, Any] = {}
    for client, device in params.data.partition.site_b.clients.items():
        node = ip_to_int(cfg.devices[device])
        print(f"  Site B {client} ({device})", flush=True)
        benign = load_device(interim_dir, "benign", device, interim["captures"]["benign"][device])
        b_frames = _node_frames(benign.frames, node)
        b_win = _node_windows(benign, node, hmi, params)
        n = b_win.windows.size
        cut = round(n * (1.0 - params.data.splits.normal_val_fraction))
        idx = np.arange(n)
        train_mask = idx < cut
        val_mask = idx >= cut + gap
        train_w = b_win.windows[train_mask]
        in_train = np.isin(np.floor(b_frames["ts"].to_numpy() / length).astype(np.int64), train_w)
        vocab = learn_vocabulary(b_frames[in_train])
        b_feat = window_features(b_frames, benign.tcp, b_win.windows, node, vocab, length)

        attack = attack_data[device]
        a_frames = _node_frames(attack.frames, node)
        a_win = _node_windows(attack, node, hmi, params)
        a_feat = window_features(a_frames, attack.tcp, a_win.windows, node, vocab, length)
        val_attack_mask = (
            a_win.windows < boundary_w - (gap - 1) if gap > 0 else a_win.windows < boundary_w
        )
        test_mask = a_win.windows > boundary_w if gap > 0 else a_win.windows >= boundary_w

        tables = {
            "train": _table(b_win, train_mask, b_feat, [], length),
            "val_normal": _table(b_win, val_mask, b_feat, [], length),
            "val_attack": _table(a_win, val_attack_mask, a_feat, episodes, length),
            "test": _table(a_win, test_mask, a_feat, episodes, length),
        }
        summaries = {}
        for part in PARTITIONS:
            sha = write_table(tables[part], out_dir / client / f"{part}.parquet")
            summaries[part] = _summary(tables[part], sha)
        nodes[client] = {
            "device": device,
            "ip": cfg.devices[device],
            "vocabulary": vocab.summary(),
            "idle_windows_dropped": {
                "benign": b_win.dropped_idle,
                "attack_day": a_win.dropped_idle,
                "attack_day_with_attacker_traffic": a_win.dropped_idle_attack,
            },
            "attack_day_attack_windows": int((a_win.attacker > 0).sum()),
            "partitions": summaries,
        }

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "site": "site_b",
        "dataset": "CIC Modbus 2023",
        "config": {
            "modbus": cfg.model_dump(mode="json"),
            "window": params.data.windows.site_b.model_dump(mode="json"),
            "splits": params.data.splits.model_dump(mode="json"),
            "partition": params.data.partition.site_b.model_dump(mode="json"),
        },
        "features": {
            "n_features": len(FEATURES),
            "features": list(FEATURES),
            "latency_features": list(LATENCY_FEATURES),
        },
        "sources": {
            split: {
                device: [
                    {"file": e["file"], "sha256": e["sha256"], "packets": e["packets"]}
                    for e in entries
                ]
                for device, entries in interim["captures"][split].items()
            }
            for split in SPLITS
        },
        "episodes": [
            {
                "episode": i,
                "start": iso(a),
                "end": iso(b),
                "duration_s": round(b - a, 1),
                "partition": "val_attack" if i < k else "test",
            }
            for i, (a, b) in enumerate(episodes)
        ],
        "attack_split_boundary": iso(boundary_w * length),
        "nodes": nodes,
    }
    write_manifest(manifest, out_dir)
    return manifest
