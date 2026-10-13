"""Week 1 feasibility spike: inspect the raw datasets for Gate G0.

HAI (Site A)
    Files, feature list consistency, sampling interval, missing values, labelled
    attack events per test file, and the features whose 6-hour block means vary most
    between blocks (candidate "operating regime" variables for the Site A partition,
    Proposal §4.1). Git LFS pointer files are reported instead of read.

CIC Modbus 2023 (Site B)
    Every capture's time span and packet count, grouped by split (benign/attack) and
    device; how much the devices' captures overlap in time (needed for evidence
    fusion, RQ2); any attack-log or label files; and any MITRE ATT&CK for ICS
    technique IDs found in text files shipped with the dataset.

The output is a feasibility artifact; it reports facts and suggestions. The G0
decisions themselves are recorded in docs/adr/0001-g0-decisions.md.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agmesh.common.artifacts import write_artifact
from agmesh.common.checksums import is_lfs_pointer
from agmesh.common.config import DEFAULT_PARAMS_PATH, load_params
from agmesh.feasibility.captures import (
    CaptureFormatError,
    Interval,
    capture_span,
    intersect,
    merge_intervals,
    total_length,
)

TIME_COLUMNS = ("time", "timestamp", "datetime")
CAPTURE_SUFFIXES = (".pcap", ".pcapng", ".cap")
TEXT_SUFFIXES = (".csv", ".txt", ".md", ".json", ".log", ".html", ".htm", ".tsv")
MITRE_ICS_ID = re.compile(r"\bT0\d{3}\b")
MAX_TEXT_SCAN_BYTES = 50 * 2**20
MIN_FUSION_OVERLAP_S = 600.0  # at least 10 minutes of shared attack time to evaluate fusion


# --------------------------------------------------------------------------- HAI


def _natural_key(path: Path) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def find_hai_files(root: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {"train": [], "test": []}
    for split in files:
        files[split] = sorted(
            (p for p in root.glob(f"{split}*") if p.is_file() and ".csv" in p.name),
            key=_natural_key,
        )
    return files


def _sniff_separator(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
    return ";" if first.count(";") > first.count(",") else ","


def split_columns(columns: list[str]) -> tuple[str | None, list[str], list[str]]:
    """Return (time column, label columns, feature columns)."""
    time_col = next((c for c in columns if c.strip().lower() in TIME_COLUMNS), None)
    labels = [
        c for c in columns if c.strip().lower().startswith("attack") or c.strip().lower() == "label"
    ]
    features = [c for c in columns if c != time_col and c not in labels]
    return time_col, labels, features


def label_events(label: np.ndarray, seconds: np.ndarray) -> list[dict[str, float]]:
    """Maximal contiguous runs where ``label`` is non-zero, with start time and duration."""
    active = label.astype(bool)
    if not active.any():
        return []
    padded = np.concatenate([[False], active, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    events = []
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        events.append(
            {
                "start_s": float(seconds[start]),
                "duration_s": float(seconds[stop - 1] - seconds[start]) + 1.0,
                "rows": int(stop - start),
            }
        )
    return events


def regime_candidates(
    block_means: list[np.ndarray], block_stds: list[np.ndarray], features: list[str], top: int = 10
) -> list[dict[str, float | str]]:
    """Features whose block means vary most relative to their within-block variation."""
    if len(block_means) < 2:
        return []
    means = np.vstack(block_means)
    within = np.nanmean(np.vstack(block_stds), axis=0)
    between = np.nanstd(means, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(within > 1e-9, between / within, np.where(between > 1e-9, np.inf, 0.0))
    order = np.argsort(-np.nan_to_num(ratio, nan=-1.0, posinf=1e12))[:top]
    return [
        {"feature": features[i], "between_within_ratio": round(float(ratio[i]), 3)}
        for i in order
        if np.isfinite(ratio[i]) and ratio[i] > 0
    ]


def inspect_hai_file(
    path: Path, split: str, block_hours: int
) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    report: dict[str, Any] = {"file": path.name, "size_bytes": path.stat().st_size}
    if is_lfs_pointer(path):
        report["status"] = "lfs_pointer"
        return report, [], []
    frame = pd.read_csv(path, sep=_sniff_separator(path), low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    time_col, label_cols, features = split_columns(list(frame.columns))
    report.update(
        status="ok",
        rows=len(frame),
        n_features=len(features),
        time_column=time_col,
        label_columns=label_cols,
        missing_values=int(frame[features].isna().sum().sum()),
    )
    block_means: list[np.ndarray] = []
    block_stds: list[np.ndarray] = []
    if time_col is None:
        report["warning"] = "no time column found"
        return report, block_means, block_stds
    times = pd.to_datetime(frame[time_col], errors="coerce")
    valid = times.notna().to_numpy()
    seconds = (times - times.min()).dt.total_seconds().to_numpy()
    deltas = np.diff(seconds[valid])
    report.update(
        start=str(times.min()),
        end=str(times.max()),
        median_interval_s=float(np.median(deltas)) if deltas.size else None,
        non_monotonic_steps=int((deltas <= 0).sum()),
        unparseable_times=int((~valid).sum()),
    )
    any_attack = np.zeros(len(frame), dtype=bool)
    if label_cols:
        labels = frame[label_cols].fillna(0).to_numpy(dtype=float)
        any_attack = (labels > 0).any(axis=1)
        events = label_events(any_attack, seconds)
        durations = [e["duration_s"] for e in events]
        report["attack_rows"] = int(any_attack.sum())
        report["events"] = {
            "count": len(events),
            "duration_s_min": min(durations) if durations else None,
            "duration_s_median": float(np.median(durations)) if durations else None,
            "duration_s_max": max(durations) if durations else None,
        }
        report["events_per_label_column"] = {
            col: len(label_events(frame[col].fillna(0).to_numpy(dtype=float) > 0, seconds))
            for col in label_cols
        }
    if split == "train" and deltas.size:
        # Summarise fixed-length blocks of normal operation for the regime analysis.
        # A block counts if it holds at least half of the rows it should.
        values = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        usable = valid & ~any_attack
        block_ids = np.where(valid, seconds // (block_hours * 3600), -1).astype(int)
        expected_rows = block_hours * 3600 / max(float(np.median(deltas)), 1e-9)
        for block in np.unique(block_ids[usable]):
            rows = values[(block_ids == block) & usable]
            if len(rows) >= 0.5 * expected_rows:
                block_means.append(np.nanmean(rows, axis=0))
                block_stds.append(np.nanstd(rows, axis=0))
        report["complete_blocks"] = len(block_means)
    report["feature_names"] = features
    return report, block_means, block_stds


def inspect_hai(root: Path, block_hours: int, n_clients: int) -> dict[str, Any]:
    if not root.is_dir():
        return {"root": root.as_posix(), "status": "missing"}
    files = find_hai_files(root)
    reports: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    all_means: list[np.ndarray] = []
    all_stds: list[np.ndarray] = []
    feature_lists: list[list[str]] = []
    for split, paths in files.items():
        for path in paths:
            print(f"  HAI {split}: {path.name}", flush=True)
            report, means, stds = inspect_hai_file(path, split, block_hours)
            if "feature_names" in report:
                feature_lists.append(report.pop("feature_names"))
            reports[split].append(report)
            all_means += means
            all_stds += stds
    features = feature_lists[0] if feature_lists else []
    consistent = all(f == features for f in feature_lists)
    test_ok = [r for r in reports["test"] if r.get("status") == "ok"]
    n_events = sum(r.get("events", {}).get("count", 0) for r in test_ok)
    return {
        "root": root.as_posix(),
        "status": "ok" if test_ok else "no_readable_test_files",
        "release_hint": root.name,
        "files": reports,
        "lfs_pointer_files": [
            r["file"]
            for split in reports.values()
            for r in split
            if r.get("status") == "lfs_pointer"
        ],
        "n_features": len(features),
        "feature_lists_consistent": consistent,
        "features": features,
        "test_attack_events": n_events,
        "regime_analysis": {
            "block_hours": block_hours,
            "complete_train_blocks": len(all_means),
            "enough_blocks_for_clients": len(all_means) >= 2 * n_clients,
            "candidates": regime_candidates(all_means, all_stds, features) if consistent else [],
        },
    }


# ------------------------------------------------------------------- CIC Modbus


def classify_capture(rel: Path, devices: list[str]) -> tuple[str, str]:
    """Return (split, device) for a capture path relative to the dataset root."""
    parts = [p.lower() for p in rel.parts[:-1]]
    split = "attack" if "attack" in parts else "benign" if "benign" in parts else "unknown"
    normalised = {d.lower().replace("_", "-"): d for d in devices}
    for part in parts:
        key = part.replace("_", "-")
        if key in normalised:
            return split, normalised[key]
    for part in parts:  # fall back to the directory directly below the split folder
        if part not in ("attack", "benign"):
            return split, part
    return split, "unknown"


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).isoformat() if ts is not None else None


def scan_mitre_ids(paths: list[Path]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in paths:
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > MAX_TEXT_SCAN_BYTES:
            continue
        ids = sorted(set(MITRE_ICS_ID.findall(path.read_text(encoding="utf-8", errors="replace"))))
        if ids:
            found[path.name] = ids
    return found


def overlap_report(groups: dict[str, list[Interval]]) -> dict[str, Any]:
    merged = {device: merge_intervals(iv) for device, iv in groups.items()}
    devices = sorted(merged)
    pairwise = {
        f"{a}|{b}": round(total_length(intersect(merged[a], merged[b])), 1)
        for a, b in combinations(devices, 2)
    }
    all_devices: float | None = None
    if len(devices) >= 2:
        common = merged[devices[0]]
        for device in devices[1:]:
            common = intersect(common, merged[device])
        all_devices = round(total_length(common), 1)
    return {
        "span_s_by_device": {d: round(total_length(merged[d]), 1) for d in devices},
        "pairwise_overlap_s": pairwise,
        "all_devices_overlap_s": all_devices,
    }


def inspect_modbus(root: Path, devices: list[str]) -> dict[str, Any]:
    if not root.is_dir():
        return {"root": root.as_posix(), "status": "missing"}
    all_files = sorted(p for p in root.rglob("*") if p.is_file())
    captures = [p for p in all_files if p.suffix.lower() in CAPTURE_SUFFIXES]
    others = [p for p in all_files if p not in captures]
    records: list[dict[str, Any]] = []
    intervals: dict[str, dict[str, list[Interval]]] = defaultdict(lambda: defaultdict(list))
    for i, path in enumerate(captures, start=1):
        rel = path.relative_to(root)
        split, device = classify_capture(rel, devices)
        print(f"  Modbus [{i}/{len(captures)}] {rel.as_posix()}", flush=True)
        record: dict[str, Any] = {
            "file": rel.as_posix(),
            "split": split,
            "device": device,
            "size_bytes": path.stat().st_size,
        }
        if is_lfs_pointer(path):
            record["status"] = "lfs_pointer"
        else:
            try:
                span = capture_span(path)
            except CaptureFormatError as exc:
                record.update(status="unreadable", error=str(exc))
            else:
                record.update(
                    status="ok",
                    format=span.fmt,
                    packets=span.n_packets,
                    start=_iso(span.first_ts),
                    end=_iso(span.last_ts),
                    duration_s=round(span.duration_s, 1) if span.duration_s is not None else None,
                    truncated=span.truncated,
                )
                if span.first_ts is not None and span.last_ts is not None:
                    intervals[split][device].append((span.first_ts, span.last_ts))
        records.append(record)

    log_like = [
        p.relative_to(root).as_posix()
        for p in others
        if p.suffix.lower() in (".csv", ".txt", ".log", ".json", ".xlsx", ".xls", ".tsv")
    ]
    manual = [
        p.relative_to(root).as_posix()
        for p in others
        if p.suffix.lower() in (".pdf", ".docx", ".xlsx")
    ]
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        counts[r["split"]][r["device"]] += 1
    return {
        "root": root.as_posix(),
        "status": "ok" if captures else "no_captures",
        "n_captures": len(captures),
        "captures_by_split_and_device": {s: dict(d) for s, d in counts.items()},
        "captures": records,
        "overlap": {split: overlap_report(groups) for split, groups in intervals.items()},
        "non_capture_files": [p.relative_to(root).as_posix() for p in others],
        "possible_label_or_log_files": log_like,
        "mitre_ids_in_text_files": scan_mitre_ids(others),
        "files_needing_manual_check": manual,
    }


# -------------------------------------------------------------------- G0 hints


def gate_hints(hai: dict[str, Any], modbus: dict[str, Any]) -> dict[str, Any]:
    """Suggestions for Gate G0, derived mechanically from the inspection.

    These are inputs to the decision, not the decision: the ADR records what is
    actually chosen and why.
    """
    attack_overlap = modbus.get("overlap", {}).get("attack", {})
    pairwise = attack_overlap.get("pairwise_overlap_s", {})
    best_pair = max(pairwise.values(), default=0.0)
    has_logs = bool(modbus.get("possible_label_or_log_files"))
    has_mitre = bool(modbus.get("mitre_ids_in_text_files"))
    candidates = hai.get("regime_analysis", {}).get("candidates", [])
    return {
        "hai_release": hai.get("release_hint"),
        "hai_usable": hai.get("status") == "ok" and not hai.get("lfs_pointer_files"),
        "site_a_partition_scheme": (
            "regime"
            if candidates and hai.get("regime_analysis", {}).get("enough_blocks_for_clients")
            else "time"
        ),
        "rq2_fusion_data": "real_overlap"
        if best_pair >= MIN_FUSION_OVERLAP_S
        else "synthetic_composition",
        "rq2_best_pairwise_attack_overlap_s": best_pair,
        "site_b_attack_labels": "attack_logs_available" if has_logs else "capture_level_only",
        "site_b_attribution_ground_truth": (
            "dataset_mitre_mapping_found" if has_mitre else "needs_mapping_protocol_or_manual_check"
        ),
    }


# ------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the raw HAI and CIC Modbus datasets")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--hai-dir", type=Path, default=None)
    parser.add_argument("--modbus-dir", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("results/feasibility/data_inspection.json")
    )
    args = parser.parse_args(argv)

    params = load_params(args.params)
    hai_dir = args.hai_dir or params.paths.hai_dir
    modbus_dir = args.modbus_dir or params.paths.modbus_dir
    site_a = params.data.partition.site_a
    devices = list(params.data.partition.site_b.clients.values())

    print(f"Inspecting HAI at {hai_dir}")
    hai = inspect_hai(hai_dir, site_a.block_hours, len(site_a.clients))
    print(f"Inspecting CIC Modbus 2023 at {modbus_dir}")
    modbus = inspect_modbus(modbus_dir, devices)
    hints = gate_hints(hai, modbus)
    path = write_artifact(
        args.output,
        kind="feasibility.data_inspection",
        data_source="runtime",
        payload={"hai": hai, "modbus": modbus, "g0_hints": hints},
        params_path=args.params,
    )
    print(f"\nWrote {path}")
    for key, value in hints.items():
        print(f"  {key}: {value}")
    missing = [
        name for name, rep in (("HAI", hai), ("Modbus", modbus)) if rep.get("status") == "missing"
    ]
    if missing:
        print(f"WARNING: not found: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
