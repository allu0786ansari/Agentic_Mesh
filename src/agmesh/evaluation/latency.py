"""Latency per segment from metric events (Proposal §8.6; Plan Week 8 validation).

    python -m agmesh.evaluation.latency --segment detection --split val
    python -m agmesh.evaluation.latency --segment detection --runs results/edge/site_b__val__*

The detection segment is the wall time from a window's emission by the replay to its
alert committed on the bus. It is read from the ``detection`` events that the edge
runtime writes to ``events.jsonl``. The latest run of each site for the split is used
unless ``--runs`` names the runs. The report is written as an artifact with p50, p95
and max per site, against the target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.common.artifacts import write_artifact

TARGETS_MS = {"detection": 1000.0}
EVENT = {"detection": "detection"}


def _latest(root: Path, site: str, split: str) -> Path | None:
    runs = sorted(p for p in root.glob(f"{site}__{split}__*") if (p / "events.jsonl").is_file())
    return runs[-1] if runs else None


def collect(run_dir: Path, segment: str) -> tuple[list[float], set[str]]:
    values, sources = [], set()
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["event"] == EVENT[segment]:
            values.append(float(event["payload"]["latency_ms"]))
            sources.add(event["data_source"])
    return values, sources


def report(runs: list[Path], segment: str) -> dict[str, Any]:
    out: dict[str, Any] = {"segment": segment, "target_p95_ms": TARGETS_MS[segment], "runs": {}}
    for run in runs:
        values, sources = collect(run, segment)
        a = np.asarray(values)
        out["runs"][run.name] = {
            "events": int(a.size),
            "p50_ms": float(np.percentile(a, 50)) if a.size else None,
            "p95_ms": float(np.percentile(a, 95)) if a.size else None,
            "max_ms": float(a.max()) if a.size else None,
            "meets_target": bool(a.size and np.percentile(a, 95) <= TARGETS_MS[segment]),
            "data_source": sorted(sources),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Latency per segment from metric events")
    parser.add_argument("--segment", choices=tuple(TARGETS_MS), default="detection")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--root", type=Path, default=Path("results/edge"))
    parser.add_argument("--runs", type=Path, nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/latency"))
    args = parser.parse_args(argv)
    runs = args.runs or [
        r
        for r in (_latest(args.root, s, args.split) for s in ("site_a", "site_b"))
        if r is not None
    ]
    if not runs:
        raise SystemExit(f"no edge runs with events under {args.root} for split {args.split}")
    rep = report(runs, args.segment)
    sources = {s for r in rep["runs"].values() for s in r["data_source"]}
    path = write_artifact(
        args.out / f"{args.segment}__{args.split}.json",
        kind="latency.report",
        data_source="runtime" if sources == {"runtime"} else "fixture",
        payload=rep,
    )
    for name, r in rep["runs"].items():
        print(
            f"{name}: {r['events']} events, p50 {r['p50_ms']:.2f} ms, p95 {r['p95_ms']:.2f} ms "
            f"(target {TARGETS_MS[args.segment]:.0f} ms) -> "
            f"{'OK' if r['meets_target'] else 'MISSED'}"
        )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
