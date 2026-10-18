"""Latency per segment from metric events (Proposal §8.6; Plan Week 8 validation).

    python -m agmesh.evaluation.latency --segment detection --split val
    python -m agmesh.evaluation.latency --segment detection --runs results/edge/site_b__val__*
    python -m agmesh.evaluation.latency --report          # X5 (ADR 0017 §5)

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


PT7_MS = {
    "detection": 1000.0,
    "investigation": 2000.0,
    "incident_to_dispatch_retrieval_only": 3000.0,
    "incident_to_dispatch_full": 24000.0,
}


def x5_report(edge_root: Path, mesh_root: Path, split: str = "val") -> dict[str, Any]:
    """X5: p50/p95 per segment and site from committed run summaries (ADR 0017 §5).

    Detection comes from the latest edge run per site. Investigation and
    incident-to-dispatch come from the latest mesh runs over mTLS (``nodes = https``),
    per mode.
    """
    sites: dict[str, Any] = {}
    for site in ("site_a", "site_b"):
        row: dict[str, Any] = {}
        edges = sorted(
            p for p in edge_root.glob(f"{site}__{split}__*") if (p / "summary.json").is_file()
        )
        if edges:
            e = json.loads((edges[-1] / "summary.json").read_text())
            d = e["detection_latency_ms"]
            row["detection"] = {
                "run": edges[-1].name,
                "n": d["n"],
                "p50_ms": d["p50"],
                "p95_ms": d["p95"],
            }
        for mode in ("retrieval_only", "full"):
            runs = []
            for p in sorted(mesh_root.glob(f"{site}__{mode}__{split}__*/summary.json")):
                s = json.loads(p.read_text())
                if s.get("nodes") == "https":
                    runs.append((p.parent.name, s))
            if not runs:
                continue
            name, s = runs[-1]
            lat = s["latency_ms"]
            if mode == "full" or "investigation" not in row:
                row["investigation" if mode == "full" else "investigation_retrieval_only"] = {
                    "run": name, "n": lat["investigation"]["n"],
                    "p50_ms": lat["investigation"]["p50"], "p95_ms": lat["investigation"]["p95"],
                }  # fmt: skip
            row[f"incident_to_dispatch_{mode}"] = {
                "run": name,
                "n": lat["incident_to_dispatch"]["n"],
                "p50_ms": lat["incident_to_dispatch"]["p50"],
                "p95_ms": lat["incident_to_dispatch"]["p95"],
            }
        for seg, v in row.items():
            target = PT7_MS.get(seg) or PT7_MS.get(seg.removesuffix("_retrieval_only"))
            v["target_p95_ms"] = target
            v["meets_target"] = bool(target is not None and v["p95_ms"] <= target)
        sites[site] = row
    return {
        "split": split,
        "protocol": "docs/adr/0017-x3-x6-week13.md §5",
        "sites": sites,
        "note": "replay runs incidents back to back; queueing is not measured",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Latency per segment from metric events")
    parser.add_argument("--segment", choices=tuple(TARGETS_MS), default="detection")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--root", type=Path, default=Path("results/edge"))
    parser.add_argument("--runs", type=Path, nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/latency"))
    parser.add_argument("--report", action="store_true", help="X5: every segment (ADR 0017 §5)")
    parser.add_argument("--mesh-root", type=Path, default=Path("results/mesh"))
    args = parser.parse_args(argv)
    if args.report:
        rep = x5_report(args.root, args.mesh_root, args.split)
        path = write_artifact(
            Path("results/X5") / f"latency_{args.split}.json",
            kind="X5.latency",
            data_source="runtime",
            payload=rep,
        )
        for site, row in rep["sites"].items():
            for seg, v in row.items():
                verdict = "OK" if v["meets_target"] else "MISSED"
                print(
                    f"{site} {seg}: p50 {v['p50_ms']:.1f} ms, p95 {v['p95_ms']:.1f} ms "
                    f"(target {v['target_p95_ms']:.0f}) {verdict}"
                )
        print(path)
        return 0
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
