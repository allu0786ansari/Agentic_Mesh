"""Summarise multi-seed result artifacts: mean and 95% bootstrap CI per configuration.

    python -m agmesh.experiments.summarise X1            # -> results/X1/summary_val.json

Reads every ``results/<experiment>/<config_id>/seed_<n>.json`` of one split, groups by
configuration, and reports for each metric the mean over seeds, a percentile
bootstrap CI and the per-seed values. Selected paired comparisons (same seeds) use
the paired bootstrap and the exact Wilcoxon signed-rank test (Proposal §10.6).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import DEFAULT_PARAMS_PATH
from agmesh.evaluation.stats import bootstrap_ci, paired_bootstrap, wilcoxon_signed_rank

METRICS = (
    "auroc",
    "auprc",
    "event_recall",
    "range_f1",
    "window_fpr",
    "ttd_median_s",
    "false_alarms_per_hour",
)
#: (label, config A, config B, metric): is A better than B on the same seeds?
COMPARISONS = (
    ("federation headroom, Site A", "site_a__central__vae", "site_a__local__vae", "auroc"),
    ("FPR transfer, Site A", "site_a__local__vae", "site_a__central__vae", "window_fpr"),
    ("VAE vs PCA, Site A central", "site_a__central__vae", "site_a__central__pca", "auroc"),
    ("VAE vs PCA, Site A local", "site_a__local__vae", "site_a__local__pca", "auroc"),
    ("AE vs PCA FPR, Site B local", "site_b__local__ae", "site_b__local__pca", "window_fpr"),
)


def load(results: Path, experiment: str, split: str) -> dict[str, dict[int, dict[str, Any]]]:
    runs: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for path in sorted((results / experiment).glob("*/seed_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("split") == split:
            runs[doc["config_id"]][int(doc["seed"])] = doc
    return dict(runs)


def summarise(runs: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for config, by_seed in sorted(runs.items()):
        seeds = sorted(by_seed)
        entry: dict[str, Any] = {"seeds": seeds}
        for metric in METRICS:
            values = [by_seed[s]["metrics"].get(metric) for s in seeds]
            finite = [float(v) for v in values if v is not None and np.isfinite(v)]
            if not finite:
                entry[metric] = None
                continue
            ci = bootstrap_ci(finite, seed=0)
            entry[metric] = {
                "mean": ci.estimate,
                "low": ci.low,
                "high": ci.high,
                "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "values": finite,
            }
        table[config] = entry
    return table


def compare(runs: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for label, a, b, metric in COMPARISONS:
        if a not in runs or b not in runs:
            continue
        seeds = sorted(set(runs[a]) & set(runs[b]))
        xa = [runs[a][s]["metrics"].get(metric) for s in seeds]
        xb = [runs[b][s]["metrics"].get(metric) for s in seeds]
        pairs = [(x, y) for x, y in zip(xa, xb, strict=True) if x is not None and y is not None]
        if len(pairs) < 2:
            continue
        pa, pb = np.array(pairs).T
        boot = paired_bootstrap(pa, pb, seed=0)
        wil = wilcoxon_signed_rank(pa, pb)
        out.append(
            {
                "comparison": label,
                "a": a,
                "b": b,
                "metric": metric,
                "seeds": len(pairs),
                "mean_difference": boot.mean_difference,
                "ci": [boot.ci.low, boot.ci.high],
                "p_paired_bootstrap": boot.p_value,
                "p_wilcoxon": wil.p_value,
            }
        )
    return out


def _cell(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "–"
    return f"{entry['mean']:.3f} [{entry['low']:.3f}, {entry['high']:.3f}]"


def markdown(table: dict[str, Any], comparisons: list[dict[str, Any]]) -> str:
    keys = ("auroc", "auprc", "event_recall", "window_fpr", "false_alarms_per_hour")
    lines = ["| Config | n | " + " | ".join(keys) + " |", "|---" * (len(keys) + 2) + "|"]
    for config, entry in table.items():
        cells = " | ".join(_cell(entry.get(k)) for k in keys)
        lines.append(f"| {config} | {len(entry['seeds'])} | {cells} |")
    if comparisons:
        lines += ["", "| Comparison | metric | mean diff [95% CI] | p (bootstrap) | p (Wilcoxon) |"]
        lines.append("|---|---|---|---|---|")
        for c in comparisons:
            lines.append(
                f"| {c['comparison']} | {c['metric']} | {c['mean_difference']:+.3f} "
                f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | {c['p_paired_bootstrap']:.3f} | "
                f"{c['p_wilcoxon']:.4f} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise multi-seed result artifacts")
    parser.add_argument("experiment", default="X1", nargs="?")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    args = parser.parse_args(argv)
    runs = load(args.results, args.experiment, args.split)
    if not runs:
        print("no result artifacts found")
        return 1
    table = summarise(runs)
    comparisons = compare(runs)
    sources = {r["data_source"] for by_seed in runs.values() for r in by_seed.values()}
    write_artifact(
        args.results / args.experiment / f"summary_{args.split}.json",
        kind=f"{args.experiment}.summary",
        data_source="runtime" if sources == {"runtime"} else "fixture",
        payload={"split": args.split, "configs": table, "comparisons": comparisons},
        params_path=args.params,
    )
    print(markdown(table, comparisons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
