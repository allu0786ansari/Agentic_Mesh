"""ADR 0004: choose the non-private training budget with a label-free rule.

    python -m agmesh.experiments.training_budget

For each site and each candidate ``max_epochs`` it trains the neural model locally
(every node) and centrally for every seed, and records how each run ended. The rule
declared in ADR 0004 picks the smallest budget for which early stopping, not the cap,
ends at least 95% of runs. Detection metrics are recorded alongside as a sensitivity
analysis only. Output: ``results/X1/training_budget.json``; test data is never read.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import DEFAULT_PARAMS_PATH, Params, load_params
from agmesh.data.access import Site
from agmesh.evaluation.detection import macro
from agmesh.experiments import x1_baselines as x1

BUDGETS = {"site_a": (100, 200, 400), "site_b": (30, 60, 120)}
KIND = {"site_a": "vae", "site_b": "ae"}
SECTION = {"site_a": "vae", "site_b": "ae"}
CAPPED_SHARE_MAX = 0.05
KEYS = ("auroc", "auprc", "event_recall", "window_fpr", "false_alarms_per_hour")


def _with_budget(params: Params, site: Site, max_epochs: int) -> Params:
    raw = params.model_dump()
    raw["models"][SECTION[site]]["max_epochs"] = max_epochs
    return Params.model_validate(raw)


def capped(info: dict[str, Any], max_epochs: int, patience: int) -> bool:
    """A run is capped if its best epoch lies within ``patience`` epochs of the cap."""
    return int(info["best_epoch"]) > max_epochs - patience


def choose(rows: list[dict[str, Any]]) -> int:
    """Smallest budget whose share of capped runs is at most CAPPED_SHARE_MAX."""
    for row in sorted(rows, key=lambda r: r["max_epochs"]):
        if row["capped_share"] <= CAPPED_SHARE_MAX:
            return int(row["max_epochs"])
    return int(max(r["max_epochs"] for r in rows))


def run(
    params: Params,
    sites: list[Site],
    partitions: Path,
    seeds: list[int],
    out: Path,
    *,
    data_source: str,
    params_path: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for site in sites:
        data = x1._load(params, site, partitions)
        kind = KIND[site]
        patience = getattr(params.models, SECTION[site]).patience
        rows = []
        for budget in BUDGETS[site]:
            p = _with_budget(params, site, budget)
            runs: list[dict[str, Any]] = []
            detection: dict[str, list[dict[str, Any]]] = {"local": [], "central": []}
            for seed in seeds:
                for mode in ("local", "central"):
                    fn = x1.run_local if mode == "local" else x1.run_central
                    per_client, training = fn(site, kind, data, p, seed)
                    detection[mode].append(macro(per_client, KEYS))
                    for node, info in training.items():
                        runs.append(
                            {
                                "seed": seed,
                                "mode": mode,
                                "node": node,
                                "epochs_run": info["epochs_run"],
                                "best_epoch": info["best_epoch"],
                                "best_val_loss": info["best_val_loss"],
                                "capped": capped(info, budget, patience),
                            }
                        )
            share = float(np.mean([r["capped"] for r in runs]))
            summary = {
                mode: {
                    k: float(np.mean([d[k] for d in ds if d[k] is not None]))
                    for k in KEYS
                    if any(d[k] is not None for d in ds)
                }
                for mode, ds in detection.items()
            }
            row = {
                "max_epochs": budget,
                "patience": patience,
                "runs": runs,
                "capped_share": share,
                "mean_best_epoch": float(np.mean([r["best_epoch"] for r in runs])),
                "detection_mean_over_seeds": summary,
            }
            rows.append(row)
            print(
                f"{site} max_epochs {budget:>3}: capped {share:.0%} of {len(runs)} runs; "
                f"mean best epoch {row['mean_best_epoch']:.0f}; "
                f"local AUROC {summary['local'].get('auroc', float('nan')):.3f} "
                f"FPR {summary['local'].get('window_fpr', float('nan')):.3f}; "
                f"central AUROC {summary['central'].get('auroc', float('nan')):.3f} "
                f"FPR {summary['central'].get('window_fpr', float('nan')):.3f}",
                flush=True,
            )
        chosen = choose(rows)
        report[site] = {"model": kind, "budgets": rows, "chosen_max_epochs": chosen}
        print(f"{site}: chosen max_epochs {chosen}", flush=True)
    write_artifact(
        out,
        kind="X1.training_budget",
        data_source=data_source,  # type: ignore[arg-type]
        payload={"protocol": "docs/adr/0004-training-budget.md", "seeds": seeds, "sites": report},
        params_path=params_path,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ADR 0004 training-budget check")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument(
        "--sites", nargs="+", choices=("site_a", "site_b"), default=["site_a", "site_b"]
    )
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/X1/training_budget.json"))
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)
    params = load_params(args.params)
    run(
        params,
        args.sites,
        args.partitions or params.paths.partitions_dir,
        args.seeds or params.project.seeds,
        args.output,
        data_source="fixture" if args.fixture else "runtime",
        params_path=args.params,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
