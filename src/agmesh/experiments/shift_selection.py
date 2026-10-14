"""ADR 0005: choose the shift-robust scoring candidate on validation, once.

    python -m agmesh.experiments.shift_selection          # -> results/X1/shift_selection.json

For every seed, the site's neural model (VAE on Site A), PCA and Isolation Forest
are trained once per node (local) and once on pooled data (central). The same
trained models are then scored with every candidate R0-R3 (:mod:`agmesh.models.shift`);
nothing is retrained per candidate. The rule declared in ADR 0005 picks the
candidate from the neural model's results; PCA and Isolation Forest are reported as
a sensitivity table only. ``--sites site_b`` reports R0 and R1 on Site B without
choosing (ADR 0005: Site B is outside the selection). The test split is never read.
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
from agmesh.models.shift import SHIFTS, SIMPLICITY

KEYS = (
    "auroc",
    "auprc",
    "event_recall",
    "range_f1",
    "window_fpr",
    "window_fpr_excl_recovery",
    "false_alarms_per_hour",
    "ttd_median_s",
    "abstained_share",
)
NEURAL = {"site_a": "vae", "site_b": "ae"}
REPORTED = {"site_a": ("vae", "pca", "iforest"), "site_b": ("ae", "pca")}
CANDIDATES: dict[str, tuple[str, ...]] = {"site_a": SHIFTS, "site_b": ("r0", "r1")}
RECALL_GUARD = 0.05
AUPRC_GUARD = 0.02
FPR_TIE = 0.01


def _mean(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def select(summary: dict[str, dict[str, float | None]]) -> dict[str, Any]:
    """The ADR 0005 rule on the neural model's seed-and-mode means."""
    r0 = summary["r0"]
    rows: list[dict[str, Any]] = []
    for shift, m in summary.items():
        recall_ok = (m["event_recall"] or 0.0) >= (r0["event_recall"] or 0.0) - RECALL_GUARD
        auprc_ok = (m["auprc"] or 0.0) >= (r0["auprc"] or 0.0) - AUPRC_GUARD
        rows.append(
            {
                "shift": shift,
                "eligible": bool(recall_ok and auprc_ok),
                "recall_guard": bool(recall_ok),
                "auprc_guard": bool(auprc_ok),
                "window_fpr": m["window_fpr"],
            }
        )
    eligible = [r for r in rows if r["eligible"] and r["window_fpr"] is not None]
    best = min(float(r["window_fpr"]) for r in eligible)
    tied = [r for r in eligible if float(r["window_fpr"]) <= best + FPR_TIE]
    chosen = str(min(tied, key=lambda r: SIMPLICITY[str(r["shift"])])["shift"])
    return {"chosen": chosen, "rows": rows, "best_fpr": best}


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
        candidates = CANDIDATES[site]
        # model -> shift -> mode -> seed -> macro metrics
        table: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for seed in seeds:
            for kind in REPORTED[site]:
                for mode in ("local", "central"):
                    fit = x1.fit_local if mode == "local" else x1.fit_central
                    fits, _training = fit(site, kind, data, params, seed)
                    for shift in candidates:
                        per_client, _ = x1.evaluate_fits(site, fits, params, shift)
                        metrics = macro(per_client, KEYS)
                        node = table.setdefault(kind, {}).setdefault(shift, {})
                        node.setdefault(mode, {})[str(seed)] = {
                            "metrics": metrics,
                            "per_client": {
                                c: {k: v.get(k) for k in KEYS} for c, v in per_client.items()
                            },
                        }
                        print(
                            f"{site} s{seed} {kind:<7} {mode:<7} {shift}  "
                            + " ".join(
                                f"{k} {metrics[k]:.3f}"
                                for k in ("auroc", "auprc", "event_recall", "window_fpr")
                                if metrics[k] is not None
                            ),
                            flush=True,
                        )
        summary: dict[str, dict[str, dict[str, Any]]] = {}
        for kind, by_shift in table.items():
            for shift, by_mode in by_shift.items():
                entry: dict[str, Any] = {}
                for k in KEYS:
                    entry[k] = _mean(
                        [r["metrics"][k] for runs in by_mode.values() for r in runs.values()]
                    )
                for mode, runs in by_mode.items():
                    entry[mode] = {k: _mean([r["metrics"][k] for r in runs.values()]) for k in KEYS}
                summary.setdefault(kind, {})[shift] = entry
        site_report: dict[str, Any] = {
            "candidates": list(candidates),
            "summary": summary,
            "runs": table,
        }
        if site == "site_a":
            neural = {
                s: {k: v for k, v in e.items() if k in KEYS}
                for s, e in summary[NEURAL[site]].items()
            }
            decision = select(neural)
            site_report["decision"] = decision
            print(f"{site}: chosen {decision['chosen']}", flush=True)
        else:
            site_report["decision"] = {"chosen": "r0", "note": "outside the selection (ADR 0005)"}
        for kind, by_shift in summary.items():
            for shift, e in by_shift.items():
                print(
                    f"{site} {kind:<7} {shift}: "
                    + " ".join(
                        f"{k} {e[k]:.3f}"
                        for k in (
                            "auroc",
                            "auprc",
                            "event_recall",
                            "window_fpr",
                            "window_fpr_excl_recovery",
                            "abstained_share",
                        )
                        if e[k] is not None
                    ),
                    flush=True,
                )
        report[site] = site_report
    write_artifact(
        out,
        kind="X1.shift_selection",
        data_source=data_source,  # type: ignore[arg-type]
        payload={
            "protocol": "docs/adr/0005-shift-robust-scoring-and-inference.md",
            "seeds": seeds,
            "split": "val",
            "guards": {"event_recall": RECALL_GUARD, "auprc": AUPRC_GUARD, "fpr_tie": FPR_TIE},
            "shift_window": params.models.scoring.shift_window.model_dump(),
            "sites": report,
        },
        params_path=params_path,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ADR 0005 shift-candidate selection")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--sites", nargs="+", choices=("site_a", "site_b"), default=["site_a"])
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/X1/shift_selection.json"))
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
