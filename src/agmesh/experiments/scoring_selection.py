"""ADR 0003: choose the anomaly-score aggregation on validation, once.

    python -m agmesh.experiments.scoring_selection

Trains the neural model of each site once per node (local) and once on pooled data
(central), then re-scores the same models with every variant in
:data:`agmesh.models.scoring.GRID`. Each variant gets per-node thresholds from normal
validation at the target FPR and is evaluated on attack validation. The selection
rule is the one declared in ADR 0003 before this ran. Everything is written to
``results/X1/scoring_selection.json``; the test split is never read.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import DEFAULT_PARAMS_PATH, Params, load_params
from agmesh.data.access import Site
from agmesh.evaluation.detection import evaluate_node, macro
from agmesh.experiments import x1_baselines as x1
from agmesh.models import inputs as mi
from agmesh.models.scoring import GRID, ErrorReference, Scoring, window_scores

KIND = {"site_a": "vae", "site_b": "ae"}
SIMPLICITY = {"mean": 0, "norm_mean": 1, "norm_top3": 2}
KEYS = ("auroc", "auprc", "event_recall", "range_f1", "window_fpr", "false_alarms_per_hour")


def _errors(model: Any, inp: mi.ModelInput) -> np.ndarray:
    from agmesh.models import nets

    return nets.reconstruction_errors(model, inp.x)


def _node_bundle(
    site: Site,
    node: x1.NodeData,
    scaler: mi.Scaler,
    model: Any,
    params: Params,
    train: mi.ModelInput,
) -> dict[str, Any]:
    vn = x1._inputs(site, node.parts["val_normal"], scaler, params, evaluation=True)
    va = x1._inputs(site, node.parts["val_attack"], scaler, params, evaluation=True)
    return {
        "reference": ErrorReference.fit(mi.per_feature(_errors(model, train), train)),
        "val_normal": (vn, _errors(model, vn)),
        "val_attack": (va, _errors(model, va)),
        "events": x1._events(site, va),
    }


def _bundles(
    site: Site, mode: str, data: dict[str, x1.NodeData], params: Params, seed: int
) -> dict[str, Any]:
    bundles: dict[str, Any] = {}
    if mode == "local":
        for client, node in data.items():
            scaler = mi.Scaler.fit(
                x1._raw(site, node.parts["train"], params), x1._clip(site, params)
            )
            train = x1._inputs(site, node.parts["train"], scaler, params, evaluation=False)
            val = x1._inputs(site, node.parts["val_normal"], scaler, params, evaluation=False)
            model, _info = x1._train_net(site, train.x, val.x, params, seed)
            bundles[client] = _node_bundle(site, node, scaler, model, params, train)
        return bundles
    scaler = mi.Scaler.pooled(
        [x1._raw(site, n.parts["train"], params) for n in data.values()], x1._clip(site, params)
    )
    trains = [
        x1._inputs(site, n.parts["train"], scaler, params, evaluation=False) for n in data.values()
    ]
    vals = [
        x1._inputs(site, n.parts["val_normal"], scaler, params, evaluation=False)
        for n in data.values()
    ]
    model, _info = x1._train_net(
        site,
        np.concatenate([t.x for t in trains]),
        np.concatenate([v.x for v in vals]),
        params,
        seed,
    )
    # one reference from the pooled training errors, as a central model would have
    pooled_ref = ErrorReference.fit(
        np.concatenate([mi.per_feature(_errors(model, t), t) for t in trains])
    )
    for (client, node), train in zip(data.items(), trains, strict=True):
        bundle = _node_bundle(site, node, scaler, model, params, train)
        bundle["reference"] = pooled_ref
        bundles[client] = bundle
    return bundles


def _evaluate(bundles: dict[str, Any], scoring: Scoring, params: Params) -> dict[str, Any]:
    per_client = {}
    for client, b in bundles.items():
        vn, en = b["val_normal"]
        va, ea = b["val_attack"]
        per_client[client] = evaluate_node(
            window_scores(en, vn, scoring, b["reference"]),
            window_scores(ea, va, scoring, b["reference"]),
            va.label,
            va.window_start,
            window_length_s=va.window_length_s,
            stride_s=va.stride_s,
            events=b["events"],
            target_fpr=params.evaluation.target_window_fpr,
            tolerance_s=params.evaluation.event_tolerance_s,
        )
    return per_client


def select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The ADR 0003 rule: best mean AUPRC; ties (0.005) by FPR, then simplicity."""
    best = max(r["criterion"] for r in rows)
    tied = [r for r in rows if r["criterion"] >= best - 0.005]
    tied.sort(key=lambda r: (r["mean_fpr"], SIMPLICITY[r["aggregation"]], r["smooth"]))
    return tied[0]


def run(
    params: Params,
    sites: list[Site],
    partitions: Path,
    out: Path,
    *,
    data_source: str,
    params_path: Path,
) -> dict[str, Any]:
    seed = params.project.seeds[0]
    report: dict[str, Any] = {}
    for site in sites:
        data = x1._load(params, site, partitions)
        bundles = {mode: _bundles(site, mode, data, params, seed) for mode in ("local", "central")}
        rows = []
        for scoring in GRID:
            entry: dict[str, Any] = {
                "variant": scoring.name,
                "aggregation": scoring.aggregation,
                "smooth": scoring.smooth,
            }
            for mode, b in bundles.items():
                per_client = _evaluate(b, scoring, params)
                entry[mode] = {"metrics": macro(per_client, KEYS), "per_client": per_client}
            auprcs = [entry[m]["metrics"]["auprc"] for m in ("local", "central")]
            fprs = [entry[m]["metrics"]["window_fpr"] for m in ("local", "central")]
            entry["criterion"] = float(np.mean([a for a in auprcs if a is not None]))
            entry["mean_fpr"] = float(np.mean([f for f in fprs if f is not None]))
            rows.append(entry)
            lm, cm = entry["local"]["metrics"], entry["central"]["metrics"]
            pairs = " | ".join(
                f"{k} {lm[k]:.3f}/{cm[k]:.3f}"
                for k in ("auprc", "auroc", "window_fpr", "event_recall")
            )
            print(f"{site} {scoring.name:<16} local/central {pairs}", flush=True)
        chosen = select(rows)
        report[site] = {"model": KIND[site], "variants": rows, "chosen": chosen["variant"]}
        print(
            f"{site}: chosen {chosen['variant']} (criterion {chosen['criterion']:.3f})", flush=True
        )
    write_artifact(
        out,
        kind="X1.scoring_selection",
        data_source=data_source,  # type: ignore[arg-type]
        payload={
            "protocol": "docs/adr/0003-scoring-selection.md",
            "seed": seed,
            "split": "val",
            "sites": report,
        },
        params_path=params_path,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ADR 0003 scoring selection on validation")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument(
        "--sites", nargs="+", choices=("site_a", "site_b"), default=["site_a", "site_b"]
    )
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/X1/scoring_selection.json"))
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)
    params = load_params(args.params)
    run(
        params,
        args.sites,
        args.partitions or params.paths.partitions_dir,
        args.output,
        data_source="fixture" if args.fixture else "runtime",
        params_path=args.params,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
