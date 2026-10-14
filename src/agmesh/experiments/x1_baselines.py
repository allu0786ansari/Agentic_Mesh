"""X1 baselines on validation data (Implementation Plan, Week 4 task 3).

    python -m agmesh.experiments.x1_baselines            # both sites, all baselines
    python -m agmesh.experiments.x1_baselines --sites site_b --models ae rule_new_peer

For each site it trains, without differential privacy and with one seed:

* ``local``   — one model per node on that node's training data;
* ``central`` — one model on all nodes' training data pooled (the oracle reference);

for the neural model (VAE on Site A, autoencoder on Site B), PCA and Isolation
Forest. Site A also runs the flattened-window VAE (``vae_flat``) for the input-mode
comparison (HAI-6), and Site B the new-peer rule (MB-15), which needs no training.

Every node's threshold is set on its own normal-validation windows at the target
FPR, and the node is evaluated on its attack-validation partition. One result
artifact per configuration is written to ``results/X1/<config_id>/seed_<n>.json``
with the macro average over nodes in ``metrics`` and the per-node values in
``per_client``. The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.common.config import DEFAULT_PARAMS_PATH, Params, load_params
from agmesh.common.provenance import sha256_bytes, sha256_file, utc_now_iso
from agmesh.data.access import Partition, Site, load_partition
from agmesh.evaluation import events as ev
from agmesh.evaluation.detection import evaluate_node, macro
from agmesh.evaluation.results import DETECTION_METRICS, write_result
from agmesh.models import inputs as mi
from agmesh.models.scoring import ErrorReference, Scoring, causal_smooth, window_scores

SITE_MODELS = {
    "site_a": ("vae", "vae_flat", "pca", "iforest"),
    "site_b": ("ae", "pca", "iforest", "rule_new_peer"),
}
EXTRA_METRICS = ("range_precision", "range_recall", "window_precision", "ttd_mean_s")


@dataclass
class NodeData:
    parts: dict[str, Partition]


def _clients(params: Params, site: Site) -> list[str]:
    if site == "site_a":
        return list(params.data.partition.site_a.clients)
    return list(params.data.partition.site_b.clients)


def _load(params: Params, site: Site, root: Path) -> dict[str, NodeData]:
    return {
        client: NodeData(
            {
                p: load_partition(params, site, client, p, root=root)
                for p in ("train", "val_normal", "val_attack")
            }
        )
        for client in _clients(params, site)
    }


def _raw(site: Site, part: Partition, params: Params) -> np.ndarray:
    return mi.site_a_raw(part, params) if site == "site_a" else mi.site_b_raw(part, params)


def _clip(site: Site, params: Params) -> float:
    return params.data.hai.scaling.clip_sd if site == "site_a" else params.models.ae.clip_sd


def _inputs(
    site: Site,
    part: Partition,
    scaler: mi.Scaler,
    params: Params,
    *,
    evaluation: bool,
    flat: bool = False,
) -> mi.ModelInput:
    if site == "site_b":
        return mi.site_b_input(part, scaler, params)
    p = params
    if flat:
        raw = p.model_dump()
        raw["models"]["vae"]["input_mode"] = "flat"
        p = Params.model_validate(raw)
    stride = params.data.windows.site_a.infer_stride_s if evaluation else None
    return mi.site_a_input(part, scaler, p, stride_s=stride)


def _events(site: Site, inp: mi.ModelInput) -> list[tuple[float, float]]:
    if site == "site_a":
        return ev.events_from_rows(inp.row_time_s, inp.row_target)
    return ev.events_from_episodes(inp.window_start, inp.episode, inp.window_length_s, inp.label)


# ---------------------------------------------------------------------------- scorers


def _train_net(
    site: Site, train: np.ndarray, val: np.ndarray, params: Params, seed: int
) -> tuple[Any, dict[str, Any]]:
    from agmesh.models import nets  # torch is imported only when a neural model runs

    if site == "site_a":
        cfg = params.models.vae
        model = nets.build("vae", train.shape[1], cfg.hidden, cfg.latent, seed=seed)
        result = nets.fit(
            model,
            train,
            val,
            lr=cfg.lr,
            batch=cfg.batch,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            seed=seed,
            kl_weight=cfg.kl_weight,
        )
    else:
        acfg = params.models.ae
        model = nets.build("ae", train.shape[1], acfg.hidden, seed=seed)
        result = nets.fit(
            model,
            train,
            val,
            lr=acfg.lr,
            batch=acfg.batch,
            max_epochs=acfg.max_epochs,
            patience=acfg.patience,
            seed=seed,
        )
    info = {
        "parameters": nets.parameter_count(model),
        "epochs_run": result.epochs_run,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "train_seconds": round(result.train_seconds, 2),
        "state_sha256": nets.state_sha256(model),
    }
    return model, info


@dataclass
class Fitted:
    """A trained detector: per-input errors (reconstruction models) or raw scores."""

    errors: Callable[[np.ndarray], np.ndarray] | None
    raw_score: Callable[[np.ndarray], np.ndarray] | None
    info: dict[str, Any]
    reference: ErrorReference | None = None

    def fit_reference(self, trains: list[mi.ModelInput]) -> None:
        if self.errors is not None:
            fn = self.errors
            self.reference = ErrorReference.fit(
                np.concatenate([mi.per_feature(fn(t.x), t) for t in trains])
            )

    def scores(self, inp: mi.ModelInput, scoring: Scoring) -> np.ndarray:
        if self.errors is not None:
            return window_scores(self.errors(inp.x), inp, scoring, self.reference)
        assert self.raw_score is not None
        return causal_smooth(self.raw_score(inp.x), inp.window_start, inp.stride_s, scoring.smooth)


def site_scoring(site: Site, params: Params) -> Scoring:
    cfg = params.models.scoring.site_a if site == "site_a" else params.models.scoring.site_b
    return Scoring(cfg.aggregation, cfg.smooth, cfg.top_k)


def _scorer(
    kind: str, site: Site, train: np.ndarray, val: np.ndarray, params: Params, seed: int
) -> Fitted:
    from agmesh.models.baselines import IForestDetector, PcaDetector

    started = time.perf_counter()
    b = params.models.baselines
    if kind in ("vae", "vae_flat", "ae"):
        model, info = _train_net(site, train, val, params, seed)
        from agmesh.models import nets

        return Fitted(lambda x: nets.reconstruction_errors(model, x), None, info)
    if kind == "pca":
        pca = PcaDetector(b.pca_variance).fit(train)
        info = {
            "components": pca.n_components,
            "train_seconds": round(time.perf_counter() - started, 2),
        }
        return Fitted(pca.errors, None, info)
    if kind == "iforest":
        forest = IForestDetector(b.iforest_trees, b.iforest_max_samples, seed).fit(train)
        return Fitted(
            None, forest.score, {"train_seconds": round(time.perf_counter() - started, 2)}
        )
    raise ValueError(f"unknown model {kind!r}")


# ------------------------------------------------------------------------------- runs


def _evaluate(
    site: Site, node: NodeData, scaler: mi.Scaler, fitted: Fitted, params: Params, flat: bool
) -> dict[str, Any]:
    val_normal = _inputs(site, node.parts["val_normal"], scaler, params, evaluation=True, flat=flat)
    val_attack = _inputs(site, node.parts["val_attack"], scaler, params, evaluation=True, flat=flat)
    scoring = site_scoring(site, params)
    return evaluate_node(
        fitted.scores(val_normal, scoring),
        fitted.scores(val_attack, scoring),
        val_attack.label,
        val_attack.window_start,
        window_length_s=val_attack.window_length_s,
        stride_s=val_attack.stride_s,
        events=_events(site, val_attack),
        target_fpr=params.evaluation.target_window_fpr,
        tolerance_s=params.evaluation.event_tolerance_s,
    )


def run_local(
    site: Site, kind: str, data: dict[str, NodeData], params: Params, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    flat = kind == "vae_flat"
    per_client: dict[str, Any] = {}
    training: dict[str, Any] = {}
    for client, node in data.items():
        scaler = mi.Scaler.fit(_raw(site, node.parts["train"], params), _clip(site, params))
        train = _inputs(site, node.parts["train"], scaler, params, evaluation=False, flat=flat)
        val = _inputs(site, node.parts["val_normal"], scaler, params, evaluation=False, flat=flat)
        fitted = _scorer(kind, site, train.x, val.x, params, seed)
        fitted.fit_reference([train])
        fitted.info.update(train_windows=len(train), inputs=train.n_inputs)
        training[client] = fitted.info
        per_client[client] = _evaluate(site, node, scaler, fitted, params, flat)
    return per_client, training


def run_central(
    site: Site, kind: str, data: dict[str, NodeData], params: Params, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    scaler = mi.Scaler.pooled(
        [_raw(site, n.parts["train"], params) for n in data.values()], _clip(site, params)
    )
    trains = [
        _inputs(site, n.parts["train"], scaler, params, evaluation=False) for n in data.values()
    ]
    vals = [
        _inputs(site, n.parts["val_normal"], scaler, params, evaluation=False)
        for n in data.values()
    ]
    x_train = np.concatenate([t.x for t in trains])
    x_val = np.concatenate([v.x for v in vals])
    fitted = _scorer(kind, site, x_train, x_val, params, seed)
    fitted.fit_reference(trains)
    fitted.info.update(train_windows=int(x_train.shape[0]), inputs=int(x_train.shape[1]))
    per_client = {
        client: _evaluate(site, node, scaler, fitted, params, False)
        for client, node in data.items()
    }
    return per_client, {"pooled": fitted.info}


def run_rule(
    site: Site, data: dict[str, NodeData], params: Params
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The new-peer rule: alert when a window has a peer never seen in training."""
    length = float(params.data.windows.site_b.length_s)
    per_client: dict[str, Any] = {}
    for client, node in data.items():
        normal, attack = node.parts["val_normal"], node.parts["val_attack"]
        col = attack.features.columns.index("new_peers")
        starts = attack.labels.time_s
        per_client[client] = evaluate_node(
            normal.features.values[:, col],
            attack.features.values[:, col],
            attack.labels.target,
            starts,
            window_length_s=length,
            stride_s=length,
            events=ev.events_from_episodes(
                starts, attack.labels.group.astype(np.int64), length, attack.labels.target
            ),
            target_fpr=params.evaluation.target_window_fpr,
            tolerance_s=params.evaluation.event_tolerance_s,
        )
    return per_client, {"rule": "score = new_peers (peers never seen in training); no training"}


def run(
    params: Params,
    sites: list[Site],
    models: list[str] | None,
    partitions: Path,
    results: Path,
    *,
    data_source: str = "runtime",
    params_path: Path = DEFAULT_PARAMS_PATH,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run every configuration for every seed (default: the first seed in params)."""
    summary: list[dict[str, Any]] = []
    for seed in seeds or params.project.seeds[:1]:
        summary += _run_seed(
            params, sites, models, partitions, results, seed, data_source, params_path
        )
    return summary


def _run_seed(
    params: Params,
    sites: list[Site],
    models: list[str] | None,
    partitions: Path,
    results: Path,
    seed: int,
    data_source: str,
    params_path: Path,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for site in sites:
        data = _load(params, site, partitions)
        manifest_sha = sha256_file(partitions / site / "manifest.json")
        for kind in models or SITE_MODELS[site]:
            if kind not in SITE_MODELS[site]:
                continue
            modes = ["local"] if kind in ("vae_flat", "rule_new_peer") else ["local", "central"]
            for mode in modes:
                started_at = utc_now_iso()
                clock = time.perf_counter()
                if kind == "rule_new_peer":
                    per_client, training = run_rule(site, data, params)
                elif mode == "local":
                    per_client, training = run_local(site, kind, data, params, seed)
                else:
                    per_client, training = run_central(site, kind, data, params, seed)
                config_id = f"{site}__{mode}__{kind}"
                metrics = macro(per_client, (*DETECTION_METRICS, *EXTRA_METRICS))
                shas = {
                    c: t.get("state_sha256") for c, t in training.items() if isinstance(t, dict)
                }
                path = write_result(
                    results,
                    experiment="X1",
                    config_id=config_id,
                    seed=seed,
                    split="val",
                    data_source=data_source,  # type: ignore[arg-type]
                    metrics=metrics,
                    manifest_sha256=manifest_sha,
                    started_at=started_at,
                    checkpoint_sha256=sha256_bytes(json.dumps(shas, sort_keys=True).encode())
                    if any(shas.values())
                    else None,
                    extra={
                        "site": site,
                        "mode": mode,
                        "model": kind,
                        "private": False,
                        "per_client": per_client,
                        "training": training,
                        "wall_seconds": round(time.perf_counter() - clock, 2),
                        "scoring": site_scoring(site, params).__dict__,
                        "evaluation": {
                            "threshold_from": "val_normal",
                            "evaluated_on": "val_attack",
                            "target_window_fpr": params.evaluation.target_window_fpr,
                            "event_tolerance_s": params.evaluation.event_tolerance_s,
                        },
                    },
                    params_path=params_path,
                )
                row = {
                    "config_id": config_id,
                    "seed": seed,
                    **{k: metrics.get(k) for k in DETECTION_METRICS},
                    "path": path.as_posix(),
                }
                summary.append(row)
                print(_format_row(row), flush=True)
    return summary


def _fmt(value: Any) -> str:
    return "  -  " if value is None else f"{value:6.3f}" if abs(value) < 100 else f"{value:6.0f}"


def _format_row(row: dict[str, Any]) -> str:
    keys = (
        "auroc",
        "auprc",
        "event_recall",
        "range_f1",
        "window_fpr",
        "ttd_median_s",
        "false_alarms_per_hour",
    )
    label = f"{row['config_id']} s{row['seed']}"
    return f"{label:<34}" + " ".join(_fmt(row[k]) for k in keys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="X1 non-private baselines on validation data")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument(
        "--sites", nargs="+", choices=("site_a", "site_b"), default=["site_a", "site_b"]
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--fixture", action="store_true", help="mark results as fixture data")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="seeds to run (default: all of project.seeds in params)",
    )
    args = parser.parse_args(argv)
    params = load_params(args.params)
    print(
        f"{'config':<34}"
        + " ".join(f"{h:>6}" for h in ("AUROC", "AUPRC", "evRec", "rngF1", "FPR", "TTD", "FA/h"))
    )
    run(
        params,
        args.sites,
        args.models,
        args.partitions or params.paths.partitions_dir,
        args.results,
        data_source="fixture" if args.fixture else "runtime",
        params_path=args.params,
        seeds=args.seeds or params.project.seeds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
