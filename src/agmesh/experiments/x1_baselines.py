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

from agmesh.common.config import DEFAULT_PARAMS_PATH, Params, ShiftWindow, load_params
from agmesh.common.provenance import sha256_bytes, sha256_file, utc_now_iso
from agmesh.data.access import Partition, Site, load_partition
from agmesh.evaluation import events as ev
from agmesh.evaluation.detection import NodeArrays, evaluate_node_full, macro, node_arrays_npz
from agmesh.evaluation.results import (
    DETECTION_METRICS,
    compare_validation,
    result_path,
    write_result,
)
from agmesh.evaluation.sealed import TEST_ACCESS_LOG, FinalAccess, add_final_argument
from agmesh.models import inputs as mi
from agmesh.models.scoring import ErrorReference, Scoring, causal_smooth, window_scores
from agmesh.models.shift import Shifted, ShiftReference, apply_shift

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
    shift_reference: ShiftReference | None = None

    def fit_reference(self, trains: list[mi.ModelInput], scoring: Scoring | None = None) -> None:
        """Training-error references for ADR 0003 normalisation and ADR 0005 shifts."""
        if self.errors is not None:
            fn = self.errors
            errs = [fn(t.x) for t in trains]
            feats = np.concatenate(
                [mi.per_feature(e, t) for e, t in zip(errs, trains, strict=True)]
            )
            self.reference = ErrorReference.fit(feats)
            if scoring is not None:
                base = np.concatenate(
                    [
                        window_scores(e, t, scoring, self.reference)
                        for e, t in zip(errs, trains, strict=True)
                    ]
                )
                self.shift_reference = ShiftReference.fit(feats, base)
        elif self.raw_score is not None and scoring is not None:
            fn_raw = self.raw_score
            self.shift_reference = ShiftReference.fit(
                None, np.concatenate([fn_raw(t.x) for t in trains])
            )

    def scores(self, inp: mi.ModelInput, scoring: Scoring) -> np.ndarray:
        if self.errors is not None:
            return window_scores(self.errors(inp.x), inp, scoring, self.reference)
        assert self.raw_score is not None
        return causal_smooth(self.raw_score(inp.x), inp.window_start, inp.stride_s, scoring.smooth)

    def shifted(
        self, inp: mi.ModelInput, scoring: Scoring, shift: str, window: ShiftWindow
    ) -> Shifted:
        """Window scores after an ADR 0005 shift candidate (``r0`` = unchanged)."""
        if shift != "r0" and scoring.smooth != 1:
            raise ValueError("shift candidates are defined for unsmoothed scores (ADR 0005)")
        if shift != "r0" and self.shift_reference is None:
            raise ValueError("fit_reference(..., scoring) must run before a shift candidate")
        if self.errors is not None:
            err = self.errors(inp.x)
            base = window_scores(err, inp, scoring, self.reference)
            feats: np.ndarray | None = mi.per_feature(err, inp)
            if shift == "r2" and scoring.aggregation != "mean":
                raise ValueError("r2 detrends per-feature errors and needs the 'mean' aggregation")
        else:
            base = self.scores(inp, scoring)
            feats = None
        ref = self.shift_reference or ShiftReference(None, float(np.median(base)))
        return apply_shift(shift, base, feats, inp.window_start, inp.stride_s, ref, window)


def site_scoring(site: Site, params: Params) -> Scoring:
    cfg = params.models.scoring.site_a if site == "site_a" else params.models.scoring.site_b
    return Scoring(cfg.aggregation, cfg.smooth, cfg.top_k)


def site_shift(site: Site, params: Params) -> str:
    cfg = params.models.scoring.site_a if site == "site_a" else params.models.scoring.site_b
    return cfg.shift


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


@dataclass
class NodeFit:
    """One node's trained detector and the scaler its inputs went through."""

    client: str
    node: NodeData
    scaler: mi.Scaler
    fitted: Fitted
    flat: bool = False


def _evaluate(
    site: Site,
    fit: NodeFit,
    params: Params,
    shift: str | None = None,
    threshold_value: float | None = None,
    part: str = "val_attack",
) -> tuple[dict[str, Any], NodeArrays]:
    """Metrics on ``part`` (``val_attack``, or ``test`` in a final run) at the threshold
    from ``val_normal``; ``test`` must already be in ``fit.node.parts``."""

    def inputs(name: str) -> mi.ModelInput:
        return _inputs(
            site, fit.node.parts[name], fit.scaler, params, evaluation=True, flat=fit.flat
        )

    val_normal, val_attack = inputs("val_normal"), inputs(part)
    scoring = site_scoring(site, params)
    chosen = shift or site_shift(site, params)
    window = params.models.scoring.shift_window
    normal = fit.fitted.shifted(val_normal, scoring, chosen, window)
    attack = fit.fitted.shifted(val_attack, scoring, chosen, window)
    return evaluate_node_full(
        normal.scores,
        attack.scores,
        val_attack.label,
        val_attack.window_start,
        window_length_s=val_attack.window_length_s,
        stride_s=val_attack.stride_s,
        events=_events(site, val_attack),
        target_fpr=params.evaluation.target_window_fpr,
        tolerance_s=params.evaluation.event_tolerance_s,
        normal_val_scored=normal.scored,
        scored=attack.scored,
        threshold_value=threshold_value,
    )


def fit_local(
    site: Site, kind: str, data: dict[str, NodeData], params: Params, seed: int
) -> tuple[list[NodeFit], dict[str, Any]]:
    flat = kind == "vae_flat"
    scoring = site_scoring(site, params)
    fits: list[NodeFit] = []
    training: dict[str, Any] = {}
    for client, node in data.items():
        scaler = mi.Scaler.fit(_raw(site, node.parts["train"], params), _clip(site, params))
        train = _inputs(site, node.parts["train"], scaler, params, evaluation=False, flat=flat)
        val = _inputs(site, node.parts["val_normal"], scaler, params, evaluation=False, flat=flat)
        fitted = _scorer(kind, site, train.x, val.x, params, seed)
        fitted.fit_reference([train], scoring)
        fitted.info.update(train_windows=len(train), inputs=train.n_inputs)
        training[client] = fitted.info
        fits.append(NodeFit(client, node, scaler, fitted, flat))
    return fits, training


def fit_central(
    site: Site, kind: str, data: dict[str, NodeData], params: Params, seed: int
) -> tuple[list[NodeFit], dict[str, Any]]:
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
    fitted.fit_reference(trains, site_scoring(site, params))
    fitted.info.update(train_windows=int(x_train.shape[0]), inputs=int(x_train.shape[1]))
    fits = [NodeFit(client, node, scaler, fitted) for client, node in data.items()]
    return fits, {"pooled": fitted.info}


def evaluate_fits(
    site: Site,
    fits: list[NodeFit],
    params: Params,
    shift: str | None = None,
    part: str = "val_attack",
) -> tuple[dict[str, Any], dict[str, NodeArrays]]:
    per_client: dict[str, Any] = {}
    arrays: dict[str, NodeArrays] = {}
    for fit in fits:
        per_client[fit.client], arrays[fit.client] = _evaluate(site, fit, params, shift, part=part)
    return per_client, arrays


def run_local(
    site: Site, kind: str, data: dict[str, NodeData], params: Params, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    fits, training = fit_local(site, kind, data, params, seed)
    return evaluate_fits(site, fits, params)[0], training


def run_central(
    site: Site, kind: str, data: dict[str, NodeData], params: Params, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    fits, training = fit_central(site, kind, data, params, seed)
    return evaluate_fits(site, fits, params)[0], training


def run_rule(
    site: Site, data: dict[str, NodeData], params: Params
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The new-peer rule: alert when a window has a peer never seen in training."""
    per_client, _arrays, info = run_rule_full(site, data, params)
    return per_client, info


def run_rule_full(
    site: Site, data: dict[str, NodeData], params: Params, part: str = "val_attack"
) -> tuple[dict[str, Any], dict[str, NodeArrays], dict[str, Any]]:
    length = float(params.data.windows.site_b.length_s)
    per_client: dict[str, Any] = {}
    arrays: dict[str, NodeArrays] = {}
    for client, node in data.items():
        normal, attack = node.parts["val_normal"], node.parts[part]
        col = attack.features.columns.index("new_peers")
        starts = attack.labels.time_s
        per_client[client], arrays[client] = evaluate_node_full(
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
    info = {"rule": "score = new_peers (peers never seen in training); no training"}
    return per_client, arrays, info


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
    final: bool = False,
    access_log: Path = TEST_ACCESS_LOG,
) -> list[dict[str, Any]]:
    """Run every configuration for every seed (default: the first seed in params).

    With ``final`` (ADR 0019 §4), each configuration is refitted exactly as on
    validation, evaluated on validation again (compared with the committed artifact)
    and then on test at the validation threshold. Only the test artifact is written,
    under ``<results>/test/``.
    """
    summary: list[dict[str, Any]] = []
    for seed in seeds or params.project.seeds[:1]:
        summary += _run_seed(
            params, sites, models, partitions, results, seed, data_source, params_path,
            final, access_log,
        )  # fmt: skip
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
    final: bool = False,
    access_log: Path = TEST_ACCESS_LOG,
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
                done = result_path(results / "test", "X1", f"{site}__{mode}__{kind}", seed)
                if final and done.is_file():
                    print(f"{done} exists: a completed test run is never repeated; skipped")
                    continue
                clock = time.perf_counter()
                config_id = f"{site}__{mode}__{kind}"
                fits: list[NodeFit] = []
                if kind == "rule_new_peer":
                    per_client, arrays, training = run_rule_full(site, data, params)
                    shift = "r0"
                else:
                    fit = fit_local if mode == "local" else fit_central
                    fits, training = fit(site, kind, data, params, seed)
                    per_client, arrays = evaluate_fits(site, fits, params)
                    shift = site_shift(site, params)
                metrics = macro(per_client, (*DETECTION_METRICS, *EXTRA_METRICS))
                if final:
                    validation = {
                        "metrics": metrics,
                        "comparison": compare_validation(
                            Path("results"), "X1", config_id, seed, metrics, None
                        ),
                    }
                    acc = FinalAccess(
                        f"X1/{config_id}", seed, log_path=access_log, params_path=params_path,
                        extra={"run_id": f"{config_id}__seed{seed}__{started_at}"},
                    )  # fmt: skip
                    with_test = {
                        c: NodeData({**n.parts, "test": load_partition(
                            params, site, c, "test", root=partitions, final=acc)})
                        for c, n in data.items()
                    }  # fmt: skip
                    if kind == "rule_new_peer":
                        per_client, arrays, _ = run_rule_full(site, with_test, params, "test")
                    else:
                        fits = [
                            NodeFit(f.client, with_test[f.client], f.scaler, f.fitted, f.flat)
                            for f in fits
                        ]
                        per_client, arrays = evaluate_fits(site, fits, params, part="test")
                    metrics = macro(per_client, (*DETECTION_METRICS, *EXTRA_METRICS))
                shas = {
                    c: t.get("state_sha256") for c, t in training.items() if isinstance(t, dict)
                }
                path = write_result(
                    results / "test" if final else results,
                    experiment="X1",
                    config_id=config_id,
                    seed=seed,
                    split="test" if final else "val",
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
                        "scoring": {**site_scoring(site, params).__dict__, "shift": shift},
                        "evaluation": {
                            "threshold_from": "val_normal",
                            "evaluated_on": "test" if final else "val_attack",
                            "target_window_fpr": params.evaluation.target_window_fpr,
                            "event_tolerance_s": params.evaluation.event_tolerance_s,
                        },
                        **(
                            {"validation": validation, "access_run_id": acc.extra["run_id"]}
                            if final
                            else {}
                        ),
                    },
                    params_path=params_path,
                    scores=node_arrays_npz(arrays),
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
    add_final_argument(parser)
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
        final=args.final,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
