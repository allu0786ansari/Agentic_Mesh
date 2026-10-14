"""Federated runner: config -> federation -> result artifact (Plan Week 6 task 6).

    python -m agmesh.federated.run --site site_a --strategy fedavg --epsilon 10
    python -m agmesh.federated.run --site site_a --strategy fedprox --epsilon 3 --h1c
    python -m agmesh.federated.run --site site_a --rounds 5 --smoke     # 5-round smoke test

A full run writes ``results/X1/<site>__<strategy>__eps<e>[__h1c][__het][__drop<p>]/seed_<n>.json``
with a per-window score file (ADR 0005). The artifact holds:

* the macro and per-client detection metrics (validation);
* the pooled AUROC from binned scores, with its bin error;
* each client's privacy ledger (ε, δ, noise, steps, every release; RDP and the looser
  basic-composition total);
* the round log (participants, stragglers, training loss, validation loss) and the
  node registry;
* timing.

Each execution opens fresh privacy ledgers under
``results/privacy/<run_id>/``, where the run id holds the config, seed and a
timestamp. A rerun is a new training run with its own budget; within a run, nothing
is ever reset. ``--smoke`` writes to ``results/federated/smoke/`` instead of
``results/X1``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import DEFAULT_PARAMS_PATH, Params, load_params
from agmesh.common.provenance import sha256_bytes, sha256_file, utc_now_iso
from agmesh.data.access import Site
from agmesh.evaluation.detection import DETECTION_KEYS
from agmesh.evaluation.results import write_result
from agmesh.experiments import x1_baselines as x1
from agmesh.federated.backends import ClientSettings, InProcessPool
from agmesh.federated.client import check_h1c
from agmesh.federated.registry import NodeRegistry
from agmesh.federated.server import FederationResult, Server, ServerConfig
from agmesh.federated.strategies import Strategy

HETEROGENEOUS_EPOCHS = (1, 2, 3)


def eps_label(epsilon: float) -> str:
    return "inf" if math.isinf(epsilon) else f"{epsilon:g}"


def config_id(
    site: str, strategy: str, epsilon: float, *, h1c: bool, heterogeneous: bool, dropout: float
) -> str:
    parts = [site, strategy, f"eps{eps_label(epsilon)}"]
    if h1c:
        parts.append("h1c")
    if heterogeneous:
        parts.append("het")
    if dropout > 0:
        parts.append(f"drop{dropout:g}")
    return "__".join(parts)


def rounds_to_target(val_losses: list[float], tolerance: float) -> int | None:
    """First round within ``tolerance`` (relative) of the run's best validation loss."""
    finite = [v for v in val_losses if np.isfinite(v)]
    if not finite:
        return None
    best = min(finite)
    for i, v in enumerate(val_losses, start=1):
        if np.isfinite(v) and v <= best * (1 + tolerance):
            return i
    return None


def _privacy_block(result: FederationResult) -> dict[str, Any]:
    out: dict[str, Any] = {"clients": result.privacy}
    eps = [p.get("epsilon") for p in result.privacy.values()]
    finite = [float(e) for e in eps if e is not None and math.isfinite(float(e))]
    out["max_client_epsilon"] = max(finite) if finite else math.inf
    return out


def federate(
    params: Params,
    site: Site,
    *,
    strategy: str,
    epsilon: float,
    seed: int,
    partitions: Path,
    rounds: int | None = None,
    h1c: bool = False,
    heterogeneous: bool = False,
    dropout: float | None = None,
    privacy_root: Path = Path("results/privacy"),
    run_id: str | None = None,
    backend: str = "inprocess",
    params_path: Path = DEFAULT_PARAMS_PATH,
    suffix: str = "",
) -> tuple[str, str, FederationResult]:
    fed = params.federated
    n_rounds = rounds or fed.max_rounds
    drop = fed.dropout_prob if dropout is None else dropout
    cid = config_id(site, strategy, epsilon, h1c=h1c, heterogeneous=heterogeneous, dropout=drop)
    cid += suffix
    rid = run_id or f"{cid}__seed{seed}__{time.strftime('%Y%m%dT%H%M%S')}"
    epochs_per_round = max(HETEROGENEOUS_EPOCHS) if heterogeneous else fed.local_epochs
    if h1c:
        check_h1c(params, site)
    h1c_eps = params.privacy.threshold_epsilon if h1c and math.isfinite(epsilon) else None
    nodes = x1._clients(params, site)
    settings = ClientSettings(
        params,
        site,
        partitions,
        rid,
        epsilon,
        seed,
        h1c_epsilon=h1c_eps,
        epochs=n_rounds * epochs_per_round,
        privacy_root=privacy_root,
    )
    cfg = ServerConfig(
        site=site,
        rounds=n_rounds,
        strategy=Strategy(strategy, fed.proximal_mu if strategy == "fedprox" else 0.0),
        local_epochs=fed.local_epochs,
        min_fit_clients=getattr(fed.min_fit_clients, site),
        round_timeout_s=fed.round_timeout_s,
        seed=seed,
        dropout_prob=drop,
        local_epoch_choices=HETEROGENEOUS_EPOCHS if heterogeneous else (),
        h1c=h1c,
        target_fpr=params.evaluation.target_window_fpr,
        threshold_sd=params.privacy.normalisation_threshold_sd,
        metric_keys=DETECTION_KEYS,
    )
    registry = NodeRegistry(privacy_root / rid / "registry.db")
    if backend == "flower":
        from agmesh.federated import flower

        fs = flower.FlowerSettings(
            str(params_path),
            site,
            str(partitions),
            rid,
            epsilon,
            seed,
            h1c_eps,
            n_rounds * epochs_per_round,
            str(privacy_root),
            nodes,
        )
        res = fed.client_resources
        result = flower.run(
            fs,
            lambda pool: Server(pool, cfg, registry),
            num_cpus_per_client=res.num_cpus,
            total_cpus=max(int(res.num_cpus), os.cpu_count() or 2),
        )
    else:
        result = Server(InProcessPool(settings, nodes), cfg, registry).run()
    return cid, rid, result


def _score_file(result: FederationResult) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    nodes: list[str] = []
    for reply in result.final_replies:
        if reply.arrays is None:
            continue
        for key, value in reply.arrays.items():
            if key == "nodes":
                nodes += [str(n) for n in value]
            else:
                merged[key] = value
    merged["nodes"] = np.array(sorted(nodes))
    return merged


def record(
    params: Params,
    site: Site,
    cid: str,
    rid: str,
    result: FederationResult,
    *,
    seed: int,
    strategy: str,
    epsilon: float,
    partitions: Path,
    results: Path,
    smoke: bool | str,
    started_at: str,
    data_source: str,
    params_path: Path,
    backend: str = "inprocess",
) -> Path:
    val_losses = [r.val_loss for r in result.rounds]
    federation = {
        "run_id": rid,
        "strategy": strategy,
        "proximal_mu": params.federated.proximal_mu if strategy == "fedprox" else 0.0,
        "rounds": len(result.rounds),
        "rounds_to_target": rounds_to_target(
            val_losses, params.federated.rounds_to_target_tolerance
        ),
        "aggregation": "uniform mean",
        "round_log": [r.__dict__ for r in result.rounds],
        "registry": result.registry,
        "pooled_auroc_hist": result.final["pooled_auroc_hist"],
        "pooled_auroc_bin_error": result.final["pooled_auroc_bin_error"],
        "seconds": round(result.seconds, 1),
        "seconds_per_round": round(result.seconds / max(1, len(result.rounds)), 2),
        "backend": backend,
    }
    privacy = _privacy_block(result)
    extra = {
        "site": site,
        "mode": "federated",
        "model": "vae" if site == "site_a" else "ae",
        "private": math.isfinite(epsilon),
        "target_epsilon": eps_label(epsilon),
        "per_client": result.final["per_client"],
        "h1c": result.h1c,
        "scoring": {**x1.site_scoring(site, params).__dict__, "shift": x1.site_shift(site, params)},
        "evaluation": {
            "threshold_from": "val_normal (per node)",
            "evaluated_on": "val_attack",
            "target_window_fpr": params.evaluation.target_window_fpr,
            "event_tolerance_s": params.evaluation.event_tolerance_s,
        },
        "model_spec": result.spec.__dict__,
    }
    if smoke:
        tag = smoke if isinstance(smoke, str) else "smoke"
        out = results / "federated" / tag / f"{cid}__seed{seed}.json"
        return write_artifact(
            out,
            kind=f"federated.{tag}",
            data_source=data_source,  # type: ignore[arg-type]
            payload={
                "config_id": cid,
                "seed": seed,
                "metrics": result.final["macro"],
                "privacy": privacy,
                "federation": federation,
                **extra,
            },
            params_path=params_path,
        )
    weights_sha = sha256_bytes(b"".join(np.ascontiguousarray(w).tobytes() for w in result.weights))
    return write_result(
        results,
        experiment="X1",
        config_id=cid,
        seed=seed,
        split="val",
        data_source=data_source,  # type: ignore[arg-type]
        metrics=result.final["macro"],
        manifest_sha256=sha256_file(partitions / site / "manifest.json"),
        started_at=started_at,
        checkpoint_sha256=weights_sha,
        privacy=privacy,
        federation=federation,
        extra=extra,
        params_path=params_path,
        scores=_score_file(result),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Federated training and evaluation")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--site", choices=("site_a", "site_b"), required=True)
    parser.add_argument("--strategy", choices=("fedavg", "fedprox"), default=None)
    parser.add_argument("--epsilon", type=float, default=None, help="inf for non-private")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--h1c", action="store_true")
    parser.add_argument("--heterogeneous", action="store_true")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--tag", default=None, help="write to results/federated/<tag>/ (e.g. g2_tuning)"
    )
    parser.add_argument("--lr", type=float, default=None, help="override the DP learning rate")
    parser.add_argument("--clip", type=float, default=None, help="override max_grad_norm")
    parser.add_argument("--backend", choices=("inprocess", "flower"), default="inprocess")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)
    params = load_params(args.params)
    if args.backend == "flower" and (args.lr is not None or args.clip is not None):
        parser.error(
            "--lr/--clip overrides are in-process only (Flower clients read the params file)"
        )
    raw = params.model_dump()
    if args.lr is not None:
        raw["privacy"]["dp_lr"] = {**raw["privacy"]["dp_lr"], args.site: args.lr}
    if args.clip is not None:
        raw["privacy"]["max_grad_norm"] = args.clip
    params = Params.model_validate(raw)
    strategy = args.strategy or params.federated.strategy
    epsilon = params.privacy.target_epsilon if args.epsilon is None else args.epsilon
    partitions = args.partitions or params.paths.partitions_dir
    seeds = args.seeds or params.project.seeds[:1]
    for seed in seeds:
        started_at = utc_now_iso()
        cid, rid, result = federate(
            params,
            args.site,
            strategy=strategy,
            epsilon=epsilon,
            seed=seed,
            partitions=partitions,
            rounds=args.rounds,
            h1c=args.h1c,
            heterogeneous=args.heterogeneous,
            dropout=args.dropout,
            privacy_root=args.results / "privacy",
            backend=args.backend,
            params_path=args.params,
            suffix=(f"__lr{args.lr:g}" if args.lr is not None else "")
            + (f"__clip{args.clip:g}" if args.clip is not None else ""),
        )
        path = record(
            params,
            args.site,
            cid,
            rid,
            result,
            seed=seed,
            strategy=strategy,
            epsilon=epsilon,
            partitions=partitions,
            results=args.results,
            smoke=args.tag or args.smoke,
            started_at=started_at,
            data_source="fixture" if args.fixture else "runtime",
            params_path=args.params,
            backend=args.backend,
        )
        m = result.final["macro"]
        print(
            f"{cid} seed {seed}: AUROC {m['auroc']:.3f} AUPRC {m['auprc']:.3f} "
            f"recall {m['event_recall']:.3f} FPR {m['window_fpr']:.3f} | pooled AUROC "
            f"{result.final['pooled_auroc_hist']:.3f}±{result.final['pooled_auroc_bin_error']:.3f} "
            f"| max ε {_privacy_block(result)['max_client_epsilon']:.3f} | "
            f"{result.seconds:.0f} s -> {path}",
            flush=True,
        )
        if result.h1c:
            print(
                "  H1c:",
                json.dumps({k: result.h1c[k] for k in ("threshold", "macro")}, default=str),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
