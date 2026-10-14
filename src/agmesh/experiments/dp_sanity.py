"""Week 5 sanity check: single-client DP training at ε ∈ {1, 3, 10} and ε = ∞.

    python -m agmesh.experiments.dp_sanity         # -> results/privacy/dp_sanity.json

For one client per site (the first in ``params``), every ε runs the whole private
pipeline:

1. resolve δ, calibrate the budget, open the persistent accountant;
2. release the noisy normalisation histograms (spent through the guard) and build
   the scaler from them;
3. DP-SGD for the planned steps (``federated.max_rounds * local_epochs`` epochs);
4. score validation with the site's frozen scoring (ADR 0003, ADR 0005).

ε = ∞ is the same pipeline with exact histograms and no clipping or noise
(:func:`~agmesh.models.dp_train.fit_reference`). It isolates the cost of privacy. The
client's non-private Week 4 result (early stopping, mean/std scaler) is copied in from
``results/X1`` for comparison when it is present. Validation only; the test split is
never read.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import DEFAULT_PARAMS_PATH, Params, load_params
from agmesh.data.access import Partition, Site
from agmesh.data.windows import gather, window_starts
from agmesh.evaluation.detection import DETECTION_KEYS
from agmesh.experiments import x1_baselines as x1
from agmesh.models import inputs as mi
from agmesh.privacy import bounds, client_budget, noisy_stats
from agmesh.privacy.accountant_store import ROOT

EPSILONS = (1.0, 3.0, 10.0, math.inf)
KIND = {"site_a": "vae", "site_b": "ae"}


def units(site: Site, part: Partition, params: Params) -> tuple[np.ndarray, list[str]]:
    """One row per privacy unit (training window) and the feature names.

    Site A: each non-overlapping 60 s training window, as its (60, features) raw rows.
    Site B: the 10 s windows themselves (``log1p`` already applied).
    """
    raw = x1._raw(site, part, params)
    if site == "site_b":
        names = [part.features.columns[i] for i in mi.site_b_columns(part, params)]
        return np.asarray(raw, dtype=np.float64), names
    win = params.data.windows.site_a
    if win.train_stride_s != win.length_s:
        raise ValueError("the privacy unit needs non-overlapping training windows (P4)")
    length = win.length_s * params.data.hai.sample_rate_hz
    starts = window_starts(part.labels.time_s, part.labels.group, length, win.train_stride_s)
    names = [part.features.columns[i] for i in mi.site_a_columns(part, params)]
    return gather(raw, starts, length).astype(np.float64), names


def _lr(params: Params, site: Site) -> float:
    return params.models.vae.lr if site == "site_a" else params.models.ae.lr


def run_one(
    params: Params,
    site: Site,
    client: str,
    node: x1.NodeData,
    epsilon: float,
    seed: int,
    *,
    run_id: str,
    root: Path,
) -> dict[str, Any]:
    from agmesh.models import dp_train, nets

    started = time.perf_counter()
    part = node.parts["train"]
    u, names = units(site, part, params)
    lo, hi = bounds.bounds_for(site, names)
    bins = getattr(params.privacy.normalisation_bins, site)
    clip = x1._clip(site, params)
    rng = np.random.default_rng(seed)
    private = math.isfinite(epsilon)
    budget = None
    if private:
        budget = client_budget.prepare(
            params,
            site=site,
            node=client,
            n_windows=len(u),
            run_id=run_id,
            target_epsilon=epsilon,
            root=root,
        )
        z_stats = budget.stats_noise_multiplier
        budget.guard.spend(noisy_stats.mechanism(z_stats))  # accounted before it is used
    else:
        z_stats = 0.0
    hist = noisy_stats.release(
        u,
        lo,
        hi,
        bins=bins,
        noise_multiplier=z_stats,
        rng=rng,
        threshold_sd=params.privacy.normalisation_threshold_sd,
    )
    scaler = noisy_stats.scaler(hist, clip)
    train = x1._inputs(site, part, scaler, params, evaluation=False)
    if train.x.shape[0] != len(u):
        raise ValueError("training inputs and privacy units differ")
    cfg = params.models.vae if site == "site_a" else params.models.ae
    model = (
        nets.build("vae", train.n_inputs, cfg.hidden, params.models.vae.latent, seed=seed)
        if site == "site_a"
        else nets.build("ae", train.n_inputs, cfg.hidden, seed=seed)
    )
    kl = params.models.vae.kl_weight
    if budget is not None:
        result = dp_train.fit_private(
            model,
            train.x,
            plan=budget.plan,
            guard=budget.guard,
            lr=_lr(params, site),
            seed=seed,
            kl_weight=kl,
        )
        plan = budget.plan.as_dict()
        privacy = budget.store.summary()
    else:
        q, steps = client_budget.calibrate_steps(params, site, len(u))
        result = dp_train.fit_reference(
            model,
            train.x,
            steps=steps,
            sample_rate=q,
            lr=_lr(params, site),
            seed=seed,
            kl_weight=kl,
        )
        plan = {"sample_rate": q, "steps": steps, "target_epsilon": "inf"}
        privacy = {"epsilon": math.inf}
    fitted = x1.Fitted(
        lambda x: nets.reconstruction_errors(model, x),
        None,
        {"state_sha256": nets.state_sha256(model), **result.info()},
    )
    fitted.fit_reference([train], x1.site_scoring(site, params))
    metrics, _arrays = x1._evaluate(site, x1.NodeFit(client, node, scaler, fitted), params)
    return {
        "site": site,
        "client": client,
        "target_epsilon": epsilon,
        "units": len(u),
        "normalisation": {"bins": bins, "noise_multiplier": z_stats},
        "plan": plan,
        "privacy": privacy,
        "training": {k: v for k, v in result.info().items() if k != "history"},
        "loss_history": result.history,
        "metrics": {k: metrics.get(k) for k in DETECTION_KEYS},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }


def _x1_reference(results: Path, site: Site, client: str, seed: int) -> dict[str, Any] | None:
    path = results / "X1" / f"{site}__local__{KIND[site]}" / f"seed_{seed}.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    per = doc.get("per_client", {}).get(client)
    return {k: per.get(k) for k in DETECTION_KEYS} if per else None


def run(
    params: Params,
    sites: list[Site],
    partitions: Path,
    out: Path,
    *,
    epsilons: tuple[float, ...] = EPSILONS,
    seed: int | None = None,
    results: Path = Path("results"),
    root: Path = ROOT,
    data_source: str = "runtime",
    params_path: Path = DEFAULT_PARAMS_PATH,
) -> list[dict[str, Any]]:
    seed = params.project.seeds[0] if seed is None else seed
    rows: list[dict[str, Any]] = []
    for site in sites:
        data = x1._load(params, site, partitions)
        client = next(iter(data))
        for eps in epsilons:
            label = "inf" if math.isinf(eps) else f"{eps:g}"
            run_id = f"dp_sanity__{site}__{client}__eps{label}__seed{seed}"
            row = run_one(params, site, client, data[client], eps, seed, run_id=run_id, root=root)
            rows.append(row)
            m = row["metrics"]
            spent = row["privacy"].get("epsilon")
            print(
                f"{site} {client} ε={label:>3}: spent {spent:.3f} "
                f"z={row['plan'].get('noise_multiplier', 0):.2f} steps {row['plan']['steps']} | "
                f"AUROC {m['auroc']:.3f} AUPRC {m['auprc']:.3f} recall {m['event_recall']:.3f} "
                f"FPR {m['window_fpr']:.3f} | {row['wall_seconds']} s",
                flush=True,
            )
        reference = _x1_reference(results, site, client, seed)
        if reference:
            rows.append(
                {
                    "site": site,
                    "client": client,
                    "target_epsilon": "x1_nonprivate",
                    "metrics": reference,
                }
            )
    write_artifact(
        out,
        kind="privacy.dp_sanity",
        data_source=data_source,  # type: ignore[arg-type]
        payload={"seed": seed, "split": "val", "rows": rows},
        params_path=params_path,
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 5 single-client DP sanity check")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument(
        "--sites", nargs="+", choices=("site_a", "site_b"), default=["site_a", "site_b"]
    )
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/privacy/dp_sanity.json"))
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)
    params = load_params(args.params)
    run(
        params,
        args.sites,
        args.partitions or params.paths.partitions_dir,
        args.output,
        results=args.results,
        root=args.output.parent,
        data_source="fixture" if args.fixture else "runtime",
        params_path=args.params,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
