"""One client's whole privacy budget, set up once before it releases anything.

``prepare`` resolves δ from the client's number of training windows (flaw P5). It fixes
the noise of the one-off releases from their configured standalone ε:

* the normalisation histograms, always;
* the H1c score histogram, if the run uses the DP federated threshold.

It then calibrates DP-SGD for the planned steps so that everything composed in RDP is
within the target ε (flaws P1, F8). It returns the plan, the persistent store and the
guard. The H1c release happens after training; it is **reserved** in the guard from
the start, so training can never eat into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from agmesh.common.config import Params
from agmesh.privacy import noisy_stats
from agmesh.privacy.accountant_store import ROOT, AccountantStore, store_path
from agmesh.privacy.budget_guard import BudgetGuard
from agmesh.privacy.calibrate import (
    Mechanism,
    PrivacyPlan,
    calibrate,
    gaussian_noise_multiplier,
    plan_steps,
)

THRESHOLD = "threshold_histogram"


@dataclass
class ClientBudget:
    plan: PrivacyPlan
    store: AccountantStore
    guard: BudgetGuard
    stats_noise_multiplier: float
    threshold: Mechanism | None


def planned_epochs(params: Params) -> int:
    """Local DP and federated runs train for the same total number of epochs."""
    return params.federated.max_rounds * params.federated.local_epochs


def batch_for(params: Params, site: str) -> int:
    return params.models.vae.batch if site == "site_a" else params.models.ae.batch


def calibrate_steps(params: Params, site: str, n_windows: int) -> tuple[float, int]:
    """Sampling rate and planned steps of a run (the ε = ∞ reference uses the same)."""
    return plan_steps(n_windows, batch_for(params, site), planned_epochs(params))


def prepare(
    params: Params,
    *,
    site: str,
    node: str,
    n_windows: int,
    run_id: str,
    target_epsilon: float | None = None,
    threshold_epsilon: float | None = None,
    epochs: int | None = None,
    root: Path = ROOT,
) -> ClientBudget:
    cfg = params.privacy
    target = cfg.target_epsilon if target_epsilon is None else target_epsilon
    if not math.isfinite(target):
        raise ValueError("prepare() is for private runs; ε = ∞ runs need no budget")
    delta = cfg.resolve_delta(n_windows)
    z_stats = gaussian_noise_multiplier(cfg.normalisation_stats_epsilon, delta)
    fixed = [noisy_stats.mechanism(z_stats)]
    threshold = None
    if threshold_epsilon is not None:
        threshold = Mechanism(
            THRESHOLD, gaussian_noise_multiplier(threshold_epsilon, delta), 1.0, 1
        )
        fixed.append(threshold)
    plan = calibrate(
        target_epsilon=target,
        delta=delta,
        n_windows=n_windows,
        batch=batch_for(params, site),
        epochs=epochs or planned_epochs(params),
        max_grad_norm=cfg.max_grad_norm,
        fixed=tuple(fixed),
    )
    store = AccountantStore(
        store_path(run_id, node, root),
        node=node,
        delta=delta,
        target_epsilon=target,
        plan=plan.as_dict(),
    )
    guard = BudgetGuard(store, reserved=(threshold,) if threshold is not None else ())
    return ClientBudget(plan, store, guard, z_stats, threshold)
