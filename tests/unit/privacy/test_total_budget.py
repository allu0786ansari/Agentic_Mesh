"""Executing the whole plan, statistics and threshold included, stays within ε (P1, F8)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("opacus")

from agmesh.common.config import Params
from agmesh.privacy import client_budget, noisy_stats
from agmesh.privacy.budget_guard import BudgetExhausted
from agmesh.privacy.calibrate import calibrate, epsilon, plan_steps


@pytest.mark.parametrize("target", [1.0, 3.0, 10.0])
def test_plan_executes_within_target(params: Params, tmp_path: Path, target: float) -> None:
    raw = params.model_dump()
    raw["federated"]["max_rounds"] = 3  # keep the test quick; the arithmetic is the same
    p = Params.model_validate(raw)
    budget = client_budget.prepare(
        p,
        site="site_a",
        node="a1",
        n_windows=3060,
        run_id=f"t{target}",
        target_epsilon=target,
        threshold_epsilon=0.1,
        root=tmp_path,
    )
    plan = budget.plan
    assert plan.planned_epsilon <= target and plan.planned_epsilon >= target - 0.01
    assert plan.epsilon_basic_composition() >= plan.planned_epsilon  # RDP is tighter
    budget.guard.spend(noisy_stats.mechanism(budget.stats_noise_multiplier))
    for _ in range(plan.steps):
        budget.guard.spend(plan.sgd.__class__("dpsgd", plan.noise_multiplier, plan.sample_rate, 1))
    budget.guard.release_reserved(client_budget.THRESHOLD)
    final = budget.store.epsilon()
    assert final <= target + 1e-9
    assert final == pytest.approx(plan.planned_epsilon, rel=1e-9)
    with pytest.raises(BudgetExhausted):
        budget.guard.spend(plan.sgd.__class__("dpsgd", plan.noise_multiplier, plan.sample_rate, 1))
    summary = budget.store.summary()
    assert summary["plan"]["fixed"][0]["name"] == "normalisation"


def test_warm_up_is_part_of_the_plan() -> None:
    _q, steps = plan_steps(1000, 100, epochs=5, warmup_epochs=2)
    assert steps == 70
    with_warmup = calibrate(
        target_epsilon=3,
        delta=1e-5,
        n_windows=1000,
        batch=100,
        epochs=5,
        max_grad_norm=1.0,
        warmup_epochs=2,
    )
    without = calibrate(
        target_epsilon=3, delta=1e-5, n_windows=1000, batch=100, epochs=5, max_grad_norm=1.0
    )
    assert with_warmup.noise_multiplier > without.noise_multiplier


def test_unreachable_targets_fail_loudly() -> None:
    from agmesh.privacy.calibrate import Mechanism

    expensive = (Mechanism("normalisation", 0.5, 1.0, 1),)
    with pytest.raises(ValueError, match="one-off releases alone"):
        calibrate(
            target_epsilon=1,
            delta=1e-5,
            n_windows=1000,
            batch=100,
            epochs=5,
            max_grad_norm=1.0,
            fixed=expensive,
        )
    assert epsilon([], 1e-5) == 0.0
