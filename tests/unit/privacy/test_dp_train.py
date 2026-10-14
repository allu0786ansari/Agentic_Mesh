"""DP-SGD training: planned steps, accounted, guarded and reproducible."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("opacus")
pytest.importorskip("torch")

from agmesh.common.config import Params
from agmesh.models import nets
from agmesh.models.dp_train import fit_private
from agmesh.privacy.accountant_store import AccountantStore
from agmesh.privacy.budget_guard import BudgetExhausted, BudgetGuard
from agmesh.privacy.calibrate import PrivacyPlan, calibrate
from tests.conftest import REPO_PARAMS


def _setup(
    tmp_path: Path, name: str, target: float = 3.0
) -> tuple[npt.NDArray[np.float32], PrivacyPlan, AccountantStore]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 12)).astype(np.float32)
    plan = calibrate(
        target_epsilon=target, delta=1e-5, n_windows=400, batch=40, epochs=3, max_grad_norm=1.0
    )
    store = AccountantStore(tmp_path / f"{name}.json", node=name, delta=1e-5, target_epsilon=target)
    return x, plan, store


def test_trains_exactly_the_plan_and_stays_within_budget(tmp_path: Path) -> None:
    x, plan, store = _setup(tmp_path, "b1")
    model = nets.build("ae", 12, [16, 8], seed=3)
    before = nets.state_sha256(model)
    result = fit_private(model, x, plan=plan, guard=BudgetGuard(store), lr=1e-2, seed=3)
    assert result.steps == plan.steps == store.steps("dpsgd")
    assert result.epsilon <= 3.0 + 1e-9 and result.epsilon == pytest.approx(store.epsilon())
    assert nets.state_sha256(model) != before
    errors = nets.reconstruction_errors(model, x)
    assert errors.shape == x.shape and np.isfinite(errors).all()
    assert result.info()["secure_mode"] is False and len(result.history) == 3


def test_reproducible_with_the_same_seed(tmp_path: Path) -> None:
    hashes = []
    for k in range(2):
        x, plan, store = _setup(tmp_path, f"r{k}")
        model = nets.build("vae", 12, [16], 4, seed=5)
        fit_private(model, x, plan=plan, guard=BudgetGuard(store), lr=1e-2, seed=5)
        hashes.append(nets.state_sha256(model))
    assert hashes[0] == hashes[1]


def test_exhausted_budget_stops_training(tmp_path: Path) -> None:
    x, plan, store = _setup(tmp_path, "a1")
    guard = BudgetGuard(store)
    model = nets.build("ae", 12, [16, 8], seed=1)
    fit_private(model, x, plan=plan, guard=guard, lr=1e-2, seed=1)
    with pytest.raises(BudgetExhausted):  # a second full plan cannot fit in the same budget
        fit_private(model, x, plan=plan, guard=guard, lr=1e-2, seed=2)
    with pytest.raises(ValueError, match="calibrated for"):
        fit_private(model, x[:10], plan=plan, guard=guard, lr=1e-2, seed=2)


def test_dp_sanity_runner_on_fixture(
    params: Params, site_a_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    import json
    import math

    from agmesh.experiments import dp_sanity

    raw = params.model_dump()
    raw["federated"]["max_rounds"] = 2
    p = Params.model_validate(raw)
    out = tmp_path / "privacy" / "dp_sanity.json"
    rows = dp_sanity.run(
        p,
        ["site_a"],
        site_a_run[1].parent,
        out,
        epsilons=(3.0, math.inf),
        results=tmp_path / "none",
        root=out.parent,
        data_source="fixture",
        params_path=REPO_PARAMS,
    )
    private, reference = rows
    assert private["privacy"]["epsilon"] <= 3.0 + 1e-9
    names = [m["name"] for m in private["privacy"]["history"]]
    assert names == ["normalisation", "dpsgd"]  # statistics spent first, then training
    assert reference["privacy"]["epsilon"] == math.inf
    assert private["plan"]["steps"] == reference["plan"]["steps"]
    for row in rows:
        assert set(row["metrics"]) >= {"auroc", "window_fpr", "event_recall"}
    stores = list((out.parent).glob("dp_sanity__site_a__*__eps3__seed11/*.json"))
    assert len(stores) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["kind"] == "privacy.dp_sanity"
