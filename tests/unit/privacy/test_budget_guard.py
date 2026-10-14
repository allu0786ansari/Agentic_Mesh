"""The guard blocks any release past the budget, counting what is reserved (flaw P2)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("opacus")

from agmesh.privacy.accountant_store import AccountantStore
from agmesh.privacy.budget_guard import BudgetExhausted, BudgetGuard
from agmesh.privacy.calibrate import Mechanism, epsilon

STEP = Mechanism("dpsgd", 1.5, 0.1, 1)


def test_guard_blocks_overspend_and_records_nothing(tmp_path: Path) -> None:
    store = AccountantStore(tmp_path / "a1.json", node="a1", delta=1e-5, target_epsilon=2.0)
    guard = BudgetGuard(store)
    spent = 0
    with pytest.raises(BudgetExhausted):
        while True:
            guard.spend(STEP)
            spent += 1
    assert spent > 0 and store.steps("dpsgd") == spent
    assert store.epsilon() <= 2.0
    assert epsilon([*store.history, STEP], 1e-5) > 2.0  # the refused step really would exceed
    with pytest.raises(BudgetExhausted):  # and it stays refused after reopening
        BudgetGuard(
            AccountantStore(tmp_path / "a1.json", node="a1", delta=1e-5, target_epsilon=2.0)
        ).spend(STEP)


def test_reserved_release_is_protected(tmp_path: Path) -> None:
    hist = Mechanism("threshold_histogram", 3.0, 1.0, 1)
    store = AccountantStore(tmp_path / "a1.json", node="a1", delta=1e-5, target_epsilon=2.0)
    guard = BudgetGuard(store, reserved=(hist,))
    with pytest.raises(BudgetExhausted):
        while True:
            guard.spend(STEP)
    before = store.epsilon()
    after = guard.release_reserved("threshold_histogram")  # always fits: it was counted
    assert after > before and after <= 2.0 + 1e-9
    with pytest.raises(KeyError):
        guard.release_reserved("threshold_histogram")
