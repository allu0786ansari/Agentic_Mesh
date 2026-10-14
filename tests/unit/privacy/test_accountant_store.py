"""The accountant persists, only grows, and cannot be reset (flaw P2)."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

pytest.importorskip("opacus")

from agmesh.privacy import accountant_store as store_mod
from agmesh.privacy.accountant_store import AccountantStore, StoreMismatchError
from agmesh.privacy.calibrate import Mechanism

STEP = Mechanism("dpsgd", 1.1, 0.05, 1)


def _open(path: Path) -> AccountantStore:
    return AccountantStore(path, node="a1", delta=1e-5, target_epsilon=10.0)


def test_epsilon_grows_and_persists_across_restart(tmp_path: Path) -> None:
    path = tmp_path / "run" / "a1.json"
    store = _open(path)
    assert store.epsilon() == 0.0
    seen = []
    for _ in range(30):
        store.record(STEP)
        seen.append(store.epsilon())
    assert all(b > a for a, b in itertools.pairwise(seen))
    assert store.steps("dpsgd") == 30 and len(store.history) == 1  # merged
    reopened = _open(path)  # what a recreated client object does
    assert reopened.epsilon() == pytest.approx(seen[-1])
    reopened.record(STEP)
    assert reopened.epsilon() > seen[-1] and reopened.steps("dpsgd") == 31
    reopened.record(Mechanism("normalisation", 5.0, 1.0, 1))
    assert len(_open(path).history) == 2
    assert not list(path.parent.glob("*.tmp"))


def test_reopening_with_other_settings_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "a1.json"
    _open(path).record(STEP)
    with pytest.raises(StoreMismatchError):
        AccountantStore(path, node="a1", delta=1e-6, target_epsilon=10.0)
    with pytest.raises(StoreMismatchError):
        AccountantStore(path, node="a1", delta=1e-5, target_epsilon=3.0)
    with pytest.raises(StoreMismatchError):
        AccountantStore(path, node="a2", delta=1e-5, target_epsilon=10.0)


def test_no_reset_path_exists() -> None:
    import inspect

    from agmesh.privacy import budget_guard, calibrate, client_budget, noisy_stats

    banned = ("reset", "clear", "truncate", "rewind", "forget", "delete")
    for module in (store_mod, budget_guard, calibrate, client_budget, noisy_stats):
        for name, obj in inspect.getmembers(module):
            assert not any(b in name.lower() for b in banned), f"{module.__name__}.{name}"
            if inspect.isclass(obj):
                for attr in vars(obj):
                    assert not any(b in attr.lower() for b in banned), f"{obj.__name__}.{attr}"
