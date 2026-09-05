from __future__ import annotations

import pytest

from federated.convergence import ConvergenceChecker


def test_convergence_requires_patience_without_meaningful_improvement() -> None:
    checker = ConvergenceChecker(patience=2, min_delta=0.01)

    assert checker.update(1.0) is False
    assert checker.update(0.995) is False
    assert checker.update(0.994) is True
    assert checker.summary()["best_loss"] == 1.0


def test_convergence_resets_stale_rounds_after_improvement() -> None:
    checker = ConvergenceChecker(patience=2, min_delta=0.01)

    checker.update(1.0)
    checker.update(0.995)
    checker.update(0.98)

    assert checker.should_stop is False
    assert checker.stale_rounds == 0


def test_convergence_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        ConvergenceChecker(patience=0)
    with pytest.raises(ValueError):
        ConvergenceChecker(min_delta=-1)
