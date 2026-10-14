"""Thresholds, isotonic calibration and the statistical tests, against hand values."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agmesh.evaluation import stats
from agmesh.evaluation.calibration import IsotonicCalibrator, pava, reliability
from agmesh.evaluation.thresholds import fpr_threshold


def test_fpr_threshold_by_hand() -> None:
    scores = np.arange(100, dtype=float)
    t = fpr_threshold(scores, 0.05)
    assert t.value == 94.0 and t.achieved_fpr == 0.05
    assert fpr_threshold(scores, 0.01).value == 98.0
    assert fpr_threshold(np.ones(10), 0.05).achieved_fpr == 0.0  # ties never exceed
    with pytest.raises(ValueError):
        fpr_threshold([], 0.01)
    with pytest.raises(ValueError):
        fpr_threshold([1.0, float("nan")], 0.01)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=300), st.floats(0.001, 0.5))
def test_fpr_threshold_never_exceeds_target(scores: list[float], target: float) -> None:
    t = fpr_threshold(scores, target)
    assert t.achieved_fpr <= target + 1e-12


def test_isotonic_by_hand() -> None:
    cal = IsotonicCalibrator().fit([1, 2, 3, 4], [0, 1, 0, 1])
    assert cal.predict([1, 2, 2.5, 3, 4]).tolist() == [0.0, 0.5, 0.5, 0.5, 1.0]
    assert cal.predict([-5, 10]).tolist() == [0.0, 1.0]
    again = IsotonicCalibrator.from_dict(cal.to_dict())
    assert again.predict([2.5]).tolist() == [0.5]
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().predict([1.0])


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(-100, 100), min_size=1, max_size=50))
def test_pava_is_monotone_and_preserves_mean(y: list[float]) -> None:
    values = np.array(y)
    fitted = pava(values, np.ones_like(values))
    assert (np.diff(fitted) >= -1e-9).all()
    assert math.isclose(fitted.sum(), values.sum(), rel_tol=1e-9, abs_tol=1e-6)


def test_reliability_by_hand() -> None:
    r = reliability([0.05, 0.05, 0.95, 0.95], [0, 0, 1, 1], bins=10)
    assert math.isclose(r.ece, 0.05)
    assert r.counts[0] == 2 and r.counts[9] == 2


def test_wilcoxon_matches_published_values() -> None:
    # exact two-sided p-values (as given by scipy.stats.wilcoxon)
    five_up = stats.wilcoxon_signed_rank([1, 2, 3, 4, 5], [0] * 5)
    assert five_up.statistic == 15 and math.isclose(five_up.p_value, 0.0625)
    one_down = stats.wilcoxon_signed_rank([1, -2, 3, 4, 5], [0] * 5)
    assert one_down.statistic == 13 and math.isclose(one_down.p_value, 6 / 32)
    assert stats.wilcoxon_signed_rank([1, 2], [1, 2]).p_value == 1.0
    tied = stats.wilcoxon_signed_rank([1, 1, -1], [0, 0, 0])
    assert 0 < tied.p_value <= 1


def test_holm_by_hand() -> None:
    assert stats.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5]).tolist() == [0.5]


def test_bootstrap_intervals() -> None:
    ci = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=1)
    assert ci.estimate == 3.0 and ci.low <= 3.0 <= ci.high
    constant = stats.paired_bootstrap([0.9] * 5, [0.8] * 5)
    assert math.isclose(constant.mean_difference, 0.1)
    assert math.isclose(constant.ci.low, 0.1) and math.isclose(constant.ci.high, 0.1)
    assert constant.p_value == 0.0
    noisy = stats.paired_bootstrap([0.1, -0.1, 0.2, -0.2, 0.0], [0.0] * 5, seed=3)
    assert noisy.p_value > 0.5
    with pytest.raises(ValueError):
        stats.paired_bootstrap([1.0], [1.0, 2.0])
