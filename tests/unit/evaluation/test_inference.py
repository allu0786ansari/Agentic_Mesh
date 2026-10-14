"""Event-level block bootstrap (ADR 0005, Part 2)."""

from __future__ import annotations

import numpy as np
import pytest

from agmesh.evaluation.detection import NodeArrays, evaluate_node_full
from agmesh.evaluation.inference import EventBootstrap, Units, compare, holm

STRIDE = 10.0


def _node(
    rng: np.random.Generator,
    *,
    fp_rate: float,
    hours: float = 6.0,
    events: int = 5,
    abstain: int = 0,
    detect: float = 0.9,
) -> tuple[dict[str, float], NodeArrays]:
    n = int(hours * 3600 / STRIDE)
    t = STRIDE * np.arange(n, dtype=np.float64)
    y = np.zeros(n, bool)
    intervals = []
    for k in range(events):
        a = int((k + 1) * n / (events + 1))
        y[a : a + 12] = True
        intervals.append((t[a], t[a + 12]))
    s = rng.normal(size=n)
    s[y] += np.where(rng.random(events) < detect, 4.0, -1.0).repeat(12)
    s = np.round(s, 1)  # ties on purpose
    # normal scores above 2.33 mark the 1% threshold; inflate some to reach fp_rate
    s[~y & (rng.random(n) < fp_rate)] += 5.0
    ok = np.ones(n, bool)
    ok[:abstain] = False
    return evaluate_node_full(
        rng.normal(size=5000),
        s,
        y,
        t,
        window_length_s=60.0,
        stride_s=STRIDE,
        events=intervals,
        target_fpr=0.01,
        tolerance_s=0.0,
        scored=ok,
    )


def test_observed_values_equal_the_point_metrics() -> None:
    rng = np.random.default_rng(0)
    m1, a1 = _node(rng, fp_rate=0.05, abstain=30)
    m2, a2 = _node(rng, fp_rate=0.01)
    plan = EventBootstrap({"a1": a1, "a2": a2}, block_s=1800, b=50, seed=1)
    obs = plan.observed({11: {"a1": a1, "a2": a2}})
    for key in (
        "auroc",
        "auprc",
        "event_recall",
        "window_fpr",
        "false_alarms_per_hour",
        "ttd_mean_s",
    ):
        assert obs[key] == pytest.approx((m1[key] + m2[key]) / 2, rel=1e-9, abs=1e-12), key


def test_units_are_blocks_and_events() -> None:
    _, arrays = _node(np.random.default_rng(2), fp_rate=0.02, hours=3.0, events=4)
    u = Units.of(arrays, 1800)
    assert u.n_events == 4 and u.n_blocks == 6
    assert (u.block[arrays.labels] == -1).all() and (u.block[~arrays.labels] >= 0).all()


def test_identical_configs_and_a_clear_difference() -> None:
    rng = np.random.default_rng(3)
    _, base = _node(rng, fp_rate=0.02)
    _, worse = _node(np.random.default_rng(3), fp_rate=0.02)
    plan = EventBootstrap({"a1": base}, block_s=1800, b=400, seed=0)
    same = compare(plan, {11: {"a1": base}}, {11: {"a1": worse}}, "window_fpr")
    assert same.observed_difference == 0 and same.ci == (0.0, 0.0) and same.p_one_sided == 1.0
    noisy = NodeArrays(
        **{
            **base.__dict__,
            "alerts": base.alerts | (~base.labels & (np.arange(base.labels.size) % 7 == 0)),
        }
    )
    diff = compare(plan, {11: {"a1": noisy}}, {11: {"a1": base}}, "window_fpr")
    assert diff.observed_difference > 0.1 and diff.ci[0] > 0 and diff.p_one_sided < 0.01


def test_coverage_of_a_known_fpr_difference() -> None:
    """Blocks are i.i.d.; A alerts on 10% of normal windows and B on 5%."""
    rng = np.random.default_rng(4)
    covered = 0
    reps = 60
    for _ in range(reps):
        _, a = _node(rng, fp_rate=0.0, hours=8.0)
        u = rng.random(a.labels.size)
        alert_a = ~a.labels & (u < 0.10)
        alert_b = ~a.labels & (u < 0.05)
        ra = NodeArrays(**{**a.__dict__, "alerts": alert_a})
        rb = NodeArrays(**{**a.__dict__, "alerts": alert_b})
        plan = EventBootstrap({"a1": ra}, block_s=1800, b=300, seed=int(rng.integers(1 << 30)))
        c = compare(plan, {1: {"a1": ra}}, {1: {"a1": rb}}, "window_fpr")
        covered += c.ci[0] <= 0.05 <= c.ci[1]
    assert covered / reps >= 0.85


def test_seeds_are_averaged_and_units_must_match() -> None:
    rng = np.random.default_rng(5)
    m1, a1 = _node(rng, fp_rate=0.01)
    m2, a2 = _node(np.random.default_rng(5), fp_rate=0.01, detect=0.0)
    plan = EventBootstrap({"a1": a1}, block_s=1800, b=20)
    obs = plan.observed({11: {"a1": a1}, 23: {"a1": a2}})
    assert obs["event_recall"] == pytest.approx((m1["event_recall"] + m2["event_recall"]) / 2)
    _, other = _node(rng, fp_rate=0.01, hours=5.0)
    with pytest.raises(ValueError, match="differ"):
        plan.draws({11: {"a1": other}})


def test_holm() -> None:
    adjusted = holm([0.01, 0.04, 0.03, float("nan")])
    assert adjusted[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(adjusted[3])
