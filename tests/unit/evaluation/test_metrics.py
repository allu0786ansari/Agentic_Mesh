"""Window- and event-level metrics against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from agmesh.evaluation import events as ev
from agmesh.evaluation import metrics as m


def test_auroc_by_hand() -> None:
    # positives 0.35, 0.8 vs negatives 0.1, 0.4: 3 of 4 pairs ordered correctly
    assert m.auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert m.auroc([0, 1], [0.5, 0.5]) == 0.5
    assert m.auroc([1, 1], [0.2, 0.3]) != m.auroc([1, 1], [0.2, 0.3])  # nan: one class


def test_auprc_by_hand() -> None:
    # ranks: hit (P=1, R=.5), miss, hit (P=2/3, R=1), miss -> 0.5*1 + 0.5*2/3
    assert math.isclose(m.auprc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]), 0.5 + 1 / 3)
    assert m.auprc([1, 0], [0.5, 0.5]) == 0.5  # tied scores move together


@settings(max_examples=50, deadline=None)
@given(st.lists(st.tuples(st.booleans(), st.floats(0, 1)), min_size=2, max_size=60))
def test_auroc_matches_pair_counting(rows: list[tuple[bool, float]]) -> None:
    y = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows])
    if y.all() or not y.any():
        return
    pos, neg = s[y], s[~y]
    pairs = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    assert math.isclose(m.auroc(y, s), pairs / (pos.size * neg.size))


def test_confusion_and_false_alarms() -> None:
    y = np.zeros(10, dtype=int)
    alerts = np.zeros(10, dtype=bool)
    alerts[[1, 2, 5]] = True  # windows 1-2 are one false alarm, window 5 another
    starts = np.arange(10) * 10.0
    c = m.confusion(y, alerts)
    assert (c.tp, c.fp, c.tn, c.fn) == (0, 3, 7, 0)
    assert c.fpr == 0.3
    assert math.isclose(m.false_alarms_per_hour(y, alerts, starts, 10.0), 2 / (100 / 3600))
    assert m.false_alarms_per_hour(y, np.zeros(10, bool), starts, 10.0) == 0.0


def test_pooled_auroc_and_its_error_bound() -> None:
    pos = np.array([0.95, 0.55])
    neg = np.array([0.05, 0.551])
    fine = m.pooled_auroc([m.score_histogram(pos)], [m.score_histogram(neg)])
    assert fine.auroc == 0.75 and fine.max_bin_error == 0.0
    coarse = m.pooled_auroc([m.score_histogram(pos, 10)], [m.score_histogram(neg, 10)])
    assert coarse.auroc == 0.875 and coarse.max_bin_error == 0.125


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.floats(0, 1), min_size=1, max_size=40),
    st.lists(st.floats(0, 1), min_size=1, max_size=40),
    st.integers(1, 3),
)
def test_pooled_auroc_error_bound_holds(pos: list[float], neg: list[float], clients: int) -> None:
    exact = m.auroc([1] * len(pos) + [0] * len(neg), pos + neg)
    split_pos = np.array_split(np.array(pos), clients)
    split_neg = np.array_split(np.array(neg), clients)
    pooled = m.pooled_auroc(
        [m.score_histogram(p, 20) for p in split_pos], [m.score_histogram(n, 20) for n in split_neg]
    )
    assert abs(pooled.auroc - exact) <= pooled.max_bin_error + 1e-12


def test_point_adjustment_inflates_f1() -> None:
    """Why point-adjusted F1 is excluded: one lucky alert scores a perfect event."""
    y = np.zeros(1000, dtype=int)
    y[400:500] = 1
    alerts = np.zeros(1000, dtype=bool)
    alerts[450] = True
    assert m.f1(y, alerts) < 0.02
    assert m.f1(y, m.point_adjust(y, alerts)) == 1.0
    # random alerts at a 2% rate: honest F1 stays near zero, point-adjusted F1 does not
    rng = np.random.default_rng(0)
    noise = rng.random(1000) < 0.02
    assert m.f1(y, noise) < 0.1
    assert m.f1(y, m.point_adjust(y, noise)) > 0.8


def test_event_recall_and_time_to_detect() -> None:
    events = [(100.0, 160.0)]
    early = ev.event_scores(events, [(150.0, 160.0)], tolerance_s=60)
    assert early.recall == 1.0 and early.ttd_s == [60.0]
    late = ev.event_scores(events, [(200.0, 210.0)], tolerance_s=60)  # inside the tolerance
    assert late.recall == 1.0 and late.ttd_median_s == 110.0
    missed = ev.event_scores(events, [(230.0, 240.0)], tolerance_s=60)
    assert missed.recall == 0.0 and math.isnan(missed.ttd_median_s)
    assert math.isnan(ev.event_scores([], [], 60).recall)


def test_range_scores_by_hand() -> None:
    half = ev.range_scores([(0.0, 10.0)], [(5.0, 15.0)])
    assert (half.precision, half.recall, half.f1) == (0.5, 0.5, 0.5)
    fragmented = ev.range_scores([(0.0, 10.0)], [(0.0, 2.0), (4.0, 6.0)])
    assert math.isclose(fragmented.recall, 0.5 * 4 / 10)  # cardinality penalty 1/2
    assert fragmented.precision == 1.0
    existence = ev.range_scores([(0.0, 10.0)], [(0.0, 2.0), (4.0, 6.0)], alpha=1.0)
    assert existence.recall == 1.0
    empty = ev.range_scores([(0.0, 10.0)], [])
    assert empty.recall == 0.0 and empty.f1 == 0.0


def test_intervals_and_row_events() -> None:
    assert ev.merge([(5.0, 8.0), (0.0, 5.0), (10.0, 12.0)]) == [(0.0, 8.0), (10.0, 12.0)]
    starts = np.array([0.0, 10.0, 20.0, 30.0])
    assert ev.alert_intervals(starts, [False, True, True, False], 10.0) == [
        (10.0, 20.0),
        (20.0, 30.0),
    ]
    t = np.arange(10, dtype=float)
    label = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    assert ev.events_from_rows(t, label) == [(1.0, 3.0), (5.0, 8.0)]


def test_episode_events() -> None:
    starts = np.array([0.0, 10.0, 20.0, 30.0, 100.0])
    episode = np.array([-1, 0, -1, 0, 1])
    assert ev.events_from_episodes(starts, episode, 10.0) == [(10.0, 40.0), (100.0, 110.0)]
