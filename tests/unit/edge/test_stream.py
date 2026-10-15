"""Streaming R3, pacing and evidence helpers (ADR 0009 §3, §8)."""

from __future__ import annotations

import numpy as np

from agmesh.common.config import ShiftWindow
from agmesh.edge import evidence as ev
from agmesh.edge.replay import Pacer
from agmesh.edge.stream import CausalR3
from agmesh.models.shift import ShiftReference, apply_shift


def test_incremental_r3_equals_batch_r3() -> None:
    rng = np.random.default_rng(3)
    window = ShiftWindow(history_s=600, gap_s=60, min_history_s=120)
    # three segments separated by gaps, with a level shift inside the second
    t = np.concatenate(
        [np.arange(0, 3000, 10), np.arange(5000, 8000, 10), np.arange(9000, 9300, 10)]
    )
    base = rng.gamma(2.0, 1.0, t.size) + np.where((t > 6000) & (t < 8000), 5.0, 0.0)
    ref = ShiftReference(None, float(np.median(base)))
    batch = apply_shift("r3", base, None, t, 10.0, ref, window)
    r3 = CausalR3(window, 10.0, ref.score_median)
    stream = [r3.update(float(ti), float(bi)) for ti, bi in zip(t, base, strict=True)]
    ok = np.array([o for _s, o in stream])
    scores = np.array([s for s, _o in stream])
    assert (ok == batch.scored).all()
    assert np.allclose(scores[ok], batch.scores[ok], rtol=0, atol=1e-12)
    assert not ok[:18].any() and ok[18]  # warm-up: gap 60 s + 120 s of history


def test_pacer_schedules_by_data_time_and_caps_gaps() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(dt: float) -> None:
        slept.append(dt)
        now[0] += dt

    pacer = Pacer(speed=10.0, max_gap_s=60.0, clock=lambda: now[0], sleep=sleep)
    assert pacer.emit(100.0) == 0.0
    assert pacer.emit(110.0) == 1.0  # 10 s of data at 10x
    assert pacer.emit(10_000.0) == 7.0  # a long gap counts as 60 s
    assert slept == [1.0, 6.0]
    fast = Pacer(speed=0.0, clock=lambda: 5.0, sleep=lambda _dt: None)
    assert fast.emit(1.0) == 5.0 and fast.emit(1e9) == 5.0


def test_evidence_helpers() -> None:
    assert [ev.count_bucket(n) for n in (0, 1, 10, 11, 100, 101)] == [
        "0",
        "1-10",
        "1-10",
        "11-100",
        "11-100",
        ">100",
    ]
    assert [ev.rate_bucket(n, 2, 8) for n in (1, 5, 9)] == ["low", "normal", "high"]
    names = ["a", "b", "c", "d", "e", "f", "g"]
    buckets = {n: [1.0, 2.0, 4.0] for n in names}
    errs = [0.5, 1.0, 3.0, 5.0, 1.5, 2.5, 3.5]
    feats = ev.top_features(errs, [1, -1, 1, -1, 1, 1, -1], names, buckets)
    assert [f.name for f in feats] == ["d", "g", "c", "f", "e"]  # >= p90, by error / p99
    assert [f.magnitude for f in feats] == ["high", "medium", "medium", "medium", "low"]
    assert feats[0].direction == "down" and feats[2].direction == "up"
    assert ev.top_features([0.1], [1.0], ["a"], {"a": [1.0, 2.0, 3.0]}) == []
    window = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 6.0]])
    assert ev.constants_changed(window, ["k1", "k2"], {"k1": 1.0, "k2": 5.0}) == ["k2"]
    a = ev.stable_id("alert", "sha256:x", "b1", 100.0)
    assert (
        a
        == ev.stable_id("alert", "sha256:x", "b1", 100.0)
        != ev.stable_id("alert", "sha256:x", "b2", 100.0)
    )
    assert ev.iso(0.0) == "1970-01-01T00:00:00.000Z"
