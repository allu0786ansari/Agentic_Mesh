"""ADR 0005 shift candidates: causality, warm-up, and what each one is for."""

from __future__ import annotations

import numpy as np
import pytest

from agmesh.common.config import ShiftWindow
from agmesh.evaluation.detection import (
    evaluate_node,
    evaluate_node_full,
    load_node_arrays,
    node_arrays_npz,
)
from agmesh.models.shift import ShiftReference, apply_shift, causal_stats, scored_mask

STRIDE = 10.0
WIN = ShiftWindow(history_s=1800, gap_s=120, min_history_s=300)


def _times(n: int, start: float = 0.0) -> np.ndarray:
    return start + STRIDE * np.arange(n, dtype=np.float64)


def test_warm_up_restarts_in_every_segment() -> None:
    t = np.concatenate([_times(100), _times(100, start=1e6)])
    ok = scored_mask(t, STRIDE, WIN)
    first = int((WIN.gap_s + WIN.min_history_s) / STRIDE)  # 42 windows
    assert not ok[:first].any() and ok[first:100].all()
    assert not ok[100 : 100 + first].any() and ok[100 + first :].all()
    shuffled = np.random.default_rng(0).permutation(t.size)
    assert (scored_mask(t[shuffled], STRIDE, WIN) == ok[shuffled]).all()


def test_history_is_causal() -> None:
    rng = np.random.default_rng(1)
    n = 600
    t = _times(n)
    x = rng.normal(size=(n, 3))
    med, mad = causal_stats(x, t, STRIDE, WIN)
    later = x.copy()
    later[400:] += 100.0  # changing the future must not change the past
    med2, mad2 = causal_stats(later, t, STRIDE, WIN)
    np.testing.assert_array_equal(med[:400], med2[:400])
    np.testing.assert_array_equal(mad[:400], mad2[:400])
    # the gap: the 12 windows just before t are never in its history
    i = 300
    hist = x[(t >= t[i] - WIN.history_s) & (t < t[i] - WIN.gap_s)]
    np.testing.assert_allclose(med[i], np.median(hist, axis=0))


def _reference(n_features: int = 4) -> ShiftReference:
    rng = np.random.default_rng(2)
    train = rng.gamma(2.0, 0.05, size=(2000, n_features))
    return ShiftReference.fit(train, train.mean(axis=1))


def test_r1_bounds_one_feature() -> None:
    ref = _reference()
    fe = np.full((2, 4), 0.1)
    fe[1, 0] = 1e6  # one feature at a new operating point
    out = apply_shift("r1", fe.mean(axis=1), fe, _times(2), STRIDE, ref, WIN)
    assert out.scored.all()
    assert out.scores[1] - out.scores[0] <= 1.0 / 4 + 1e-12
    assert np.all((out.scores >= 0) & (out.scores <= 1))


def test_r2_removes_an_offset_and_keeps_a_spike() -> None:
    ref = _reference()
    n = 600
    t = _times(n)
    fe = np.full((n, 4), 0.1)
    fe[:, 0] += 5.0  # persistent: a new operating point for the whole segment
    fe[500:506, 1] += 5.0  # sudden: one minute
    r0 = apply_shift("r0", fe.mean(axis=1), fe, t, STRIDE, ref, WIN)
    r2 = apply_shift("r2", fe.mean(axis=1), fe, t, STRIDE, ref, WIN)
    assert r0.scores[100] > 1.0  # R0 is dominated by the offset
    assert r2.scored[100] and r2.scores[100] == pytest.approx(0.0)
    assert r2.scores[502] == pytest.approx(5.0 / 4)
    assert not r2.scored[:42].any() and np.isnan(r2.scores[:42]).all()


def test_r3_is_median_plus_k_mad() -> None:
    rng = np.random.default_rng(3)
    n = 800
    t = _times(n)
    base = rng.normal(1.0, 0.1, n)
    base[400:] += 3.0  # level shift: R3 re-centres after the history catches up
    ref = ShiftReference.fit(None, base[:200])
    out = apply_shift("r3", base, None, t, STRIDE, ref, WIN)
    med, mad = causal_stats(base, t, STRIDE, WIN)
    i = 700
    assert out.scores[i] == pytest.approx((base[i] - med[i]) / mad[i])
    assert abs(np.nanmedian(out.scores[650:])) < 1.0
    with pytest.raises(ValueError):
        apply_shift("r9", base, None, t, STRIDE, ref, WIN)


def test_abstention_counts_attacks_as_missed_and_skips_normals() -> None:
    n = 200
    t = _times(n)
    y = np.zeros(n, bool)
    y[5:8] = True  # an attack inside the warm-up
    y[150:153] = True
    s = np.where(y, 10.0, 0.0) + np.linspace(0, 0.1, n)
    ok = np.ones(n, bool)
    ok[:42] = False
    kw = {
        "window_length_s": 60.0,
        "stride_s": STRIDE,
        "events": [(t[5], t[8]), (t[150], t[153])],
        "target_fpr": 0.01,
        "tolerance_s": 0.0,
    }
    nv = np.linspace(0, 1, 500)
    plain = evaluate_node(nv, s, y, t, **kw)  # type: ignore[arg-type]
    metrics, arrays = evaluate_node_full(nv, s, y, t, scored=ok, **kw)  # type: ignore[arg-type]
    assert plain["event_recall"] == 1.0 and metrics["event_recall"] == 0.5
    assert metrics["abstained_attack_windows"] == 3
    assert metrics["abstained_share"] == pytest.approx(42 / n)
    assert not arrays.alerts[:42].any() and np.isnan(arrays.scores[:42]).all()
    assert arrays.event_detected.tolist() == [False, True]
    assert (arrays.event_id[y] >= 0).all() and (arrays.event_id[~y] == -1).all()
    back = load_node_arrays(node_arrays_npz({"a1": arrays}))
    np.testing.assert_array_equal(back["a1"].alerts, arrays.alerts)
    assert back["a1"].threshold == arrays.threshold
