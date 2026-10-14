"""Shift-robust scoring candidates, applied after the model (ADR 0005).

``r0``  the ADR 0003 window score, unchanged.
``r1``  bounded evidence: each feature's window error is replaced by its mid-rank
        quantile in the node's own *training* errors (probability integral
        transform), and the score is the mean over features. One feature can add at
        most ``1 / n_features`` to the score.
``r2``  causal error detrending: each feature's window error minus the median of
        that feature's errors over the causal history, floored at 0, then averaged.
        A persistent offset (a new operating point) fades out; a sudden change stays.
``r3``  robust adaptive threshold, written as a score: ``(s - median) / MAD`` of the
        window score over the causal history. A fixed threshold ``k`` on it is the
        rule ``s > median + k * MAD``; ``k`` is calibrated on normal validation like
        any other threshold.

The causal history of a window starting at ``t`` is every earlier window of the same
segment starting in ``[t - history_s, t - gap_s)``. A segment is a run of windows
less than 1.5 strides apart. R2 and R3 **abstain** while ``t - segment start <
gap_s + min_history_s``: the window raises no alarm, an abstained attack window
counts as missed, and abstained normal windows are left out of the FPR.

Models without per-feature errors (Isolation Forest) use their raw score as a single
"feature": R1 is then a monotone transform and changes nothing, so it equals R0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from agmesh.common.config import ShiftWindow

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
ShiftId = Literal["r0", "r1", "r2", "r3"]
SHIFTS: tuple[ShiftId, ...] = ("r0", "r1", "r2", "r3")
#: The MAD in R3 is floored at this fraction of the node's median training score.
MAD_FLOOR_REL = 1e-3
#: Tie-break order of ADR 0005 (simpler first).
SIMPLICITY: dict[str, int] = {"r0": 0, "r1": 1, "r3": 2, "r2": 3}
MAD_TO_SD = 1.4826


@dataclass(frozen=True)
class Shifted:
    """Window scores after a shift candidate; ``scored`` is False where it abstains."""

    scores: F64
    scored: Bool

    @property
    def abstained_share(self) -> float:
        return float(1.0 - self.scored.mean()) if self.scored.size else 0.0


@dataclass(frozen=True)
class ShiftReference:
    """What a node knows from its own training windows (no labels)."""

    sorted_errors: F64 | None  # (training windows, features), each column sorted
    score_median: float

    @classmethod
    def fit(
        cls, train_feature_errors: npt.ArrayLike | None, train_scores: npt.ArrayLike
    ) -> ShiftReference:
        sorted_errors = None
        if train_feature_errors is not None:
            e = np.asarray(train_feature_errors, dtype=np.float64)
            if e.ndim != 2 or e.shape[0] == 0:
                raise ValueError("training feature errors must be a non-empty 2-D array")
            sorted_errors = np.sort(e, axis=0)
        return cls(sorted_errors, float(np.median(np.asarray(train_scores, dtype=np.float64))))

    def quantiles(self, feature_errors: npt.ArrayLike) -> F64:
        """Mid-rank quantile of each error in its feature's training distribution."""
        if self.sorted_errors is None:
            raise ValueError("this reference has no per-feature training errors")
        e = np.asarray(feature_errors, dtype=np.float64)
        ref = self.sorted_errors
        if e.ndim != 2 or e.shape[1] != ref.shape[1]:
            raise ValueError("feature errors do not match the reference")
        out = np.empty_like(e)
        for j in range(e.shape[1]):
            lo = np.searchsorted(ref[:, j], e[:, j], side="left")
            hi = np.searchsorted(ref[:, j], e[:, j], side="right")
            out[:, j] = (lo + hi) / (2.0 * ref.shape[0])
        return out


def _segments(t_sorted: F64, stride_s: float) -> npt.NDArray[np.int64]:
    breaks = np.diff(t_sorted) > 1.5 * stride_s
    return np.concatenate([[0], np.cumsum(breaks)]).astype(np.int64)


def scored_mask(window_start_s: npt.ArrayLike, stride_s: float, window: ShiftWindow) -> Bool:
    """True where the causal history is long enough for R2/R3 to score."""
    t = np.asarray(window_start_s, dtype=np.float64)
    order = np.argsort(t, kind="mergesort")
    ts = t[order]
    seg = _segments(ts, stride_s)
    first = np.zeros_like(ts)
    starts = np.flatnonzero(np.concatenate([[True], np.diff(seg) > 0]))
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else ts.size
        first[s:e] = ts[s]
    ok_sorted = ts - first >= window.gap_s + window.min_history_s - 1e-9
    out = np.empty(t.size, dtype=bool)
    out[order] = ok_sorted
    return out


def causal_stats(
    values: npt.ArrayLike, window_start_s: npt.ArrayLike, stride_s: float, window: ShiftWindow
) -> tuple[F64, F64]:
    """Median and MAD (scaled to an SD) of ``values`` over each window's causal history.

    ``values`` is (windows,) or (windows, features); NaN where the history is empty.
    """
    v = np.asarray(values, dtype=np.float64)
    squeeze = v.ndim == 1
    if squeeze:
        v = v[:, None]
    t = np.asarray(window_start_s, dtype=np.float64)
    if v.shape[0] != t.size:
        raise ValueError("values and window starts differ in length")
    order = np.argsort(t, kind="mergesort")
    ts, vs = t[order], v[order]
    seg = _segments(ts, stride_s)
    seg_first = np.searchsorted(seg, seg, side="left")
    lo = np.maximum(np.searchsorted(ts, ts - window.history_s, side="left"), seg_first)
    hi = np.searchsorted(ts, ts - window.gap_s, side="left")
    if window.gap_s == 0:
        hi = np.arange(ts.size)  # strictly earlier windows only
    med = np.full_like(vs, np.nan)
    mad = np.full_like(vs, np.nan)
    for i in range(ts.size):
        a, b = int(lo[i]), int(hi[i])
        if b > a:
            block = vs[a:b]
            m = np.median(block, axis=0)
            med[i] = m
            mad[i] = MAD_TO_SD * np.median(np.abs(block - m), axis=0)
    out_med = np.empty_like(med)
    out_mad = np.empty_like(mad)
    out_med[order] = med
    out_mad[order] = mad
    if squeeze:
        return out_med[:, 0], out_mad[:, 0]
    return out_med, out_mad


def apply_shift(
    shift: str,
    base_scores: npt.ArrayLike,
    feature_errors: npt.ArrayLike | None,
    window_start_s: npt.ArrayLike,
    stride_s: float,
    reference: ShiftReference,
    window: ShiftWindow,
) -> Shifted:
    """Turn a model's window scores (and per-feature errors, if any) into candidate scores."""
    base = np.asarray(base_scores, dtype=np.float64)
    if shift == "r0":
        return Shifted(base.copy(), np.ones(base.size, dtype=bool))
    if shift == "r1":
        if feature_errors is None or reference.sorted_errors is None:
            return Shifted(base.copy(), np.ones(base.size, dtype=bool))
        return Shifted(reference.quantiles(feature_errors).mean(axis=1), np.ones(base.size, bool))
    ok = scored_mask(window_start_s, stride_s, window)
    if shift == "r2":
        x = base[:, None] if feature_errors is None else np.asarray(feature_errors, np.float64)
        med, _ = causal_stats(x, window_start_s, stride_s, window)
        scores = np.maximum(x - med, 0.0).mean(axis=1)
    elif shift == "r3":
        med1, mad1 = causal_stats(base, window_start_s, stride_s, window)
        floor = MAD_FLOOR_REL * max(abs(reference.score_median), 1e-12)
        scores = (base - med1) / np.maximum(mad1, floor)
    else:
        raise ValueError(f"unknown shift candidate {shift!r}")
    ok = ok & np.isfinite(scores)
    scores = np.where(ok, scores, np.nan)
    return Shifted(scores, ok)
