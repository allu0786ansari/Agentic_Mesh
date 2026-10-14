"""From reconstruction errors to window anomaly scores (ADR 0003).

``mean``       mean squared error over all inputs (Week 4 baseline).
``norm_mean``  per-feature error (averaged over a window's segments), standardised
               by that feature's median and inter-quartile range on *training*
               windows, averaged over features.
``norm_top3``  as ``norm_mean`` but the mean of the ``top_k`` largest standardised
               feature errors, so one strongly deviating feature is not diluted.

Scores may then be smoothed causally: the mean of the current window and up to
``smooth - 1`` previous windows, only across consecutive windows (start times one
stride apart). The same per-feature errors feed the evidence records (Week 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from agmesh.models.inputs import ModelInput, per_feature

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
Aggregation = Literal["mean", "norm_mean", "norm_top3"]
IQR_FLOOR = 1e-6


@dataclass(frozen=True)
class ErrorReference:
    """Per-feature median and inter-quartile range of training reconstruction errors."""

    median: F64
    iqr: F64

    @classmethod
    def fit(cls, feature_errors: npt.ArrayLike) -> ErrorReference:
        e = np.asarray(feature_errors, dtype=np.float64)
        q25, q50, q75 = np.quantile(e, [0.25, 0.5, 0.75], axis=0)
        return cls(q50, np.maximum(q75 - q25, IQR_FLOOR))

    def standardise(self, feature_errors: npt.ArrayLike) -> F64:
        out: F64 = (np.asarray(feature_errors, dtype=np.float64) - self.median) / self.iqr
        return out


@dataclass(frozen=True)
class Scoring:
    aggregation: Aggregation = "mean"
    smooth: int = 1
    top_k: int = 3

    @property
    def name(self) -> str:
        return f"{self.aggregation}__s{self.smooth}"


def aggregate(
    errors: F32, inp: ModelInput, scoring: Scoring, reference: ErrorReference | None
) -> F64:
    """Window scores from per-input squared errors (before smoothing)."""
    if scoring.aggregation == "mean":
        out: F64 = errors.astype(np.float64).mean(axis=1)
        return out
    if reference is None:
        raise ValueError(f"{scoring.aggregation} needs an ErrorReference fitted on training errors")
    z = reference.standardise(per_feature(errors, inp))
    if scoring.aggregation == "norm_mean":
        out = z.mean(axis=1)
        return out
    k = min(scoring.top_k, z.shape[1])
    out = np.sort(z, axis=1)[:, -k:].mean(axis=1)
    return out


def causal_smooth(
    scores: npt.ArrayLike, window_start_s: npt.ArrayLike, stride_s: float, smooth: int
) -> F64:
    """Mean over the current and up to ``smooth - 1`` previous consecutive windows."""
    s = np.asarray(scores, dtype=np.float64)
    if smooth <= 1 or s.size == 0:
        return s.copy()
    t = np.asarray(window_start_s, dtype=np.float64)
    order = np.argsort(t, kind="mergesort")
    st, tt = s[order], t[order]
    consecutive = np.concatenate(
        [[False], np.abs(np.diff(tt) - stride_s) < 1e-6 * max(stride_s, 1.0)]
    )
    out_sorted = np.empty_like(st)
    run_start = 0
    for i in range(st.size):
        if not consecutive[i]:
            run_start = i
        lo = max(run_start, i - smooth + 1)
        out_sorted[i] = st[lo : i + 1].mean()
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out


def window_scores(
    errors: F32, inp: ModelInput, scoring: Scoring, reference: ErrorReference | None
) -> F64:
    """Aggregate, then smooth, the reconstruction errors of one model input."""
    raw = aggregate(errors, inp, scoring, reference)
    return causal_smooth(raw, inp.window_start, inp.stride_s, scoring.smooth)


AGGREGATIONS: tuple[Aggregation, ...] = ("mean", "norm_mean", "norm_top3")
GRID: tuple[Scoring, ...] = tuple(
    Scoring(agg, smooth) for agg in AGGREGATIONS for smooth in (1, 3, 6)
)
