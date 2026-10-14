"""Isotonic calibration of anomaly scores and reliability curves (Proposal §6.5).

:class:`IsotonicCalibrator` fits a non-decreasing step function from scores to the
probability of an attack, with the pool-adjacent-violators algorithm, on a node's
validation windows (normal and attack). Scores outside the fitted range map to the
end values. :func:`reliability` reports how well calibrated probabilities match the
observed attack frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def pava(y: FloatArray, weights: FloatArray) -> FloatArray:
    """Weighted least-squares non-decreasing fit to ``y`` (pool adjacent violators)."""
    values: list[float] = []
    wsum: list[float] = []
    counts: list[int] = []
    for yi, wi in zip(y.tolist(), weights.tolist(), strict=True):
        values.append(yi)
        wsum.append(wi)
        counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            w = wsum[-2] + wsum[-1]
            v = (values[-2] * wsum[-2] + values[-1] * wsum[-1]) / w
            c = counts[-2] + counts[-1]
            values[-2:], wsum[-2:], counts[-2:] = [v], [w], [c]
    return np.repeat(np.array(values), counts)


@dataclass
class IsotonicCalibrator:
    x: FloatArray | None = None  # distinct fitted scores, ascending
    p: FloatArray | None = None  # calibrated probability at each

    def fit(self, scores: npt.ArrayLike, labels: npt.ArrayLike) -> IsotonicCalibrator:
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if s.size == 0 or s.size != y.size:
            raise ValueError("scores and labels must be non-empty and the same length")
        unique, inverse = np.unique(s, return_inverse=True)
        weights = np.bincount(inverse).astype(np.float64)
        means = np.bincount(inverse, weights=y) / weights
        self.x = unique
        self.p = pava(means, weights)
        return self

    def predict(self, scores: npt.ArrayLike) -> FloatArray:
        if self.x is None or self.p is None:
            raise RuntimeError("calibrator is not fitted")
        out: FloatArray = np.interp(np.asarray(scores, dtype=np.float64), self.x, self.p)
        return out

    def to_dict(self) -> dict[str, Any]:
        if self.x is None or self.p is None:
            raise RuntimeError("calibrator is not fitted")
        return {"x": self.x.tolist(), "p": self.p.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IsotonicCalibrator:
        return cls(np.asarray(data["x"], dtype=np.float64), np.asarray(data["p"], dtype=np.float64))


@dataclass(frozen=True)
class Reliability:
    bin_edges: list[float]
    mean_predicted: list[float]
    observed: list[float]
    counts: list[int]
    ece: float  # expected calibration error


def reliability(probs: npt.ArrayLike, labels: npt.ArrayLike, bins: int = 10) -> Reliability:
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.minimum((p * bins).astype(np.int64), bins - 1)
    counts = np.bincount(index, minlength=bins)
    sum_p = np.bincount(index, weights=p, minlength=bins)
    sum_y = np.bincount(index, weights=y, minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = np.where(counts > 0, sum_p / counts, np.nan)
        obs = np.where(counts > 0, sum_y / counts, np.nan)
    gaps = np.abs(np.nan_to_num(mean_p) - np.nan_to_num(obs)) * counts
    ece = float(gaps.sum() / counts.sum()) if counts.sum() else float("nan")
    return Reliability(edges.tolist(), mean_p.tolist(), obs.tolist(), counts.tolist(), ece)
