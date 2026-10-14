"""Window-level detection metrics (Proposal §10.2).

Conventions: ``y`` is 1 for attack windows and 0 for normal ones; higher scores mean
more anomalous; a window raises an alert when its score is strictly above the
threshold. Functions return ``nan`` when a metric is undefined (for example AUROC on
data with only one class) instead of raising, so that a report can show the gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.ArrayLike
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _as_binary(y: Array) -> npt.NDArray[np.bool_]:
    labels = np.asarray(y)
    if labels.size and not np.isin(labels, (0, 1, True, False)).all():
        raise ValueError("labels must be 0/1")
    return labels.astype(bool)


def _average_ranks(values: FloatArray) -> FloatArray:
    """1-based ranks with ties given their average rank."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    boundaries = np.flatnonzero(np.diff(ordered)) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [values.size]])
    for a, b in zip(starts, stops, strict=True):
        ranks[order[a:b]] = (a + b + 1) / 2.0
    return ranks


def auroc(y: Array, scores: Array) -> float:
    """Area under the ROC curve (Mann-Whitney U statistic, ties count one half)."""
    labels = _as_binary(y)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def auprc(y: Array, scores: Array) -> float:
    """Average precision: the precision at each recall step, weighted by the step."""
    labels = _as_binary(y)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    hits = labels[order].astype(np.float64)
    # evaluate only at distinct thresholds so tied scores move together
    last_of_group = np.concatenate([np.flatnonzero(np.diff(s_sorted)), [s.size - 1]])
    tp = np.cumsum(hits)[last_of_group]
    fp = (last_of_group + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    steps = np.diff(np.concatenate([[0.0], recall]))
    return float((steps * precision).sum())


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else float("nan")

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")


def confusion(y: Array, alerts: Array) -> Confusion:
    labels = _as_binary(y)
    pred = _as_binary(alerts)
    return Confusion(
        tp=int((labels & pred).sum()),
        fp=int((~labels & pred).sum()),
        tn=int((~labels & ~pred).sum()),
        fn=int((labels & ~pred).sum()),
    )


def alerts_at(scores: Array, threshold: float) -> npt.NDArray[np.bool_]:
    """Alert where the score is strictly above the threshold."""
    flagged: npt.NDArray[np.bool_] = np.asarray(scores, dtype=np.float64) > threshold
    return flagged


def false_alarms_per_hour(y: Array, alerts: Array, window_start_s: Array, stride_s: float) -> float:
    """False alarms per hour of normal time.

    A false alarm is a maximal run of consecutive alerting normal windows (windows
    ``stride_s`` apart), so one long false alert counts once. Normal time is the
    number of normal windows times ``stride_s``.
    """
    labels = _as_binary(y)
    pred = _as_binary(alerts)
    starts = np.asarray(window_start_s, dtype=np.float64)
    order = np.argsort(starts, kind="mergesort")
    false = (pred & ~labels)[order]
    t = starts[order]
    hours = (~labels).sum() * stride_s / 3600.0
    if hours == 0:
        return float("nan")
    if not false.any():
        return 0.0
    idx = np.flatnonzero(false)
    new_run = np.concatenate([[True], np.diff(t[idx]) > stride_s * 1.0001])
    return float(new_run.sum() / hours)


# ---------------------------------------------------------- federated (pooled) AUROC


def score_histogram(scores: Array, bins: int = 1000) -> IntArray:
    """Counts of calibrated probabilities in ``bins`` equal bins on [0, 1]."""
    p = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    index = np.minimum((p * bins).astype(np.int64), bins - 1)
    return np.bincount(index, minlength=bins).astype(np.int64)


@dataclass(frozen=True)
class PooledAuroc:
    auroc: float
    max_bin_error: float  # bound on |binned - exact| from pairs that share a bin


def pooled_auroc(pos_hists: list[IntArray], neg_hists: list[IntArray]) -> PooledAuroc:
    """AUROC over all clients from their binned score histograms (Proposal §7.5).

    Pairs in different bins are ordered exactly; pairs in the same bin are counted as
    ties (one half). The true AUROC therefore lies within ``max_bin_error`` of the
    estimate, and that bound is computed from the histograms alone.
    """
    pos = np.sum(pos_hists, axis=0).astype(np.float64)
    neg = np.sum(neg_hists, axis=0).astype(np.float64)
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return PooledAuroc(float("nan"), float("nan"))
    neg_below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    same_bin = pos * neg
    value = (pos * neg_below).sum() + 0.5 * same_bin.sum()
    pairs = n_pos * n_neg
    return PooledAuroc(float(value / pairs), float(0.5 * same_bin.sum() / pairs))


# ------------------------------------------------------------- point adjustment (demo)


def point_adjust(y: Array, alerts: Array) -> npt.NDArray[np.bool_]:
    """Point adjustment: one alert inside a labelled run marks the whole run detected.

    Implemented only to show, in tests, why it is **not** used: it rewards almost
    any detector (Kim et al., 2022). No reported metric calls it.
    """
    labels = _as_binary(y)
    pred = _as_binary(alerts).copy()
    padded = np.concatenate([[False], labels, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        if pred[a:b].any():
            pred[a:b] = True
    return pred


def f1(y: Array, alerts: Array) -> float:
    c = confusion(y, alerts)
    p, r = c.precision, c.recall
    if np.isnan(p) or np.isnan(r) or p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
