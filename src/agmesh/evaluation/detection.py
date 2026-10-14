"""Evaluate one node's detector at its operating point (Proposal §10.2, §10.3).

The threshold comes from the node's **normal validation** scores at the target
window FPR. The evaluation partition (attack validation now, test after G4) holds
normal and attack windows; on it the harness computes threshold-free metrics,
window FPR and precision on its normal windows, event recall and time to detect,
range-based F1 and false alarms per hour.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt

from agmesh.evaluation import events as ev
from agmesh.evaluation import metrics as m
from agmesh.evaluation.thresholds import fpr_threshold

Interval = tuple[float, float]
#: The headline detection metrics, in report order.
DETECTION_KEYS = (
    "auroc",
    "auprc",
    "event_recall",
    "range_f1",
    "window_fpr",
    "window_fpr_excl_recovery",
    "ttd_median_s",
    "false_alarms_per_hour",
    "abstained_share",
)


@dataclass(frozen=True)
class NodeArrays:
    """Per-window record of one node's evaluation, kept for event-level inference."""

    scores: npt.NDArray[np.float32]  # NaN where the detector abstained
    labels: npt.NDArray[np.bool_]
    scored: npt.NDArray[np.bool_]
    alerts: npt.NDArray[np.bool_]
    window_start_s: npt.NDArray[np.float64]
    event_id: npt.NDArray[np.int32]  # index into the node's events; -1 for normal windows
    event_detected: npt.NDArray[np.bool_]
    event_ttd_s: npt.NDArray[np.float32]
    stride_s: float
    window_length_s: float
    threshold: float

    def to_npz(self, prefix: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, value in self.__dict__.items():
            out[f"{prefix}__{name}"] = np.asarray(value)
        return out

    @classmethod
    def from_npz(cls, data: Any, prefix: str) -> NodeArrays:
        kw: dict[str, Any] = {}
        for f in fields(cls):
            value = data[f"{prefix}__{f.name}"]
            kw[f.name] = float(value) if f.name in _SCALARS else np.asarray(value)
        return cls(**kw)


_SCALARS = ("stride_s", "window_length_s", "threshold")


def node_arrays_npz(arrays: dict[str, NodeArrays]) -> dict[str, np.ndarray]:
    """Flatten per-node records into one ``np.savez`` mapping (keys ``<node>__<field>``)."""
    out: dict[str, np.ndarray] = {"nodes": np.array(sorted(arrays))}
    for node, record in arrays.items():
        out.update(record.to_npz(node))
    return out


def load_node_arrays(data: Any) -> dict[str, NodeArrays]:
    return {str(n): NodeArrays.from_npz(data, str(n)) for n in data["nodes"]}


#: ADR 0005 sensitivity metric: post-attack recovery excluded (TaPR's ambiguous section).
RECOVERY_S = 900.0


def fpr_excluding_recovery(
    labels: npt.ArrayLike,
    alerts: npt.ArrayLike,
    scored: npt.ArrayLike,
    window_start_s: npt.ArrayLike,
    window_length_s: float,
    recovery_s: float = RECOVERY_S,
) -> float:
    """Window FPR over scored normal windows that start at least ``recovery_s``
    after the end of the most recent attack window (or before any attack)."""
    y = np.asarray(labels, dtype=bool)
    a = np.asarray(alerts, dtype=bool)
    ok = np.asarray(scored, dtype=bool)
    t = np.asarray(window_start_s, dtype=np.float64)
    attack_end = np.sort(t[y]) + window_length_s
    if attack_end.size == 0:
        clean0 = ~y & ok
        return float(a[clean0].mean()) if clean0.any() else float("nan")
    pos = np.searchsorted(attack_end, t, side="right")  # attacks that ended before t
    last_end = np.where(pos > 0, attack_end[np.maximum(pos - 1, 0)], -np.inf)
    clean = ~y & ok & (t - last_end >= recovery_s)
    return float(a[clean].mean()) if clean.any() else float("nan")


def window_event_ids(
    window_start_s: npt.ArrayLike,
    labels: npt.ArrayLike,
    window_length_s: float,
    events: list[Interval],
) -> npt.NDArray[np.int32]:
    """Map every attack window to the event it overlaps (nearest start if none)."""
    t = np.asarray(window_start_s, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    out = np.full(t.size, -1, dtype=np.int32)
    if not events:
        return out
    starts = np.array([a for a, _ in events])
    ends = np.array([b for _, b in events])
    for i in np.flatnonzero(y):
        overlap = (starts < t[i] + window_length_s) & (ends > t[i])
        hit = np.flatnonzero(overlap)
        out[i] = int(hit[0]) if hit.size else int(np.argmin(np.abs(starts - t[i])))
    return out


def evaluate_node(
    normal_val_scores: npt.ArrayLike,
    scores: npt.ArrayLike,
    labels: npt.ArrayLike,
    window_start_s: npt.ArrayLike,
    *,
    window_length_s: float,
    stride_s: float,
    events: list[Interval],
    target_fpr: float,
    tolerance_s: float,
    normal_val_scored: npt.ArrayLike | None = None,
    scored: npt.ArrayLike | None = None,
) -> dict[str, Any]:
    return evaluate_node_full(
        normal_val_scores,
        scores,
        labels,
        window_start_s,
        window_length_s=window_length_s,
        stride_s=stride_s,
        events=events,
        target_fpr=target_fpr,
        tolerance_s=tolerance_s,
        normal_val_scored=normal_val_scored,
        scored=scored,
    )[0]


def evaluate_node_full(
    normal_val_scores: npt.ArrayLike,
    scores: npt.ArrayLike,
    labels: npt.ArrayLike,
    window_start_s: npt.ArrayLike,
    *,
    window_length_s: float,
    stride_s: float,
    events: list[Interval],
    target_fpr: float,
    tolerance_s: float,
    normal_val_scored: npt.ArrayLike | None = None,
    scored: npt.ArrayLike | None = None,
) -> tuple[dict[str, Any], NodeArrays]:
    """Metrics at the operating point, and the per-window record behind them.

    ``scored`` / ``normal_val_scored`` mark windows where the detector gives a score
    (ADR 0005 warm-up). An abstained window raises no alert; abstained attack windows
    count as missed and rank lowest, abstained normal windows are left out of the
    FPR, false alarms per hour and the threshold-free metrics.
    """
    nv = np.asarray(normal_val_scores, dtype=np.float64)
    nv_ok = (
        np.ones(nv.size, bool) if normal_val_scored is None else np.asarray(normal_val_scored, bool)
    )
    threshold = fpr_threshold(nv[nv_ok], target_fpr)
    s = np.asarray(scores, dtype=np.float64)
    ok = np.ones(s.size, bool) if scored is None else np.asarray(scored, bool)
    y = np.asarray(labels).astype(bool)
    starts = np.asarray(window_start_s, dtype=np.float64)
    alerts = m.alerts_at(np.where(ok, s, -np.inf), threshold.value) & ok
    keep = ok | y  # abstained normal windows are not part of the evaluation
    ranked = s.copy()
    if (~ok).any():
        lowest = float(np.min(s[ok])) - 1.0 if ok.any() else 0.0
        ranked[~ok] = lowest
    yk, rk, ak, tk = y[keep], ranked[keep], alerts[keep], starts[keep]
    confusion = m.confusion(yk, ak)
    intervals = ev.alert_intervals(starts, alerts, window_length_s)
    detected = ev.event_scores(events, intervals, tolerance_s)
    ranges = ev.range_scores(events, intervals) if events else None
    metrics = {
        "auroc": m.auroc(yk, rk),
        "auprc": m.auprc(yk, rk),
        "event_recall": detected.recall,
        "events": detected.n_events,
        "events_detected": detected.detected,
        "range_precision": ranges.precision if ranges else float("nan"),
        "range_recall": ranges.recall if ranges else float("nan"),
        "range_f1": ranges.f1 if ranges else float("nan"),
        "window_fpr": confusion.fpr,
        "window_precision": confusion.precision,
        "window_recall": confusion.recall,
        "ttd_median_s": detected.ttd_median_s,
        "ttd_mean_s": detected.ttd_mean_s,
        "false_alarms_per_hour": m.false_alarms_per_hour(yk, ak, tk, stride_s),
        "threshold": threshold.value,
        "threshold_val_fpr": threshold.achieved_fpr,
        "windows": int(y.size),
        "attack_windows": int(y.sum()),
        "window_fpr_excl_recovery": fpr_excluding_recovery(
            y, alerts, ok, starts, window_length_s, RECOVERY_S
        ),
        "abstained_share": float(1.0 - ok.mean()) if ok.size else 0.0,
        "abstained_attack_windows": int((~ok & y).sum()),
    }
    arrays = NodeArrays(
        scores=np.where(ok, s, np.nan).astype(np.float32),
        labels=y,
        scored=ok,
        alerts=alerts,
        window_start_s=starts,
        event_id=window_event_ids(starts, y, window_length_s, events),
        event_detected=np.array(detected.flags, dtype=bool),
        event_ttd_s=np.array(detected.ttd_by_event_s, dtype=np.float32),
        stride_s=float(stride_s),
        window_length_s=float(window_length_s),
        threshold=float(threshold.value),
    )
    return metrics, arrays


def macro(per_client: dict[str, dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float | None]:
    """Mean of each metric over the clients where it is defined (None if nowhere)."""
    out: dict[str, float | None] = {}
    for key in keys:
        values = [float(v[key]) for v in per_client.values() if v.get(key) is not None]
        finite = [x for x in values if np.isfinite(x)]
        out[key] = float(np.mean(finite)) if finite else None
    return out
