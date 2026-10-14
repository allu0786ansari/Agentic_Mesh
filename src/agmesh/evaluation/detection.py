"""Evaluate one node's detector at its operating point (Proposal §10.2, §10.3).

The threshold comes from the node's **normal validation** scores at the target
window FPR. The evaluation partition (attack validation now, test after G4) holds
normal and attack windows; on it the harness computes threshold-free metrics,
window FPR and precision on its normal windows, event recall and time to detect,
range-based F1 and false alarms per hour.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from agmesh.evaluation import events as ev
from agmesh.evaluation import metrics as m
from agmesh.evaluation.thresholds import fpr_threshold

Interval = tuple[float, float]


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
) -> dict[str, Any]:
    threshold = fpr_threshold(normal_val_scores, target_fpr)
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    starts = np.asarray(window_start_s, dtype=np.float64)
    alerts = m.alerts_at(s, threshold.value)
    confusion = m.confusion(y, alerts)
    intervals = ev.alert_intervals(starts, alerts, window_length_s)
    detected = ev.event_scores(events, intervals, tolerance_s)
    ranges = ev.range_scores(events, intervals) if events else None
    return {
        "auroc": m.auroc(y, s),
        "auprc": m.auprc(y, s),
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
        "false_alarms_per_hour": m.false_alarms_per_hour(y, alerts, starts, stride_s),
        "threshold": threshold.value,
        "threshold_val_fpr": threshold.achieved_fpr,
        "windows": int(y.size),
        "attack_windows": int(y.sum()),
    }


def macro(per_client: dict[str, dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float | None]:
    """Mean of each metric over the clients where it is defined (None if nowhere)."""
    out: dict[str, float | None] = {}
    for key in keys:
        values = [float(v[key]) for v in per_client.values() if v.get(key) is not None]
        finite = [x for x in values if np.isfinite(x)]
        out[key] = float(np.mean(finite)) if finite else None
    return out
