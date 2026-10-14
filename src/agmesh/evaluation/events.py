"""Event-level detection metrics (Proposal §10.2).

Times are seconds. An *event* is a labelled interval ``[start, end)``; an *alert
window* is the interval ``[start, start + length)`` of a window whose score crossed
the threshold. An alert is raised when its window completes, at ``start + length``.

* Event recall: an event is detected if any alert window overlaps
  ``[start, end + tolerance)``.
* Time to detect: for a detected event, the earliest alert time minus the event
  start (never negative).
* Range-based precision, recall and F1 (Tatbul et al., NeurIPS 2018) with a flat
  positional bias and the reciprocal cardinality factor; ``alpha`` weights the
  existence reward in recall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Interval = tuple[float, float]


def merge(intervals: list[Interval]) -> list[Interval]:
    """Union of intervals; touching intervals are joined."""
    out: list[Interval] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def alert_intervals(
    window_start_s: npt.ArrayLike, alerts: npt.ArrayLike, length_s: float
) -> list[Interval]:
    starts = np.asarray(window_start_s, dtype=np.float64)[np.asarray(alerts, dtype=bool)]
    return [(float(s), float(s) + length_s) for s in np.sort(starts)]


def _overlap(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


@dataclass(frozen=True)
class EventScores:
    n_events: int
    detected: int
    recall: float
    ttd_s: list[float]
    flags: tuple[bool, ...] = ()  # per event, in input order
    ttd_by_event_s: tuple[float, ...] = ()  # per event; NaN if missed

    @property
    def ttd_median_s(self) -> float:
        return float(np.median(self.ttd_s)) if self.ttd_s else float("nan")

    @property
    def ttd_mean_s(self) -> float:
        return float(np.mean(self.ttd_s)) if self.ttd_s else float("nan")


def event_scores(events: list[Interval], alerts: list[Interval], tolerance_s: float) -> EventScores:
    """Event recall and time to detect (see module docstring)."""
    if not events:
        return EventScores(0, 0, float("nan"), [])
    ordered = sorted(alerts)
    alert_starts = np.array([a for a, _ in ordered], dtype=np.float64)
    detected = 0
    ttd: list[float] = []
    flags: list[bool] = []
    by_event: list[float] = []
    for start, end in events:
        reach = (start, end + tolerance_s)
        # only alerts starting before the reach ends can overlap it
        hi = int(np.searchsorted(alert_starts, reach[1], side="left"))
        hits = [a for a in ordered[:hi] if _overlap(a, reach) > 0]
        if hits:
            detected += 1
            ttd.append(max(0.0, min(b for _, b in hits) - start))
        flags.append(bool(hits))
        by_event.append(ttd[-1] if hits else float("nan"))
    return EventScores(
        len(events), detected, detected / len(events), ttd, tuple(flags), tuple(by_event)
    )


def _gamma(count: int) -> float:
    return 1.0 if count <= 1 else 1.0 / count


@dataclass(frozen=True)
class RangeScores:
    precision: float
    recall: float
    f1: float


def range_scores(events: list[Interval], alerts: list[Interval], alpha: float = 0.0) -> RangeScores:
    """Range-based precision, recall and F1 (Tatbul et al., 2018)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    real = merge(events)
    predicted = merge(alerts)

    def recall_of(r: Interval) -> float:
        overlaps = [_overlap(r, p) for p in predicted]
        hits = [o for o in overlaps if o > 0]
        existence = 1.0 if hits else 0.0
        size = sum(hits) / (r[1] - r[0]) if r[1] > r[0] else existence
        return alpha * existence + (1 - alpha) * _gamma(len(hits)) * size

    def precision_of(p: Interval) -> float:
        hits = [o for o in (_overlap(p, r) for r in real) if o > 0]
        return _gamma(len(hits)) * sum(hits) / (p[1] - p[0]) if p[1] > p[0] else 0.0

    recall = float(np.mean([recall_of(r) for r in real])) if real else float("nan")
    if not predicted:
        precision = float("nan")
    else:
        precision = float(np.mean([precision_of(p) for p in predicted]))
    if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return RangeScores(precision, recall, f1)


def events_from_rows(
    time_s: npt.ArrayLike, label: npt.ArrayLike, period_s: float = 1.0
) -> list[Interval]:
    """Events from row-level labels: maximal runs of consecutive labelled samples."""
    t = np.asarray(time_s, dtype=np.float64)
    flags = np.asarray(label).astype(bool)
    if not flags.any():
        return []
    order = np.argsort(t, kind="mergesort")
    t, flags = t[order], flags[order]
    idx = np.flatnonzero(flags)
    breaks = np.flatnonzero(np.diff(t[idx]) > period_s * 1.5)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks, [idx.size - 1]])
    return [
        (float(t[idx[a]]), float(t[idx[b]]) + period_s) for a, b in zip(starts, stops, strict=True)
    ]


def events_from_episodes(
    window_start_s: npt.ArrayLike,
    episode: npt.ArrayLike,
    length_s: float,
    label: npt.ArrayLike | None = None,
) -> list[Interval]:
    """Site B events: one interval per attacker episode, from its labelled windows.

    ``episode`` is the episode index of each window (-1 for none). Episodes are
    defined by time for the whole site, so with ``label`` only the windows labelled
    as attacks on *this* node count: a node the attacker never reached has no events.
    The event runs from the first such window's start to the last one's end.
    """
    starts = np.asarray(window_start_s, dtype=np.float64)
    ids = np.asarray(episode).astype(np.int64).copy()
    if label is not None:
        ids[~np.asarray(label).astype(bool)] = -1
    events: list[Interval] = []
    for e in np.unique(ids[ids >= 0]):
        member = starts[ids == e]
        events.append((float(member.min()), float(member.max()) + length_s))
    return sorted(events)
