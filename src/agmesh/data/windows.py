"""Fixed-length windows over 1 Hz Site A partitions.

Windows never cross a ``run_id`` boundary (a contiguous stretch of one partition)
and never span a gap in the timestamps. Training windows use ``stride == length``,
so each reading belongs to exactly one training window: that window is the DP
privacy unit (Proposal §7.3).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


def window_starts(
    times_s: npt.NDArray[np.float64],
    run_ids: npt.ArrayLike,
    length: int,
    stride: int,
    *,
    sample_period_s: float = 1.0,
) -> IntArray:
    """Row indices where a complete window of ``length`` rows starts.

    Windows start at offsets 0, stride, 2*stride, ... from the start of each run and
    are kept only if their ``length`` rows are consecutive samples (no time gap).
    """
    if length <= 0 or stride <= 0:
        raise ValueError("length and stride must be positive")
    runs = np.asarray(run_ids)
    n = len(times_s)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    boundaries = np.flatnonzero(runs[1:] != runs[:-1]) + 1
    run_starts = np.concatenate([[0], boundaries])
    run_stops = np.concatenate([boundaries, [n]])
    tolerance = 0.5 * sample_period_s
    expected_span = (length - 1) * sample_period_s
    starts: list[IntArray] = []
    for begin, stop in zip(run_starts, run_stops, strict=True):
        candidates = np.arange(begin, stop - length + 1, stride, dtype=np.int64)
        if candidates.size == 0:
            continue
        span = times_s[candidates + length - 1] - times_s[candidates]
        starts.append(candidates[np.abs(span - expected_span) <= tolerance])
    return np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)


def gather(values: FloatArray, starts: IntArray, length: int) -> FloatArray:
    """Stack windows into an array of shape (n_windows, length, n_features)."""
    if starts.size == 0:
        return np.empty((0, length, values.shape[1]), dtype=values.dtype)
    index = starts[:, None] + np.arange(length)[None, :]
    stacked: FloatArray = values[index]
    return stacked


def segment_means(windows: FloatArray, segments: int) -> FloatArray:
    """Summarise each window as ``segments`` equal-length means per feature, flattened.

    Output shape is (n_windows, segments * n_features), ordered segment-major
    (all features of segment 0, then segment 1, ...).
    """
    n, length, n_features = windows.shape
    if length % segments:
        raise ValueError(f"window length {length} is not divisible by {segments} segments")
    reshaped = windows.reshape(n, segments, length // segments, n_features)
    return reshaped.mean(axis=2).reshape(n, segments * n_features).astype(np.float32)


def window_labels(label: npt.ArrayLike, starts: IntArray, length: int) -> npt.NDArray[np.bool_]:
    """A window is an attack window if any of its rows is labelled."""
    flags = np.asarray(label).astype(bool)
    if starts.size == 0:
        return np.zeros(0, dtype=bool)
    cumulative = np.concatenate([[0], np.cumsum(flags, dtype=np.int64)])
    flagged: npt.NDArray[np.bool_] = (cumulative[starts + length] - cumulative[starts]) > 0
    return flagged


def frame_window_starts(frame: pd.DataFrame, length: int, stride: int) -> IntArray:
    """:func:`window_starts` for a Site A partition frame (``time`` and ``run_id`` columns)."""
    times = frame["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64) / 1e9
    return window_starts(times.astype(np.float64), frame["run_id"].to_numpy(), length, stride)
