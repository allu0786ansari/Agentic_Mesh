"""Windows never cross runs or gaps; training windows never overlap; events and clusters."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from agmesh.data.events import cluster_times, runs
from agmesh.data.windows import gather, segment_means, window_labels, window_starts


def test_windows_stay_inside_runs_and_skip_gaps() -> None:
    times = np.concatenate([np.arange(0, 130), np.arange(200, 330)]).astype(float)
    run_ids = np.array(["a"] * 130 + ["b"] * 130)
    starts = window_starts(times, run_ids, length=60, stride=60)
    assert starts.tolist() == [0, 60, 130, 190]
    # a gap inside one run drops the window that would span it
    times[100] = 100.5
    times[101:130] += 5
    starts = window_starts(times, np.array(["a"] * 260), length=60, stride=60)
    assert 60 not in starts.tolist()


@given(
    n=st.integers(min_value=0, max_value=500),
    length=st.integers(min_value=1, max_value=60),
)
def test_training_windows_never_overlap(n: int, length: int) -> None:
    times = np.arange(n, dtype=float)
    starts = window_starts(times, np.zeros(n), length=length, stride=length)
    covered = np.zeros(n, dtype=int)
    for s in starts:
        covered[s : s + length] += 1
    assert covered.max(initial=0) <= 1
    assert (starts + length <= n).all()


def test_segment_means_and_labels() -> None:
    values = np.arange(120 * 2, dtype=np.float32).reshape(120, 2)
    starts = np.array([0, 60])
    windows = gather(values, starts, 60)
    assert windows.shape == (2, 60, 2)
    means = segment_means(windows, 6)
    assert means.shape == (2, 12)
    assert means[0, 0] == values[0:10, 0].mean()
    assert means[0, 3] == values[10:20, 1].mean()
    label = np.zeros(120)
    label[119] = 1
    assert window_labels(label, starts, 60).tolist() == [False, True]


def test_runs_and_clusters() -> None:
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert runs(mask) == [(1, 3), (5, 6), (7, 10)]
    assert runs(np.zeros(4, dtype=bool)) == []
    times = np.array([10.0, 0.0, 5.0, 400.0, 401.0, 1000.0])
    assert cluster_times(times, 125.0) == [(0.0, 10.0), (400.0, 401.0), (1000.0, 1000.0)]
    assert cluster_times(np.empty(0), 1.0) == []
