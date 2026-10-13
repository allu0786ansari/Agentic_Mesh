"""Contiguous labelled intervals ("events") in a row-ordered label vector."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Maximal runs of True in ``mask`` as (start, stop) row slices, stop exclusive."""
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def cluster_times(times: npt.NDArray[np.float64], max_gap: float) -> list[tuple[float, float]]:
    """Group sorted timestamps into (first, last) clusters split at gaps above ``max_gap``."""
    if times.size == 0:
        return []
    ordered = np.sort(times)
    breaks = np.flatnonzero(np.diff(ordered) > max_gap)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks, [ordered.size - 1]])
    return [(float(ordered[a]), float(ordered[b])) for a, b in zip(starts, stops, strict=True)]
