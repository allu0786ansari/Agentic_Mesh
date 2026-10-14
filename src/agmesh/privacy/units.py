"""The privacy unit as data: one row per training window (Proposal §7.3; flaw P4)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from agmesh.common.config import Params
from agmesh.data.access import Partition
from agmesh.data.windows import gather, window_starts
from agmesh.models import inputs as mi


def training_units(
    site: str, part: Partition, params: Params
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Every privacy unit's raw values and the feature names.

    Site A: each non-overlapping 60 s training window, as its (60, features) raw rows.
    Site B: the 10 s windows themselves (``log1p`` already applied).
    """
    if site == "site_b":
        cols = mi.site_b_columns(part, params)
        names = [part.features.columns[i] for i in cols]
        return np.asarray(mi.site_b_raw(part, params), dtype=np.float64), names
    win = params.data.windows.site_a
    if win.train_stride_s != win.length_s:
        raise ValueError("the privacy unit needs non-overlapping training windows (P4)")
    length = win.length_s * params.data.hai.sample_rate_hz
    starts = window_starts(part.labels.time_s, part.labels.group, length, win.train_stride_s)
    names = [part.features.columns[i] for i in mi.site_a_columns(part, params)]
    raw = mi.site_a_raw(part, params)
    return gather(raw, starts, length).astype(np.float64), names


def n_training_units(site: str, part: Partition, params: Params) -> int:
    """How many privacy units a partition holds (cheap; no values are gathered)."""
    if site == "site_b":
        return len(part.features.values)
    win = params.data.windows.site_a
    length = win.length_s * params.data.hai.sample_rate_hz
    return len(window_starts(part.labels.time_s, part.labels.group, length, win.train_stride_s))
