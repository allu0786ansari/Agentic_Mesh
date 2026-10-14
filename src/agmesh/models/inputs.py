"""Model inputs built from label-free partitions.

Site A (HAI)
    Rows are 1 Hz samples of the 56 modelled features. They are standardised with a
    scaler fitted on training rows and clipped to +-``clip_sd`` (HAI-4), then cut into
    60-second windows that never cross a run. With ``input_mode: segment_means``
    each window becomes six 10-second means per feature (6 x 56 = 336 inputs,
    HAI-6); ``flat`` keeps all 60 x 56 values.
Site B (Modbus)
    Rows are already 10-second windows. Features in ``exclude_features`` are dropped
    (MB-15), the rest are ``log1p``-transformed (counts and latency are heavy-tailed)
    and standardised with a scaler fitted on training windows.

Labels travel next to the inputs (``label``, ``window_start``, ``episode``) for
evaluation only; models receive ``x``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from agmesh.common.config import Params
from agmesh.data.access import Partition
from agmesh.data.windows import gather, segment_means, window_labels, window_starts

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
STD_FLOOR = 1e-6


@dataclass(frozen=True)
class Scaler:
    mean: F64
    std: F64
    clip: float

    @classmethod
    def fit(cls, values: npt.ArrayLike, clip: float) -> Scaler:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("scaler needs a non-empty 2-D array")
        std = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(std < STD_FLOOR, 1.0, std), clip)

    @classmethod
    def pooled(cls, scalers_inputs: list[npt.ArrayLike], clip: float) -> Scaler:
        return cls.fit(
            np.concatenate([np.asarray(v, dtype=np.float64) for v in scalers_inputs]), clip
        )

    def transform(self, values: npt.ArrayLike) -> F32:
        z = (np.asarray(values, dtype=np.float64) - self.mean) / self.std
        return np.clip(z, -self.clip, self.clip).astype(np.float32)


@dataclass(frozen=True)
class ModelInput:
    """One partition turned into model inputs, with evaluation labels alongside."""

    x: F32  # (windows, inputs)
    window_start: F64  # epoch seconds
    window_length_s: float
    stride_s: float
    label: npt.NDArray[np.bool_]  # window contains an attack row (A) or attacker traffic (B)
    base_features: tuple[str, ...]
    segments: int  # inputs per base feature (Site A segments; 1 on Site B)
    row_time_s: F64 = field(default_factory=lambda: np.empty(0))  # Site A rows, for events
    row_target: npt.NDArray[np.int8] = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    episode: npt.NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_inputs(self) -> int:
        return int(self.x.shape[1])


def site_a_columns(partition: Partition, params: Params) -> list[int]:
    excluded = set(params.models.vae.exclude_features)
    unknown = excluded - set(partition.features.columns)
    if unknown:
        raise ValueError(f"models.vae.exclude_features names unknown features: {sorted(unknown)}")
    return [i for i, c in enumerate(partition.features.columns) if c not in excluded]


def site_a_raw(partition: Partition, params: Params) -> F32:
    cols = site_a_columns(partition, params)
    out: F32 = np.ascontiguousarray(partition.features.values[:, cols])
    return out


def site_a_input(
    partition: Partition, scaler: Scaler, params: Params, *, stride_s: int | None = None
) -> ModelInput:
    """Windows of standardised rows, reduced as ``models.vae.input_mode`` says."""
    win = params.data.windows.site_a
    vae = params.models.vae
    stride = stride_s or win.train_stride_s
    length = win.length_s * params.data.hai.sample_rate_hz
    labels = partition.labels
    starts = window_starts(labels.time_s, labels.group, length, stride)
    cols = site_a_columns(partition, params)
    scaled = scaler.transform(site_a_raw(partition, params))
    windows = gather(scaled, starts, length)
    if vae.input_mode == "segment_means":
        x = segment_means(windows, vae.segments_per_window)
        segments = vae.segments_per_window
    else:
        # segment-major order like segment_means: all features of second 0, then 1, ...
        x = windows.reshape(windows.shape[0], -1)
        segments = length
    return ModelInput(
        x=np.ascontiguousarray(x, dtype=np.float32),
        window_start=labels.time_s[starts] if starts.size else np.empty(0),
        window_length_s=float(win.length_s),
        stride_s=float(stride),
        label=window_labels(labels.target, starts, length),
        base_features=tuple(partition.features.columns[i] for i in cols),
        segments=segments,
        row_time_s=labels.time_s,
        row_target=labels.target,
    )


def site_b_columns(partition: Partition, params: Params) -> list[int]:
    excluded = set(params.models.ae.exclude_features)
    unknown = excluded - set(partition.features.columns)
    if unknown:
        raise ValueError(f"models.ae.exclude_features names unknown features: {sorted(unknown)}")
    return [i for i, c in enumerate(partition.features.columns) if c not in excluded]


def site_b_raw(partition: Partition, params: Params) -> F64:
    cols = site_b_columns(partition, params)
    values = partition.features.values[:, cols].astype(np.float64)
    if params.models.ae.transform == "log1p":
        values = np.log1p(np.maximum(values, 0.0))
    return values


def site_b_input(partition: Partition, scaler: Scaler, params: Params) -> ModelInput:
    cols = site_b_columns(partition, params)
    win = params.data.windows.site_b
    return ModelInput(
        x=scaler.transform(site_b_raw(partition, params)),
        window_start=partition.labels.time_s,
        window_length_s=float(win.length_s),
        stride_s=float(win.train_stride_s),
        label=partition.labels.target.astype(bool),
        base_features=tuple(partition.features.columns[i] for i in cols),
        segments=1,
        episode=partition.labels.group.astype(np.int64),
    )


def per_feature(errors: F32, inp: ModelInput) -> F32:
    """Average per-input errors over segments to one error per base feature."""
    n, d = errors.shape
    f = len(inp.base_features)
    if d != f * inp.segments:
        raise ValueError("error vector does not match the input layout")
    out: F32 = errors.reshape(n, inp.segments, f).mean(axis=1).astype(np.float32)
    return out
