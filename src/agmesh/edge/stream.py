"""Window-at-a-time scoring with the frozen model (ADR 0009 §8).

The batch path (``x1.Fitted.shifted``) scores a whole partition at once. Here each
window is built from its own rows, as it becomes complete:

* **Site A:** a 60 s window at the 10 s inference stride. Its rows are scaled and
  reduced to six 10 s segment means.
* **Site B:** one 10 s counter row.

Each window is scored on its own. R3 (ADR 0005) carries its causal history
incrementally:

* the median and MAD of the earlier base scores of the same segment that start in
  ``[t - history_s, t - gap_s)``;
* abstention for the first ``gap_s + min_history_s`` of a segment;
* a new segment where consecutive window starts are more than 1.5 strides apart.

These are the batch definitions (``models.shift``), and a test checks that the two
paths agree.
"""

from __future__ import annotations

import collections
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from agmesh.common.config import Params, ShiftWindow
from agmesh.data.access import Partition, Site
from agmesh.data.windows import segment_means, window_starts
from agmesh.edge.bundle import NodeCalibration
from agmesh.models import inputs as mi
from agmesh.models.shift import MAD_FLOOR_REL, MAD_TO_SD

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Window:
    """One complete window of one node's stream (private to the node)."""

    node: str
    part: str
    index: int
    t_start: float
    t_end: float
    x: F32  # (1, n_inputs) model input
    counters: dict[str, float] | None = None  # Site B raw counters of the window
    constants: F64 | None = None  # Site A raw values of the training-constant variables


@dataclass
class WindowSource:
    """The windows of one node's partition, in time order, built from its rows."""

    site: Site
    node: str
    part: str
    partition: Partition
    scaler: mi.Scaler
    params: Params
    constant_values: F64 | None = None  # (rows, n_constants), Site A
    _starts: npt.NDArray[np.int64] = field(init=False)

    def __post_init__(self) -> None:
        p = self.params
        if self.site == "site_a":
            win = p.data.windows.site_a
            self.length = win.length_s * p.data.hai.sample_rate_hz
            self.stride = win.infer_stride_s
            labels = self.partition.labels
            self._starts = window_starts(labels.time_s, labels.group, self.length, self.stride)
            self._raw: npt.NDArray[Any] = mi.site_a_raw(self.partition, p)
            self.window_length_s = float(win.length_s)
        else:
            self._starts = np.arange(len(self.partition.labels.time_s), dtype=np.int64)
            self._raw = mi.site_b_raw(self.partition, p)
            self._counters = self.partition.features.values.astype(np.float64)
            self._counter_names = list(self.partition.features.columns)
            self.window_length_s = float(p.data.windows.site_b.length_s)
            self.stride = p.data.windows.site_b.train_stride_s

    def __len__(self) -> int:
        return int(self._starts.size)

    def t_start(self, k: int) -> float:
        return float(self.partition.labels.time_s[self._starts[k]])

    def t_end(self, k: int) -> float:
        return self.t_start(k) + self.window_length_s

    def label(self, k: int) -> bool:
        """Evaluation only: whether the window contains attack rows (never in evidence)."""
        s = int(self._starts[k])
        target = self.partition.labels.target
        if self.site == "site_a":
            return bool(target[s : s + self.length].any())
        return bool(target[s])

    def window(self, k: int) -> Window:
        s = int(self._starts[k])
        if self.site == "site_a":
            rows = self.scaler.transform(self._raw[s : s + self.length])
            vae = self.params.models.vae
            if vae.input_mode == "segment_means":
                x = segment_means(rows[None, :, :], vae.segments_per_window)
            else:
                x = rows.reshape(1, -1)
            consts = (
                self.constant_values[s : s + self.length]
                if self.constant_values is not None
                else None
            )
            return Window(self.node, self.part, k, self.t_start(k), self.t_end(k), x, None, consts)
        x = self.scaler.transform(self._raw[s : s + 1])
        counters = dict(zip(self._counter_names, self._counters[s].tolist(), strict=True))
        return Window(self.node, self.part, k, self.t_start(k), self.t_end(k), x, counters)


class CausalR3:
    """Incremental R3: ``(s - median) / MAD`` over the causal history of the segment."""

    def __init__(self, window: ShiftWindow, stride_s: float, score_median: float) -> None:
        self.window, self.stride = window, stride_s
        self.floor = MAD_FLOOR_REL * max(abs(score_median), 1e-12)
        self.history: collections.deque[tuple[float, float]] = collections.deque()
        self.prev_t: float | None = None
        self.seg_first = 0.0

    def update(self, t: float, base: float) -> tuple[float, bool]:
        w = self.window
        if self.prev_t is not None and t < self.prev_t:
            raise ValueError("windows must arrive in time order")
        if self.prev_t is None or t - self.prev_t > 1.5 * self.stride:
            self.seg_first = t
            self.history.clear()
        self.prev_t = t
        while self.history and self.history[0][0] < t - w.history_s:
            self.history.popleft()
        if w.gap_s == 0:
            block = [v for _t, v in self.history]
        else:
            block = [v for th, v in self.history if th < t - w.gap_s]
        self.history.append((t, base))
        warm = t - self.seg_first >= w.gap_s + w.min_history_s - 1e-9
        if not block:
            return float("nan"), False
        arr = np.asarray(block, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(MAD_TO_SD * np.median(np.abs(arr - med)))
        score = (base - med) / max(mad, self.floor)
        ok = warm and bool(np.isfinite(score))
        return (score if ok else float("nan")), ok


@dataclass(frozen=True)
class Scored:
    window: Window
    score: float
    scored: bool
    feature_errors: F64  # (features,) segment-averaged squared error
    residuals: F64  # (features,) segment-averaged input minus reconstruction


class NodeScorer:
    """Scores one node's windows with the frozen network and its calibration."""

    def __init__(self, site: Site, calibration: NodeCalibration, net: Any, params: Params) -> None:
        self.site, self.cal, self.net, self.params = site, calibration, net, params
        cfg = params.models.scoring.site_a if site == "site_a" else params.models.scoring.site_b
        self.aggregation, self.top_k, self.shift = cfg.aggregation, cfg.top_k, cfg.shift
        if cfg.smooth != 1:
            raise ValueError("the edge runtime implements unsmoothed scores (ADR 0003 choice)")
        self.n_features = len(calibration.features)
        self.median = np.asarray(calibration.error_median) if calibration.error_median else None
        self.iqr = np.asarray(calibration.error_iqr) if calibration.error_iqr else None
        self.r3: dict[str, CausalR3] = {}

    def _r3(self, part: str, stride_s: float) -> CausalR3:
        if part not in self.r3:
            self.r3[part] = CausalR3(
                self.params.models.scoring.shift_window, stride_s, self.cal.score_median
            )
        return self.r3[part]

    def score(self, w: Window, stride_s: float) -> Scored:
        import torch

        with torch.no_grad():
            recon, _mu, _lv = self.net(torch.from_numpy(np.ascontiguousarray(w.x)))
        rec = recon.numpy()
        err = (rec - w.x) ** 2  # float32, as the batch path
        segments = err.shape[1] // self.n_features
        fe = (
            err.reshape(segments, self.n_features)
            .mean(axis=0)
            .astype(np.float32)
            .astype(np.float64)
        )
        res = (w.x - rec).reshape(segments, self.n_features).mean(axis=0).astype(np.float64)
        if self.aggregation == "mean":
            base = float(err.astype(np.float64).mean())
        else:
            assert self.median is not None and self.iqr is not None
            z = (fe - self.median) / self.iqr
            if self.aggregation == "norm_mean":
                base = float(z.mean())
            else:
                k = min(self.top_k, z.size)
                base = float(np.sort(z)[-k:].mean())
        if self.shift == "r0":
            return Scored(w, base, True, fe, res)
        if self.shift != "r3":
            raise ValueError(f"the edge runtime implements r0 and r3, not {self.shift}")
        value, ok = self._r3(w.part, stride_s).update(w.t_start, base)
        return Scored(w, value, ok, fe, res)


def merged(sources: list[WindowSource]) -> Iterator[tuple[WindowSource, int]]:
    """All windows of a site's sources in order of window end (then node, part)."""
    keys = []
    for si, src in enumerate(sources):
        for k in range(len(src)):
            keys.append((src.t_end(k), src.node, src.part, k, si))
    keys.sort()
    for _t, _node, _part, k, si in keys:
        yield sources[si], k
