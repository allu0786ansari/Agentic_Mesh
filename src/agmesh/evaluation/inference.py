"""Event-level inference: a two-stage stratified block bootstrap (ADR 0005, Part 2).

The uncertainty that matters for a detection claim is which attacks and which
normal periods happened to be recorded, not which random seed initialised a model.
So the resampling units are:

* **attack events**: an event carries all of its attack windows;
* **normal blocks**: normal windows grouped into non-overlapping blocks of
  ``block_s`` seconds within each continuous segment, so autocorrelation inside a
  block is kept.

Units are resampled with replacement **within each node** (stratified), and the
same draws are applied to every configuration and seed (paired). Each resample
becomes a vector of multiplicities, which turn into window weights. Every metric is
recomputed from the stored per-window records (:class:`NodeArrays`) at the stored
thresholds, so thresholds are not re-fitted. For each draw, a configuration's value
is the mean over nodes of the per-node metric, then the mean over seeds; the seeds
are averaged out, not used as replicates.

Metrics and what they resample over:

* event recall and mean time to detect: events;
* window FPR and false alarms per hour: normal blocks;
* AUROC and AUPRC: both.

The bootstrap reports the **mean** time to detect, because a weighted median is
unstable with about four events per node. Range F1 is not resampled; it is reported
as a point estimate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from agmesh.evaluation.detection import NodeArrays

F64 = npt.NDArray[np.float64]
METRICS = ("auroc", "auprc", "event_recall", "ttd_mean_s", "window_fpr", "false_alarms_per_hour")
#: Normal-block length per site (ADR 0005).
BLOCK_S = {"site_a": 1800.0, "site_b": 3600.0}
CHUNK = 500

Records = Mapping[int, Mapping[str, NodeArrays]]  # seed -> node -> record


def _segment_ids(t: F64, stride_s: float) -> npt.NDArray[np.int64]:
    order = np.argsort(t, kind="mergesort")
    ts = t[order]
    seg_sorted = np.concatenate([[0], np.cumsum(np.diff(ts) > 1.5 * stride_s)]).astype(np.int64)
    out = np.empty(t.size, dtype=np.int64)
    out[order] = seg_sorted
    return out


@dataclass(frozen=True)
class Units:
    """Resampling units of one node; identical across configurations and seeds."""

    n_events: int
    block: npt.NDArray[np.int64]  # block id of each normal window; -1 for attack windows
    n_blocks: int
    window_start_s: F64
    labels: npt.NDArray[np.bool_]

    @classmethod
    def of(cls, arrays: NodeArrays, block_s: float) -> Units:
        t = np.asarray(arrays.window_start_s, dtype=np.float64)
        y = np.asarray(arrays.labels, dtype=bool)
        seg = _segment_ids(t, arrays.stride_s)
        first = np.zeros_like(t)
        for s in np.unique(seg):
            first[seg == s] = t[seg == s].min()
        raw = seg * 1_000_000 + np.floor((t - first) / block_s).astype(np.int64)
        block = np.full(t.size, -1, dtype=np.int64)
        n_blocks = 0
        if (~y).any():
            _, dense = np.unique(raw[~y], return_inverse=True)
            block[~y] = dense
            n_blocks = int(dense.max()) + 1
        return cls(int(arrays.event_detected.size), block, n_blocks, t, y)

    def matches(self, arrays: NodeArrays) -> bool:
        return (
            arrays.event_detected.size == self.n_events
            and np.array_equal(arrays.window_start_s, self.window_start_s)
            and np.array_equal(arrays.labels, self.labels)
        )


@dataclass(frozen=True)
class _Prepared:
    """One node record reduced to what the weighted metrics need."""

    order: npt.NDArray[np.int64]  # kept windows, score descending
    group_starts: npt.NDArray[np.int64]  # tie groups within ``order``
    pos: F64  # in ``order``
    unit_event: npt.NDArray[np.int64]  # in ``order``; -1 for normal windows
    unit_block: npt.NDArray[np.int64]  # in ``order``; -1 for attack windows
    detected: F64  # per event
    ttd: F64  # per event; 0 where missed
    block_fp: F64
    block_n: F64
    block_onsets: F64
    stride_s: float


def _prepare(arrays: NodeArrays, units: Units) -> _Prepared:
    y = np.asarray(arrays.labels, dtype=bool)
    ok = np.asarray(arrays.scored, dtype=bool)
    keep = np.flatnonzero(ok | y)
    s = np.asarray(arrays.scores, dtype=np.float64)
    finite = s[ok]
    lowest = float(finite.min()) - 1.0 if finite.size else 0.0
    ranked = np.where(ok, s, lowest)[keep]
    order_local = np.argsort(-ranked, kind="mergesort")
    order = keep[order_local]
    sorted_scores = ranked[order_local]
    group_starts = np.flatnonzero(np.concatenate([[True], np.diff(sorted_scores) != 0]))
    eid = np.asarray(arrays.event_id, dtype=np.int64)
    alerts = np.asarray(arrays.alerts, dtype=bool)
    k = units.n_blocks
    normal_ok = ~y & ok
    blocks = units.block
    fp = np.bincount(blocks[normal_ok & alerts], minlength=k).astype(np.float64)
    n = np.bincount(blocks[normal_ok], minlength=k).astype(np.float64)
    # false-alarm onsets: a run of alerting scored normal windows, attributed to its first block
    t = units.window_start_s
    idx = np.flatnonzero(normal_ok & alerts)
    idx = idx[np.argsort(t[idx], kind="mergesort")]
    onset: npt.NDArray[np.bool_] = np.zeros(idx.size, dtype=bool)
    if idx.size:
        onset = np.concatenate([[True], np.diff(t[idx]) > arrays.stride_s * 1.0001])
    onsets = np.bincount(blocks[idx[onset]], minlength=k).astype(np.float64)
    det = np.asarray(arrays.event_detected, dtype=np.float64)
    ttd = np.nan_to_num(np.asarray(arrays.event_ttd_s, dtype=np.float64), nan=0.0)
    return _Prepared(
        order=order,
        group_starts=group_starts,
        pos=y[order].astype(np.float64),
        unit_event=np.where(y[order], eid[order], -1),
        unit_block=np.where(y[order], -1, blocks[order]),
        detected=det,
        ttd=ttd,
        block_fp=fp,
        block_n=n,
        block_onsets=onsets,
        stride_s=float(arrays.stride_s),
    )


def _ratio(num: F64, den: F64) -> F64:
    out: F64 = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
    return out


def _node_draws(p: _Prepared, ce: F64, ck: F64) -> dict[str, F64]:
    """Per-draw metrics of one node; ``ce`` (B, events) and ``ck`` (B, blocks) are counts."""
    out: dict[str, F64] = {
        "event_recall": _ratio(ce @ p.detected, ce.sum(axis=1)),
        "ttd_mean_s": _ratio(ce @ (p.detected * p.ttd), ce @ p.detected),
        "window_fpr": _ratio(ck @ p.block_fp, ck @ p.block_n),
        "false_alarms_per_hour": _ratio(
            ck @ p.block_onsets, (ck @ p.block_n) * p.stride_s / 3600.0
        ),
    }
    b = ce.shape[0]
    auroc = np.full(b, np.nan)
    auprc = np.full(b, np.nan)
    ev_idx = np.maximum(p.unit_event, 0)
    bl_idx = np.maximum(p.unit_block, 0)
    is_ev = p.unit_event >= 0
    for lo in range(0, b, CHUNK):
        hi = min(lo + CHUNK, b)
        we = ce[lo:hi][:, ev_idx] if ce.shape[1] else np.zeros((hi - lo, ev_idx.size))
        wk = ck[lo:hi][:, bl_idx] if ck.shape[1] else np.zeros((hi - lo, bl_idx.size))
        w = np.where(is_ev, we, wk)
        gp = np.add.reduceat(w * p.pos, p.group_starts, axis=1)
        gn = np.add.reduceat(w * (1.0 - p.pos), p.group_starts, axis=1)
        tp, tn = gp.sum(axis=1), gn.sum(axis=1)
        neg_before = np.cumsum(gn, axis=1) - gn
        correct = (gp * (tn[:, None] - neg_before - gn)).sum(axis=1) + 0.5 * (gp * gn).sum(axis=1)
        auroc[lo:hi] = _ratio(correct, tp * tn)
        cum_tot = np.cumsum(gp + gn, axis=1)
        prec = np.divide(np.cumsum(gp, axis=1), cum_tot, out=np.zeros_like(gp), where=cum_tot > 0)
        auprc[lo:hi] = _ratio((gp * prec).sum(axis=1), tp)
    out["auroc"] = auroc
    out["auprc"] = auprc
    return out


def _counts(rng: np.random.Generator, n_units: int, b: int) -> F64:
    if n_units == 0:
        return np.zeros((b, 0))
    draws = rng.integers(0, n_units, size=(b, n_units))
    flat = (draws + n_units * np.arange(b)[:, None]).ravel()
    return np.bincount(flat, minlength=b * n_units).reshape(b, n_units).astype(np.float64)


class EventBootstrap:
    """Shared resampling plan for one site: same draws for every configuration."""

    def __init__(
        self, template: Mapping[str, NodeArrays], *, block_s: float, b: int = 10_000, seed: int = 0
    ) -> None:
        self.units = {node: Units.of(a, block_s) for node, a in sorted(template.items())}
        self.b = b
        rng = np.random.default_rng(seed)
        self._draws = {
            node: (_counts(rng, u.n_events, b), _counts(rng, u.n_blocks, b))
            for node, u in self.units.items()
        }

    def _check(self, records: Records) -> None:
        for seed, nodes in records.items():
            if set(nodes) != set(self.units):
                raise ValueError(f"seed {seed}: nodes {sorted(nodes)} differ from the plan")
            for node, arrays in nodes.items():
                if not self.units[node].matches(arrays):
                    raise ValueError(f"seed {seed}, node {node}: windows or events differ")

    def _evaluate(self, records: Records, ones: bool) -> dict[str, F64]:
        self._check(records)
        per_seed: list[dict[str, F64]] = []
        for nodes in records.values():
            node_values: dict[str, list[F64]] = {m: [] for m in METRICS}
            for node, arrays in nodes.items():
                u = self.units[node]
                ce, ck = self._draws[node]
                if ones:
                    ce, ck = np.ones((1, u.n_events)), np.ones((1, u.n_blocks))
                for m, v in _node_draws(_prepare(arrays, u), ce, ck).items():
                    node_values[m].append(v)
            per_seed.append({m: _nanmean(np.vstack(v)) for m, v in node_values.items()})
        return {m: _nanmean(np.vstack([d[m] for d in per_seed])) for m in METRICS}

    def draws(self, records: Records) -> dict[str, F64]:
        """(B,) values of every metric: mean over nodes, then over seeds."""
        return self._evaluate(records, ones=False)

    def observed(self, records: Records) -> dict[str, float]:
        return {m: float(v[0]) for m, v in self._evaluate(records, ones=True).items()}


def _nanmean(stack: F64) -> F64:
    finite = np.isfinite(stack)
    total = np.where(finite, stack, 0.0).sum(axis=0)
    count = finite.sum(axis=0)
    out: F64 = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
    return out


@dataclass(frozen=True)
class Comparison:
    metric: str
    observed_difference: float
    ci: tuple[float, float]
    p_one_sided: float  # H1: mean(A) - mean(B) > 0
    b: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "observed_difference": self.observed_difference,
            "ci": list(self.ci),
            "p_one_sided": self.p_one_sided,
            "resamples": self.b,
        }


def difference(
    draws_a: Mapping[str, F64],
    draws_b: Mapping[str, F64],
    observed_a: Mapping[str, float],
    observed_b: Mapping[str, float],
    metric: str,
    *,
    level: float = 0.95,
) -> Comparison:
    """Paired comparison from precomputed draws (same plan for both)."""
    diff = draws_a[metric] - draws_b[metric]
    diff = diff[np.isfinite(diff)]
    observed = float(observed_a[metric] - observed_b[metric])
    if diff.size == 0:
        return Comparison(metric, observed, (float("nan"), float("nan")), float("nan"), 0)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(diff, [alpha, 1.0 - alpha])
    p = float((1 + np.sum(diff <= 0)) / (diff.size + 1))
    return Comparison(metric, observed, (float(lo), float(hi)), p, int(diff.size))


def compare(
    plan: EventBootstrap, a: Records, b: Records, metric: str, *, level: float = 0.95
) -> Comparison:
    """Paired comparison of two configurations on the same draws."""
    return difference(
        plan.draws(a), plan.draws(b), plan.observed(a), plan.observed(b), metric, level=level
    )


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order (NaN stays NaN)."""
    idx = [i for i, p in enumerate(p_values) if np.isfinite(p)]
    m = len(idx)
    adjusted = [float("nan")] * len(p_values)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: p_values[j])):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted
