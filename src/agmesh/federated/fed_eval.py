"""Federated evaluation: aggregate client summaries; pooled AUROC and H1c from histograms.

**Score grid.** Scores are binned on a fixed public grid. The transform is
``t(x) = sign(x) * log1p(|x|)``, with 1,000 equal bins over ``[-12, 12]``, so
``|x|`` up to about 160,000, since Site B's normalised scores can be large. This
covers the R0 and R3 score ranges with fine resolution near the usual thresholds.
Out-of-range scores go to the end bins.

**Pooled AUROC.** It is computed from the summed normal and attack histograms, with
ties inside a bin counted as half. Its error bound is half the share of
(attack, normal) pairs that fall in the same bin; the exact pooled AUROC lies within
the bound.

**H1c threshold.** Clients release their normal-score histograms on a coarser grid of
``bins`` bins (default 100, groups of 10 grid bins), so that each bin holds more
windows than the noise. The server sums the noisy histograms and zeroes every bin
below the noise level at which, with probability 0.95, *no* noise-only bin survives
(a Bonferroni cut ``Phi^-1(1 - 0.05 / bins)`` noise SDs, about 3.3 for 100 bins).
That is post-processing. A single surviving noise bin high in the tail would
otherwise set the threshold on its own. The server then takes the smallest bin upper
edge with at most ``target_fpr`` of the mass above it. Using the upper edge errs
towards fewer false alarms.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from agmesh.federated.protocol import EvalReply

F64 = npt.NDArray[np.float64]
GRID_BINS = 1000
GRID_LIMIT = 12.0


def transform(x: npt.ArrayLike) -> F64:
    v = np.asarray(x, dtype=np.float64)
    out: F64 = np.sign(v) * np.log1p(np.abs(v))
    return out


def inverse(t: float) -> float:
    return float(math.copysign(math.expm1(abs(t)), t))


def edges() -> F64:
    out: F64 = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_BINS + 1)
    return out


def bin_scores(scores: npt.ArrayLike) -> F64:
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    idx = np.floor((transform(s) + GRID_LIMIT) / (2 * GRID_LIMIT) * GRID_BINS).astype(np.int64)
    idx = np.clip(idx, 0, GRID_BINS - 1)
    out: F64 = np.bincount(idx, minlength=GRID_BINS).astype(np.float64)
    return out


def pooled_auroc(normal: F64, attack: F64) -> tuple[float, float]:
    """AUROC from binned scores and its tie bound (``nan`` if a class is empty)."""
    n_total, a_total = float(normal.sum()), float(attack.sum())
    if n_total <= 0 or a_total <= 0:
        return float("nan"), float("nan")
    below = np.cumsum(normal) - normal  # normal windows in lower bins
    auroc = float((attack * (below + 0.5 * normal)).sum() / (a_total * n_total))
    bound = float(0.5 * (attack * normal).sum() / (a_total * n_total))
    return auroc, bound


def coarsen(counts: F64, bins: int) -> F64:
    """Sum groups of grid bins into ``bins`` bins (``GRID_BINS`` must be a multiple)."""
    if GRID_BINS % bins:
        raise ValueError(f"{bins} bins do not divide the {GRID_BINS}-bin grid")
    out: F64 = np.asarray(counts, dtype=np.float64).reshape(bins, -1).sum(axis=1)
    return out


def bonferroni_cut(bins: int, alpha: float = 0.05) -> float:
    """Noise SDs above which no noise-only bin survives with probability 1 - alpha."""
    from statistics import NormalDist

    return float(NormalDist().inv_cdf(1.0 - alpha / bins))


def h1c_threshold(
    summed: F64, *, noise_sd: float, target_fpr: float
) -> tuple[float, dict[str, Any]]:
    """Pooled threshold (score units) from summed noisy normal-score histograms."""
    bins = summed.size
    cut = bonferroni_cut(bins) if noise_sd > 0 else 0.0
    clean = np.where(summed > cut * noise_sd, summed, 0.0)
    total = float(clean.sum())
    e = np.linspace(-GRID_LIMIT, GRID_LIMIT, bins + 1)
    if total <= 0:
        return float("inf"), {"mass": 0.0, "cut_sd": cut}
    above = np.cumsum(clean[::-1])[::-1] - clean  # mass strictly above each bin
    ok = np.flatnonzero(above <= target_fpr * total)
    k = int(ok[0]) if ok.size else bins - 1
    info = {
        "mass": total,
        "bin": k,
        "bins": bins,
        "cut_sd": cut,
        "kept_bins": int((clean > 0).sum()),
    }
    return inverse(float(e[k + 1])), info


def macro(replies: list[EvalReply], keys: tuple[str, ...]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in keys:
        vals = [r.metrics.get(key) for r in replies]
        finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
        out[key] = float(np.mean(finite)) if finite else None
    return out


def summary(replies: list[EvalReply], keys: tuple[str, ...]) -> dict[str, Any]:
    normal = np.sum([r.normal_hist for r in replies if r.normal_hist is not None], axis=0)
    attack = np.sum([r.attack_hist for r in replies if r.attack_hist is not None], axis=0)
    auroc, bound = pooled_auroc(np.asarray(normal), np.asarray(attack))
    return {
        "macro": macro(replies, keys),
        "per_client": {r.node: r.metrics for r in replies},
        "pooled_auroc_hist": auroc,
        "pooled_auroc_bin_error": bound,
        "val_loss_mean": float(np.mean([r.val_loss for r in replies])),
    }
