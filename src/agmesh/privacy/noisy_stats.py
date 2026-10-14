"""DP normalisation statistics from noisy histograms (Proposal §7.3; Plan Week 5 task 4).

A model's inputs are standardised per feature. The centre and scale are computed from
the client's training data, so releasing them, or using them in a released model,
must be private.

**Mechanism.** Every privacy unit (one training window) adds a total weight of 1 to
every feature's histogram. A Site B unit is one row and adds 1 to one bin. A Site A
unit holds 60 rows, each adding 1/60 to its bin, so the histogram describes rows,
which is what the non-private scaler standardises. A window's mean would understate
the spread of fast signals such as vibration. The bins are ``bins`` equal-width bins
over the feature's **public** range (:mod:`agmesh.privacy.bounds`); a value outside
the range goes to the end bin. For one feature, a unit's contribution has L2 norm at
most 1 (all its rows in one bin). Adding or removing a unit therefore changes the
concatenated ``(features, bins)`` vector by L2 norm at most ``sqrt(d)``. The
release adds Gaussian noise with standard deviation ``z * sqrt(d)`` to every bin.
That is one Gaussian mechanism ``(z, q=1, 1 step)`` for the accountant, composed with
DP-SGD in RDP.

**Post-processing (free).** Noisy counts below ``threshold`` noise standard deviations
are set to zero. This stops noise in the many empty bins from adding up in the
cumulative distribution. Then:

* centre = the histogram mean (bin centres);
* scale = the histogram standard deviation with Sheppard's correction (subtract one
  twelfth of the squared bin width from the variance), floored at bin width / sqrt(12),
  which is the spread of a value known only to lie in one bin.

These estimate the same mean and standard deviation as the non-private
:class:`~agmesh.models.inputs.Scaler`, so the cost of privacy is not confounded with
a change of standardisation. (A median and IQR were tried first. They fail on Site
B's zero-inflated counts, whose IQR is 0.) Histograms from several clients can be summed before
post-processing; that is how the federated normalisation round uses them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from agmesh.privacy.calibrate import Mechanism

if TYPE_CHECKING:
    from agmesh.models.inputs import Scaler

F64 = npt.NDArray[np.float64]
IQR_TO_SD = 1.349
NAME = "normalisation"


@dataclass(frozen=True)
class Histograms:
    lo: F64  # (d,)
    hi: F64  # (d,)
    counts: F64  # (d, bins), noisy unless noise_multiplier is 0
    noise_multiplier: float
    threshold_sd: float

    @property
    def n_features(self) -> int:
        return int(self.counts.shape[0])

    @property
    def bins(self) -> int:
        return int(self.counts.shape[1])

    @property
    def noise_sd(self) -> float:
        return self.noise_multiplier * math.sqrt(self.n_features)

    def __add__(self, other: Histograms) -> Histograms:
        if not (np.array_equal(self.lo, other.lo) and np.array_equal(self.hi, other.hi)):
            raise ValueError("histograms over different ranges cannot be added")
        if self.counts.shape != other.counts.shape:
            raise ValueError("histograms with different bins cannot be added")
        # independent noise adds in variance
        z = math.hypot(self.noise_multiplier, other.noise_multiplier)
        return Histograms(self.lo, self.hi, self.counts + other.counts, z, self.threshold_sd)


def exact_counts(values: npt.ArrayLike, lo: F64, hi: F64, bins: int) -> F64:
    """(d, bins) histogram with total weight 1 per unit and feature, values clipped in range.

    ``values`` is (units, features), one value per unit, or (units, rows, features),
    where each of a unit's rows has weight 1 / rows.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 2:
        v = v[:, None, :]
    if v.ndim != 3 or v.shape[2] != lo.size:
        raise ValueError("values must be (units, features) or (units, rows, features)")
    rows = v.shape[1]
    flat = v.reshape(-1, lo.size)
    width = (hi - lo) / bins
    idx = np.floor((flat - lo) / np.where(width > 0, width, 1.0)).astype(np.int64)
    idx = np.clip(idx, 0, bins - 1)
    out = np.zeros((lo.size, bins))
    for j in range(lo.size):
        out[j] = np.bincount(idx[:, j], minlength=bins) / rows
    return out


def release(
    values: npt.ArrayLike,
    lo: F64,
    hi: F64,
    *,
    bins: int,
    noise_multiplier: float,
    rng: np.random.Generator,
    threshold_sd: float = 3.0,
) -> Histograms:
    """The Gaussian-noised histograms of one client's units (``noise_multiplier`` 0: exact)."""
    counts = exact_counts(values, lo, hi, bins)
    if noise_multiplier > 0:
        counts = counts + rng.normal(0.0, noise_multiplier * math.sqrt(lo.size), counts.shape)
    return Histograms(lo, hi, counts, noise_multiplier, threshold_sd)


def mechanism(noise_multiplier: float) -> Mechanism:
    return Mechanism(NAME, noise_multiplier, 1.0, 1)


def _quantile(cdf: F64, edges: F64, p: float) -> float:
    total = cdf[-1]
    if total <= 0:
        return float(0.5 * (edges[0] + edges[-1]))
    target = p * total
    k = int(np.searchsorted(cdf, target, side="left"))
    k = min(k, cdf.size - 1)
    before = cdf[k - 1] if k > 0 else 0.0
    inside = cdf[k] - before
    frac = (target - before) / inside if inside > 0 else 0.5
    return float(edges[k] + frac * (edges[k + 1] - edges[k]))


def cleaned(h: Histograms) -> F64:
    """Counts with noise-only bins removed (below ``threshold_sd`` noise SDs)."""
    floor = h.threshold_sd * h.noise_sd if h.noise_multiplier > 0 else 0.0
    out: F64 = np.where(h.counts > floor, h.counts, 0.0)
    return out


def centre_scale(h: Histograms) -> tuple[F64, F64]:
    """Mean and standard deviation per feature from (summed) noisy histograms."""
    clean = cleaned(h)
    centre = np.empty(h.n_features)
    scale = np.empty(h.n_features)
    for j in range(h.n_features):
        width = (h.hi[j] - h.lo[j]) / h.bins
        mids = h.lo[j] + width * (np.arange(h.bins) + 0.5)
        total = clean[j].sum()
        floor = width / math.sqrt(12.0)
        if total <= 0:
            centre[j], scale[j] = 0.5 * (h.lo[j] + h.hi[j]), floor
            continue
        mean = float((clean[j] * mids).sum() / total)
        var = float((clean[j] * (mids - mean) ** 2).sum() / total) - width**2 / 12.0
        centre[j] = mean
        scale[j] = max(math.sqrt(max(var, 0.0)), floor)
    return centre, scale


def median_iqr(h: Histograms) -> tuple[F64, F64]:
    """Median and IQR/1.349 per feature (kept for comparison; not the default)."""
    clean = cleaned(h)
    centre = np.empty(h.n_features)
    scale = np.empty(h.n_features)
    for j in range(h.n_features):
        edges = np.linspace(h.lo[j], h.hi[j], h.bins + 1)
        cdf = np.cumsum(clean[j])
        q25, q50, q75 = (_quantile(cdf, edges, p) for p in (0.25, 0.5, 0.75))
        centre[j] = q50
        scale[j] = max((q75 - q25) / IQR_TO_SD, (h.hi[j] - h.lo[j]) / h.bins)
    return centre, scale


def scaler(h: Histograms, clip: float) -> Scaler:
    from agmesh.models.inputs import Scaler  # lazy: the server imports this module

    centre, scale = centre_scale(h)
    return Scaler(centre, scale, clip)
