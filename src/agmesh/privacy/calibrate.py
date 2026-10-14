"""RDP accounting and up-front noise calibration (Proposal §7.3; flaws P1, F8).

Every release a client makes is a **mechanism**: a (sub)sampled Gaussian with a noise
multiplier ``z`` (noise standard deviation divided by L2 sensitivity), a sampling
rate ``q`` and a number of compositions ``steps``.

* A DP-SGD step is ``(z, q = batch / N, 1)`` under Poisson sampling.
* A one-off release of a whole-data statistic, such as the normalisation histograms
  or the H1c score histogram, is ``(z, 1, 1)``. That is the plain Gaussian mechanism.

All of a client's mechanisms compose in Rényi DP, the tightest standard composition
for Gaussian mechanisms, and are converted to (ε, δ) once, for the client's δ. The
order grid is wider than Opacus's default, because small ε with many steps needs
large orders; the default grid makes Opacus warn that the bound is loose.

Calibration is done **once, before training**, for the planned total: rounds x local
epochs x steps per epoch, plus any warm-up. Fixed one-off releases are included, so
the final ε of a run that executes its plan is at most the target.
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

#: Rényi orders: step 0.02 below 11, where the optimum usually lies; integers above
#: (fractional large orders are numerically unstable in the sampled-Gaussian bound).
#: Opacus's default grid (step 0.1 up to 11) can be loose by about 0.01 ε.
ALPHAS: tuple[float, ...] = (
    *[1.0 + x / 50.0 for x in range(1, 500)],  # 1.02 ... 10.98, step 0.02
    *[float(a) for a in range(11, 64)],  # integer orders: exact and stable
    *[64.0, 80.0, 96.0, 128.0, 160.0, 192.0, 256.0, 320.0, 384.0, 512.0],  # integers only
)
NOISE_SEARCH = (1e-3, 1e4)
TOLERANCE = 1e-3  # calibrated ε lies in [target - TOLERANCE, target]


@dataclass(frozen=True)
class Mechanism:
    """One (sub)sampled Gaussian mechanism, composed ``steps`` times."""

    name: str
    noise_multiplier: float
    sample_rate: float
    steps: int

    def __post_init__(self) -> None:
        if not (self.noise_multiplier > 0 and 0 < self.sample_rate <= 1 and self.steps >= 0):
            raise ValueError(f"invalid mechanism {self}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Mechanism:
        return cls(
            str(raw["name"]),
            float(raw["noise_multiplier"]),
            float(raw["sample_rate"]),
            int(raw["steps"]),
        )


@functools.lru_cache(maxsize=4096)
def _rdp_one_step(noise_multiplier: float, sample_rate: float) -> tuple[float, ...]:
    """RDP of one composition at every order (RDP adds linearly over compositions)."""
    from opacus.accountants.analysis.rdp import compute_rdp

    rdp = compute_rdp(q=sample_rate, noise_multiplier=noise_multiplier, steps=1, orders=ALPHAS)
    return tuple(float(x) for x in np.atleast_1d(rdp))


def epsilon(mechanisms: list[Mechanism], delta: float) -> float:
    """ε of the RDP composition of ``mechanisms`` at ``delta`` (0 if nothing was released)."""
    from opacus.accountants.analysis.rdp import get_privacy_spent

    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    active = [m for m in mechanisms if m.steps > 0]
    if not active:
        return 0.0
    total = np.zeros(len(ALPHAS))
    for m in active:
        total += m.steps * np.asarray(_rdp_one_step(m.noise_multiplier, m.sample_rate))
    with warnings.catch_warnings():
        # Opacus warns when the best order is at the edge of the grid. The bound is then
        # merely conservative (a larger ε than necessary), never invalid.
        warnings.simplefilter("ignore", UserWarning)
        eps, _alpha = get_privacy_spent(orders=list(ALPHAS), rdp=total, delta=delta)
    return float(eps)


def _bisect(target: float, eps_of: Any) -> float:
    """Smallest noise multiplier whose ε is at most ``target`` (ε falls as noise grows)."""
    lo, hi = NOISE_SEARCH
    if eps_of(hi) > target:
        raise ValueError(f"target ε {target} is unreachable even with noise {hi}")
    if eps_of(lo) <= target:
        return lo
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if eps_of(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-6 or target - eps_of(hi) <= TOLERANCE:
            break
    return hi


def gaussian_noise_multiplier(target_epsilon: float, delta: float) -> float:
    """Noise multiplier of a single Gaussian release that alone is (ε, δ)-DP."""
    return _bisect(target_epsilon, lambda z: epsilon([Mechanism("g", z, 1.0, 1)], delta))


def plan_steps(
    n_windows: int, batch: int, epochs: int, warmup_epochs: int = 0
) -> tuple[float, int]:
    """Poisson sampling rate and total steps for ``epochs`` (+ warm-up) passes."""
    if n_windows <= 0 or batch <= 0 or epochs <= 0:
        raise ValueError("n_windows, batch and epochs must be positive")
    q = min(1.0, batch / n_windows)
    per_epoch = max(1, round(1.0 / q))
    return q, per_epoch * (epochs + warmup_epochs)


@dataclass(frozen=True)
class PrivacyPlan:
    """A client's whole budget, fixed before it releases anything."""

    target_epsilon: float
    delta: float
    n_windows: int
    sample_rate: float
    steps: int
    noise_multiplier: float
    max_grad_norm: float
    fixed: tuple[Mechanism, ...]  # one-off releases (normalisation, thresholds)

    @property
    def sgd(self) -> Mechanism:
        return Mechanism("dpsgd", self.noise_multiplier, self.sample_rate, self.steps)

    @property
    def planned_epsilon(self) -> float:
        return epsilon([*self.fixed, self.sgd], self.delta)

    def epsilon_basic_composition(self) -> float:
        """The looser bound if every part were converted to (ε, δ) separately and summed."""
        n = 1 + len(self.fixed)
        d = self.delta / n
        return sum(epsilon([m], d) for m in (*self.fixed, self.sgd))

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_epsilon": self.target_epsilon,
            "delta": self.delta,
            "n_windows": self.n_windows,
            "sample_rate": self.sample_rate,
            "steps": self.steps,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "fixed": [m.as_dict() for m in self.fixed],
            "planned_epsilon": self.planned_epsilon,
            "accountant": "rdp",
        }


def calibrate(
    *,
    target_epsilon: float,
    delta: float,
    n_windows: int,
    batch: int,
    epochs: int,
    max_grad_norm: float,
    fixed: tuple[Mechanism, ...] = (),
    warmup_epochs: int = 0,
) -> PrivacyPlan:
    """Noise multiplier for DP-SGD such that everything planned stays within ε."""
    q, steps = plan_steps(n_windows, batch, epochs, warmup_epochs)
    floor = epsilon(list(fixed), delta)
    if floor >= target_epsilon:
        raise ValueError(
            f"the one-off releases alone cost ε = {floor:.3f} >= target {target_epsilon}"
        )
    z = _bisect(
        target_epsilon,
        lambda z: epsilon([*fixed, Mechanism("dpsgd", z, q, steps)], delta),
    )
    return PrivacyPlan(target_epsilon, delta, n_windows, q, steps, z, max_grad_norm, fixed)
