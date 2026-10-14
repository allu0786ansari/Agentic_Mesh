"""FPR-targeted thresholds from normal validation scores (Proposal §6.5, §10.3).

The threshold is chosen on a node's **normal validation** windows only, so that at
most ``target_fpr`` of them would raise an alert (alerts are ``score > threshold``).
It is then applied unchanged to attack validation and, after Gate G4, to test data,
where the false-positive rate is measured, not fixed by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Threshold:
    value: float
    target_fpr: float
    achieved_fpr: float  # on the normal validation windows it was fitted on
    n_normal: int


def fpr_threshold(normal_scores: npt.ArrayLike, target_fpr: float) -> Threshold:
    """Smallest threshold whose false-positive rate on ``normal_scores`` is <= target."""
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must lie in (0, 1)")
    scores = np.sort(np.asarray(normal_scores, dtype=np.float64))
    n = scores.size
    if n == 0:
        raise ValueError("no normal validation scores")
    if np.isnan(scores).any():
        raise ValueError("normal validation scores contain NaN")
    allowed = math.floor(target_fpr * n)  # windows that may exceed the threshold
    value = float(scores[n - 1 - allowed])
    achieved = float((scores > value).sum() / n)
    return Threshold(value=value, target_fpr=target_fpr, achieved_fpr=achieved, n_normal=n)
