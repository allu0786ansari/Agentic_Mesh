"""Messages between server and clients: plain arrays and numbers, never raw data.

These are the only things a client releases to the server:

* the noisy normalisation histograms (DP; accounted);
* model updates after DP-SGD (DP; accounted);
* the noisy validation score histogram for the H1c threshold (DP; accounted, optional);
* evaluation summaries (see ``EvalReply``).

``EvalReply`` is the experimenter's measurement of a trained model on the client's
validation data, used to report results. The deployed system does not release it,
and nothing in training depends on it. In particular, the round count is fixed in
advance; there is no early stopping on these numbers. A deployment would keep the
client's own threshold and metrics local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

Weights = list[npt.NDArray[np.float32]]


@dataclass(frozen=True)
class ModelSpec:
    kind: str  # "vae" or "ae"
    n_inputs: int
    hidden: tuple[int, ...]
    latent: int


@dataclass(frozen=True)
class HelloReply:
    """Round 0: the client's model input size and its noisy normalisation histograms."""

    node: str
    spec: ModelSpec
    lo: npt.NDArray[np.float64]
    hi: npt.NDArray[np.float64]
    counts: npt.NDArray[np.float64]  # (features, bins), Gaussian-noised unless ε = ∞
    noise_multiplier: float
    threshold_sd: float


@dataclass(frozen=True)
class Normalisation:
    """The pooled scaler the server broadcasts after round 0 (post-processing of DP data)."""

    centre: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]


@dataclass(frozen=True)
class FitInstruction:
    round: int
    weights: Weights
    norm: Normalisation
    local_epochs: int
    proximal_mu: float
    seed: int


@dataclass(frozen=True)
class EvalInstruction:
    weights: Weights
    norm: Normalisation
    full: bool  # False: validation loss only (per-round curve); True: all metrics
    threshold: float | None = None  # H1c: evaluate at this pooled threshold
    shift: str | None = None  # scoring override (H1c is defined on R0 scores)
    part: str = "val_attack"  # "test" only in a final run (ADR 0019 §4)


@dataclass(frozen=True)
class FitReply:
    node: str
    round: int
    weights: Weights | None  # None when the client declined
    declined: str | None
    train_loss: float | None
    epsilon: float | None
    steps: int
    seconds: float


@dataclass(frozen=True)
class EvalReply:
    node: str
    val_loss: float  # mean reconstruction loss on normal validation windows
    metrics: dict[str, Any] = field(default_factory=dict)
    # binned scores on the site's public score grid, for the pooled AUROC
    normal_hist: npt.NDArray[np.float64] | None = None
    attack_hist: npt.NDArray[np.float64] | None = None
    arrays: dict[str, np.ndarray] | None = None  # per-window record (experimenter only)


@dataclass(frozen=True)
class ThresholdReply:
    """H1c: the client's noisy histogram of normal-validation scores."""

    node: str
    counts: npt.NDArray[np.float64] | None
    noise_sd: float
    declined: str | None = None
