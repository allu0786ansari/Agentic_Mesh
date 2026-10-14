"""Classical local baselines (Proposal §6.4, §10.5): PCA and Isolation Forest.

Neither is federated: Isolation Forest has no gradients to aggregate, and PCA is a
reference point for what a linear model of normal behaviour already achieves.

* :class:`PcaDetector` keeps the components that explain ``pca_variance`` of the
  training variance; the score is the mean squared reconstruction error, and the
  per-input errors are available for evidence.
* :class:`IForestDetector` scores with the negated Isolation Forest score, so
  higher means more anomalous like every other detector.
* :class:`RuleDetector` scores with one raw input column (the new-peer rule, MB-15).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

F32 = npt.NDArray[np.float32]


@dataclass
class PcaDetector:
    variance: float
    model: PCA | None = None

    def fit(self, x: F32) -> PcaDetector:
        self.model = PCA(n_components=self.variance, svd_solver="full").fit(x)
        return self

    def errors(self, x: F32) -> F32:
        if self.model is None:
            raise RuntimeError("not fitted")
        recon = self.model.inverse_transform(self.model.transform(x))
        out: F32 = ((x - recon) ** 2).astype(np.float32)
        return out

    def score(self, x: F32) -> F32:
        out: F32 = self.errors(x).mean(axis=1).astype(np.float32)
        return out

    @property
    def n_components(self) -> int:
        return int(self.model.n_components_) if self.model is not None else 0


@dataclass
class IForestDetector:
    trees: int
    max_samples: int
    seed: int
    model: IsolationForest | None = None

    def fit(self, x: F32) -> IForestDetector:
        self.model = IsolationForest(
            n_estimators=self.trees,
            max_samples=min(self.max_samples, x.shape[0]),
            random_state=self.seed,
            n_jobs=1,
        ).fit(x)
        return self

    def score(self, x: F32) -> F32:
        if self.model is None:
            raise RuntimeError("not fitted")
        out: F32 = (-self.model.score_samples(x)).astype(np.float32)
        return out


@dataclass
class RuleDetector:
    """Score = one raw feature column (no training); used for the new-peer rule."""

    column: int
    name: str = field(default="rule")

    def score(self, raw: npt.ArrayLike) -> F32:
        out: F32 = np.asarray(raw, dtype=np.float32)[:, self.column]
        return out
