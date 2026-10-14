"""FedAvg and FedProx aggregation.

Both average the client models **uniformly**. Weighting by a client's number of
windows would make the aggregate depend on a data-dependent count that the privacy
analysis does not cover. Our clients are of similar size (Site A 3,060-3,580 windows,
Site B 59,000-65,000), so uniform weights change little. FedProx differs only on the
client: each local step adds mu * (w - w_global). In DP runs that term is added after
clipping and noise (``dp_train.fit_private``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agmesh.federated.protocol import Weights


@dataclass(frozen=True)
class Strategy:
    name: str  # "fedavg" or "fedprox"
    proximal_mu: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in ("fedavg", "fedprox"):
            raise ValueError(f"unknown strategy {self.name!r}")
        if self.name == "fedavg" and self.proximal_mu != 0.0:
            raise ValueError("FedAvg has no proximal term")

    @property
    def client_mu(self) -> float:
        return self.proximal_mu if self.name == "fedprox" else 0.0

    def aggregate(self, updates: list[Weights]) -> Weights:
        if not updates:
            raise ValueError("nothing to aggregate")
        shapes = [[a.shape for a in u] for u in updates]
        if any(s != shapes[0] for s in shapes):
            raise ValueError("client updates have different shapes")
        return [
            np.mean(np.stack([u[i] for u in updates]), axis=0).astype(np.float32)
            for i in range(len(updates[0]))
        ]
