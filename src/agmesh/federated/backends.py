"""Client pools: how the server reaches its clients.

``InProcessPool`` runs clients in this process, in a fixed order, so runs are
deterministic. It builds a **new client object for every call**, as Flower's
simulation does with its ClientApp. A client therefore keeps nothing between rounds
except what is on disk (its privacy ledger), which is what the persistence tests
rely on. Partitions and calibrations are cached under the hood, so recreating
clients is cheap.

The Flower adapter (``flower.py``) offers the same interface over Flower messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agmesh.common.config import Params
from agmesh.data.access import Site
from agmesh.federated.client import FederatedClient
from agmesh.federated.protocol import (
    EvalInstruction,
    EvalReply,
    FitInstruction,
    FitReply,
    HelloReply,
    ThresholdReply,
)
from agmesh.privacy.accountant_store import ROOT


@dataclass
class ClientSettings:
    params: Params
    site: Site
    partitions: Path
    run_id: str
    epsilon: float
    seed: int
    h1c_epsilon: float | None = None
    epochs: int | None = None
    privacy_root: Path = ROOT


@dataclass
class InProcessPool:
    settings: ClientSettings
    nodes: list[str]
    calls: dict[str, int] = field(default_factory=dict)

    def _client(self, node: str) -> FederatedClient:
        s = self.settings
        self.calls[node] = self.calls.get(node, 0) + 1
        return FederatedClient(
            s.params,
            s.site,
            node,
            partitions=s.partitions,
            run_id=s.run_id,
            epsilon=s.epsilon,
            seed=s.seed,
            node_index=self.nodes.index(node),
            h1c_epsilon=s.h1c_epsilon,
            epochs=s.epochs,
            privacy_root=s.privacy_root,
        )

    def hello(self, node: str) -> HelloReply:
        return self._client(node).hello()

    def fit(self, node: str, ins: FitInstruction) -> FitReply:
        return self._client(node).fit(ins)

    def evaluate(self, node: str, ins: EvalInstruction) -> EvalReply:
        return self._client(node).evaluate(ins)

    def threshold(self, node: str, ins: EvalInstruction) -> ThresholdReply:
        return self._client(node).threshold(ins)

    def privacy(self, node: str) -> dict[str, Any]:
        return self._client(node).privacy()
