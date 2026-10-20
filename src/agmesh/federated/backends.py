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

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agmesh.common.config import Params
from agmesh.data.access import Site
from agmesh.evaluation.sealed import FinalAccess
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
    final: FinalAccess | None = None  # a --final run: clients may evaluate on test


@dataclass
class InProcessPool:
    settings: ClientSettings
    nodes: list[str]
    calls: dict[str, int] = field(default_factory=dict)
    #: a node's index in its site's full client list, when this pool holds only some
    #: (the local-only baseline), so its noise and batch seeds match the federation's
    index_of: dict[str, int] = field(default_factory=dict)

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
            node_index=self.index_of.get(node, self.nodes.index(node)),
            h1c_epsilon=s.h1c_epsilon,
            epochs=s.epochs,
            privacy_root=s.privacy_root,
            final=s.final,
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


CENTRAL = "central"


@dataclass
class CentralPool(InProcessPool):
    """The centralised baseline (ADR 0016 §2) through the federation's own pipeline.

    * **Training:** one client, ``central``, holds every node's training windows
      under one privacy budget.
    * **Evaluation:** every node evaluates the global model on its own validation
      data, with its own threshold, as in the federation. Evaluation-only node
      clients release nothing, so they open no budget.
    """

    @property
    def train_nodes(self) -> list[str]:
        return [CENTRAL]

    def _client(self, node: str) -> FederatedClient:
        s = self.settings
        self.calls[node] = self.calls.get(node, 0) + 1
        if node == CENTRAL:
            return FederatedClient(
                s.params,
                s.site,
                CENTRAL,
                partitions=s.partitions,
                run_id=s.run_id,
                epsilon=s.epsilon,
                seed=s.seed,
                node_index=len(self.nodes),
                epochs=s.epochs,
                privacy_root=s.privacy_root,
                train_nodes=list(self.nodes),
            )
        return FederatedClient(
            s.params,
            s.site,
            node,
            partitions=s.partitions,
            run_id=s.run_id,
            epsilon=math.inf,  # evaluates only
            seed=s.seed,
            node_index=self.nodes.index(node),
            privacy_root=s.privacy_root,
            final=s.final,
        )
