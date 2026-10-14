"""A data-free client pool for testing the server's round logic."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agmesh.federated.protocol import (
    EvalInstruction,
    EvalReply,
    FitInstruction,
    FitReply,
    HelloReply,
    ModelSpec,
    ThresholdReply,
)

SPEC = ModelSpec("ae", 4, (3,), 0)


@dataclass
class FakePool:
    nodes: list[str]
    slow: set[str] = field(default_factory=set)  # always exceed the timeout
    exhausted_after: dict[str, int] = field(default_factory=dict)  # decline after n fits
    fits: dict[str, int] = field(default_factory=dict)
    seen_epochs: list[int] = field(default_factory=list)

    def hello(self, node: str) -> HelloReply:
        lo, hi = np.zeros(4), np.ones(4)
        counts = np.zeros((4, 10))
        counts[:, 5] = 100.0
        return HelloReply(node, SPEC, lo, hi, counts, 0.0, 3.0)

    def fit(self, node: str, ins: FitInstruction) -> FitReply:
        self.seen_epochs.append(ins.local_epochs)
        done = self.fits.get(node, 0)
        if node in self.exhausted_after and done >= self.exhausted_after[node]:
            return FitReply(node, ins.round, None, "budget exhausted", None, 1.0, 0, 0.0)
        self.fits[node] = done + 1
        bump = float(self.nodes.index(node) + 1)
        new = [w + bump for w in ins.weights]
        seconds = 1e9 if node in self.slow else 0.01
        return FitReply(node, ins.round, new, None, 0.5, 1.0, 1, seconds)

    def evaluate(self, node: str, ins: EvalInstruction) -> EvalReply:
        return EvalReply(node, float(np.mean([w.mean() for w in ins.weights])), {"auroc": 0.9})

    def threshold(self, node: str, ins: EvalInstruction) -> ThresholdReply:
        return ThresholdReply(node, np.zeros(100), 0.0)

    def privacy(self, node: str) -> dict[str, Any]:
        return {"epsilon": math.inf}
