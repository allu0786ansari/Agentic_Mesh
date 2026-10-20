"""The federated server: round loop, aggregation, stragglers, federated evaluation.

The server never holds data. It imports no data loader. It sees only what clients
release (``protocol.py``) through a :class:`ClientPool`, which the backend provides:
in-process objects, or Flower messages. A contract test checks both the imports and
that this module never opens a file.

**Round 0, normalisation.** Every client sends its noisy normalisation histograms.
The server sums them and derives one pooled scaler, which it broadcasts. This is
post-processing of DP releases.

**Rounds 1..R.** Every round runs: there is no early stopping, because the privacy
plan was calibrated for R rounds. For each node, in a fixed order:

* skipped if its budget is exhausted;
* dropped with probability ``dropout_prob`` (injected stragglers, seeded);
* otherwise it trains ``local_epochs`` (fixed, or drawn from ``local_epoch_choices``
  for the heterogeneous arm X2).

A reply slower than ``round_timeout_s`` is discarded as a straggler. Its privacy
spend is not refunded, which is conservative. The round aggregates if at least
``min_fit_clients`` replies arrive; otherwise the global model is kept. Each
miss counts towards ``degraded`` in the registry.

After every round the clients report the validation loss of the global model, for
the convergence curve. After the last round they report every metric. With H1c,
they release noisy normal-score histograms, the server sets the pooled threshold, and
the clients are evaluated at it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from agmesh.federated import fed_eval
from agmesh.federated.protocol import (
    EvalInstruction,
    EvalReply,
    FitInstruction,
    FitReply,
    HelloReply,
    ModelSpec,
    Normalisation,
    ThresholdReply,
    Weights,
)
from agmesh.federated.registry import NodeRegistry
from agmesh.federated.strategies import Strategy
from agmesh.privacy import noisy_stats


class ClientPool(Protocol):
    nodes: list[str]  # every node evaluates the global model

    def hello(self, node: str) -> HelloReply: ...

    def fit(self, node: str, ins: FitInstruction) -> FitReply: ...

    def evaluate(self, node: str, ins: EvalInstruction) -> EvalReply: ...

    def threshold(self, node: str, ins: EvalInstruction) -> ThresholdReply: ...

    def privacy(self, node: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ServerConfig:
    site: str
    rounds: int
    strategy: Strategy
    local_epochs: int
    min_fit_clients: int
    round_timeout_s: float
    seed: int
    dropout_prob: float = 0.0
    local_epoch_choices: tuple[int, ...] = ()
    h1c: bool = False
    target_fpr: float = 0.01
    threshold_sd: float = 3.0
    metric_keys: tuple[str, ...] = ()
    final: bool = False  # also evaluate on test after validation (ADR 0019 §4)


@dataclass
class RoundLog:
    round: int
    participants: list[str]
    missed: list[str]
    declined: list[str]
    aggregated: bool
    train_loss: float | None
    val_loss: float
    seconds: float


@dataclass
class FederationResult:
    spec: ModelSpec
    norm: Normalisation
    weights: Weights
    rounds: list[RoundLog]
    final: dict[str, Any]
    final_replies: list[EvalReply]
    h1c: dict[str, Any] | None
    registry: dict[str, dict[str, object]]
    privacy: dict[str, dict[str, Any]]
    seconds: float
    extras: dict[str, Any] = field(default_factory=dict)


def initial_weights(spec: ModelSpec, seed: int) -> Weights:
    from agmesh.models import nets

    model = (
        nets.build("vae", spec.n_inputs, list(spec.hidden), spec.latent, seed=seed)
        if spec.kind == "vae"
        else nets.build("ae", spec.n_inputs, list(spec.hidden), seed=seed)
    )
    return [v.detach().numpy().astype(np.float32).copy() for v in model.state_dict().values()]


def pooled_normalisation(replies: list[HelloReply]) -> Normalisation:
    hists = [
        noisy_stats.Histograms(r.lo, r.hi, r.counts, r.noise_multiplier, r.threshold_sd)
        for r in replies
    ]
    total = hists[0]
    for h in hists[1:]:
        total = total + h
    centre, scale = noisy_stats.centre_scale(total)
    return Normalisation(centre, scale)


class Server:
    def __init__(self, pool: ClientPool, config: ServerConfig, registry: NodeRegistry) -> None:
        self.pool, self.cfg, self.registry = pool, config, registry
        self.h1c_arrays: dict[str, list[EvalReply]] = {}

    def _eval_all(self, ins: EvalInstruction) -> list[EvalReply]:
        return [self.pool.evaluate(node, ins) for node in self.pool.nodes]

    def _index(self, node: str, default: int) -> int:
        index_of: dict[str, int] = getattr(self.pool, "index_of", {}) or {}
        return index_of.get(node, default)

    def _train_nodes(self) -> list[str]:
        """Who sends normalisation histograms and trains: every node, unless the pool
        names its own training clients (the centralised baseline, ADR 0016 §2)."""
        train: list[str] | None = getattr(self.pool, "train_nodes", None)
        return list(train) if train else list(self.pool.nodes)

    def run(self) -> FederationResult:
        cfg = self.cfg
        started = time.perf_counter()
        train_nodes = self._train_nodes()
        for node in train_nodes:
            self.registry.register(node, cfg.site, "vae" if cfg.site == "site_a" else "ae")
        hello = [self.pool.hello(node) for node in train_nodes]
        if len(hello) < cfg.min_fit_clients:
            raise RuntimeError("too few clients for the normalisation round")
        spec = hello[0].spec
        if any(h.spec != spec for h in hello):
            raise RuntimeError("clients disagree on the model spec")
        norm = pooled_normalisation(hello)
        weights = initial_weights(spec, cfg.seed)
        logs: list[RoundLog] = []
        for rnd in range(1, cfg.rounds + 1):
            t0 = time.perf_counter()
            rng = np.random.default_rng([cfg.seed, rnd])
            updates: list[Weights] = []
            participants, missed, declined, losses = [], [], [], []
            for idx, node in enumerate(train_nodes):
                if self.registry.status(node)[0] == "exhausted":
                    continue
                drop = rng.random() < cfg.dropout_prob
                epochs = (
                    int(rng.choice(cfg.local_epoch_choices))
                    if cfg.local_epoch_choices
                    else cfg.local_epochs
                )
                if drop:
                    missed.append(node)
                    self.registry.missed(node)
                    continue
                ins = FitInstruction(
                    rnd,
                    weights,
                    norm,
                    epochs,
                    cfg.strategy.client_mu,
                    seed=cfg.seed * 100_000 + rnd * 100 + self._index(node, idx),
                )
                reply = self.pool.fit(node, ins)
                if reply.declined is not None:
                    declined.append(node)
                    self.registry.exhausted(node)
                elif reply.weights is None or reply.seconds > cfg.round_timeout_s:
                    missed.append(node)
                    self.registry.missed(node)
                else:
                    updates.append(reply.weights)
                    participants.append(node)
                    self.registry.succeeded(node)
                    if reply.train_loss is not None:
                        losses.append(reply.train_loss)
            aggregated = len(updates) >= cfg.min_fit_clients
            if aggregated:
                weights = cfg.strategy.aggregate(updates)
            curve = self._eval_all(EvalInstruction(weights, norm, full=False))
            logs.append(
                RoundLog(
                    rnd,
                    participants,
                    missed,
                    declined,
                    aggregated,
                    float(np.mean(losses)) if losses else None,
                    float(np.mean([r.val_loss for r in curve])),
                    time.perf_counter() - t0,
                )
            )
        final_replies = self._eval_all(EvalInstruction(weights, norm, full=True))
        final = fed_eval.summary(final_replies, cfg.metric_keys)
        h1c = self._h1c(weights, norm) if cfg.h1c else None
        extras: dict[str, Any] = (
            {"h1c_replies": self.h1c_arrays}
            if cfg.h1c and h1c and h1c.get("threshold") is not None
            else {}
        )
        if cfg.final:
            extras["test"] = self._test(weights, norm, h1c)
        return FederationResult(
            spec,
            norm,
            weights,
            logs,
            final,
            final_replies,
            h1c,
            self.registry.snapshot(),
            {node: self.pool.privacy(node) for node in train_nodes},
            time.perf_counter() - started,
            extras,
        )

    def _test(
        self, weights: Weights, norm: Normalisation, h1c: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The final model on test, at the thresholds fixed on validation (ADR 0019 §4).

        Each client reads its test partition once in the run; the H1c variants reuse
        it. The H1c pooled threshold is the one released from the validation
        histograms above, so no new release (and no privacy cost) happens here.
        """
        cfg = self.cfg
        replies = self._eval_all(EvalInstruction(weights, norm, full=True, part="test"))
        out: dict[str, Any] = {
            "final": fed_eval.summary(replies, cfg.metric_keys),
            "replies": replies,
        }
        tau = (h1c or {}).get("threshold")
        if cfg.h1c and tau is not None:
            at_tau = self._eval_all(
                EvalInstruction(weights, norm, full=True, threshold=tau, shift="r0", part="test")
            )
            per_node = self._eval_all(
                EvalInstruction(weights, norm, full=True, shift="r0", part="test")
            )
            out["h1c"] = {
                "threshold": tau,
                "threshold_from": "validation histograms (this run)",
                "macro": fed_eval.macro(at_tau, cfg.metric_keys),
                "per_client": {r.node: r.metrics for r in at_tau},
                "r0_per_node_thresholds": {
                    "macro": fed_eval.macro(per_node, cfg.metric_keys),
                    "per_client": {r.node: r.metrics for r in per_node},
                },
            }
            out["h1c_replies"] = {"pooled": at_tau, "r0_per_node": per_node}
        return out

    def _h1c(self, weights: Weights, norm: Normalisation) -> dict[str, Any]:
        cfg = self.cfg
        replies = [
            self.pool.threshold(node, EvalInstruction(weights, norm, full=False))
            for node in self.pool.nodes
        ]
        released = [r for r in replies if r.counts is not None]
        if not released:
            return {"threshold": None, "declined": [r.node for r in replies]}
        summed = np.sum([r.counts for r in released], axis=0)
        noise_sd = float(np.sqrt(np.sum([r.noise_sd**2 for r in released])))
        tau, info = fed_eval.h1c_threshold(summed, noise_sd=noise_sd, target_fpr=cfg.target_fpr)
        at_tau = self._eval_all(
            EvalInstruction(weights, norm, full=True, threshold=tau, shift="r0")
        )
        per_node = self._eval_all(EvalInstruction(weights, norm, full=True, shift="r0"))
        # per-window records at both thresholds, for the event-level H1c test (ADR 0016 §4)
        self.h1c_arrays = {"pooled": at_tau, "r0_per_node": per_node}
        return {
            "threshold": tau,
            "noise_sd_summed": noise_sd,
            "released_by": [r.node for r in released],
            "declined": [r.node for r in replies if r.counts is None],
            **info,
            "scoring": "r0 (H1c is defined on scores that use no training statistic)",
            "macro": fed_eval.macro(at_tau, cfg.metric_keys),
            "per_client": {r.node: r.metrics for r in at_tau},
            "r0_per_node_thresholds": {
                "macro": fed_eval.macro(per_node, cfg.metric_keys),
                "per_client": {r.node: r.metrics for r in per_node},
            },
        }
