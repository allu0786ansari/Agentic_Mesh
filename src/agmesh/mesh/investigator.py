"""The Investigator: same-site evidence, at most 3 hops, within a deadline (ADR 0012 §3).

A **hop** is one evidence request to one node.

* **Site B:** the site's nodes, alerting nodes first.
* **Site A:** the alerting nodes only, for adjacent windows.

Each hop:

* goes through that node's circuit breaker;
* is retried once on a transient error, after an exponential backoff with jitter;
* is recorded in ``hops`` before the next one starts.

When the overall deadline passes, the investigation stops and returns what it has,
marked degraded. A hop refused by an open breaker also marks it degraded.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agmesh.common.schemas import EvidenceRecord
from agmesh.mesh.breaker import CircuitBreaker
from agmesh.mesh.nodeclient import NodeClient, TransientNodeError
from agmesh.mesh.sessions import SessionStore
from agmesh.observability.tracing import KIND, TOOL, noop_tracer

SITE_NODES = {"site_a": ("a1", "a2", "a3", "a4"), "site_b": ("b1", "b2", "b3")}
PAD_S = {"site_a": 120.0, "site_b": 60.0}
BACKOFF_S = 0.05


@dataclass
class Investigation:
    records: list[EvidenceRecord] = field(default_factory=list)
    hops: list[tuple[str, str, float]] = field(default_factory=list)  # node, outcome, ms
    degraded: bool = False
    reason: str | None = None


def plan_hops(site: str, alerting: Sequence[str], max_hops: int) -> list[str]:
    nodes = sorted(set(alerting))
    if site == "site_b":
        nodes += [n for n in SITE_NODES[site] if n not in nodes]
    return nodes[:max_hops]


class Investigator:
    def __init__(
        self,
        client: NodeClient,
        store: SessionStore,
        *,
        max_hops: int = 3,
        connect_timeout_s: float = 0.2,
        read_timeout_s: float = 0.5,
        max_retries: int = 1,
        deadline_s: float = 3.0,
        breakers: dict[str, CircuitBreaker] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        seed: int = 0,
        tracer: Any = None,
    ) -> None:
        self.client, self.store = client, store
        self.max_hops, self.max_retries, self.deadline_s = max_hops, max_retries, deadline_s
        self.timeout_s = connect_timeout_s + read_timeout_s
        self.breakers = breakers if breakers is not None else {}
        self.clock, self.sleep = clock, sleep
        self.rng = random.Random(seed)  # noqa: S311 (backoff jitter, not security)
        self.tracer = tracer if tracer is not None else noop_tracer()

    def breaker(self, node: str) -> CircuitBreaker:
        if node not in self.breakers:
            self.breakers[node] = CircuitBreaker(node, clock=self.clock)
        return self.breakers[node]

    def run(
        self, session_id: str, site: str, alerting: Sequence[str], start_s: float, end_s: float
    ) -> Investigation:
        out = Investigation()
        t0 = self.clock()
        lo, hi = start_s - PAD_S[site], end_s + PAD_S[site]
        # resume: hops already made in this session count against the budget and are
        # not repeated; their evidence comes back from the session store
        done = {node for _i, node, _o in self.store.hops(session_id)}
        for node in sorted(done):
            prior = self.store.step(session_id, f"hop:{node}")
            if prior is not None:
                out.records += [EvidenceRecord.model_validate(r) for r in prior["records"]]
        budget = self.max_hops - len(done)
        todo = [n for n in plan_hops(site, alerting, self.max_hops) if n not in done][
            : max(budget, 0)
        ]
        for node in todo:
            if self.clock() - t0 >= self.deadline_s:
                out.degraded, out.reason = True, "investigation deadline"
                break
            br = self.breaker(node)
            hop_ns = time.time_ns()
            if not br.allow():
                self._hop(session_id, node, "breaker_open", 0.0, out, hop_ns)
                out.degraded, out.reason = True, f"breaker open for {node}"
                continue
            outcome, started = "error", self.clock()
            for attempt in range(self.max_retries + 1):
                remaining = self.deadline_s - (self.clock() - t0)
                if remaining <= 0:
                    outcome = "deadline"
                    break
                try:
                    found = self.client.evidence(
                        node, lo, hi, min(self.timeout_s, remaining), session_id=session_id
                    )
                except TransientNodeError:
                    if attempt < self.max_retries:
                        self.sleep(BACKOFF_S * (2**attempt) * (1 + self.rng.random()))
                        continue
                    outcome = "transient_error"
                except Exception:  # non-transient: never retried
                    outcome = "error"
                    break
                else:
                    seen = {r.evidence_id for r in out.records}
                    out.records += [r for r in found if r.evidence_id not in seen]
                    outcome = "ok"
                    break
            ms = 1000.0 * (self.clock() - started)
            if outcome == "ok":
                br.success()
            else:
                br.failure()
                out.degraded = True
                out.reason = out.reason or f"{node}: {outcome}"
            self._hop(session_id, node, outcome, ms, out, hop_ns)
        out.records.sort(key=lambda r: (r.window_start, r.node_id))
        return out

    def _hop(
        self,
        session_id: str,
        node: str,
        outcome: str,
        ms: float,
        out: Investigation,
        start_ns: int | None = None,
    ) -> None:
        span = self.tracer.start_span(
            "hop",
            start_time=start_ns,
            attributes={
                KIND: TOOL,
                "tool.name": "node_evidence",
                "agmesh.node": node,
                "agmesh.outcome": outcome,
                "agmesh.ms": ms,
            },
        )
        span.end()
        mine = [r.model_dump() for r in out.records if r.node_id == node]
        self.store.record_step(session_id, f"hop:{node}", outcome, {"records": mine})
        self.store.record_hop(session_id, node, outcome, ms)
        out.hops.append((node, outcome, ms))
