"""X6 chaos harness (ADR 0017 §6): real node API processes over mTLS, faults injected.

The report collector writes ``results/X6/chaos.json`` when ``AGMESH_CHAOS_REPORT`` is
set (``make chaos``).
"""

from __future__ import annotations

import json
import os
import signal
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from agmesh.bus.sqlite_bus import SqliteBus
from agmesh.edge.evidence_store import EvidenceStore
from agmesh.mesh.attribution import AttributionAgent
from agmesh.mesh.breaker import CircuitBreaker
from agmesh.mesh.graph import Mesh
from agmesh.mesh.investigator import Investigator
from agmesh.mesh.nodeclient import HttpsNodeClient
from agmesh.mesh.sessions import TERMINAL, SessionStore
from agmesh.mesh.triage import Triage
from agmesh.security import ca
from agmesh.security.jwt import TokenMinter
from agmesh.security.tls import Pki, client_context
from agmesh.stack import down, up
from tests.mesh_doubles import alert_of, fixed_rank, record
from tests.security.conftest import free_port

T0 = 1_675_242_222.0
NODES = ("b1", "b2", "b3")
REPORT: dict[str, Any] = {}


@dataclass
class Stack:
    root: Path
    pki: Pki
    edge: Path
    stack_yaml: Path
    ports: dict[str, int]
    pids: dict[str, int] = field(default_factory=dict)
    state: Path = Path()

    def start(self, delays: dict[str, float] | None = None) -> None:
        self.pids = up("site_b", self.edge, self.stack_yaml, self.state, fault_delay_s=delays)

    def kill(self, node: str) -> None:
        os.kill(self.pids[node], signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.waitpid(self.pids[node], os.WNOHANG)
                os.kill(self.pids[node], 0)
            except (ProcessLookupError, ChildProcessError):
                return
            time.sleep(0.05)

    def client(self) -> HttpsNodeClient:
        p = self.pki
        return HttpsNodeClient(
            {n: f"https://127.0.0.1:{port}" for n, port in self.ports.items()},
            client_context(p.cert("mesh"), p.key("mesh"), p.cert("ca-server")),
            TokenMinter.from_file(p.root / "jwt-mesh.key"),
            connect_timeout_s=0.2,
        )


@pytest.fixture(scope="session")
def pki(tmp_path_factory: pytest.TempPathFactory) -> Pki:
    root = tmp_path_factory.mktemp("chaos_pki")
    ca.build(root)
    return Pki(root)


@pytest.fixture
def stack(pki: Pki, tmp_path: Path) -> Iterator[Stack]:
    edge = tmp_path / "edge"
    for node in NODES:
        store = EvidenceStore(edge / f"evidence_{node}.db")
        for i in range(60):  # one record per minute over an hour
            store.put(record(node, T0 + 60 * i, [("req_write", "up", "high")]))
        store.close()
    ports = {n: free_port() for n in NODES}
    stack_yaml = tmp_path / "stack.yaml"
    stack_yaml.write_text(
        yaml.safe_dump(
            {"host": "127.0.0.1", "pki": str(pki.root), "nodes": ports, "approval_api": free_port()}
        )
    )
    s = Stack(tmp_path, pki, edge, stack_yaml, ports, state=tmp_path / "state")
    yield s
    down(s.state)


@dataclass
class Rig:
    mesh: Mesh
    store: SessionStore
    bus: SqliteBus
    transitions: list[tuple[str, str, str]]
    client: HttpsNodeClient


def make_rig(stack: Stack, db: Path) -> Rig:
    store, bus = SessionStore(db), SqliteBus(db)
    transitions: list[tuple[str, str, str]] = []
    breakers = {n: CircuitBreaker(n, on_transition=lambda *t: transitions.append(t)) for n in NODES}
    client = stack.client()
    inv = Investigator(client, store, breakers=breakers)  # production timeouts and deadline
    agent = AttributionAgent(fixed_rank(["T0806", "T0814"]), None, k=10)
    return Rig(Mesh(store, bus, inv, agent), store, bus, transitions, client)


def incident(t: float, nodes: tuple[str, ...] = ("b1",)) -> dict[str, Any]:
    tri = Triage(60.0)
    for n in nodes:
        r = record(n, t, [("req_write", "up", "high")])
        tri.feed(alert_of(r), "m-" + r.evidence_id)
    (inc,) = tri.drain()
    return inc.summary()


def terminal_and_unique(store: SessionStore) -> tuple[bool, bool]:
    terminal = all(s.status in TERMINAL for s in store.all_sessions())
    pairs = [(a["session_id"], a["action_type"]) for a in store.actions()]
    return terminal, len(pairs) == len(set(pairs))


@pytest.fixture(scope="session")
def chaos_report() -> Iterator[dict[str, Any]]:
    yield REPORT
    if os.environ.get("AGMESH_CHAOS_REPORT"):
        from agmesh.common.artifacts import write_artifact

        write_artifact(
            Path("results/X6/chaos.json"),
            kind="X6.chaos",
            data_source="runtime",
            payload={
                "protocol": "docs/adr/0017-x3-x6-week13.md §6",
                "scenarios": REPORT,
                "all_passed": all(v.get("passed") for v in REPORT.values()),
            },
        )
        print(json.dumps(REPORT, indent=1))
