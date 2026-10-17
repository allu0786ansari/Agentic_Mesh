"""Node APIs as separate processes; the Investigator across the process boundary (ADR 0014 §1)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from agmesh.edge.evidence_store import EvidenceStore
from agmesh.mesh.investigator import Investigator
from agmesh.mesh.nodeclient import HttpsNodeClient
from agmesh.mesh.sessions import SessionStore
from agmesh.security import ca
from agmesh.security.jwt import TokenMinter
from agmesh.security.tls import Pki, client_context
from agmesh.stack import down, up
from tests.mesh_doubles import record
from tests.security.conftest import free_port

T0 = 1_675_242_222.0


def test_three_node_processes_serve_the_investigator(tmp_path: Path) -> None:
    pki = Pki(tmp_path / "pki")
    ca.build(pki.root)
    edge = tmp_path / "edge"
    for node in ("b1", "b2", "b3"):
        store = EvidenceStore(edge / f"evidence_{node}.db")
        if node != "b3":
            store.put(record(node, T0, [("req_write", "up", "high")]))
        store.close()
    ports = {n: free_port() for n in ("b1", "b2", "b3")}
    stack = tmp_path / "stack.yaml"
    stack.write_text(
        yaml.safe_dump(
            {"host": "127.0.0.1", "pki": str(pki.root), "nodes": ports, "approval_api": free_port()}
        )
    )
    state = tmp_path / "state"
    pids = up("site_b", edge, stack, state)
    try:
        assert sorted(pids) == ["b1", "b2", "b3"] and os.getpid() not in pids.values()
        client = HttpsNodeClient(
            {n: f"https://127.0.0.1:{p}" for n, p in ports.items()},
            client_context(pki.cert("mesh"), pki.key("mesh"), pki.cert("ca-server")),
            TokenMinter.from_file(pki.root / "jwt-mesh.key"),
            connect_timeout_s=1.0,
        )
        sessions = SessionStore(tmp_path / "storage.db")
        sid = sessions.open("site_b", ["x"], "r")[0].session_id
        inv = Investigator(client, sessions, deadline_s=10.0, read_timeout_s=3.0)
        res = inv.run(sid, "site_b", ["b1"], T0, T0 + 10)
        client.close()
        assert [h[:2] for h in res.hops] == [("b1", "ok"), ("b2", "ok"), ("b3", "ok")]
        assert sorted(r.node_id for r in res.records) == ["b1", "b2"] and not res.degraded
    finally:
        stopped = down(state)
    assert stopped == ["b1", "b2", "b3"]
    for pid in pids.values():
        assert not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text()
