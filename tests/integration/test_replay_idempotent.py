"""Same alert twice: one session, one dispatch (Plan Week 10 validation; ADR 0012 §5-6)."""

from __future__ import annotations

from pathlib import Path

from agmesh.bus.sqlite_bus import SqliteBus
from agmesh.mesh.attribution import AttributionAgent
from agmesh.mesh.graph import Mesh
from agmesh.mesh.investigator import Investigator
from agmesh.mesh.runner import consume
from agmesh.mesh.sessions import SessionStore
from agmesh.mesh.triage import Triage
from tests.mesh_doubles import FakeNodes, alert_of, fixed_rank, record

T0 = 1_675_242_222.0


def test_duplicate_alerts_and_rerun_dispatch_once(tmp_path: Path) -> None:
    db = tmp_path / "storage.db"
    bus, store = SqliteBus(db), SessionStore(db)
    recs = [record("b1", T0 + 10 * i, [("req_write", "up", "high")]) for i in range(3)]
    far = [record("b2", T0 + 900, [("req_read", "up", "low")])]
    for r in recs + far:
        a = alert_of(r)
        assert bus.publish("alerts.site_b", a, a["alert_id"])
        assert not bus.publish(
            "alerts.site_b", a, a["alert_id"]
        )  # the replayed alert is stored once
    nodes = FakeNodes({"b1": recs, "b2": far, "b3": []})
    agent = AttributionAgent(fixed_rank(["T0806", "T0861"]), None, k=10)
    mesh = Mesh(store, bus, Investigator(nodes, store, sleep=lambda _s: None), agent)
    out = consume(mesh, bus, Triage(60.0), "site_b", "retrieval_only", "r1")
    assert len(out) == 2 and store.count() == 2  # two incidents (900 s apart)
    assert bus.pending("alerts.site_b") == 0 and bus.stats("alerts.site_b")["acked"] == 4
    actions = store.count("actions")
    # the whole stream replayed: same alert IDs -> same sessions, no new dispatch
    for r in recs + far:
        bus.publish("alerts.site_b", alert_of(r), alert_of(r)["alert_id"] + "-replay")
    consume(mesh, bus, Triage(60.0), "site_b", "retrieval_only", "r1")
    assert store.count() == 2 and store.count("actions") == actions
    pairs = [(a["session_id"], a["action_type"]) for a in store.actions()]
    assert len(pairs) == len(set(pairs))  # exactly one action per (session, type)
