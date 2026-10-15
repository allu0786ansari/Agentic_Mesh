"""The full graph on synthetic incidents (Plan Week 10 validation; ADR 0012 §1-6).

* A Site B write flood runs every agent step and ends ``COMPLETED``:
  * the four steps are persisted;
  * the hops are bounded;
  * the high-severity incident proposes a high-impact action, which waits for
    approval.
* A node that fails mid-investigation leaves the session ``COMPLETED_DEGRADED``.
* The same alert published twice opens one session and dispatches once
  (``test_replay_idempotent`` covers the runner path).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agmesh.bus.sqlite_bus import SqliteBus
from agmesh.common.schemas import EvidenceRecord
from agmesh.mesh.attribution import AttributionAgent
from agmesh.mesh.graph import Mesh
from agmesh.mesh.investigator import Investigator
from agmesh.mesh.sessions import SessionStore
from agmesh.mesh.triage import Triage
from tests.mesh_doubles import FakeNodes, ScriptedLLM, alert_of, fixed_rank, record

T0 = 1_675_242_222.0
ORDER = [
    "T0806",
    "T1692.001",
    "T0814",
    "T0861",
    "T0801",
    "T0888",
    "T0846",
    "T0866",
    "T0802",
    "T0836",
]


def _mesh(
    tmp: Path, nodes: FakeNodes, reply: dict[str, Any] | str | Exception
) -> tuple[Mesh, SessionStore, SqliteBus]:
    store, bus = SessionStore(tmp / "storage.db"), SqliteBus(tmp / "storage.db")
    inv = Investigator(nodes, store, sleep=lambda _s: None)
    agent = AttributionAgent(
        fixed_rank(ORDER),
        ScriptedLLM(reply),
        k=10,
        mitigations={"T0806": [["M0807", "Network Allowlists"]]},
    )
    return Mesh(store, bus, inv, agent), store, bus


def _incident(records: list[EvidenceRecord]) -> dict[str, Any]:
    tri = Triage(60.0)
    for r in records:
        tri.feed(alert_of(r), "m-" + r.evidence_id)
    (inc,) = tri.drain()
    return inc.summary()


def test_full_graph_completes_with_four_persisted_steps(tmp_path: Path) -> None:
    recs = [record("b1", T0 + 10 * i, [("req_write", "up", "high")]) for i in range(4)]
    nodes = FakeNodes({"b1": recs, "b2": [], "b3": []})
    mesh, store, bus = _mesh(
        tmp_path,
        nodes,
        {"technique_id": "T0806", "tactic": "x", "confidence": 0.9, "rationale": "flood"},
    )
    state = mesh.run("site_b", _incident(recs), "full", "r1")
    sid = state["session_id"]
    assert state["status"] == "COMPLETED" and store.get(sid).status == "COMPLETED"
    assert {"triage", "investigate", "attribute", "dispatch"} <= set(store.steps(sid))
    assert store.get(sid).hop_count == 3 and nodes.calls == ["b1", "b2", "b3"]
    att = state["attribution"]
    assert att["technique_id"] == "T0806" and att["mitigations"] == [
        ["M0807", "Network Allowlists"]
    ]
    acts = {a["action_type"]: a["status"] for a in store.actions(sid)}
    assert acts == {
        "operator_alert": "dispatched",
        "increase_monitoring": "dispatched",
        "isolate_or_block": "pending_approval",
    }
    assert bus.pending("actions.site_b") == 3
    assert set(state["latency_ms"]) == {"investigation", "incident_to_dispatch"}


def test_failing_node_degrades_and_no_evidence_abstains(tmp_path: Path) -> None:
    recs = [record("b2", T0, [("req_write", "up", "high")])]
    nodes = FakeNodes({"b1": [], "b2": recs, "b3": []}, {"b1": ["error"]})
    mesh, _store, _ = _mesh(
        tmp_path,
        nodes,
        {"technique_id": "T0806", "tactic": "x", "confidence": 0.9, "rationale": "r"},
    )
    inc = _incident(recs)
    inc["nodes"] = ["b1", "b2"]
    state = mesh.run("site_b", inc, "full", "r1")
    assert state["status"] == "COMPLETED_DEGRADED"
    empty = FakeNodes({"b1": [], "b2": [], "b3": []})
    mesh2, store2, _ = _mesh(
        tmp_path / "two",
        empty,
        {"technique_id": "T0806", "tactic": "x", "confidence": 0.9, "rationale": "r"},
    )
    s2 = mesh2.run("site_b", inc, "full", "r2")
    assert s2["status"] == "ABSTAINED" and store2.actions() == []


def test_rerun_of_a_finished_session_changes_nothing(tmp_path: Path) -> None:
    recs = [record("b1", T0, [("req_write", "up", "high")])]
    nodes = FakeNodes({"b1": recs, "b2": [], "b3": []})
    mesh, store, bus = _mesh(
        tmp_path,
        nodes,
        {"technique_id": "T0806", "tactic": "x", "confidence": 0.9, "rationale": "r"},
    )
    inc = _incident(recs)
    first = mesh.run("site_b", inc, "full", "r1")
    calls = len(nodes.calls)
    again = mesh.run("site_b", inc, "full", "r1")
    assert again["session_id"] == first["session_id"] and again["status"] == "COMPLETED"
    assert len(nodes.calls) == calls  # terminal session: no new hops
    assert (
        store.count() == 1
        and store.count("actions") == 3
        and bus.stats("actions.site_b")["messages"] == 3
    )
