"""One trace per session, linked from the session store (ADR 0014 §7; EA3)."""

from __future__ import annotations

import json
from pathlib import Path

from agmesh.bus.sqlite_bus import SqliteBus
from agmesh.mesh.attribution import AttributionAgent
from agmesh.mesh.graph import Mesh
from agmesh.mesh.investigator import Investigator
from agmesh.mesh.sessions import SessionStore
from agmesh.mesh.triage import Triage
from agmesh.observability import tracing
from tests.mesh_doubles import FakeNodes, ScriptedLLM, alert_of, fixed_rank, record

T0 = 1_675_242_222.0
REPLY = {"technique_id": "T0806", "tactic": "x", "confidence": 0.9, "rationale": "flood"}


def test_each_session_gets_one_trace_with_its_spans(tmp_path: Path) -> None:
    tracer, provider = tracing.setup(tmp_path / "traces")
    store, bus = SessionStore(tmp_path / "storage.db"), SqliteBus(tmp_path / "storage.db")
    nodes = FakeNodes({"b1": [record("b1", T0, [("req_write", "up", "high")])], "b2": [], "b3": []})
    inv = Investigator(nodes, store, sleep=lambda _s: None, tracer=tracer)
    agent = AttributionAgent(
        fixed_rank(["T0806", "T0814"]), ScriptedLLM(REPLY), k=2, mitigations={}, tracer=tracer
    )
    mesh = Mesh(store, bus, inv, agent, tracer=tracer)
    tri = Triage(60.0)
    recs = [record("b1", T0 + 10 * i, [("req_write", "up", "high")]) for i in range(2)]
    for r in recs:
        tri.feed(alert_of(r), "m-" + r.evidence_id)
    (inc,) = tri.drain()
    state = mesh.run("site_b", inc.summary(), "full", "run-x")
    provider.shutdown()
    sid = state["session_id"]

    assert tracing.check_run(tmp_path / "traces", [sid]) == []
    doc = json.loads((tmp_path / "traces" / f"{sid}.json").read_text())
    (t,) = doc["traces"]
    names = sorted(s["name"] for s in t["spans"])
    assert names == sorted(
        [
            "session",
            "triage",
            "investigate",
            "hop",
            "hop",
            "hop",
            "attribute",
            "retrieve",
            "llm",
            "dispatch",
        ]
    )
    root = next(s for s in t["spans"] if s["name"] == "session")
    assert root["attributes"]["session.id"] == sid
    assert root["attributes"]["agmesh.technique_id"] == "T0806"
    ids = {s["context"]["trace_id"] for s in t["spans"]}
    assert len(ids) == 1
    assert store.traces(sid) == [(sid, t["trace_id"])]
    text = json.dumps(doc)
    assert "Evidence:" not in text  # no prompt or evidence text in traces


def test_check_run_reports_missing_traces(tmp_path: Path) -> None:
    root = {"name": "session", "parent_id": None, "attributes": {"session.id": "other"}}
    (tmp_path / "s1.json").write_text(json.dumps({"traces": [{"spans": [root]}]}))
    problems = tracing.check_run(tmp_path, ["s1", "s2"])
    assert problems == ["s1: root span carries another session.id", "s2: no trace file"]


def test_without_a_tracer_nothing_is_linked(tmp_path: Path) -> None:
    store, bus = SessionStore(tmp_path / "storage.db"), SqliteBus(tmp_path / "storage.db")
    nodes = FakeNodes({"b1": [record("b1", T0, [("req_write", "up", "high")])], "b2": [], "b3": []})
    inv = Investigator(nodes, store, sleep=lambda _s: None)
    mesh = Mesh(store, bus, inv, AttributionAgent(fixed_rank(["T0806"]), None, k=1, mitigations={}))
    tri = Triage(60.0)
    r = record("b1", T0, [("req_write", "up", "high")])
    tri.feed(alert_of(r), "m1")
    (inc,) = tri.drain()
    mesh.run("site_b", inc.summary(), "retrieval_only", "r")
    assert store.traces() == []
