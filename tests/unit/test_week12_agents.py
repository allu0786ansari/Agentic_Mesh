from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from agents.dispatcher.action_dispatcher import ActionDispatcher
from agents.graph import build_real_graph
from agents.security.kb_builder import MITREKnowledgeBaseBuilder
from agents.security.security_agent import SecurityAgent
from agents.tracing import configure_phoenix
from edge.bus import AsyncBus
from storage.sqlite_schema import connect, seed_registry

configure_phoenix("http://127.0.0.1:6006/v1/traces")


class FakeLLM:
    def __init__(self, response: dict):
        self.response = response

    def invoke_json(self, prompt: str) -> dict:
        return dict(self.response)


def embed(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 384), dtype=np.float32)


def test_kb_parses_and_indexes_stix(tmp_path: Path) -> None:
    stix = tmp_path / "bundle.json"
    stix.write_text(json.dumps({"objects": [{"type": "attack-pattern", "name": "Valid Accounts", "description": "Use of valid accounts", "external_references": [{"source_name": "mitre-attack-ics", "external_id": "T1078"}], "kill_chain_phases": [{"phase_name": "credential-access"}], "x_mitre_platforms": ["ics"]}]}), encoding="utf-8")
    builder = MITREKnowledgeBaseBuilder(tmp_path / "qdrant", embedder=embed)
    assert len(builder.parse_stix(stix)) == 1
    assert builder.build(stix) == 1
    assert builder.count() == 1


def test_security_confidence_falls_back_to_unknown(tmp_path: Path) -> None:
    builder = MITREKnowledgeBaseBuilder(tmp_path / "qdrant", embedder=embed)
    builder.index([{"text": "valid accounts", "technique_id": "T1078", "name": "Valid Accounts", "tactic": ["credential-access"], "platform": ["ics"], "source": "test"}])
    agent = SecurityAgent(llm=FakeLLM({"technique_id": "T1078", "tactic": "credential-access", "confidence": 0.2, "reasoning": "weak", "recommended_mitigations": []}), qdrant_client=builder.client, embedder=embed)
    result = asyncio.run(agent.run({"session_id": "s12", "investigation_evidence": [{"status": "success"}]}))
    assert result["mitre_match"]["technique_id"] == "UNKNOWN"


def test_full_graph_dispatches_action_and_updates_alert(tmp_path: Path) -> None:
    metadata = tmp_path / "node.json"
    metadata.write_text(json.dumps({"node_id": "node_01", "source": "test", "data_type": "timeseries", "model_type": "vae", "feature_count": 1, "train_rows": 1, "test_rows": 1}), encoding="utf-8")
    database = tmp_path / "mesh.db"
    seed_registry(database, [metadata])
    qdrant = tmp_path / "qdrant"
    builder = MITREKnowledgeBaseBuilder(qdrant, embedder=embed)
    builder.index([{"text": "impact", "technique_id": "T1490", "name": "Inhibit System Recovery", "tactic": ["impact"], "platform": ["ics"], "source": "test"}])
    bus = AsyncBus()
    triage = FakeLLM({"escalate": True, "severity_level": "high", "reasoning": "critical"})
    security = SecurityAgent(llm=FakeLLM({"technique_id": "T1490", "tactic": "impact", "confidence": 0.9, "reasoning": "impact evidence", "recommended_mitigations": ["isolate"]}), qdrant_client=builder.client, embedder=embed, database_path=database)
    dispatcher = ActionDispatcher(database_path=database, bus=bus)
    class Triage:
        async def run(self, state):
            from agents.triage.triage_agent import TriageAgent
            return await TriageAgent(llm=triage, database_path=database).run(state)
    class Investigator:
        async def run(self, state):
            return {**state, "investigation_evidence": [{"status": "success"}], "hop_count": 1, "status": "investigated"}
    graph = build_real_graph(triage_agent=Triage(), investigator_agent=Investigator(), security_agent=security, dispatcher=dispatcher)
    result = asyncio.run(graph.ainvoke({"session_id": "s12-full", "alert": {"node_id": "node_01", "score": 0.9, "embedding": [0.0] * 128}}))
    assert result["status"] == "complete"
    assert result["mitre_match"]["technique_id"] == "T1490"
    message = asyncio.run(bus.get("actions.node_01", timeout=1))
    assert message["action"] == "isolate_node"
    with connect(database) as connection:
        assert connection.execute("SELECT status FROM alert_records WHERE alert_id = ?", ("s12-full",)).fetchone()[0] == "dispatched"
