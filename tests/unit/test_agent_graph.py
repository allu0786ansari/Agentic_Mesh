from __future__ import annotations

from agents.graph import app
from agents.tracing import configure_phoenix

configure_phoenix("http://127.0.0.1:6006/v1/traces")


def test_high_alert_runs_full_graph() -> None:
    result = app.invoke({"session_id": "s1", "alert": {"node_id": "node_01", "score": 0.9}})
    assert result["status"] == "complete"
    assert result["hop_count"] == 1
    assert result["mitre_match"]["technique_id"] == "UNKNOWN"
    assert result["remediation"]["action"] == "investigate"


def test_low_alert_dispatches_without_investigation() -> None:
    result = app.invoke({"session_id": "s2", "alert": {"node_id": "node_01", "score": 0.0}})
    assert result["status"] == "complete"
    assert result.get("hop_count") or 0 == 0
    assert result.get("investigation_evidence") is None
