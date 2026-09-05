from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agents.llm import LocalPhiModel
from agents.state import AgentState
from agents.tracing import get_tracer
from edge.embedding.insight_embedding import QdrantEmbeddingStore
from storage.sqlite_schema import connect, initialise


class TriageAgent:
    """Retrieve similar alerts and use Phi to produce a structured triage decision."""

    def __init__(
        self,
        *,
        llm: LocalPhiModel,
        memory_store: QdrantEmbeddingStore | None = None,
        database_path: str | Path | None = None,
        memory_limit: int = 5,
    ) -> None:
        if memory_limit < 1:
            raise ValueError("memory_limit must be positive")
        self.llm = llm
        self.memory_store = memory_store
        self.database_path = Path(database_path) if database_path else None
        self.memory_limit = memory_limit
        self.tracer = get_tracer("agentic-mesh-triage")

    async def run(self, state: AgentState) -> AgentState:
        alert = dict(state.get("alert") or {})
        with self.tracer.span("triage_agent", session_id=state.get("session_id", "")):
            similar = self._retrieve_memory(alert)
            prompt = self._prompt(alert, similar)
            decision = self.llm.invoke_json(prompt)
            triage = self._validate_decision(decision, alert)
            alert.update(triage)
            if self.database_path:
                self._record_alert(state, alert)
            return {**state, "alert": alert, "status": "triaged", "investigation_evidence": similar}

    def _retrieve_memory(self, alert: dict[str, Any]) -> list[dict[str, Any]]:
        embedding = alert.get("embedding")
        if self.memory_store is None or embedding is None:
            return []
        matches = self.memory_store.search(np.asarray(embedding, dtype=np.float32), self.memory_limit)
        return [{"score": float(item.score), "payload": item.payload} for item in matches]

    @staticmethod
    def _prompt(alert: dict[str, Any], similar: list[dict[str, Any]]) -> str:
        return (
            "Assess this industrial anomaly. Return JSON with exactly these fields: "
            "escalate (boolean), severity_level (low|medium|high), reasoning (string).\n"
            f"Alert: {json.dumps({key: value for key, value in alert.items() if key != 'embedding'}, default=str)}\n"
            f"Similar historical alerts: {json.dumps(similar, default=str)}"
        )

    @staticmethod
    def _validate_decision(decision: dict[str, Any], alert: dict[str, Any]) -> dict[str, Any]:
        severity = str(decision.get("severity_level", "medium")).lower()
        if severity not in {"low", "medium", "high"}:
            severity = "medium"
        return {
            "escalate": bool(decision.get("escalate", severity != "low")),
            "severity": severity,
            "severity_level": severity,
            "reasoning": str(decision.get("reasoning", "No reasoning returned")),
        }

    def _record_alert(self, state: AgentState, alert: dict[str, Any]) -> None:
        initialise(self.database_path)
        with connect(self.database_path) as database:
            node_id = str(alert.get("node_id", "unknown"))
            exists = database.execute("SELECT 1 FROM node_registry WHERE node_id = ?", (node_id,)).fetchone()
            if exists:
                database.execute(
                    "INSERT OR REPLACE INTO alert_records(alert_id,node_id,severity,score,status) VALUES(?,?,?,?,?)",
                    (state.get("session_id", "unknown"), node_id, alert.get("severity"), float(alert.get("score", 0.0)), "triaged"),
                )


__all__ = ["TriageAgent"]
