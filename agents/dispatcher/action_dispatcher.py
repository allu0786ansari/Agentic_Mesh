from __future__ import annotations

import time
from pathlib import Path

from agents.state import AgentState
from agents.tracing import get_tracer
from edge.bus import AsyncBus
from storage.sqlite_schema import connect, initialise


class ActionDispatcher:
    """Publish structured remediation actions and persist dispatch status."""

    ACTIONS = {"credential-access": "rotate_credentials", "impact": "isolate_node", "execution": "block_process", "unknown": "investigate"}

    def __init__(self, *, database_path: str | Path | None = None, bus: AsyncBus | None = None):
        self.database_path = Path(database_path) if database_path else None
        self.bus = bus or AsyncBus()
        self.tracer = get_tracer("agentic-mesh-dispatcher")

    async def run(self, state: AgentState) -> AgentState:
        started = time.perf_counter()
        alert = state.get("alert", {})
        match = state.get("mitre_match", {})
        tactic = str(match.get("tactic", "unknown")).lower()
        action = self.ACTIONS.get(tactic, "investigate")
        payload = {"session_id": state.get("session_id", ""), "node_id": alert.get("node_id"), "action": action, "severity": alert.get("severity", "medium"), "technique_id": match.get("technique_id", "UNKNOWN"), "confidence": match.get("confidence", 0.0), "reasoning": match.get("reasoning", "")}
        with self.tracer.span("action_dispatcher", session_id=state.get("session_id", "")):
            await self.bus.publish(f"actions.{alert.get('node_id', 'global')}", payload)
            if self.database_path:
                self._mark_dispatched(state)
        payload["latency_ms"] = (time.perf_counter() - started) * 1000.0
        return {**state, "remediation": payload, "status": "complete"}

    def _mark_dispatched(self, state: AgentState) -> None:
        initialise(self.database_path)
        with connect(self.database_path) as database:
            database.execute("UPDATE alert_records SET status = ? WHERE alert_id = ?", ("dispatched", state.get("session_id", "unknown")))


__all__ = ["ActionDispatcher"]
