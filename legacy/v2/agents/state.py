from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state contract passed through the Week 10 agent graph."""

    session_id: str
    alert: dict[str, Any]
    investigation_evidence: list[dict[str, Any]]
    hop_count: int
    mitre_match: dict[str, Any]
    remediation: dict[str, Any]
    status: str
    trace_id: str
    error: str


__all__ = ["AgentState"]
