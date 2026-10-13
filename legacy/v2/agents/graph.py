from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from agents.state import AgentState
from agents.tracing import get_tracer

tracer = get_tracer("agentic-mesh-week10")


def triage_node(state: AgentState) -> AgentState:
    alert = dict(state.get("alert", {}))
    score = float(alert.get("score", 0.0))
    severity = "high" if score >= float(alert.get("high_threshold", 0.8)) else "medium" if score > 0 else "low"
    with tracer.span("triage", session_id=state.get("session_id", ""), severity=severity):
        return {**state, "alert": {**alert, "severity": severity}, "status": "triaged"}


def investigator_node(state: AgentState) -> AgentState:
    evidence = list(state.get("investigation_evidence") or [])
    alert = state.get("alert", {})
    with tracer.span("investigator", session_id=state.get("session_id", "")):
        evidence.append({"source": alert.get("node_id", "unknown"), "score": alert.get("score", 0.0), "status": "collected"})
        return {**state, "investigation_evidence": evidence, "hop_count": int(state.get("hop_count") or 0) + 1, "status": "investigated"}


def security_node(state: AgentState) -> AgentState:
    evidence = state.get("investigation_evidence", [])
    with tracer.span("security", session_id=state.get("session_id", "")):
        match = {"technique_id": "UNKNOWN", "confidence": 0.0, "evidence_count": len(evidence)}
        return {**state, "mitre_match": match, "status": "classified"}


def dispatcher_node(state: AgentState) -> AgentState:
    with tracer.span("dispatcher", session_id=state.get("session_id", "")):
        remediation = {"action": "investigate", "severity": state.get("alert", {}).get("severity", "low")}
        return {**state, "remediation": remediation, "status": "complete"}


def route_after_triage(state: AgentState) -> str:
    return "investigator" if state.get("alert", {}).get("severity") in {"medium", "high"} else "dispatcher"


def build_graph(
    *,
    triage: Callable[[AgentState], AgentState] = triage_node,
    investigator: Callable[[AgentState], AgentState] = investigator_node,
    security: Callable[[AgentState], AgentState] = security_node,
    dispatcher: Callable[[AgentState], AgentState] = dispatcher_node,
):
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage)
    graph.add_node("investigator", investigator)
    graph.add_node("security", security)
    graph.add_node("dispatcher", dispatcher)
    graph.set_entry_point("triage")
    graph.add_conditional_edges("triage", route_after_triage, {"investigator": "investigator", "dispatcher": "dispatcher"})
    graph.add_edge("investigator", "security")
    graph.add_edge("security", "dispatcher")
    graph.add_edge("dispatcher", END)
    return graph.compile()


def build_real_graph(*, triage_agent: Any, investigator_agent: Any, security_agent: Any | None = None, dispatcher: Any | None = None):
    """Compile the production graph from injected Week 11/12 agent services."""

    async def triage(state: AgentState) -> AgentState:
        return await triage_agent.run(state)

    async def investigator(state: AgentState) -> AgentState:
        return await investigator_agent.run(state)

    async def security(state: AgentState) -> AgentState:
        if security_agent is None:
            return {**state, "mitre_match": state.get("mitre_match") or {"technique_id": "UNKNOWN", "confidence": 0.0}, "status": "classified"}
        return await security_agent.run(state)

    async def dispatch(state: AgentState) -> AgentState:
        if dispatcher is None:
            return {**state, "remediation": {"action": "investigate", "severity": state.get("alert", {}).get("severity", "low")}, "status": "complete"}
        return await dispatcher.run(state)

    graph = StateGraph(AgentState)
    graph.add_node("triage", triage)
    graph.add_node("investigator", investigator)
    graph.add_node("security", security)
    graph.add_node("dispatcher", dispatch)
    graph.set_entry_point("triage")
    graph.add_conditional_edges("triage", lambda state: "investigator" if state.get("alert", {}).get("escalate", True) else "dispatcher", {"investigator": "investigator", "dispatcher": "dispatcher"})
    graph.add_edge("investigator", "security")
    graph.add_edge("security", "dispatcher")
    graph.add_edge("dispatcher", END)
    return graph.compile()


app = build_graph()

__all__ = ["AgentState", "app", "build_graph", "build_real_graph", "route_after_triage"]
