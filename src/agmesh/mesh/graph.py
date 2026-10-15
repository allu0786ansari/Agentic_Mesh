"""The mesh as a LangGraph state machine (Proposal §8.1; ADR 0012 §1).

The graph runs ``triage → investigate → attribute → dispatch``.

**Persistence.** Every node:

* writes its step to ``session_steps`` before it returns;
* moves the session's status through the allowed transitions.

**Resuming.** A node whose step is already recorded reuses the stored result, so a
replayed or resumed session never repeats completed work.

**Terminal states.**

* No evidence, or an attribution that abstains, ends the session ``ABSTAINED``.
* Otherwise it ends ``COMPLETED``, or ``COMPLETED_DEGRADED`` when the investigation
  hit its deadline or an open breaker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agmesh.bus.base import Bus
from agmesh.common.schemas import EvidenceRecord
from agmesh.mesh import dispatcher
from agmesh.mesh.attribution import Attribution, AttributionAgent
from agmesh.mesh.investigator import Investigator
from agmesh.mesh.sessions import TERMINAL, SessionStore


class MeshState(TypedDict, total=False):
    session_id: str
    site: str
    run_id: str
    mode: str
    incident: dict[str, Any]
    evidence: list[dict[str, Any]]
    hop_count: int
    degraded: bool
    attribution: dict[str, Any]
    actions: list[dict[str, Any]]
    status: str
    created: bool
    t_open: float
    latency_ms: dict[str, float]


@dataclass
class Mesh:
    store: SessionStore
    bus: Bus
    investigator: Investigator
    agent: AttributionAgent

    # ----------------------------------------------------------------- helpers
    def _advance(self, session_id: str, to: str) -> str:
        current = self.store.get(session_id).status
        if current == to or current in TERMINAL:
            return current
        return self.store.transition(session_id, to).status

    def _finish(self, state: MeshState, to: str) -> MeshState:
        sid = state["session_id"]
        if self.store.get(sid).status not in TERMINAL:
            self.store.transition(sid, to)
        return {**state, "status": self.store.get(sid).status}

    # ------------------------------------------------------------------- nodes
    def triage(self, state: MeshState) -> MeshState:
        inc = state["incident"]
        session, created = self.store.open(
            state["site"], inc["alert_ids"], state["run_id"], mode=state["mode"]
        )
        self.store.record_step(session.session_id, "triage", "ok", inc)
        return {
            **state,
            "session_id": session.session_id,
            "created": created,
            "status": self.store.get(session.session_id).status,
            "t_open": time.perf_counter(),
            "latency_ms": {},
        }

    def investigate(self, state: MeshState) -> MeshState:
        sid = state["session_id"]
        prior = self.store.step(sid, "investigate")
        if prior is None:
            self._advance(sid, "INVESTIGATING")
            inc = state["incident"]
            res = self.investigator.run(
                sid, state["site"], inc["nodes"], inc["start_s"], inc["end_s"]
            )
            prior = {
                "records": [r.model_dump() for r in res.records],
                "hops": [list(h) for h in res.hops],
                "degraded": res.degraded,
                "reason": res.reason,
                "ms": 1000.0 * (time.perf_counter() - state["t_open"]),
            }
            self.store.record_step(sid, "investigate", "degraded" if res.degraded else "ok", prior)
        latency = {**state.get("latency_ms", {}), "investigation": float(prior["ms"])}
        return {
            **state,
            "evidence": prior["records"],
            "degraded": bool(prior["degraded"]),
            "hop_count": self.store.get(sid).hop_count,
            "latency_ms": latency,
        }

    def attribute(self, state: MeshState) -> MeshState:
        sid = state["session_id"]
        prior = self.store.step(sid, "attribute")
        if prior is None:
            if not state["evidence"]:
                self._advance(sid, "INVESTIGATING")
                self.store.record_step(sid, "attribute", "abstained", {"reason": "no evidence"})
                return {**self._finish(state, "ABSTAINED"), "attribution": {}}
            self._advance(sid, "ATTRIBUTING")
            records = [EvidenceRecord.model_validate(r) for r in state["evidence"]]
            prior = self.agent.attribute(state["site"], records, state["mode"]).to_dict()
            self.store.record_step(sid, "attribute", prior["attribution_mode"], prior)
        if prior.get("technique_id") is None:
            return {**self._finish(state, "ABSTAINED"), "attribution": prior}
        return {**state, "attribution": prior}

    def dispatch(self, state: MeshState) -> MeshState:
        sid = state["session_id"]
        prior = self.store.step(sid, "dispatch")
        if prior is None:
            self._advance(sid, "DISPATCHING")
            att = Attribution(**state["attribution"])
            done = dispatcher.dispatch(
                self.store, self.bus, sid, state["site"], att, state["incident"]["severity"]
            )
            prior = {"actions": done, "ms": 1000.0 * (time.perf_counter() - state["t_open"])}
            self.store.record_step(sid, "dispatch", "ok", prior)
        out = self._finish(state, "COMPLETED_DEGRADED" if state.get("degraded") else "COMPLETED")
        latency = {**state.get("latency_ms", {}), "incident_to_dispatch": float(prior["ms"])}
        return {**out, "actions": prior["actions"], "latency_ms": latency}

    # ------------------------------------------------------------------ routes
    def _after_triage(self, state: MeshState) -> str:
        return END if state["status"] in TERMINAL else "investigate"

    def _after_attribute(self, state: MeshState) -> str:
        return END if state.get("status") in TERMINAL else "dispatch"

    def compile(self) -> Any:
        g = StateGraph(MeshState)
        g.add_node("triage", self.triage)
        g.add_node("investigate", self.investigate)
        g.add_node("attribute", self.attribute)
        g.add_node("dispatch", self.dispatch)
        g.set_entry_point("triage")
        g.add_conditional_edges(
            "triage", self._after_triage, {"investigate": "investigate", END: END}
        )
        g.add_edge("investigate", "attribute")
        g.add_conditional_edges(
            "attribute", self._after_attribute, {"dispatch": "dispatch", END: END}
        )
        g.add_edge("dispatch", END)
        return g.compile()

    def run(self, site: str, incident: dict[str, Any], mode: str, run_id: str) -> MeshState:
        app = self.compile()
        state: MeshState = {"site": site, "incident": incident, "mode": mode, "run_id": run_id}
        result: MeshState = app.invoke(state)
        return result
