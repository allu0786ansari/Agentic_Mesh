from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from agents.llm import LocalPhiModel
from agents.state import AgentState
from agents.tracing import get_tracer
from agents.investigator.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError


class InvestigatorAgent:
    """Use Phi to select nodes and query bounded corroborating evidence."""

    def __init__(
        self,
        *,
        llm: LocalPhiModel,
        node_urls: dict[str, str],
        breaker_registry: CircuitBreakerRegistry | None = None,
        hop_timeout_seconds: float = 0.5,
        max_hops: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not node_urls or not 0 < hop_timeout_seconds <= 30 or not 1 <= max_hops <= 3:
            raise ValueError("node_urls, hop timeout, and max_hops must be valid")
        self.llm = llm
        self.node_urls = dict(node_urls)
        self.breakers = breaker_registry or CircuitBreakerRegistry()
        self.hop_timeout_seconds = hop_timeout_seconds
        self.max_hops = max_hops
        self.transport = transport
        self.tracer = get_tracer("agentic-mesh-investigator")

    async def run(self, state: AgentState) -> AgentState:
        alert = dict(state.get("alert") or {})
        with self.tracer.span("investigator_agent", session_id=state.get("session_id", "")):
            selected_nodes = self._select_nodes(alert)
            evidence = list(state.get("investigation_evidence") or [])
            remaining = self.max_hops - int(state.get("hop_count") or 0)
            for node_id in selected_nodes[: max(0, remaining)]:
                evidence.append(await self._query_node(node_id, alert))
            return {
                **state,
                "investigation_evidence": evidence,
                "hop_count": int(state.get("hop_count") or 0) + min(len(selected_nodes), max(0, remaining)),
                "status": "investigated",
            }

    def _select_nodes(self, alert: dict[str, Any]) -> list[str]:
        requested = alert.get("target_nodes")
        if isinstance(requested, list):
            candidates = [str(node) for node in requested if str(node) in self.node_urls]
        else:
            candidates = []
        if not candidates:
            prompt = (
                "Select investigation targets. Return JSON only as {\"nodes\":[...]} "
                f"using only these node IDs: {json.dumps(list(self.node_urls))}. Alert: {json.dumps(alert, default=str)}"
            )
            decision = self.llm.invoke_json(prompt)
            candidates = [str(node) for node in decision.get("nodes", []) if str(node) in self.node_urls]
        source = str(alert.get("node_id", ""))
        ordered = [source] if source in self.node_urls else []
        ordered.extend(node for node in candidates if node not in ordered)
        ordered.extend(node for node in self.node_urls if node not in ordered)
        return ordered[: self.max_hops]

    async def _query_node(self, node_id: str, alert: dict[str, Any]) -> dict[str, Any]:
        async def request() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=self.hop_timeout_seconds, transport=self.transport) as client:
                response = await client.post(f"{self.node_urls[node_id].rstrip('/')}/embeddings/search", json={"embedding": alert.get("embedding", [0.0] * 128), "limit": 5})
                response.raise_for_status()
                return {"node_id": node_id, "status": "success", "response": response.json()}

        try:
            return await self.breakers.for_node(node_id).call(request)
        except CircuitOpenError as exc:
            return {"node_id": node_id, "status": "circuit_open", "error": str(exc)}
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            return {"node_id": node_id, "status": "unavailable", "error": str(exc)}


__all__ = ["InvestigatorAgent"]
