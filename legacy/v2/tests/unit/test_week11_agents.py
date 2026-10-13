from __future__ import annotations

import asyncio

import httpx

from agents.graph import build_real_graph
from agents.investigator.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from agents.investigator.investigator_agent import InvestigatorAgent
from agents.tracing import configure_phoenix
from agents.triage.triage_agent import TriageAgent

configure_phoenix("http://127.0.0.1:6006/v1/traces")


class FakeLLM:
    def __init__(self, response: dict):
        self.response = response

    def invoke_json(self, prompt: str) -> dict:
        return dict(self.response)


def test_circuit_breaker_opens_after_failures() -> None:
    async def scenario() -> None:
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10)

        async def fail() -> None:
            raise RuntimeError("down")

        for _ in range(2):
            try:
                await breaker.call(fail)
            except RuntimeError:
                pass
        assert breaker.state is CircuitState.OPEN
        try:
            await breaker.call(fail)
        except CircuitOpenError:
            pass
        else:
            raise AssertionError("open breaker allowed an operation")

    asyncio.run(scenario())


def test_real_graph_uses_triage_and_bounded_investigation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"matches": [{"node_id": "node_02", "score": 0.8}]})

    transport = httpx.MockTransport(handler)
    triage = TriageAgent(llm=FakeLLM({"escalate": True, "severity_level": "high", "reasoning": "high score"}))
    investigator = InvestigatorAgent(llm=FakeLLM({"nodes": ["node_02", "node_03"]}), node_urls={"node_01": "http://node-1", "node_02": "http://node-2", "node_03": "http://node-3"}, transport=transport, max_hops=3)
    graph = build_real_graph(triage_agent=triage, investigator_agent=investigator)

    result = asyncio.run(graph.ainvoke({"session_id": "s11", "alert": {"node_id": "node_01", "score": 0.9, "embedding": [0.0] * 128}}))

    assert result["status"] == "complete"
    assert result["alert"]["escalate"] is True
    assert result["hop_count"] == 3
    assert len(result["investigation_evidence"]) == 3
