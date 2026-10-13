from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any

from agents.metrics import log_event
from agents.state import AgentState
from agents.tracing import get_tracer
from edge.bus import AsyncBus


class GraphRunner:
    """Consume alerts.global and execute one graph session per alert."""

    def __init__(
        self,
        graph: Any,
        *,
        bus: AsyncBus | None = None,
        topic: str = "alerts.global",
        metrics_path: str = "logs/metrics.jsonl",
        executor: Any | None = None,
    ) -> None:
        if not topic.strip():
            raise ValueError("topic must not be empty")
        self.graph = graph
        self.bus = bus or AsyncBus()
        self.topic = topic
        self.metrics_path = metrics_path
        self.executor = executor
        self.tracer = get_tracer("agentic-mesh-graph-runner")

    async def process_one(self, alert: dict[str, Any]) -> AgentState:
        if not isinstance(alert, dict):
            raise TypeError("alert message must be a dictionary")
        session_id = str(alert.get("session_id") or f"alert-{time.time_ns()}")
        state: AgentState = {"session_id": session_id, "alert": alert}
        started = time.perf_counter()
        with self.tracer.span("graph_session", session_id=session_id):
            result = await self._invoke(state)
        duration_ms = (time.perf_counter() - started) * 1000.0
        log_event(
            "agent_graph_completed",
            {
                "session_id": session_id,
                "status": result.get("status"),
                "hop_count": int(result.get("hop_count") or 0),
                "duration_ms": duration_ms,
                "action": result.get("remediation", {}).get("action") if isinstance(result.get("remediation"), dict) else None,
            },
            self.metrics_path,
        )
        return result

    async def _invoke(self, state: AgentState) -> AgentState:
        if hasattr(self.graph, "ainvoke"):
            result = self.graph.ainvoke(state)
            return await result if inspect.isawaitable(result) else result
        if not hasattr(self.graph, "invoke"):
            raise TypeError("graph must expose invoke or ainvoke")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, self.graph.invoke, state)
        return result

    async def run(self, *, max_sessions: int | None = None, idle_timeout: float | None = None) -> list[AgentState]:
        if max_sessions is not None and max_sessions < 1:
            raise ValueError("max_sessions must be positive when provided")
        if idle_timeout is not None and idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive when provided")
        results: list[AgentState] = []
        while max_sessions is None or len(results) < max_sessions:
            try:
                message = await self.bus.get(self.topic, timeout=idle_timeout)
            except TimeoutError:
                break
            if isinstance(message, str):
                message = json.loads(message)
            results.append(await self.process_one(message))
        return results


__all__ = ["GraphRunner"]
