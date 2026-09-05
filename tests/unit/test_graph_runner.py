from __future__ import annotations

import asyncio
from pathlib import Path

from agents.graph_runner import GraphRunner
from agents.tracing import configure_phoenix
from edge.bus import AsyncBus

configure_phoenix("http://127.0.0.1:6006/v1/traces")


class SyncGraph:
    def invoke(self, state):
        return {**state, "status": "complete", "hop_count": 2, "remediation": {"action": "investigate"}}


class AsyncGraph:
    async def ainvoke(self, state):
        return {**state, "status": "complete", "hop_count": 3, "remediation": {"action": "isolate_node"}}


def test_runner_processes_sync_graph_without_blocking(tmp_path: Path) -> None:
    async def scenario():
        bus = AsyncBus()
        await bus.drain("alerts.global")
        await bus.publish("alerts.global", {"session_id": "sync-1", "node_id": "node_01", "score": 0.9})
        runner = GraphRunner(SyncGraph(), bus=bus, metrics_path=tmp_path / "metrics.jsonl")
        results = await runner.run(max_sessions=1)
        return results

    results = asyncio.run(scenario())
    assert results[0]["status"] == "complete"
    assert results[0]["hop_count"] == 2
    assert "agent_graph_completed" in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8")


def test_runner_processes_async_graph_and_idle_timeout() -> None:
    async def scenario():
        bus = AsyncBus()
        await bus.drain("alerts.global")
        await bus.publish("alerts.global", '{"session_id":"async-1","node_id":"node_01","score":0.9}')
        return await GraphRunner(AsyncGraph(), bus=bus).run(idle_timeout=0.01)

    results = asyncio.run(scenario())
    assert len(results) == 1
    assert results[0]["remediation"]["action"] == "isolate_node"
