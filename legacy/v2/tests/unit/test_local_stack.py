from __future__ import annotations

import asyncio
from pathlib import Path

from scripts.run_local_stack import run_stack


def test_local_stack_processes_five_alerts(tmp_path: Path) -> None:
    import argparse

    args = argparse.Namespace(
        count=5,
        node_id="node_01",
        score=0.9,
        prefix="test-stack",
        endpoint="http://127.0.0.1:6006/v1/traces",
        idle_timeout=1.0,
        metrics_path=tmp_path / "metrics.jsonl",
    )
    results = asyncio.run(run_stack(args))
    assert len(results) == 5
    assert all(result["status"] == "complete" for result in results)


def test_real_agent_stack_uses_injected_services(tmp_path: Path) -> None:
    import argparse

    class Agent:
        async def run(self, state):
            return {**state, "status": "complete", "remediation": {"action": "test"}}

    args = argparse.Namespace(
        count=1,
        node_id="node_01",
        score=0.9,
        prefix="real-stack",
        endpoint="http://127.0.0.1:6006/v1/traces",
        idle_timeout=1.0,
        metrics_path=tmp_path / "metrics.jsonl",
        real_agents=True,
        model="unused",
        qdrant_path=tmp_path / "qdrant",
        database=tmp_path / "mesh.db",
        node_urls={"node_01": "http://127.0.0.1:8443"},
    )
    results = asyncio.run(run_stack(args, service_factory=lambda bus: (Agent(), Agent(), Agent(), Agent())))
    assert results[0]["status"] == "complete"
