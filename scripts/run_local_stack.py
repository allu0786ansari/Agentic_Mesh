from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any
from pathlib import Path

from agents.dispatcher.action_dispatcher import ActionDispatcher
from agents.graph import app, build_real_graph
from agents.graph_runner import GraphRunner
from agents.investigator.investigator_agent import InvestigatorAgent
from agents.llm import LocalPhiModel
from agents.security.security_agent import SecurityAgent
from agents.triage.triage_agent import TriageAgent
from agents.tracing import configure_phoenix
from edge.bus import AsyncBus


async def run_stack(args: argparse.Namespace, *, service_factory: Any | None = None) -> list[dict]:
    configure_phoenix(args.endpoint)
    bus = AsyncBus()
    await bus.drain("alerts.global")
    graph = app
    if getattr(args, "real_agents", False):
        if service_factory is None:
            llm = LocalPhiModel(model=args.model)
            security = SecurityAgent(llm=llm, qdrant_path=args.qdrant_path, database_path=args.database)
            triage = TriageAgent(llm=llm, database_path=args.database)
            investigator = InvestigatorAgent(llm=llm, node_urls=dict(args.node_urls), max_hops=3)
            dispatcher = ActionDispatcher(database_path=args.database, bus=bus)
        else:
            triage, investigator, security, dispatcher = service_factory(bus)
        graph = build_real_graph(
            triage_agent=triage,
            investigator_agent=investigator,
            security_agent=security,
            dispatcher=dispatcher,
        )
    runner = GraphRunner(graph, bus=bus, metrics_path=args.metrics_path)
    runner_task = asyncio.create_task(runner.run(max_sessions=args.count, idle_timeout=args.idle_timeout))
    for index in range(args.count):
        await bus.publish(
            "alerts.global",
            {
                "session_id": f"{args.prefix}-{index + 1}",
                "node_id": args.node_id,
                "score": args.score,
                "embedding": [0.0] * 128,
            },
        )
    return await runner_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local alerts.global to graph stack")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--node-id", default="node_01")
    parser.add_argument("--score", type=float, default=0.9)
    parser.add_argument("--prefix", default="week13-stack")
    parser.add_argument("--endpoint", default="http://127.0.0.1:6006/v1/traces")
    parser.add_argument("--idle-timeout", type=float, default=2.0)
    parser.add_argument("--metrics-path", type=Path, default=Path("logs/metrics.jsonl"))
    parser.add_argument("--real-agents", action="store_true")
    parser.add_argument("--model", default="agmesh-phi3.5")
    parser.add_argument("--qdrant-path", type=Path, default=Path("qdrant_data_week12"))
    parser.add_argument("--database", type=Path, default=Path("agmesh.db"))
    parser.add_argument("--node-urls", nargs="*", default=["node_01=http://127.0.0.1:8443", "node_02=http://127.0.0.1:8444", "node_03=http://127.0.0.1:8445"])
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    parsed_urls: dict[str, str] = {}
    for value in args.node_urls:
        if "=" not in value:
            raise ValueError("--node-urls values must use node_id=url format")
        node_id, url = value.split("=", 1)
        parsed_urls[node_id] = url
    args.node_urls = parsed_urls
    results = asyncio.run(run_stack(args))
    if len(results) != args.count or any(result.get("status") != "complete" for result in results):
        raise RuntimeError("local stack did not complete all graph sessions")
    print(json.dumps({"sessions": len(results), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
