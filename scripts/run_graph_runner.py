from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agents.graph import app
from agents.graph_runner import GraphRunner
from agents.tracing import configure_phoenix
from edge.bus import AsyncBus


async def main_async(args: argparse.Namespace) -> None:
    configure_phoenix(args.endpoint)
    runner = GraphRunner(app, bus=AsyncBus(), metrics_path=args.metrics_path)
    results = await runner.run(max_sessions=args.max_sessions, idle_timeout=args.idle_timeout)
    print(json.dumps({"processed_sessions": len(results), "results": results}, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the alerts.global Agentic Mesh graph subscriber")
    parser.add_argument("--endpoint", default="http://127.0.0.1:6006/v1/traces")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--idle-timeout", type=float, default=None)
    parser.add_argument("--metrics-path", type=Path, default=Path("logs/metrics.jsonl"))
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
