from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone

from agents.tracing import configure_phoenix
from edge.bus import AsyncBus


async def publish(args: argparse.Namespace) -> None:
    configure_phoenix(args.endpoint)
    bus = AsyncBus()
    for index in range(args.count):
        message = {
            "session_id": f"{args.prefix}-{index + 1}",
            "node_id": args.node_id,
            "score": args.score,
            "embedding": [0.0] * 128,
            "target_nodes": args.target_nodes,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await bus.publish("alerts.global", message)
        print(json.dumps(message))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject labeled alerts into alerts.global")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--node-id", default="node_01")
    parser.add_argument("--score", type=float, default=0.9)
    parser.add_argument("--target-nodes", nargs="+", default=["node_01", "node_02", "node_03"])
    parser.add_argument("--prefix", default="week13-alert")
    parser.add_argument("--endpoint", default="http://127.0.0.1:6006/v1/traces")
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    asyncio.run(publish(args))


if __name__ == "__main__":
    main()
