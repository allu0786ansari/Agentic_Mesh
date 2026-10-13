from __future__ import annotations

import argparse
import json

from agents.tracing import configure_phoenix


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agentic Mesh graph with mandatory Phoenix tracing")
    parser.add_argument("--endpoint", default="http://127.0.0.1:6006/v1/traces")
    parser.add_argument("--session-id", default="week10-phoenix")
    parser.add_argument("--node-id", default="node_01")
    parser.add_argument("--score", type=float, default=0.9)
    args = parser.parse_args()
    configure_phoenix(args.endpoint)
    from agents.graph import app

    result = app.invoke({
        "session_id": args.session_id,
        "alert": {"node_id": args.node_id, "score": args.score},
    })
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
