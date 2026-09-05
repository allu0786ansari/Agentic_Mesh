from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from edge.config import Config
from edge.main import train_node


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all HAI VAE nodes in an isolated run")
    parser.add_argument("--partitions-root", type=Path, default=Path("data/partitions"))
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--nodes", nargs="+", default=["node_01", "node_02", "node_03"])
    args = parser.parse_args()

    base_config = Config.from_env()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_root = args.checkpoint_root or Path(base_config.checkpoint_root) / run_id
    metrics_path = Path(base_config.metrics_log_path).with_name(f"metrics_{run_id}.jsonl")
    manifest_path = Path(base_config.run_manifest_path).with_name(f"week4_training_{run_id}.json")
    results = []
    for node_id in args.nodes:
        config = replace(
            base_config,
            node_id=node_id,
            metrics_log_path=str(metrics_path),
            run_manifest_path=str(manifest_path),
        )
        result = train_node(
            config,
            partitions_root=args.partitions_root,
            checkpoint_root=checkpoint_root,
            max_windows=args.max_windows,
            run_id=run_id,
        )
        results.append(
            {
                "run_id": run_id,
                "node_id": node_id,
                "epochs": result.epochs,
                "samples": result.samples,
                "epsilon": result.epsilon,
                "checkpoint": str(result.checkpoint_path),
            }
        )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "nodes": results,
                "manifest": str(manifest_path),
                "metrics": str(metrics_path),
                "checkpoints": str(checkpoint_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
