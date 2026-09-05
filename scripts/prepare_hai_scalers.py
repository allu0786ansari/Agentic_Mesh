from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge.preprocessing import MinMaxScalerArtifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit reusable HAI scalers on normal training rows")
    parser.add_argument("--partitions-root", type=Path, default=Path("data/partitions"))
    parser.add_argument("--scalers-root", type=Path, default=Path("results/scalers"))
    parser.add_argument("--nodes", nargs="+", default=["node_01", "node_02", "node_03"])
    args = parser.parse_args()
    outputs = []
    for node_id in args.nodes:
        source = args.partitions_root / node_id / "train.parquet"
        destination = args.scalers_root / f"{node_id}.json"
        scaler = MinMaxScalerArtifact.fit(source, normal_only=True)
        scaler.save(destination)
        outputs.append({"node_id": node_id, "source": str(source), "scaler": str(destination)})
    print(json.dumps({"scalers": outputs}, indent=2))


if __name__ == "__main__":
    main()
