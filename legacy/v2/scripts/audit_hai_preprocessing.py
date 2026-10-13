from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_node(node_root: Path) -> dict:
    meta_path = node_root / "meta.json"
    train_path = node_root / "train.parquet"
    test_path = node_root / "test.parquet"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    expected = metadata["feature_columns"]
    actual_train = [column for column in train.columns if column not in {"time", "label"}]
    actual_test = [column for column in test.columns if column not in {"time", "label"}]
    numeric = train[expected].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    return {
        "node_id": metadata["node_id"],
        "feature_order_matches_train": actual_train == expected,
        "feature_order_matches_test": actual_test == expected,
        "feature_count": len(expected),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_nan_count": int(np.isnan(numeric).sum()),
        "train_inf_count": int(np.isinf(numeric).sum()),
        "train_min": float(np.min(numeric)),
        "train_max": float(np.max(numeric)),
        "normal_only_scaler_declared": bool(metadata.get("normal_only_scaler", False)),
        "scaler_metadata_path": metadata.get("scaler_metadata_path"),
        "train_sha256": sha256(train_path),
        "test_sha256": sha256(test_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HAI preprocessing and partition identity")
    parser.add_argument("--partitions-root", type=Path, default=Path("data/partitions"))
    parser.add_argument("--output", type=Path, default=Path("results/preprocessing_audit.json"))
    parser.add_argument("--nodes", nargs="+", default=["node_01", "node_02", "node_03"])
    args = parser.parse_args()
    result = {"nodes": [audit_node(args.partitions_root / node) for node in args.nodes]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
