from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from edge.embedding.insight_embedding import QdrantEmbeddingStore
from edge.models.isolation_forest import IsolationForestDetector
from edge.tabular_loader import TabularPartition
from scripts.init_qdrant import initialise


def distribution(values: np.ndarray) -> dict[str, float | int]:
    percentiles = np.percentile(values, [50, 95, 99])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(percentiles[0]),
        "p95": float(percentiles[1]),
        "p99": float(percentiles[2]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def evaluate(detector: IsolationForestDetector, train_path: Path, test_path: Path) -> dict:
    train = TabularPartition(train_path, normal_only=True, feature_columns=detector.feature_columns, scaler=detector.scaler)
    test = TabularPartition(test_path, normal_only=False, feature_columns=detector.feature_columns, scaler=detector.scaler)
    validation_values = train.values[int(len(train) * 0.8):]
    validation_scores = detector.anomaly_scores(validation_values)
    test_scores = detector.anomaly_scores(test.values)
    threshold = float(np.quantile(validation_scores, 0.95))
    labels = test.labels
    predictions = (test_scores >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    return {
        "validation": {"windows": int(validation_scores.size), "threshold": threshold, "scores": distribution(validation_scores)},
        "test": {
            "scores": distribution(test_scores),
            "positive_rows": int(labels.sum()),
            "negative_rows": int((labels == 0).sum()),
            "roc_auc": float(roc_auc_score(labels, test_scores)) if np.unique(labels).size == 2 else None,
            "pr_auc": float(average_precision_score(labels, test_scores)) if np.unique(labels).size == 2 else None,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_positive_rate": float(fp / max(tn + fp, 1)),
            "true_positive": int(((labels == 1) & (predictions == 1)).sum()),
            "false_positive": fp,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Isolation Forest nodes 04-06")
    parser.add_argument("--partitions-root", type=Path, default=Path("data/partitions"))
    parser.add_argument("--results-root", type=Path, default=Path("results/modbus_iforest"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--qdrant-path", type=Path, default=Path("qdrant_data"))
    parser.add_argument("--nodes", nargs="+", default=["node_04", "node_05", "node_06"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--store-insights", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.results_root / run_id
    checkpoint_dir = args.checkpoint_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.store_insights:
        initialise(args.qdrant_path)
        store = QdrantEmbeddingStore(args.qdrant_path)
    records = []
    for node_id in args.nodes:
        node_root = args.partitions_root / node_id
        train_path = node_root / "train.parquet"
        test_path = node_root / "test.parquet"
        train_partition = TabularPartition(train_path, normal_only=True)
        detector = IsolationForestDetector.fit(train_partition, seed=args.seed, n_estimators=args.n_estimators)
        checkpoint = detector.save(
            checkpoint_dir / f"{node_id}.joblib",
            {"run_id": run_id, "node_id": node_id, "seed": args.seed, "n_estimators": args.n_estimators},
        )
        evaluation = evaluate(detector, train_path, test_path)
        record = {
            "run_id": run_id,
            "node_id": node_id,
            "source_train": str(train_path),
            "source_test": str(test_path),
            "feature_columns": detector.feature_columns,
            "feature_count": len(detector.feature_columns),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "config": {"seed": args.seed, "n_estimators": args.n_estimators},
            "evaluation": evaluation,
        }
        if args.store_insights:
            normal_mean = float(evaluation["validation"]["scores"]["mean"])
            test_mean = float(evaluation["test"]["scores"]["mean"])
            vector = np.zeros(128, dtype=np.float32)
            vector[0] = test_mean
            vector[1] = normal_mean
            store.store(vector, {"run_id": run_id, "node_id": node_id, "model_type": "isolation_forest"}, f"{run_id}-{node_id}")
            record["insight_embedding"] = {"collection": "insight_embeddings", "dimensions": 128}
        records.append(record)
        (run_dir / f"{node_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    summary = {"run_id": run_id, "nodes": records}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "summary": str(run_dir / "summary.json"), "checkpoints": str(checkpoint_dir)}, indent=2))


if __name__ == "__main__":
    main()
