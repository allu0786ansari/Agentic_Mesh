from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from edge.bus import EmbeddingPublisher
from edge.data_loader import HAIWindowDataset
from edge.embedding.insight_embedding import InsightEmbeddingGenerator, QdrantEmbeddingStore
from edge.models.isolation_forest import IsolationForestDetector
from edge.tabular_loader import TabularPartition
from evaluation.inference import load_checkpoint_model, reconstruction_errors


def vae_records(node_id: str, checkpoint: Path, partition_root: Path, scaler: Path, run_id: str) -> list[dict]:
    model, _ = load_checkpoint_model(checkpoint)
    validation = HAIWindowDataset(partition_root / "train.parquet", normal_only=True, scaler_path=scaler, start_fraction=0.8)
    test = HAIWindowDataset(partition_root / "test.parquet", normal_only=False, scaler_path=scaler)
    threshold = float(np.quantile(reconstruction_errors(model, validation), 0.95))
    all_errors = reconstruction_errors(model, test)
    labels = test.window_labels
    loader = DataLoader(test, batch_size=256, shuffle=False)
    records = []
    offset = 0
    for inputs in loader:
        errors = all_errors[offset : offset + len(inputs)]
        embeddings = InsightEmbeddingGenerator.from_vae(model, inputs)
        for index, error in enumerate(errors):
            if error >= threshold:
                records.append({"run_id": run_id, "node_id": node_id, "model_type": "vae", "score": float(error), "threshold": threshold, "embedding": embeddings[index], "index": offset + index, "label": int(labels[offset + index]), "feature_count": len(test.feature_columns), "explanation": {"score_type": "reconstruction_mse", "window_start": int(test.window_starts[offset + index])}})
        offset += len(inputs)
    return records


def iforest_records(node_id: str, checkpoint: Path, partition_root: Path, run_id: str) -> list[dict]:
    detector, _ = IsolationForestDetector.load(checkpoint)
    train = TabularPartition(partition_root / "train.parquet", normal_only=True, feature_columns=detector.feature_columns, scaler=detector.scaler)
    test = TabularPartition(partition_root / "test.parquet", normal_only=False, feature_columns=detector.feature_columns, scaler=detector.scaler)
    validation_scores = detector.anomaly_scores(train.values[int(len(train) * 0.8):])
    threshold = float(np.quantile(validation_scores, 0.95))
    scores = detector.anomaly_scores(test.values)
    labels = test.labels
    transformed = detector.scaler.transform(test.values)
    embeddings = InsightEmbeddingGenerator.from_isolation_forest(scores, transformed)
    return [
        {"run_id": run_id, "node_id": node_id, "model_type": "isolation_forest", "score": float(score), "threshold": threshold, "embedding": embeddings[index], "index": index, "label": int(labels[index]), "feature_count": len(test.feature_columns), "explanation": {"score_type": "isolation_forest_decision_function", "feature_count": len(test.feature_columns)}}
        for index, score in enumerate(scores)
        if score >= threshold
    ]


async def main_async(args: argparse.Namespace) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    collection = f"insight_embeddings_{run_id}"
    store = QdrantEmbeddingStore(args.qdrant_path, collection=collection)
    publisher = EmbeddingPublisher()
    records: list[dict] = []
    audit_path = args.results_root / run_id / "publish_audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=False)
    for node_id in args.vae_nodes:
        records.extend(vae_records(node_id, args.vae_checkpoint_root / f"{node_id}.pt", args.partitions_root / node_id, args.scalers_root / f"{node_id}.json", run_id))
    for node_id in args.iforest_nodes:
        records.extend(iforest_records(node_id, args.iforest_checkpoint_root / f"{node_id}.joblib", args.partitions_root / node_id, run_id))
    for index, record in enumerate(records):
        embedding = record.pop("embedding")
        point_id = f"{run_id}-{record['node_id']}-{record['index']}"
        checkpoint_root = args.vae_checkpoint_root if record["model_type"] == "vae" else args.iforest_checkpoint_root
        suffix = ".pt" if record["model_type"] == "vae" else ".joblib"
        checkpoint_path = checkpoint_root / f"{record['node_id']}{suffix}"
        scaler_path = args.scalers_root / f"{record['node_id']}.json" if record["model_type"] == "vae" else None
        record["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        record["scaler_sha256"] = hashlib.sha256(scaler_path.read_bytes()).hexdigest() if scaler_path else None
        normalized_id = store.store(embedding, record, point_id)
        message = {**record, "point_id": normalized_id}
        await publisher.publish(record["node_id"], message)
        with audit_path.open("a", encoding="utf-8") as audit:
            audit.write(json.dumps({"event": "embedding_published", "payload": message}) + "\n")
    by_node = {}
    for record in records:
        by_node[record["node_id"]] = by_node.get(record["node_id"], 0) + 1
    output = {
        "run_id": run_id,
        "records": len(records),
        "records_by_node": by_node,
        "qdrant_path": str(args.qdrant_path),
        "qdrant_collection": collection,
        "topics": sorted(f"embeddings.{node_id}" for node_id in by_node),
        "audit_log": str(audit_path),
    }
    output_path = args.results_root / run_id / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 6 inference, embedding, Qdrant, and bus publishing")
    parser.add_argument("--partitions-root", type=Path, default=Path("data/partitions"))
    parser.add_argument("--vae-checkpoint-root", type=Path, default=Path("checkpoints/20260821T024039Z"))
    parser.add_argument("--iforest-checkpoint-root", type=Path, default=Path("checkpoints/20260821T214400Z"))
    parser.add_argument("--scalers-root", type=Path, default=Path("results/scalers"))
    parser.add_argument("--qdrant-path", type=Path, default=Path("qdrant_data"))
    parser.add_argument("--results-root", type=Path, default=Path("results/week6_inference"))
    parser.add_argument("--vae-nodes", nargs="+", default=["node_01", "node_02", "node_03"])
    parser.add_argument("--iforest-nodes", nargs="+", default=["node_04", "node_05", "node_06"])
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
