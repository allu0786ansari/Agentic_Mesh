from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Week 6 embedding separation and retrieval quality")
    parser.add_argument("--qdrant-path", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/week6_embedding_quality.json"))
    args = parser.parse_args()
    client = QdrantClient(path=str(args.qdrant_path))
    points, _ = client.scroll(args.collection, limit=100000, with_payload=True, with_vectors=True)
    if not points:
        raise ValueError("Embedding collection is empty")
    vectors = np.asarray([point.vector for point in points], dtype=np.float32)
    labels = np.asarray([int(point.payload.get("label", 0)) for point in points], dtype=np.int8)
    result = {
        "collection": args.collection,
        "points": int(len(points)),
        "dimensions": sorted(set(vectors.shape[1:])),
        "finite": bool(np.isfinite(vectors).all()),
        "mean_norm": float(np.linalg.norm(vectors, axis=1).mean()),
        "positive_labels": int(labels.sum()),
        "negative_labels": int((labels == 0).sum()),
    }
    if np.unique(labels).size == 2 and len(points) >= 10:
        result["label_auc_from_embedding_axis"] = float(roc_auc_score(labels, vectors[:, 0]))
        classifier = KNeighborsClassifier(n_neighbors=min(5, len(points) - 1), metric="cosine")
        indices = np.arange(len(points))
        split = max(1, int(len(points) * 0.8))
        classifier.fit(vectors[indices[:split]], labels[indices[:split]])
        result["knn_label_accuracy"] = float(accuracy_score(labels[indices[split:]], classifier.predict(vectors[indices[split:]])))
    else:
        result["label_auc_from_embedding_axis"] = None
        result["knn_label_accuracy"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
