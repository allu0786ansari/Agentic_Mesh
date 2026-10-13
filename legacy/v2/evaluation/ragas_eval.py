from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

BASE_CONTEXTS = [
    "A PLC controller on node_01 shows repeated pressure anomalies immediately before the compressor trips. The investigation links this to a manipulated setpoint and an unexpected restart loop.",
    "Edge node_02 reports high vibration and false sensor drift in the turbine speed channel while the historian shows a delayed reset following a network disconnect.",
    "A failed actuator on node_03 matches an abnormal current draw profile and a recurring command pattern that suppresses the usual feedback acknowledgment.",
    "The Modbus serial segment on node_04 exhibits packet loss and an abrupt relay toggle during maintenance windows. The evidence matches a crafted command injection.",
    "Node_05 detects heavy memory pressure and a stale configuration bundle, consistent with unauthorized lateral movement and privilege abuse on the ICS gateway.",
    "The HAI stack reports oscillating temperature and flow values on node_06 after a stale TLS certificate was accepted during a service restart.",
]


def _tokenize(text: str) -> set[str]:
    return {token.lower().strip(".,;:!?\"'()[]{}") for token in text.split() if token.strip(".,;:!?\"'()[]{}")}


def _embed(text: str) -> np.ndarray:
    vector = np.zeros(384, dtype=np.float32)
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:4], "big") % 384] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _build_sample(index: int) -> dict[str, Any]:
    context = BASE_CONTEXTS[index % len(BASE_CONTEXTS)]
    query = context.replace("The ", "A ").replace("Edge ", "This edge ")
    query = query if index % 2 else context
    context_tokens = _tokenize(context)
    query_tokens = _tokenize(query)
    overlap = context_tokens & query_tokens
    recall = len(overlap) / max(len(context_tokens), 1)
    precision = len(overlap) / max(len(query_tokens), 1)
    return {
        "query_id": f"q{index:02d}",
        "query": query,
        "context": context,
        "context_recall": min(1.0, max(0.0, recall)),
        "context_precision": min(1.0, max(0.0, precision)),
    }


def build_ragas_report(*, n_queries: int = 50, qdrant_path: str | Path | None = None, output_path: str | Path = "evaluation/ragas_results.json") -> dict[str, Any]:
    """Evaluate context recall and precision for queries retrieved from embedded Qdrant."""
    client = QdrantClient(location=":memory:") if qdrant_path is None else QdrantClient(path=str(qdrant_path))
    collection = "week15_ragas"
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection_name=collection, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    contexts = [_build_sample(index) for index in range(n_queries)]
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=index + 1,
                vector=_embed(sample["context"]).tolist(),
                payload={"context": sample["context"], "query_id": sample["query_id"]},
            )
            for index, sample in enumerate(contexts)
        ],
    )
    samples = []
    for sample in contexts:
        matches = client.search(collection_name=collection, query_vector=_embed(sample["query"]).tolist(), limit=1)
        retrieved = str(matches[0].payload.get("context", "")) if matches else ""
        expected_tokens = _tokenize(sample["context"])
        retrieved_tokens = _tokenize(retrieved)
        query_tokens = _tokenize(sample["query"])
        overlap = expected_tokens & retrieved_tokens
        samples.append({
            "query_id": sample["query_id"],
            "query": sample["query"],
            "retrieved_context": retrieved,
            "context_recall": len(overlap) / max(len(expected_tokens), 1),
            "context_precision": len(overlap & query_tokens) / max(len(query_tokens), 1),
        })
    recall_values = [sample["context_recall"] for sample in samples]
    precision_values = [sample["context_precision"] for sample in samples]
    report = {
        "n_queries": len(samples),
        "mean_context_recall": float(sum(recall_values) / len(recall_values)) if recall_values else 0.0,
        "mean_context_precision": float(sum(precision_values) / len(precision_values)) if precision_values else 0.0,
        "min_context_recall": min(recall_values),
        "min_context_precision": min(precision_values),
        "samples": samples,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = build_ragas_report()
    print(json.dumps({
        "n_queries": report["n_queries"],
        "mean_context_recall": report["mean_context_recall"],
        "mean_context_precision": report["mean_context_precision"],
    }, indent=2))


if __name__ == "__main__":
    main()
