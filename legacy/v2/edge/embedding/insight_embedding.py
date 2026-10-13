from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID, NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import torch

from edge.models.vae import VAE


class QdrantEmbeddingStore:
    """Embedded Qdrant store for fixed-size insight vectors and metadata."""

    def __init__(self, path: str | Path = "qdrant_data", collection: str = "insight_embeddings"):
        self.client = QdrantClient(path=str(path))
        self.collection = collection
        collections = {item.name for item in self.client.get_collections().collections}
        if collection not in collections:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=128, distance=Distance.COSINE),
            )

    def store(self, embedding: np.ndarray, payload: dict[str, Any], point_id: str) -> str:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size != 128:
            raise ValueError(f"Insight embedding must have 128 dimensions, got {vector.size}")
        if not np.isfinite(vector).all():
            raise ValueError("Insight embedding contains non-finite values")
        try:
            normalized_id = str(UUID(point_id))
        except ValueError:
            normalized_id = str(uuid5(NAMESPACE_URL, point_id))
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=normalized_id, vector=vector.tolist(), payload=payload)],
        )
        return normalized_id

    def search(self, embedding: np.ndarray, limit: int = 5) -> list[Any]:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size != 128 or not np.isfinite(vector).all():
            raise ValueError("Search embedding must be a finite 128-dimensional vector")
        return self.client.search(collection_name=self.collection, query_vector=vector.tolist(), limit=limit)


class InsightEmbeddingGenerator:
    """Create comparable, finite, L2-normalized 128-dim node embeddings."""

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.size < 128:
            vector = np.pad(vector, (0, 128 - vector.size))
        vector = vector[:128]
        norm = np.linalg.norm(vector)
        if norm == 0 or not np.isfinite(norm):
            return np.zeros(128, dtype=np.float32)
        return (vector / norm).astype(np.float32)

    @classmethod
    def from_vae(cls, model: VAE, inputs: torch.Tensor) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            mu, logvar = model.encode(inputs)
        vectors = torch.cat((mu, logvar), dim=1).cpu().numpy()
        return np.vstack([cls._normalize(vector) for vector in vectors])

    @classmethod
    def from_isolation_forest(cls, scores: np.ndarray, transformed_values: np.ndarray) -> np.ndarray:
        values = np.asarray(transformed_values, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
        vectors = np.concatenate((scores, values), axis=1)
        return np.vstack([cls._normalize(vector) for vector in vectors])
