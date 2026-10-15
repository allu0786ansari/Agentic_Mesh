"""Vector stores: embedded Qdrant (Proposal §8.4) and an in-memory twin for tests.

Both expose the same small interface:

* ``create`` (dimension, cosine distance);
* ``upsert`` (vectors with payloads);
* ``search`` (top ``limit`` by cosine);
* ``count`` and ``dim``.

Search is exact: Qdrant's local mode compares every point, and
:class:`MemoryStore` uses a matrix product. A ranking is therefore deterministic,
with ties broken by point order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
COLLECTIONS = ("mitre_ics", "nist_80082", "incident_memory")


class Store(Protocol):
    def create(self, collection: str, dim: int) -> None: ...

    def upsert(self, collection: str, vectors: F32, payloads: list[dict[str, Any]]) -> None: ...

    def search(
        self, collection: str, vector: F32, limit: int
    ) -> list[tuple[float, dict[str, Any]]]: ...

    def count(self, collection: str) -> int: ...

    def dim(self, collection: str) -> int: ...


class MemoryStore:
    def __init__(self) -> None:
        self.data: dict[str, tuple[int, list[F32], list[dict[str, Any]]]] = {}

    def create(self, collection: str, dim: int) -> None:
        self.data[collection] = (dim, [], [])

    def upsert(self, collection: str, vectors: F32, payloads: list[dict[str, Any]]) -> None:
        dim, vecs, pays = self.data[collection]
        if vectors.shape[1] != dim:
            raise ValueError(
                f"{collection}: vectors of dimension {vectors.shape[1]}, collection {dim}"
            )
        vecs.extend(np.asarray(vectors, dtype=np.float32))
        pays.extend(payloads)

    def search(
        self, collection: str, vector: F32, limit: int
    ) -> list[tuple[float, dict[str, Any]]]:
        dim, vecs, pays = self.data[collection]
        if vector.shape[-1] != dim:
            raise ValueError(
                f"{collection}: query of dimension {vector.shape[-1]}, collection {dim}"
            )
        if not vecs:
            return []
        scores = np.stack(vecs) @ np.asarray(vector, dtype=np.float32).reshape(-1)
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(float(scores[i]), pays[i]) for i in order]

    def count(self, collection: str) -> int:
        return len(self.data[collection][1])

    def dim(self, collection: str) -> int:
        return self.data[collection][0]


class QdrantStore:
    """Embedded Qdrant (``QdrantClient(path=...)``), dependency group ``kb``."""

    def __init__(self, path: Path) -> None:
        from qdrant_client import QdrantClient

        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))

    def create(self, collection: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

    def upsert(self, collection: str, vectors: F32, payloads: list[dict[str, Any]]) -> None:
        from qdrant_client.models import PointStruct

        start = self.count(collection)
        points = [
            PointStruct(id=start + i, vector=v.tolist(), payload=p)
            for i, (v, p) in enumerate(zip(vectors, payloads, strict=True))
        ]
        for i in range(0, len(points), 256):
            self.client.upsert(collection, points=points[i : i + 256], wait=True)

    def search(
        self, collection: str, vector: F32, limit: int
    ) -> list[tuple[float, dict[str, Any]]]:
        res = self.client.query_points(
            collection,
            query=np.asarray(vector).reshape(-1).tolist(),
            limit=limit,
            with_payload=True,
        )
        return [(float(p.score), dict(p.payload or {})) for p in res.points]

    def count(self, collection: str) -> int:
        return int(self.client.count(collection, exact=True).count)

    def dim(self, collection: str) -> int:
        params: Any = self.client.get_collection(collection).config.params.vectors
        return int(params.size)

    def close(self) -> None:
        self.client.close()
