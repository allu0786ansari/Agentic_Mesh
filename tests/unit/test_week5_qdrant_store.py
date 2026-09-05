from pathlib import Path

import numpy as np
import pytest

from edge.embedding.insight_embedding import QdrantEmbeddingStore
from scripts.init_qdrant import initialise


def test_qdrant_store_round_trip(tmp_path: Path) -> None:
    qdrant_path = tmp_path / "qdrant"
    initialise(qdrant_path)
    store = QdrantEmbeddingStore(qdrant_path)
    vector = np.ones(128, dtype=np.float32)
    store.store(vector, {"node_id": "node_04"}, "point-1")
    matches = store.search(vector, limit=1)
    assert matches
    assert matches[0].payload["node_id"] == "node_04"


def test_qdrant_store_rejects_wrong_dimension(tmp_path: Path) -> None:
    initialise(tmp_path / "qdrant")
    store = QdrantEmbeddingStore(tmp_path / "qdrant")
    with pytest.raises(ValueError):
        store.store(np.ones(8, dtype=np.float32), {}, "bad")
