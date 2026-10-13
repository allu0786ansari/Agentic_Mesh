from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTIONS = {
    "insight_embeddings": 128,
    "mitre_kb": 384,
    "agent_memory": 384,
}


def initialise(path: str | Path) -> list[str]:
    """Create the Week 3 embedded Qdrant collections idempotently."""
    client = QdrantClient(path=str(path))
    existing = {collection.name for collection in client.get_collections().collections}
    for name, size in COLLECTIONS.items():
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )
    return sorted(collection.name for collection in client.get_collections().collections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise embedded Qdrant collections")
    parser.add_argument("--path", type=Path, default=Path("qdrant_data"))
    args = parser.parse_args()
    print(initialise(args.path))


if __name__ == "__main__":
    main()
