from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class MITREKnowledgeBaseBuilder:
    """Build a 384-dimensional embedded MITRE ATT&CK for ICS knowledge base."""

    def __init__(self, qdrant_path: str | Path, *, collection: str = "mitre_kb", embedder: Callable[[list[str]], np.ndarray] | None = None):
        self.client = QdrantClient(path=str(qdrant_path))
        self.collection = collection
        self.embedder = embedder or self._sentence_transformer_embedder()
        existing = {item.name for item in self.client.get_collections().collections}
        if collection not in existing:
            self.client.create_collection(collection_name=collection, vectors_config=VectorParams(size=384, distance=Distance.COSINE))

    @staticmethod
    def _sentence_transformer_embedder() -> Callable[[list[str]], np.ndarray]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("sentence-transformers is required to build the MITRE knowledge base") from exc
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return lambda texts: np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    @staticmethod
    def parse_stix(path: str | Path, *, chunk_size: int = 1200) -> list[dict[str, Any]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        objects = payload.get("objects", payload if isinstance(payload, list) else [])
        chunks: list[dict[str, Any]] = []
        for item in objects:
            if item.get("type") != "attack-pattern" or item.get("revoked") or item.get("x_mitre_deprecated"):
                continue
            references = item.get("external_references", [])
            reference = next((ref for ref in references if str(ref.get("source_name", "")).startswith("mitre-attack")), {})
            technique_id = reference.get("external_id")
            if not technique_id:
                continue
            description = " ".join(str(item.get(key, "")) for key in ("name", "description", "x_mitre_detection"))
            phases = [phase.get("phase_name") for phase in item.get("kill_chain_phases", []) if phase.get("phase_name")]
            for index in range(0, max(1, len(description)), chunk_size):
                text = description[index : index + chunk_size]
                chunks.append({"text": text, "technique_id": technique_id, "name": item.get("name", ""), "tactic": phases, "platform": item.get("x_mitre_platforms", ["ics"]), "source": "mitre-attack-ics"})
        return chunks

    @staticmethod
    def parse_nist(path: str | Path, *, chunk_size: int = 1200) -> list[dict[str, Any]]:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return [{"text": text[index : index + chunk_size], "technique_id": "NIST-800-82", "name": "NIST SP 800-82 ICS guidance", "tactic": [], "platform": ["ics"], "source": "nist-sp-800-82"} for index in range(0, max(1, len(text)), chunk_size)]

    def index(self, documents: Iterable[dict[str, Any]], *, batch_size: int = 64) -> int:
        items = list(documents)
        if not items:
            raise ValueError("documents must not be empty")
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            vectors = self.embedder([item["text"] for item in batch])
            if vectors.shape != (len(batch), 384) or not np.isfinite(vectors).all():
                raise ValueError("embedder must return finite vectors with shape (batch, 384)")
            points = []
            for offset, (item, vector) in enumerate(zip(batch, vectors)):
                key = f"{item['source']}:{item['technique_id']}:{start + offset}:{hashlib.sha256(item['text'].encode()).hexdigest()}"
                points.append(PointStruct(id=str(uuid5(NAMESPACE_URL, key)), vector=vector.tolist(), payload=item))
            self.client.upsert(collection_name=self.collection, points=points)
        return len(items)

    def build(self, stix_path: str | Path, nist_path: str | Path | None = None) -> int:
        documents = self.parse_stix(stix_path)
        if nist_path is not None:
            documents.extend(self.parse_nist(nist_path))
        return self.index(documents)

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)


__all__ = ["MITREKnowledgeBaseBuilder"]
