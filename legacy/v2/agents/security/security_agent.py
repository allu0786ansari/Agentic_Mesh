from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agents.llm import LocalPhiModel
from agents.state import AgentState
from agents.tracing import get_tracer
from qdrant_client import QdrantClient
from storage.sqlite_schema import connect, initialise


class SecurityAgent:
    """Retrieve MITRE evidence and classify the investigation with Phi."""

    def __init__(self, *, llm: LocalPhiModel, qdrant_path: str | Path | None = None, qdrant_client: QdrantClient | None = None, collection: str = "mitre_kb", embedder: Callable[[list[str]], np.ndarray] | None = None, confidence_threshold: float = 0.6, database_path: str | Path | None = None):
        if qdrant_client is None and qdrant_path is None:
            raise ValueError("qdrant_path or qdrant_client is required")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.llm = llm
        self.client = qdrant_client or QdrantClient(path=str(qdrant_path))
        self.collection = collection
        self.embedder = embedder or self._default_embedder()
        self.confidence_threshold = confidence_threshold
        self.database_path = Path(database_path) if database_path else None
        self.tracer = get_tracer("agentic-mesh-security")

    @staticmethod
    def _default_embedder() -> Callable[[list[str]], np.ndarray]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return lambda texts: np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    async def run(self, state: AgentState) -> AgentState:
        evidence = list(state.get("investigation_evidence") or [])
        with self.tracer.span("security_agent", session_id=state.get("session_id", "")):
            text = "\n".join(json.dumps(item, default=str) for item in evidence)
            vector = np.asarray(self.embedder([text]), dtype=np.float32)
            if vector.shape != (1, 384) or not np.isfinite(vector).all():
                raise ValueError("security embedder must return one finite 384-dimensional vector")
            matches = self.client.search(collection_name=self.collection, query_vector=vector[0].tolist(), limit=5)
            retrieved = [{"score": float(item.score), "payload": item.payload} for item in matches]
            decision = self.llm.invoke_json(self._prompt(state, evidence, retrieved))
            classification = self._validate(decision, retrieved)
            if self.database_path:
                self._update_alert(state, classification)
            return {**state, "mitre_match": {**classification, "retrieved_matches": retrieved}, "status": "classified"}

    @staticmethod
    def _prompt(state: AgentState, evidence: list[dict[str, Any]], matches: list[dict[str, Any]]) -> str:
        return (
            "Classify this industrial cybersecurity investigation against MITRE ATT&CK for ICS. "
            "Return JSON with technique_id, tactic, confidence, reasoning, recommended_mitigations. "
            f"Evidence: {json.dumps(evidence, default=str)}\nRetrieved knowledge: {json.dumps(matches, default=str)}"
        )

    def _validate(self, decision: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
        confidence = float(decision.get("confidence", 0.0))
        if not 0 <= confidence <= 1:
            confidence = 0.0
        technique = str(decision.get("technique_id", "UNKNOWN")) if confidence >= self.confidence_threshold else "UNKNOWN"
        mitigations = decision.get("recommended_mitigations", [])
        if not isinstance(mitigations, list):
            mitigations = [str(mitigations)]
        return {"technique_id": technique, "tactic": str(decision.get("tactic", "UNKNOWN")), "confidence": confidence, "reasoning": str(decision.get("reasoning", "No reasoning returned")), "recommended_mitigations": mitigations}

    def _update_alert(self, state: AgentState, classification: dict[str, Any]) -> None:
        initialise(self.database_path)
        with connect(self.database_path) as database:
            database.execute("UPDATE alert_records SET status = ? WHERE alert_id = ?", ("classified", state.get("session_id", "unknown")))


__all__ = ["SecurityAgent"]
