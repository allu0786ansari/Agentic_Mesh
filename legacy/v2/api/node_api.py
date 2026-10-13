from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from agents.auth import JWTAuthManager
from edge.embedding.insight_embedding import QdrantEmbeddingStore
from edge.embedding.schemas import InsightEmbeddingQuery, InsightEmbeddingRecord


def create_app(
    qdrant_path: str | Path | None = None,
    collection: str = "insight_embeddings",
    auth_manager: JWTAuthManager | None = None,
) -> FastAPI:
    application = FastAPI(title="Agentic Mesh Node API", version="0.1.0")
    store = QdrantEmbeddingStore(qdrant_path or os.getenv("QDRANT_PATH", "./qdrant_data"), collection)
    bearer = HTTPBearer(auto_error=False)

    def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> str | None:
        if auth_manager is None:
            return None
        if credentials is None:
            raise HTTPException(status_code=401, detail="Bearer token required")
        try:
            return str(auth_manager.validate_token(credentials.credentials)["sub"])
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "collection": store.collection}

    @application.post("/embeddings")
    def store_embedding(record: InsightEmbeddingRecord, _: str | None = Depends(require_auth)) -> dict[str, str]:
        try:
            normalized_id = store.store(
                np.asarray(record.embedding, dtype=np.float32),
                record.payload.model_dump(),
                record.point_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "stored", "point_id": normalized_id}

    @application.post("/embeddings/search")
    def search_embeddings(query: InsightEmbeddingQuery, _: str | None = Depends(require_auth)) -> list[dict[str, Any]]:
        try:
            matches = store.search(np.asarray(query.embedding, dtype=np.float32), query.limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return [{"id": str(item.id), "score": item.score, "payload": item.payload} for item in matches]

    return application


app = create_app()
