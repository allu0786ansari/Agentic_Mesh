from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InsightPayload(BaseModel):
    run_id: str
    node_id: str
    model_type: Literal["vae", "isolation_forest"]
    score: float
    threshold: float
    index: int = Field(ge=0)
    label: int | None = Field(default=None, ge=0, le=1)
    checkpoint_sha256: str | None = None
    scaler_sha256: str | None = None
    feature_count: int = Field(default=0, ge=0)
    explanation: dict[str, object] = Field(default_factory=dict)


class InsightEmbeddingRecord(BaseModel):
    embedding: list[float] = Field(min_length=128, max_length=128)
    payload: InsightPayload
    point_id: str


class InsightEmbeddingQuery(BaseModel):
    embedding: list[float] = Field(min_length=128, max_length=128)
    limit: int = Field(default=5, ge=1, le=100)
