"""Outbound message contracts: evidence records and alerts (Plan §6.3, ADR 0009).

These models are the only shapes in which operational data leaves a node. They
forbid extra fields. The only numbers they carry are ``score``, ``threshold`` and
``calibrated_prob``. Every string is an identifier, an ISO time, an enumeration
value, or a feature name that the node's feature list allows
(:func:`check_features`). ``tests/contract/test_evidence_schema.py`` checks this,
including with random raw values.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Direction = Literal["up", "down"]
Magnitude = Literal["low", "medium", "high"]
CountBucket = Literal["0", "1-10", "11-100", ">100"]
RateBucket = Literal["low", "normal", "high"]
SiteId = Literal["site_a", "site_b"]
DataSource = Literal["runtime", "fixture", "synthetic"]
NODE_PATTERN = r"^[ab][1-9]$"
ID_PATTERN = r"^[0-9a-f]{32}$"
ISO_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"
VERSION_PATTERN = r"^sha256:[0-9a-f]{64}$"
RUN_PATTERN = r"^[A-Za-z0-9_.:+-]{1,160}$"
FEATURE_PATTERN = r"^[A-Za-z][A-Za-z0-9_]{0,40}$"
MAX_TOP_FEATURES = 5


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TopFeature(_Strict):
    name: str = Field(pattern=FEATURE_PATTERN)
    direction: Direction
    magnitude: Magnitude


class ProtocolSummary(_Strict):
    read_ops: CountBucket
    write_ops: CountBucket
    diagnostic_ops: CountBucket
    exception_responses: CountBucket
    request_rate: RateBucket
    new_peer: bool
    unseen_address: bool
    oversized_quantity: bool
    malformed_frames: bool
    stacked_frames: bool
    slow_responses: bool


class Provenance(_Strict):
    run_id: str = Field(pattern=RUN_PATTERN)
    data_source: DataSource


class EvidenceRecord(_Strict):
    evidence_id: str = Field(pattern=ID_PATTERN)
    site_id: SiteId
    node_id: str = Field(pattern=NODE_PATTERN)
    window_start: str = Field(pattern=ISO_PATTERN)
    window_end: str = Field(pattern=ISO_PATTERN)
    model_version: str = Field(pattern=VERSION_PATTERN)
    score: float = Field(allow_inf_nan=False)
    threshold: float = Field(allow_inf_nan=False)
    calibrated_prob: float = Field(ge=0.0, le=1.0)
    top_features: list[TopFeature] = Field(max_length=MAX_TOP_FEATURES)
    constants_changed: list[str] = Field(default_factory=list, max_length=32)
    protocol_summary: ProtocolSummary | None = None
    provenance: Provenance

    @field_validator("constants_changed")
    @classmethod
    def _names(cls, value: list[str]) -> list[str]:
        import re

        bad = [v for v in value if not re.fullmatch(FEATURE_PATTERN, v)]
        if bad:
            raise ValueError(f"not feature names: {bad}")
        return value

    def check_site(self) -> None:
        """Site A carries no protocol summary; Site B carries one and no constants."""
        if self.site_id == "site_b" and self.protocol_summary is None:
            raise ValueError("a Site B evidence record needs a protocol_summary")
        if self.site_id == "site_a" and self.protocol_summary is not None:
            raise ValueError("protocol_summary is for Site B only")
        if self.site_id == "site_b" and self.constants_changed:
            raise ValueError("constants_changed is for Site A only")
        if not self.node_id.startswith(self.site_id[-1]):
            raise ValueError(f"node {self.node_id} is not at {self.site_id}")


class Alert(_Strict):
    alert_id: str = Field(pattern=ID_PATTERN)
    evidence_id: str = Field(pattern=ID_PATTERN)
    site_id: SiteId
    node_id: str = Field(pattern=NODE_PATTERN)
    window_end: str = Field(pattern=ISO_PATTERN)
    calibrated_prob: float = Field(ge=0.0, le=1.0)
    latency_only: bool
    run_id: str = Field(pattern=RUN_PATTERN)
    data_source: DataSource


def check_features(record: EvidenceRecord, allowed: Iterable[str]) -> None:
    """Every feature name in a record must be one the node is allowed to name."""
    names = set(allowed)
    used = [f.name for f in record.top_features] + list(record.constants_changed)
    unknown = [n for n in used if n not in names]
    if unknown:
        raise ValueError(f"feature names not in the node's list: {unknown}")


#: The numeric fields an outbound record may carry (Plan §6.3).
NUMERIC_FIELDS = frozenset({"score", "threshold", "calibrated_prob"})
