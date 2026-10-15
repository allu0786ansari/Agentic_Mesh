"""Evidence records and alerts from one scored window (Proposal §6.6, ADR 0009 §3-4).

Everything here takes a node's *private* inputs: per-feature errors, residuals, raw
counters and raw constant values. It returns only the minimised outbound models of
:mod:`agmesh.common.schemas`. Raw numbers never pass through. The record carries
bucket labels, directions and feature names.

IDs are deterministic: the hash of ``(model_version, node, window_start)``. Replaying
the same stream through the same model therefore reproduces the same evidence and
alert IDs. The bus and the session store use this to de-duplicate (ADR 0009 §5).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt

from agmesh.common.provenance import sha256_bytes
from agmesh.common.schemas import (
    MAX_TOP_FEATURES,
    Alert,
    CountBucket,
    EvidenceRecord,
    Magnitude,
    ProtocolSummary,
    Provenance,
    RateBucket,
    TopFeature,
)

LATENCY_FEATURES = frozenset({"latency_median_ms", "slow_responses"})


def iso(t_s: float) -> str:
    """Epoch seconds as ISO-8601 UTC with milliseconds and a ``Z`` suffix."""
    text = datetime.fromtimestamp(float(t_s), tz=UTC).isoformat(timespec="milliseconds")
    return text.replace("+00:00", "Z")


def stable_id(kind: str, model_version: str, node: str, window_start_s: float) -> str:
    key = f"{kind}|{model_version}|{node}|{float(window_start_s):.3f}"
    return sha256_bytes(key.encode())[:32]


def count_bucket(n: float) -> CountBucket:
    if n <= 0:
        return "0"
    if n <= 10:
        return "1-10"
    if n <= 100:
        return "11-100"
    return ">100"


def rate_bucket(n: float, p1: float, p99: float) -> RateBucket:
    if n < p1:
        return "low"
    if n > p99:
        return "high"
    return "normal"


def top_features(
    feature_errors: npt.ArrayLike,
    residuals: npt.ArrayLike,
    names: Sequence[str],
    buckets: Mapping[str, Sequence[float]],
) -> list[TopFeature]:
    """Up to five features at or above their own p90, ordered by error / p99."""
    e = np.asarray(feature_errors, dtype=np.float64)
    r = np.asarray(residuals, dtype=np.float64)
    picked: list[tuple[float, TopFeature]] = []
    for j, name in enumerate(names):
        p90, p99, p999 = (float(v) for v in buckets[name])
        if not e[j] >= p90:
            continue
        magnitude: Magnitude = "high" if e[j] > p999 else "medium" if e[j] > p99 else "low"
        ratio = e[j] / p99 if p99 > 0 else np.inf
        picked.append(
            (
                ratio,
                TopFeature(name=name, direction="up" if r[j] > 0 else "down", magnitude=magnitude),
            )
        )
    picked.sort(key=lambda item: (-item[0], item[1].name))
    return [tf for _ratio, tf in picked[:MAX_TOP_FEATURES]]


def protocol_summary(
    counters: Mapping[str, float], request_rate: Sequence[float]
) -> ProtocolSummary:
    c = counters
    return ProtocolSummary(
        read_ops=count_bucket(c["req_read"]),
        write_ops=count_bucket(c["req_write"]),
        diagnostic_ops=count_bucket(c["req_other"]),
        exception_responses=count_bucket(c["resp_exception"]),
        request_rate=rate_bucket(c["req_recv"], float(request_rate[0]), float(request_rate[1])),
        new_peer=c["new_peers"] > 0,
        unseen_address=(c["unseen_addr"] + c["unseen_addr_write"]) > 0,
        oversized_quantity=c["oversized_qty"] > 0,
        malformed_frames=c["malformed_frames"] > 0,
        stacked_frames=c["stacked_frames"] > 0,
        slow_responses=c["slow_responses"] > 0,
    )


def constants_changed(
    window_values: npt.ArrayLike, names: Sequence[str], constants: Mapping[str, float]
) -> list[str]:
    """Names of training-constant variables that take another value in the window."""
    v = np.asarray(window_values, dtype=np.float64)
    out = []
    for j, name in enumerate(names):
        ref = float(constants[name])
        if np.any(np.abs(v[:, j] - ref) > 1e-6 * max(1.0, abs(ref))):
            out.append(name)
    return out


def build(
    *,
    site: str,
    node: str,
    window_start_s: float,
    window_end_s: float,
    model_version: str,
    score: float,
    threshold: float,
    calibrated_prob: float,
    features: list[TopFeature],
    run_id: str,
    data_source: str,
    summary: ProtocolSummary | None = None,
    changed: Sequence[str] = (),
) -> tuple[EvidenceRecord, Alert]:
    evidence_id = stable_id("evidence", model_version, node, window_start_s)
    record = EvidenceRecord(
        evidence_id=evidence_id,
        site_id=site,  # type: ignore[arg-type]
        node_id=node,
        window_start=iso(window_start_s),
        window_end=iso(window_end_s),
        model_version=model_version,
        score=float(score),
        threshold=float(threshold),
        calibrated_prob=float(np.clip(calibrated_prob, 0.0, 1.0)),
        top_features=list(features),
        constants_changed=list(changed),
        protocol_summary=summary,
        provenance=Provenance(run_id=run_id, data_source=data_source),  # type: ignore[arg-type]
    )
    record.check_site()
    alert = Alert(
        alert_id=stable_id("alert", model_version, node, window_start_s),
        evidence_id=evidence_id,
        site_id=record.site_id,
        node_id=node,
        window_end=record.window_end,
        calibrated_prob=record.calibrated_prob,
        latency_only=bool(features) and all(f.name in LATENCY_FEATURES for f in features),
        run_id=run_id,
        data_source=record.provenance.data_source,
    )
    return record, alert
