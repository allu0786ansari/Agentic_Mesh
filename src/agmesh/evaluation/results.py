"""Result artifacts (Implementation Plan §6.5; WR-3).

Every reported number comes from a file at
``results/<experiment>/<config_id>/seed_<n>.json`` written by :func:`write_result`.
The schema is the :class:`ResultArtifact` model; the checker validates files against
it before anything is reported.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agmesh.common.artifacts import DataSource, write_artifact
from agmesh.common.provenance import sha256_file, utc_now_iso

RESULT_SCHEMA_VERSION = "1.0"
#: Metrics every detection result (X1-X3) must report.
DETECTION_METRICS = (
    "auroc",
    "auprc",
    "event_recall",
    "range_f1",
    "window_fpr",
    "ttd_median_s",
    "false_alarms_per_hour",
)
#: Provenance fields the checker requires (lockfile lives under environment).
REQUIRED_PROVENANCE = ("git_sha", "params_sha256", "manifest_sha256", "started_at", "finished_at")
_CONFIG_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")


class ResultArtifact(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str
    kind: Literal["result"]
    experiment: str = Field(pattern=r"^X[1-8]$")
    config_id: str
    seed: int
    split: Literal["val", "test"]
    data_source: Literal["runtime", "fixture", "synthetic"]
    provenance: dict[str, Any]
    metrics: dict[str, float | None]
    privacy: dict[str, Any] | None = None
    federation: dict[str, Any] | None = None

    @field_validator("config_id")
    @classmethod
    def _safe_config_id(cls, value: str) -> str:
        if not _CONFIG_ID.match(value):
            raise ValueError("config_id may contain only letters, digits, '_', '.', '-'")
        return value


def result_path(root: Path, experiment: str, config_id: str, seed: int) -> Path:
    return root / experiment / config_id / f"seed_{seed}.json"


def scores_path(result: Path) -> Path:
    """The per-window score file that sits beside a result artifact (ADR 0005)."""
    return result.with_name(result.stem + ".scores.npz")


def write_scores(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return {"path": path.name, "sha256": sha256_file(path)}


def write_result(
    root: Path,
    *,
    experiment: str,
    config_id: str,
    seed: int,
    split: Literal["val", "test"],
    data_source: DataSource,
    metrics: dict[str, float | None],
    manifest_sha256: str,
    started_at: str,
    checkpoint_sha256: str | None = None,
    privacy: dict[str, Any] | None = None,
    federation: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    params_path: Path = Path("configs/params.yaml"),
    scores: dict[str, np.ndarray] | None = None,
) -> Path:
    """Validate and write one result artifact; returns its path.

    ``scores`` (per-window arrays) is written first to ``seed_<n>.scores.npz`` and its
    hash recorded under ``scores_file``, so event-level inference can be re-run.
    """
    payload: dict[str, Any] = {
        "experiment": experiment,
        "config_id": config_id,
        "seed": seed,
        "split": split,
        "metrics": metrics,
    }
    if privacy is not None:
        payload["privacy"] = privacy
    if federation is not None:
        payload["federation"] = federation
    if extra:
        payload.update(extra)
    if scores is not None:
        target = result_path(root, experiment, config_id, seed)
        payload["scores_file"] = write_scores(scores_path(target), scores)
    provenance = {
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
    }
    ResultArtifact.model_validate(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": "result",
            "data_source": data_source,
            "provenance": provenance,
            **payload,
        }
    )
    return write_artifact(
        result_path(root, experiment, config_id, seed),
        kind="result",
        data_source=data_source,
        payload=payload,
        params_path=params_path,
        extra_provenance=provenance,
    )
