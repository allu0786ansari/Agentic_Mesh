"""Result artifacts and metric events.

Two rules from the plan are enforced here:

* WR-3 — every reported number comes from a JSON artifact with provenance.
* E9 — every artifact and event says where its data came from (``data_source``), so
  fixture or synthetic results can never be mistaken for runtime results.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

from agmesh.common.provenance import build_provenance, utc_now_iso

DataSource = Literal["runtime", "fixture", "synthetic"]
DATA_SOURCES: tuple[str, ...] = get_args(DataSource)

ARTIFACT_SCHEMA_VERSION = "1.0"
EVENT_SCHEMA_VERSION = "1.0"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _sanitise(value: Any) -> Any:
    """Replace non-finite floats, which JSON cannot represent, with strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else ("-inf" if value < 0 else "nan")
    if isinstance(value, dict):
        return {key: _sanitise(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitise(item) for item in value]
    return value


def _check_source(data_source: str) -> None:
    if data_source not in DATA_SOURCES:
        raise ValueError(f"data_source must be one of {DATA_SOURCES}, got {data_source!r}")


def write_artifact(
    path: Path | str,
    *,
    kind: str,
    data_source: DataSource,
    payload: dict[str, Any],
    params_path: Path | str | None = "configs/params.yaml",
    extra_provenance: dict[str, Any] | None = None,
) -> Path:
    """Write a JSON artifact atomically and return its path.

    The file is written to a temporary name and renamed, so a crash never leaves a
    half-written artifact that a later checker could mistake for a complete one.
    """
    _check_source(data_source)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": kind,
        "data_source": data_source,
        "provenance": build_provenance(params_path=params_path, extra=extra_provenance),
        **_sanitise(payload),
    }
    text = json.dumps(document, indent=2, sort_keys=False, default=_json_default)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        Path(tmp_name).replace(target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def read_artifact(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        document: dict[str, Any] = json.load(handle)
    return document


def write_event(
    path: Path | str,
    event_type: str,
    payload: dict[str, Any],
    *,
    run_id: str,
    data_source: DataSource,
    session_id: str | None = None,
) -> None:
    """Append one structured metric event as a JSON line."""
    _check_source(data_source)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "ts": utc_now_iso(),
        "event": event_type,
        "run_id": run_id,
        "data_source": data_source,
    }
    if session_id is not None:
        record["session_id"] = session_id
    record["payload"] = _sanitise(payload)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")
