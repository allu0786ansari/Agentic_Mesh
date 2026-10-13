"""Deterministic partition files and their manifest (Implementation Plan §6.2).

A manifest describes one site's partitions: which source files went in (with
SHA-256), the configuration that shaped the split, and for every node and partition
its row and window counts, time range, events and file hash. It contains no wall
clock times or machine details, so running the same code on the same data with the
same parameters produces a byte-identical manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from agmesh.common.provenance import sha256_file

MANIFEST_NAME = "manifest.json"
PARTITIONS = ("train", "val_normal", "val_attack", "test")


def write_table(frame: pd.DataFrame, path: Path) -> str:
    """Write a parquet file deterministically and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index(drop=True).to_parquet(
        path, engine="pyarrow", index=False, compression="zstd", row_group_size=262_144
    )
    return sha256_file(path)


def iso(value: Any) -> str | None:
    """ISO-8601 UTC string for a timestamp-like value (epoch seconds or datetime)."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    stamp = (
        pd.Timestamp(value, unit="s", tz="UTC")
        if isinstance(value, int | float)
        else pd.Timestamp(value)
    )
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def write_manifest(manifest: dict[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_manifest(directory: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    return data
