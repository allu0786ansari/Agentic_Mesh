import json
import sqlite3
from pathlib import Path

import pandas as pd

from scripts.partition_data import partition_frame, write_partitions
from storage.sqlite_schema import seed_registry


def test_partition_frame_is_deterministic_and_complete() -> None:
    frame = pd.DataFrame({"value": range(17), "label": [0, 1] * 8 + [0]})

    first = partition_frame(frame, node_count=5, seed=42)
    second = partition_frame(frame, node_count=5, seed=42)

    assert [part.to_dict() for part in first] == [part.to_dict() for part in second]
    assert sum(len(part) for part in first) == len(frame)
    assert sorted(value for part in first for value in part["value"]) == list(range(17))


def test_write_partitions_and_seed_registry(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="s"),
            "sensor_a": range(20),
            "label": [0, 1] * 10,
        }
    )
    partitions_dir = tmp_path / "partitions"
    metadata = write_partitions(
        frame,
        partitions_dir,
        source_name="test-source",
        node_count=5,
        seed=7,
        model_type="vae",
        data_type="timeseries",
    )

    assert len(metadata) == 5
    metadata_paths = sorted(partitions_dir.glob("node_*/meta.json"))
    assert len(metadata_paths) == 5
    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["feature_count"] == 1
        assert (path.parent / "train.parquet").exists()
        assert (path.parent / "test.parquet").exists()

    database_path = tmp_path / "agmesh.db"
    assert seed_registry(database_path, metadata_paths) == 5
    with sqlite3.connect(database_path) as database:
        rows = database.execute(
            "SELECT node_id, model_type, data_type FROM node_registry ORDER BY node_id"
        ).fetchall()
    assert rows == [(f"node_{index:02d}", "vae", "timeseries") for index in range(1, 6)]
