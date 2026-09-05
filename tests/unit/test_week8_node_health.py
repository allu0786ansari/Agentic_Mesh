from __future__ import annotations

import json
from pathlib import Path

from storage.sqlite_schema import connect, record_node_round, seed_registry


def test_node_becomes_degraded_after_three_missed_rounds(tmp_path: Path) -> None:
    metadata = tmp_path / "node_01.json"
    metadata.write_text(
        json.dumps(
            {
                "node_id": "node_01",
                "source": "hai",
                "data_type": "timeseries",
                "model_type": "vae",
                "feature_count": 38,
                "train_rows": 10,
                "test_rows": 5,
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "mesh.db"
    seed_registry(database, [metadata])

    assert record_node_round(database, "node_01", 1, False) == "active"
    assert record_node_round(database, "node_01", 2, False) == "active"
    assert record_node_round(database, "node_01", 3, False) == "degraded"

    with connect(database) as connection:
        row = connection.execute(
            "SELECT status, consecutive_misses, last_seen_round FROM node_registry WHERE node_id = ?",
            ("node_01",),
        ).fetchone()
    assert tuple(row) == ("degraded", 3, None)


def test_participation_resets_missed_rounds(tmp_path: Path) -> None:
    metadata = tmp_path / "node_01.json"
    metadata.write_text(
        json.dumps(
            {
                "node_id": "node_01",
                "source": "hai",
                "data_type": "timeseries",
                "model_type": "vae",
                "feature_count": 38,
                "train_rows": 10,
                "test_rows": 5,
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "mesh.db"
    seed_registry(database, [metadata])

    record_node_round(database, "node_01", 1, False)
    assert record_node_round(database, "node_01", 2, True) == "active"

    with connect(database) as connection:
        row = connection.execute(
            "SELECT status, consecutive_misses, last_seen_round FROM node_registry WHERE node_id = ?",
            ("node_01",),
        ).fetchone()
    assert tuple(row) == ("active", 0, 2)
