from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS node_registry (
    node_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    data_type TEXT NOT NULL CHECK (data_type IN ('timeseries', 'tabular')),
    model_type TEXT NOT NULL CHECK (model_type IN ('vae', 'isolation_forest')),
    feature_count INTEGER NOT NULL CHECK (feature_count >= 0),
    train_rows INTEGER NOT NULL CHECK (train_rows >= 0),
    test_rows INTEGER NOT NULL CHECK (test_rows >= 0),
    metadata_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    consecutive_misses INTEGER NOT NULL DEFAULT 0,
    last_seen_round INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alert_records (
    alert_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES node_registry(node_id),
    severity TEXT,
    score REAL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fl_rounds (
    round_id INTEGER PRIMARY KEY,
    global_loss REAL,
    participating_nodes INTEGER NOT NULL,
    epsilon REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    database = sqlite3.connect(Path(path))
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA foreign_keys = ON")
    return database


def initialise(path: str | Path) -> None:
    with connect(path) as database:
        database.executescript(SCHEMA)
        columns = {row[1] for row in database.execute("PRAGMA table_info(node_registry)")}
        migrations = {
            "status": "ALTER TABLE node_registry ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
            "consecutive_misses": "ALTER TABLE node_registry ADD COLUMN consecutive_misses INTEGER NOT NULL DEFAULT 0",
            "last_seen_round": "ALTER TABLE node_registry ADD COLUMN last_seen_round INTEGER",
        }
        for column, statement in migrations.items():
            if column not in columns:
                database.execute(statement)


def record_node_round(database_path: str | Path, node_id: str, round_number: int, participated: bool) -> str:
    """Update node liveness and mark a node degraded after three missed rounds."""
    if not node_id.strip() or round_number < 1:
        raise ValueError("node_id must be non-empty and round_number must be positive")
    initialise(database_path)
    with connect(database_path) as database:
        row = database.execute(
            "SELECT consecutive_misses FROM node_registry WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown node_id: {node_id}")
        misses = 0 if participated else int(row[0]) + 1
        status = "active" if participated else ("degraded" if misses >= 3 else "active")
        database.execute(
            """
            UPDATE node_registry
            SET status = ?, consecutive_misses = ?, last_seen_round = CASE WHEN ? THEN ? ELSE last_seen_round END
            WHERE node_id = ?
            """,
            (status, misses, int(participated), round_number, node_id),
        )
        return status


def seed_registry(database_path: str | Path, metadata_paths: Iterable[str | Path]) -> int:
    initialise(database_path)
    count = 0
    with connect(database_path) as database:
        for metadata_path in metadata_paths:
            payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            database.execute(
                """
                INSERT INTO node_registry (
                    node_id, source, data_type, model_type, feature_count,
                    train_rows, test_rows, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    source = excluded.source,
                    data_type = excluded.data_type,
                    model_type = excluded.model_type,
                    feature_count = excluded.feature_count,
                    train_rows = excluded.train_rows,
                    test_rows = excluded.test_rows,
                    metadata_json = excluded.metadata_json
                """,
                (
                    payload["node_id"],
                    payload["source"],
                    payload["data_type"],
                    payload["model_type"],
                    payload["feature_count"],
                    payload["train_rows"],
                    payload["test_rows"],
                    json.dumps(payload, sort_keys=True),
                ),
            )
            count += 1
    return count
