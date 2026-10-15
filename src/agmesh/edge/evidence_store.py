"""Per-node evidence store (SQLite, WAL), read later by the node API (Week 11).

Only validated :class:`~agmesh.common.schemas.EvidenceRecord` objects are stored, as
their JSON. Inserts are idempotent on ``evidence_id``, so replay never duplicates a
record. Queries are by node and time range, which is what the Investigator asks for
(adjacent windows, Proposal §8.2).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agmesh.common.provenance import utc_now_iso
from agmesh.common.schemas import EvidenceRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
  evidence_id TEXT PRIMARY KEY,
  node_id TEXT NOT NULL,
  window_start TEXT NOT NULL,
  window_end TEXT NOT NULL,
  record TEXT NOT NULL,
  stored_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_time ON evidence (node_id, window_end);
"""


class EvidenceStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)

    def put(self, record: EvidenceRecord) -> bool:
        """Store a record; False when it was already there (idempotent replay)."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.evidence_id,
                record.node_id,
                record.window_start,
                record.window_end,
                record.model_dump_json(),
                utc_now_iso(),
            ),
        )
        return cur.rowcount == 1

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        row = self.conn.execute(
            "SELECT record FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        return EvidenceRecord.model_validate_json(row[0]) if row else None

    def between(self, node_id: str, start_iso: str, end_iso: str) -> list[EvidenceRecord]:
        """Records of a node whose window ends in ``[start_iso, end_iso]``, in time order."""
        rows = self.conn.execute(
            "SELECT record FROM evidence WHERE node_id = ? AND window_end BETWEEN ? AND ? "
            "ORDER BY window_end",
            (node_id, start_iso, end_iso),
        ).fetchall()
        return [EvidenceRecord.model_validate_json(r[0]) for r in rows]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0])

    def close(self) -> None:
        self.conn.close()
