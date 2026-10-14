"""The node registry (Plan §6.4): active or degraded, with consecutive missed rounds.

A node that misses ``degrade_after`` consecutive rounds (no reply, timeout or
dropout) is marked ``degraded``. A successful round sets it back to ``active`` and
clears the count. A node that declines because its budget is exhausted is not
"missing"; it is recorded as ``exhausted`` and stays so.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """CREATE TABLE IF NOT EXISTS node_registry (
  node_id TEXT PRIMARY KEY, site_id TEXT NOT NULL, model_type TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active', missed_rounds INTEGER NOT NULL DEFAULT 0)"""


class NodeRegistry:
    def __init__(self, path: Path | str = ":memory:", degrade_after: int = 3) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        # the Flower simulation runs the server in its own thread
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute(SCHEMA)
        self.degrade_after = degrade_after

    def register(self, node: str, site: str, model_type: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO node_registry (node_id, site_id, model_type) VALUES (?, ?, ?)",
            (node, site, model_type),
        )
        self.db.commit()

    def succeeded(self, node: str) -> None:
        self.db.execute(
            "UPDATE node_registry SET status='active', missed_rounds=0 "
            "WHERE node_id=? AND status != 'exhausted'",
            (node,),
        )
        self.db.commit()

    def missed(self, node: str) -> None:
        self.db.execute(
            "UPDATE node_registry SET missed_rounds = missed_rounds + 1 WHERE node_id=?", (node,)
        )
        self.db.execute(
            "UPDATE node_registry SET status='degraded' "
            "WHERE node_id=? AND missed_rounds >= ? AND status='active'",
            (node, self.degrade_after),
        )
        self.db.commit()

    def exhausted(self, node: str) -> None:
        self.db.execute("UPDATE node_registry SET status='exhausted' WHERE node_id=?", (node,))
        self.db.commit()

    def status(self, node: str) -> tuple[str, int]:
        row = self.db.execute(
            "SELECT status, missed_rounds FROM node_registry WHERE node_id=?", (node,)
        ).fetchone()
        if row is None:
            raise KeyError(node)
        return str(row[0]), int(row[1])

    def snapshot(self) -> dict[str, dict[str, object]]:
        rows = self.db.execute(
            "SELECT node_id, status, missed_rounds FROM node_registry ORDER BY node_id"
        ).fetchall()
        return {r[0]: {"status": r[1], "missed_rounds": r[2]} for r in rows}
