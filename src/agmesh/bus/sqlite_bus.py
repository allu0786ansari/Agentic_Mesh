"""SQLite-backed bus (Plan §6.4 ``bus_messages``; ADR 0009 §5).

Safe across processes:

* the database runs in WAL mode;
* every claim runs inside ``BEGIN IMMEDIATE``, so two consumers can never lease the
  same message.

The table is the Plan's, plus ``failed_at`` for dead letters.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from agmesh.bus.base import (
    DEFAULT_LEASE_S,
    MAX_ATTEMPTS,
    Clock,
    Message,
    check_topic,
    stamp,
    system_clock,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS bus_messages (
  msg_id TEXT PRIMARY KEY, topic TEXT NOT NULL, payload TEXT NOT NULL,
  created_at TEXT NOT NULL, available_at TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
  lease_owner TEXT, lease_until TEXT, acked_at TEXT, failed_at TEXT
);
CREATE INDEX IF NOT EXISTS bus_ready ON bus_messages (topic, acked_at, failed_at, available_at);
"""


class SqliteBus:
    def __init__(
        self, path: Path, *, clock: Clock = system_clock, max_attempts: int = MAX_ATTEMPTS
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.clock, self.max_attempts = path, clock, max_attempts
        self.conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)

    def publish(self, topic: str, payload: dict[str, Any], msg_id: str) -> bool:
        check_topic(topic)
        now = stamp(self.clock())
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO bus_messages (msg_id, topic, payload, created_at, available_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (msg_id, topic, json.dumps(payload, sort_keys=True), now, now),
        )
        return cur.rowcount == 1

    def claim(
        self, topic: str, consumer: str, *, lease_s: float = DEFAULT_LEASE_S, limit: int = 1
    ) -> list[Message]:
        check_topic(topic)
        t = self.clock()
        now, until = stamp(t), stamp(t + lease_s)
        free = (
            "acked_at IS NULL AND failed_at IS NULL AND (lease_until IS NULL OR lease_until <= ?)"
        )
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                f"UPDATE bus_messages SET failed_at = ?, lease_owner = NULL, lease_until = NULL "  # noqa: S608 (fixed SQL fragment, values bound)
                f"WHERE topic = ? AND attempts >= ? AND {free}",
                (now, topic, self.max_attempts, now),
            )
            rows = self.conn.execute(
                f"SELECT msg_id, payload, attempts FROM bus_messages WHERE topic = ? "  # noqa: S608 (fixed SQL fragment, values bound)
                f"AND available_at <= ? AND {free} ORDER BY created_at, msg_id LIMIT ?",
                (topic, now, now, limit),
            ).fetchall()
            for msg_id, _payload, _attempts in rows:
                self.conn.execute(
                    "UPDATE bus_messages SET lease_owner = ?, lease_until = ?, "
                    "attempts = attempts + 1 WHERE msg_id = ?",
                    (consumer, until, msg_id),
                )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return [Message(m, topic, json.loads(p), a + 1) for m, p, a in rows]

    def ack(self, msg_id: str, consumer: str) -> bool:
        cur = self.conn.execute(
            "UPDATE bus_messages SET acked_at = ?, lease_owner = NULL, lease_until = NULL "
            "WHERE msg_id = ? AND lease_owner = ? AND acked_at IS NULL",
            (stamp(self.clock()), msg_id, consumer),
        )
        return cur.rowcount == 1

    def release(self, msg_id: str, consumer: str, delay_s: float = 0.0) -> bool:
        cur = self.conn.execute(
            "UPDATE bus_messages SET lease_owner = NULL, lease_until = NULL, available_at = ? "
            "WHERE msg_id = ? AND lease_owner = ? AND acked_at IS NULL",
            (stamp(self.clock() + delay_s), msg_id, consumer),
        )
        return cur.rowcount == 1

    def pending(self, topic: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM bus_messages WHERE topic = ? AND acked_at IS NULL "
            "AND failed_at IS NULL",
            (topic,),
        ).fetchone()
        return int(row[0])

    def dead(self, topic: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT msg_id FROM bus_messages WHERE topic = ? AND failed_at IS NOT NULL "
            "ORDER BY created_at",
            (topic,),
        ).fetchall()
        return [r[0] for r in rows]

    def stats(self, topic: str) -> dict[str, int]:
        row = self.conn.execute(
            "SELECT COUNT(*), SUM(acked_at IS NOT NULL), SUM(failed_at IS NOT NULL), "
            "SUM(attempts) FROM bus_messages WHERE topic = ?",
            (topic,),
        ).fetchone()
        return {
            "messages": int(row[0] or 0),
            "acked": int(row[1] or 0),
            "dead": int(row[2] or 0),
            "attempts": int(row[3] or 0),
        }

    def close(self) -> None:
        self.conn.close()
