"""Session store (Plan §6.4 ``sessions``, ``hops``, ``actions``; Proposal §8.3; ADR 0009 §6).

* The **idempotency key** is the SHA-256 of ``site_id``, ``|``, and the sorted IDs of
  the opening alerts joined by ``,``. Opening the same incident twice, for example
  after an alert is redelivered, returns the existing session.
* **Status changes** follow :data:`TRANSITIONS`. Anything else raises. Terminal
  states set ``closed_at``.
* **Hops** are recorded before the graph advances. ``hop_count`` can never exceed
  ``max_hops``.
* **Actions** are unique per ``(session_id, action_type)``, so dispatch is
  idempotent (Proposal §8.2).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agmesh.common.provenance import sha256_bytes, utc_now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY,
  idempotency_key TEXT UNIQUE NOT NULL, site_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('OPEN','INVESTIGATING','ATTRIBUTING','DISPATCHING',
         'COMPLETED','COMPLETED_DEGRADED','ABSTAINED','FAILED')),
  opened_at TEXT NOT NULL, closed_at TEXT, hop_count INTEGER NOT NULL DEFAULT 0,
  attribution_mode TEXT, trace_id TEXT, run_id TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS hops (session_id TEXT NOT NULL, hop_index INTEGER NOT NULL,
  node_id TEXT NOT NULL, outcome TEXT NOT NULL, latency_ms REAL, started_at TEXT NOT NULL,
  PRIMARY KEY (session_id, hop_index));

CREATE TABLE IF NOT EXISTS actions (action_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
  action_type TEXT NOT NULL, impact TEXT NOT NULL CHECK (impact IN ('low','high')),
  status TEXT NOT NULL CHECK (status IN ('dispatched','pending_approval','approved','rejected')),
  technique_id TEXT, tactic TEXT, confidence REAL, payload TEXT NOT NULL, trace_id TEXT,
  created_at TEXT NOT NULL, UNIQUE (session_id, action_type));

CREATE TABLE IF NOT EXISTS session_alerts (session_id TEXT NOT NULL, alert_id TEXT NOT NULL,
  PRIMARY KEY (session_id, alert_id));
"""

TERMINAL = frozenset({"COMPLETED", "COMPLETED_DEGRADED", "ABSTAINED", "FAILED"})
TRANSITIONS: dict[str, frozenset[str]] = {
    "OPEN": frozenset({"INVESTIGATING", "FAILED"}),
    "INVESTIGATING": frozenset({"ATTRIBUTING", "ABSTAINED", "FAILED"}),
    "ATTRIBUTING": frozenset({"DISPATCHING", "ABSTAINED", "FAILED"}),
    "DISPATCHING": frozenset({"COMPLETED", "COMPLETED_DEGRADED", "FAILED"}),
}


class TransitionError(ValueError):
    pass


def idempotency_key(site_id: str, alert_ids: Iterable[str]) -> str:
    ids = sorted(set(alert_ids))
    if not ids:
        raise ValueError("a session is opened by at least one alert")
    return sha256_bytes(f"{site_id}|{','.join(ids)}".encode())


@dataclass(frozen=True)
class Session:
    session_id: str
    idempotency_key: str
    site_id: str
    status: str
    opened_at: str
    closed_at: str | None
    hop_count: int
    attribution_mode: str | None
    trace_id: str | None
    run_id: str


class SessionStore:
    def __init__(self, path: Path, *, max_hops: int = 3) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.max_hops = max_hops
        self.conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ sessions
    def open(
        self, site_id: str, alert_ids: Iterable[str], run_id: str, *, mode: str | None = None
    ) -> tuple[Session, bool]:
        """The session for this incident, and whether this call created it."""
        ids = sorted(set(alert_ids))
        key = idempotency_key(site_id, ids)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, idempotency_key, site_id, status, "
                "opened_at, attribution_mode, run_id) VALUES (?, ?, ?, 'OPEN', ?, ?, ?)",
                (uuid.uuid4().hex, key, site_id, utc_now_iso(), mode, run_id),
            )
            created = cur.rowcount == 1
            session = self._by_key(key)
            if created:
                self.conn.executemany(
                    "INSERT INTO session_alerts VALUES (?, ?)",
                    [(session.session_id, a) for a in ids],
                )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return session, created

    def _by_key(self, key: str) -> Session:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE idempotency_key = ?", (key,)
        ).fetchone()
        return Session(*row)

    def get(self, session_id: str) -> Session:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return Session(*row)

    def alerts(self, session_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT alert_id FROM session_alerts WHERE session_id = ? ORDER BY alert_id",
            (session_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def transition(self, session_id: str, to: str) -> Session:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            current = self.get(session_id).status
            if to not in TRANSITIONS.get(current, frozenset()):
                raise TransitionError(f"{session_id}: {current} -> {to} is not allowed")
            closed = utc_now_iso() if to in TERMINAL else None
            self.conn.execute(
                "UPDATE sessions SET status = ?, closed_at = COALESCE(?, closed_at) "
                "WHERE session_id = ? AND status = ?",
                (to, closed, session_id, current),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return self.get(session_id)

    def open_sessions(self) -> list[Session]:
        """Sessions that are not terminal (for resuming after a restart)."""
        marks = ",".join("?" * len(TERMINAL))
        rows = self.conn.execute(
            f"SELECT * FROM sessions WHERE status NOT IN ({marks}) ORDER BY opened_at",  # noqa: S608
            tuple(sorted(TERMINAL)),
        ).fetchall()
        return [Session(*r) for r in rows]

    # ---------------------------------------------------------------------- hops
    def record_hop(
        self, session_id: str, node_id: str, outcome: str, latency_ms: float | None
    ) -> int:
        """Persist the next hop; refuses a hop beyond ``max_hops``."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            count = self.get(session_id).hop_count
            if count >= self.max_hops:
                raise TransitionError(f"{session_id}: hop {count + 1} exceeds {self.max_hops}")
            self.conn.execute(
                "INSERT INTO hops VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, count, node_id, outcome, latency_ms, utc_now_iso()),
            )
            self.conn.execute(
                "UPDATE sessions SET hop_count = hop_count + 1 WHERE session_id = ?", (session_id,)
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return count

    def hops(self, session_id: str) -> list[tuple[int, str, str]]:
        rows = self.conn.execute(
            "SELECT hop_index, node_id, outcome FROM hops WHERE session_id = ? ORDER BY hop_index",
            (session_id,),
        ).fetchall()
        return [(int(i), str(n), str(o)) for i, n, o in rows]

    # ------------------------------------------------------------------- actions
    def record_action(
        self,
        session_id: str,
        action_type: str,
        impact: str,
        status: str,
        payload: dict[str, Any],
        *,
        technique_id: str | None = None,
        tactic: str | None = None,
        confidence: float | None = None,
        trace_id: str | None = None,
    ) -> tuple[str, bool]:
        """Insert an action once per (session, type); returns its id and whether it is new."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO actions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                uuid.uuid4().hex,
                session_id,
                action_type,
                impact,
                status,
                technique_id,
                tactic,
                confidence,
                json.dumps(payload, sort_keys=True),
                trace_id,
                utc_now_iso(),
            ),
        )
        row = self.conn.execute(
            "SELECT action_id FROM actions WHERE session_id = ? AND action_type = ?",
            (session_id, action_type),
        ).fetchone()
        return str(row[0]), cur.rowcount == 1

    def count(self, table: str = "sessions") -> int:
        if table not in ("sessions", "hops", "actions"):
            raise ValueError(table)
        return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # noqa: S608

    def close(self) -> None:
        self.conn.close()
