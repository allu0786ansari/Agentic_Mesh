"""At-least-once delivery across processes (Plan Week 8 validation; ADR 0009 §5-6).

A consumer process claims an alert, opens its session (the handler's commit), and is
killed before it acknowledges. After the lease expires, a second consumer gets the
same message again. Opening the session again is a no-op, because the idempotency
key is the same. The alert is therefore processed once and acknowledged once.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from agmesh.bus.sqlite_bus import SqliteBus
from agmesh.mesh.sessions import SessionStore

TOPIC = "alerts.site_b"
LEASE_S = 0.6


def _consumer(db: str, name: str, die_before_ack: bool) -> None:
    bus = SqliteBus(Path(db))
    store = SessionStore(Path(db))
    deadline = time.time() + 10
    while time.time() < deadline:
        msgs = bus.claim(TOPIC, name, lease_s=LEASE_S)
        if not msgs:
            time.sleep(0.05)
            continue
        for m in msgs:
            store.open(m.payload["site_id"], [m.payload["alert_id"]], "test")  # handler commits
            if die_before_ack:
                os._exit(17)  # killed between commit and ack
            bus.ack(m.msg_id, name)
        return


def _spawn(db: Path, name: str, *, die_before_ack: bool) -> subprocess.CompletedProcess[str]:
    """Run a consumer in a separate interpreter (no multiprocessing helpers left behind)."""
    code = (
        "import sys; from tests.integration.test_bus_redelivery import _consumer; "
        "_consumer(sys.argv[1], sys.argv[2], sys.argv[3] == '1')"
    )
    return subprocess.run(
        [sys.executable, "-c", code, str(db), name, "1" if die_before_ack else "0"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_killed_consumer_redelivers_and_is_processed_once(tmp_path: Path) -> None:
    db = tmp_path / "storage.db"
    bus = SqliteBus(db)
    SessionStore(db)  # create the tables before the children start
    alert = {"alert_id": "f" * 32, "site_id": "site_b"}
    assert bus.publish(TOPIC, alert, alert["alert_id"])
    assert not bus.publish(TOPIC, alert, alert["alert_id"])  # duplicate replay ignored

    first = _spawn(db, "c1", die_before_ack=True)
    assert first.returncode == 17, first.stderr
    assert bus.pending(TOPIC) == 1  # not acknowledged

    second = _spawn(db, "c2", die_before_ack=False)
    assert second.returncode == 0, second.stderr

    stats = bus.stats(TOPIC)
    assert stats == {"messages": 1, "acked": 1, "dead": 0, "attempts": 2}
    store = SessionStore(db)
    assert store.count() == 1  # processed once thanks to the idempotency key
    (session,) = store.open_sessions()
    assert store.alerts(session.session_id) == [alert["alert_id"]]
