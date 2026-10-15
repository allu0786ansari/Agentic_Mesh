"""Session store: idempotent opening, transitions, hop limit, unique actions (ADR 0009 §6)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agmesh.mesh.sessions import SessionStore, TransitionError, idempotency_key


def test_idempotency_key_is_site_plus_sorted_alerts() -> None:
    key = idempotency_key("site_b", ["z9", "a1", "a1"])
    assert key == hashlib.sha256(b"site_b|a1,z9").hexdigest()
    assert key == idempotency_key("site_b", ["a1", "z9"])
    assert key != idempotency_key("site_a", ["a1", "z9"])
    with pytest.raises(ValueError):
        idempotency_key("site_b", [])


def test_open_twice_returns_the_same_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "storage.db")
    s1, created1 = store.open("site_b", ["a2", "a1"], "run")
    s2, created2 = SessionStore(tmp_path / "storage.db").open("site_b", ["a1", "a2"], "run")
    assert created1 and not created2 and s1.session_id == s2.session_id
    assert store.count() == 1 and store.alerts(s1.session_id) == ["a1", "a2"]
    assert s1.status == "OPEN" and s1.closed_at is None


def test_transitions_and_terminal_states(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "storage.db")
    s, _ = store.open("site_a", ["x"], "run")
    with pytest.raises(TransitionError):
        store.transition(s.session_id, "COMPLETED")  # must pass through the agents
    for to in ("INVESTIGATING", "ATTRIBUTING", "DISPATCHING", "COMPLETED_DEGRADED"):
        s = store.transition(s.session_id, to)
    assert s.closed_at is not None and store.open_sessions() == []
    with pytest.raises(TransitionError):
        store.transition(s.session_id, "FAILED")  # terminal is final
    t, _ = store.open("site_a", ["y"], "run")
    store.transition(t.session_id, "INVESTIGATING")
    assert store.transition(t.session_id, "ABSTAINED").status == "ABSTAINED"
    u, _ = store.open("site_a", ["z"], "run")
    assert [o.session_id for o in store.open_sessions()] == [u.session_id]


def test_hop_limit_and_unique_actions(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "storage.db", max_hops=3)
    s, _ = store.open("site_b", ["x"], "run")
    for i, node in enumerate(("b1", "b2", "b3")):
        assert store.record_hop(s.session_id, node, "ok", 12.0) == i
    with pytest.raises(TransitionError):
        store.record_hop(s.session_id, "b1", "ok", 1.0)
    assert store.get(s.session_id).hop_count == 3 and len(store.hops(s.session_id)) == 3
    a1, new1 = store.record_action(s.session_id, "notify_operator", "low", "dispatched", {"m": 1})
    a2, new2 = store.record_action(s.session_id, "notify_operator", "low", "dispatched", {"m": 2})
    assert new1 and not new2 and a1 == a2 and store.count("actions") == 1
