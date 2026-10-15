"""High-impact actions never execute without an approval (Plan Week 10 validation; ADR 0012 §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agmesh.bus.sqlite_bus import SqliteBus
from agmesh.mesh import approvals
from agmesh.mesh.attribution import Attribution
from agmesh.mesh.dispatcher import dispatch, proposed
from agmesh.mesh.sessions import SessionStore

KEY = b"0123456789abcdef-test-key"


def _att(conf: float | None = 0.9, tid: str | None = "T0806") -> Attribution:
    return Attribution(
        "full", "full", tid, ["Impair Process Control"], conf, [tid] if tid else [], [], "text"
    )


def test_policy() -> None:
    assert proposed(_att(), "high") == [
        ("operator_alert", "low"),
        ("increase_monitoring", "low"),
        ("isolate_or_block", "high"),
    ]
    assert proposed(_att(), "medium") == [("operator_alert", "low"), ("increase_monitoring", "low")]
    assert proposed(_att(None), "high") == [
        ("operator_alert", "low"),
        ("increase_monitoring", "low"),
    ]  # no confidence
    assert proposed(_att(tid=None), "high") == []
    assert all(a != "change_threshold" for a, _ in proposed(_att(), "high"))


def test_high_impact_waits_for_a_valid_approval(tmp_path: Path) -> None:
    db = tmp_path / "storage.db"
    store, bus = SessionStore(db), SqliteBus(db)
    sid = store.open("site_b", ["a"], "r")[0].session_id
    done = dispatch(store, bus, sid, "site_b", _att(), "high")
    high = next(a for a in done if a["impact"] == "high")
    assert high["status"] == "pending_approval"
    exp = 2_000_000_000
    good = approvals.sign(high["action_id"], "approved", "operator1", exp, KEY)
    bad = [
        (high["action_id"], "approved", "operator1", exp, "0" * 64),  # tampered
        (high["action_id"], "approved", "operator2", exp, good),  # another approver
        (high["action_id"], "rejected", "operator1", exp, good),  # another decision
        (
            high["action_id"],
            "approved",
            "operator1",
            1,
            approvals.sign(high["action_id"], "approved", "operator1", 1, KEY),
        ),  # expired
    ]
    for args in bad:
        with pytest.raises(approvals.ApprovalError):
            approvals.decide(store, bus, *args, key=KEY, clock=lambda: 1_900_000_000)
        assert (
            next(a for a in store.actions(sid) if a["action_id"] == high["action_id"])["status"]
            == "pending_approval"
        )
    low = next(a for a in done if a["impact"] == "low")
    with pytest.raises(approvals.ApprovalError, match="only high-impact"):
        approvals.decide(
            store,
            bus,
            low["action_id"],
            "approved",
            "op",
            exp,
            approvals.sign(low["action_id"], "approved", "op", exp, KEY),
            key=KEY,
            clock=lambda: 0,
        )
    assert (
        approvals.decide(
            store,
            bus,
            high["action_id"],
            "approved",
            "operator1",
            exp,
            good,
            key=KEY,
            clock=lambda: 0,
        )
        == "approved"
    )
    with pytest.raises(approvals.ApprovalError, match="not pending"):
        approvals.decide(
            store,
            bus,
            high["action_id"],
            "approved",
            "operator1",
            exp,
            good,
            key=KEY,
            clock=lambda: 0,
        )
    assert bus.stats("approvals")["messages"] == 1
    with pytest.raises(approvals.ApprovalError, match="not set"):
        approvals.sign("x", "approved", "op", exp, b"short")
