"""X6 chaos scenarios (ADR 0017 §6; EA1): every session terminal, no duplicate dispatch."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from tests.chaos.conftest import T0, Stack, incident, make_rig, terminal_and_unique
from tests.mesh_doubles import alert_of, record


def _hops(rig: Any, sid: str) -> list[tuple[str, str]]:
    return [(n, o) for _i, n, o in rig.store.hops(sid)]


def test_node_down_before_a_hop(stack: Stack, chaos_report: dict[str, Any]) -> None:
    stack.start()
    stack.kill("b2")
    rig = make_rig(stack, stack.root / "storage.db")
    state = rig.mesh.run("site_b", incident(T0 + 600), "retrieval_only", "x6")
    hops = _hops(rig, state["session_id"])
    terminal, unique = terminal_and_unique(rig.store)
    ok = state["status"] == "COMPLETED_DEGRADED" and ("b2", "transient_error") in hops
    chaos_report["node_down_before_hop"] = {
        "status": state["status"],
        "hops": hops,
        "terminal": terminal,
        "unique_dispatch": unique,
        "passed": ok and terminal and unique,
    }
    assert ok and terminal and unique


def test_node_killed_during_a_hop(stack: Stack, chaos_report: dict[str, Any]) -> None:
    stack.start({"b2": 0.4})  # b2 answers after 0.4 s, inside the 0.5 s read timeout
    rig = make_rig(stack, stack.root / "storage.db")
    killer = threading.Timer(0.15, stack.kill, args=("b2",))
    killer.start()
    state = rig.mesh.run("site_b", incident(T0 + 600), "retrieval_only", "x6")
    killer.join()
    hops = _hops(rig, state["session_id"])
    terminal, unique = terminal_and_unique(rig.store)
    ok = state["status"] == "COMPLETED_DEGRADED" and hops[1][0] == "b2" and hops[1][1] != "ok"
    chaos_report["node_killed_during_hop"] = {
        "status": state["status"],
        "hops": hops,
        "terminal": terminal,
        "unique_dispatch": unique,
        "passed": ok and terminal and unique,
    }
    assert ok and terminal and unique


def test_node_killed_after_a_hop_opens_its_breaker(
    stack: Stack, chaos_report: dict[str, Any]
) -> None:
    stack.start()
    rig = make_rig(stack, stack.root / "storage.db")
    first = rig.mesh.run("site_b", incident(T0 + 300), "retrieval_only", "x6")
    stack.kill("b2")
    later = [
        rig.mesh.run("site_b", incident(T0 + 900 + 600 * i), "retrieval_only", "x6")
        for i in range(3)
    ]
    terminal, unique = terminal_and_unique(rig.store)
    outcomes = [dict(_hops(rig, s["session_id"])).get("b2") for s in later]
    ok = (
        first["status"] == "COMPLETED"
        and all(s["status"] == "COMPLETED_DEGRADED" for s in later)
        and ("b2", "closed", "open") in rig.transitions
        and outcomes[-1] == "breaker_open"
    )
    chaos_report["node_killed_after_hop"] = {
        "statuses": [first["status"], *[s["status"] for s in later]],
        "b2_outcomes": outcomes,
        "breaker_transitions": rig.transitions,
        "terminal": terminal,
        "unique_dispatch": unique,
        "passed": ok and terminal and unique,
    }
    assert ok and terminal and unique


def test_slow_node_times_out_within_the_deadline(
    stack: Stack, chaos_report: dict[str, Any]
) -> None:
    stack.start({"b3": 1.0})  # above the 0.5 s read timeout
    rig = make_rig(stack, stack.root / "storage.db")
    started = time.monotonic()
    state = rig.mesh.run("site_b", incident(T0 + 600), "retrieval_only", "x6")
    seconds = time.monotonic() - started
    hops = dict(_hops(rig, state["session_id"]))
    terminal, unique = terminal_and_unique(rig.store)
    ok = state["status"] == "COMPLETED_DEGRADED" and hops.get("b3") in (
        "transient_error",
        "deadline",
    )
    ok = ok and seconds < 3.0 + 1.5  # the 3 s investigation deadline, plus the session's own work
    chaos_report["slow_node"] = {
        "status": state["status"],
        "hops": hops,
        "session_seconds": round(seconds, 2),
        "terminal": terminal,
        "unique_dispatch": unique,
        "passed": ok and terminal and unique,
    }
    assert ok and terminal and unique


def test_mesh_killed_mid_session_resumes_from_sqlite(
    stack: Stack, chaos_report: dict[str, Any]
) -> None:
    stack.start({"b1": 0.3})  # sessions take long enough to be interrupted
    db = stack.root / "storage.db"
    rig = make_rig(stack, db)
    n_incidents = 8
    for i in range(n_incidents):  # 8 incidents, 10 min apart
        r = record("b1", T0 + 600 * i, [("req_write", "up", "high")])
        a = alert_of(r)
        rig.bus.publish("alerts.site_b", a, a["alert_id"])
    cmd = [sys.executable, "-m", "tests.chaos.worker", str(db), str(stack.stack_yaml)]
    first = subprocess.Popen(cmd, cwd=Path.cwd())
    time.sleep(1.6)
    first.kill()
    first.wait()
    interrupted = rig.store.count()
    time.sleep(1.2)  # the killed worker's 1 s leases expire
    second = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True, timeout=120)
    terminal, unique = terminal_and_unique(rig.store)
    sessions = rig.store.count()
    ok = (
        second.returncode == 0
        and sessions == n_incidents
        and rig.bus.pending("alerts.site_b") == 0
        and 0 < interrupted < n_incidents
    )
    chaos_report["mesh_killed_and_restarted"] = {
        "sessions_before_kill": interrupted,
        "sessions_after_restart": sessions,
        "incidents": n_incidents,
        "terminal": terminal,
        "unique_dispatch": unique,
        "passed": ok and terminal and unique,
    }
    assert ok and terminal and unique, second.stdout + second.stderr


def test_duplicate_alert_replay_dispatches_once(stack: Stack, chaos_report: dict[str, Any]) -> None:
    stack.start()
    db = stack.root / "storage.db"
    rig = make_rig(stack, db)
    from agmesh.mesh.runner import consume
    from agmesh.mesh.triage import Triage

    alerts = [alert_of(record("b1", T0 + 10 * i, [("req_write", "up", "high")])) for i in range(3)]
    for a in alerts:
        rig.bus.publish("alerts.site_b", a, a["alert_id"])
        rig.bus.publish("alerts.site_b", a, a["alert_id"])  # the same message again
    consume(rig.mesh, rig.bus, Triage(60.0), "site_b", "retrieval_only", "x6")
    actions = rig.store.count("actions")
    for a in alerts:  # the whole stream replayed under new message keys
        rig.bus.publish("alerts.site_b", a, a["alert_id"] + "-replay")
    consume(rig.mesh, rig.bus, Triage(60.0), "site_b", "retrieval_only", "x6")
    terminal, unique = terminal_and_unique(rig.store)
    ok = rig.store.count() == 1 and rig.store.count("actions") == actions
    chaos_report["duplicate_alert_replay"] = {
        "sessions": rig.store.count(),
        "actions": actions,
        "terminal": terminal,
        "unique_dispatch": unique,
        "passed": ok and terminal and unique,
    }
    assert ok and terminal and unique
