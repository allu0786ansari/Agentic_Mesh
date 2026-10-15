"""Investigator and breaker: hop bound, retries, deadline, resume (ADR 0012 §3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agmesh.common.schemas import EvidenceRecord
from agmesh.mesh.breaker import CircuitBreaker
from agmesh.mesh.investigator import Investigator, plan_hops
from agmesh.mesh.sessions import SessionStore
from tests.mesh_doubles import FakeNodes, record

T0 = 1_675_223_500.0


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _nodes() -> dict[str, list[EvidenceRecord]]:
    return {
        n: [record(n, T0 + 10 * i, [("req_write", "up", "high")]) for i in range(3)]
        for n in ("b1", "b2", "b3")
    }


def _inv(
    tmp: Path, nodes: FakeNodes, clock: Clock, **kw: object
) -> tuple[Investigator, SessionStore, str]:
    store = SessionStore(tmp / "s.db")
    sid = store.open("site_b", ["x"], "r")[0].session_id
    inv = Investigator(nodes, store, clock=clock, sleep=clock.advance, **kw)  # type: ignore[arg-type]
    return inv, store, sid


def test_breaker_opens_after_two_failures_and_probes() -> None:
    clock, moves = Clock(), []
    br = CircuitBreaker("b1", clock=clock, on_transition=lambda n, f, t: moves.append((f, t)))
    br.failure()
    assert br.allow()
    br.failure()
    assert not br.allow()
    clock.advance(30)
    assert br.allow() and br.state == "half_open"
    br.failure()
    assert br.state == "open" and not br.allow()
    clock.advance(30)
    assert br.allow()
    br.success()
    assert moves == [
        ("closed", "open"),
        ("open", "half_open"),
        ("half_open", "open"),
        ("open", "half_open"),
        ("half_open", "closed"),
    ]


def test_plan_hops_same_site_only() -> None:
    assert plan_hops("site_b", ["b2"], 3) == ["b2", "b1", "b3"]
    assert plan_hops("site_a", ["a3", "a1", "a3"], 3) == ["a1", "a3"]
    assert plan_hops("site_b", [], 2) == ["b1", "b2"]


def test_retry_once_on_transient_never_on_error(tmp_path: Path) -> None:
    clock = Clock()
    nodes = FakeNodes(
        _nodes(), {"b1": ["transient"], "b2": ["error"], "b3": ["transient", "transient"]}
    )
    inv, store, sid = _inv(tmp_path, nodes, clock)
    res = inv.run(sid, "site_b", ["b1"], T0, T0 + 30)
    assert nodes.calls == ["b1", "b1", "b2", "b3", "b3"]
    assert [h[1] for h in res.hops] == ["ok", "error", "transient_error"]
    assert res.degraded and {r.node_id for r in res.records} == {"b1"}
    assert store.get(sid).hop_count == 3


def test_deadline_degrades_and_resume_never_exceeds_hops(tmp_path: Path) -> None:
    clock = Clock()
    nodes = FakeNodes(_nodes(), {"b1": ["slow", "slow"]}, advance=clock.advance)
    inv, store, sid = _inv(tmp_path, nodes, clock, deadline_s=1.0)
    res = inv.run(sid, "site_b", ["b1"], T0, T0 + 30)
    assert res.degraded and res.reason is not None
    again = inv.run(sid, "site_b", ["b1"], T0, T0 + 30)  # resume after a crash
    assert store.get(sid).hop_count <= 3
    assert {r.evidence_id for r in again.records} >= {r.evidence_id for r in res.records}


@settings(max_examples=60, deadline=None)
@given(
    faults=st.dictionaries(
        st.sampled_from(["b1", "b2", "b3"]),
        st.lists(st.sampled_from(["ok", "transient", "error", "slow"]), max_size=4),
    ),
    deadline=st.floats(0.1, 5.0),
    reruns=st.integers(1, 3),
)
def test_hop_count_never_exceeds_three(
    tmp_path_factory: pytest.TempPathFactory,
    faults: dict[str, list[str]],
    deadline: float,
    reruns: int,
) -> None:
    clock = Clock()
    nodes = FakeNodes(_nodes(), {k: list(v) for k, v in faults.items()}, advance=clock.advance)
    inv, store, sid = _inv(tmp_path_factory.mktemp("h"), nodes, clock, deadline_s=deadline)
    for _ in range(reruns):
        inv.run(sid, "site_b", ["b2"], T0, T0 + 30)
    assert store.get(sid).hop_count <= 3 and len(store.hops(sid)) <= 3
