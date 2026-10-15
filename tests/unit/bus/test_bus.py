"""Bus semantics, identical for the SQLite and in-memory buses (ADR 0009 §5)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agmesh.bus.base import Bus
from agmesh.bus.memory import MemoryBus
from agmesh.bus.sqlite_bus import SqliteBus

T = "alerts.site_b"


class FakeClock:
    def __init__(self) -> None:
        self.t = 1_700_000_000.0

    def __call__(self) -> float:
        return self.t


Factory = Callable[[FakeClock], Bus]


@pytest.fixture(params=["memory", "sqlite"])
def make(request: pytest.FixtureRequest, tmp_path: Path) -> Factory:
    if request.param == "memory":
        return lambda clock: MemoryBus(clock=clock, max_attempts=3)
    return lambda clock: SqliteBus(tmp_path / "bus.db", clock=clock, max_attempts=3)


def test_publish_is_idempotent_and_fifo(make: Factory) -> None:
    clock = FakeClock()
    bus = make(clock)
    assert bus.publish(T, {"n": 1}, "m1")
    clock.t += 0.001
    assert bus.publish(T, {"n": 2}, "m2")
    assert not bus.publish(T, {"n": 99}, "m1")  # a replayed alert is stored once
    got = bus.claim(T, "c", limit=10)
    assert [m.msg_id for m in got] == ["m1", "m2"] and got[0].payload == {"n": 1}
    assert bus.pending(T) == 2
    with pytest.raises(ValueError, match="unknown topic"):
        bus.publish("alerts.site_c", {}, "x")


def test_lease_ack_and_redelivery(make: Factory) -> None:
    clock = FakeClock()
    bus = make(clock)
    bus.publish(T, {"n": 1}, "m1")
    (m,) = bus.claim(T, "c1", lease_s=10)
    assert m.attempts == 1
    assert bus.claim(T, "c2") == []  # leased
    clock.t += 11  # c1 died without acknowledging
    (again,) = bus.claim(T, "c2", lease_s=10)
    assert again.msg_id == "m1" and again.attempts == 2
    assert not bus.ack("m1", "c1")  # the old holder no longer owns it
    assert bus.ack("m1", "c2")
    assert not bus.ack("m1", "c2")
    clock.t += 100
    assert bus.claim(T, "c3") == [] and bus.pending(T) == 0


def test_release_with_backoff_and_dead_letters(make: Factory) -> None:
    clock = FakeClock()
    bus = make(clock)
    bus.publish(T, {}, "poison")
    for attempt in (1, 2, 3):
        (m,) = bus.claim(T, "c", lease_s=5)
        assert m.attempts == attempt
        assert bus.release("poison", "c", delay_s=2)
        assert bus.claim(T, "c") == []  # not yet available
        clock.t += 2
    assert bus.claim(T, "c") == []  # max_attempts reached: dead-lettered
    assert bus.dead(T) == ["poison"] and bus.pending(T) == 0


def test_sqlite_claims_are_exclusive_across_connections(tmp_path: Path) -> None:
    clock = FakeClock()
    a = SqliteBus(tmp_path / "bus.db", clock=clock)
    b = SqliteBus(tmp_path / "bus.db", clock=clock)
    for i in range(20):
        a.publish(T, {"i": i}, f"m{i:02d}")
    seen: list[str] = []
    while True:
        batch = a.claim(T, "a", limit=3) + b.claim(T, "b", limit=3)
        if not batch:
            break
        seen += [m.msg_id for m in batch]
    assert sorted(seen) == [f"m{i:02d}" for i in range(20)]  # each exactly once
    assert a.stats(T)["attempts"] == 20
