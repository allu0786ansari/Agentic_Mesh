"""Straggler tolerance: minimum clients, dropout, timeouts, degraded nodes (Plan W6)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from agmesh.federated.registry import NodeRegistry
from agmesh.federated.server import Server, ServerConfig
from agmesh.federated.strategies import Strategy
from tests.integration.fake_pool import FakePool


def _cfg(**kw: object) -> ServerConfig:
    base: dict[str, object] = {
        "site": "site_a",
        "rounds": 6,
        "strategy": Strategy("fedavg"),
        "local_epochs": 1,
        "min_fit_clients": 3,
        "round_timeout_s": 60.0,
        "seed": 7,
    }
    base.update(kw)
    return ServerConfig(**base)  # type: ignore[arg-type]


def test_rounds_complete_with_three_of_four_and_slow_node_degrades() -> None:
    pool = FakePool(["a1", "a2", "a3", "a4"], slow={"a4"})
    result = Server(pool, _cfg(), NodeRegistry()).run()
    assert all(r.aggregated for r in result.rounds)
    assert all(r.participants == ["a1", "a2", "a3"] for r in result.rounds)
    assert result.registry["a4"] == {"status": "degraded", "missed_rounds": 6}
    assert result.registry["a1"]["status"] == "active"


def test_two_of_three_on_site_b_and_skipped_rounds_keep_the_model() -> None:
    pool = FakePool(["b1", "b2", "b3"], slow={"b3"})
    ok = Server(pool, _cfg(site="site_b", min_fit_clients=2), NodeRegistry()).run()
    assert all(r.aggregated and len(r.participants) == 2 for r in ok.rounds)
    pool2 = FakePool(["b1", "b2", "b3"], slow={"b2", "b3"})
    bad = Server(pool2, _cfg(site="site_b", min_fit_clients=2, rounds=2), NodeRegistry()).run()
    assert not any(r.aggregated for r in bad.rounds)  # one reply is too few
    assert bad.rounds[0].val_loss == bad.rounds[1].val_loss  # global model unchanged


def test_dropout_is_seeded_and_recovery_clears_misses() -> None:
    runs = [
        Server(
            FakePool(["a1", "a2", "a3", "a4"]), _cfg(dropout_prob=0.3, rounds=12), NodeRegistry()
        ).run()
        for _ in range(2)
    ]
    assert [r.missed for r in runs[0].rounds] == [r.missed for r in runs[1].rounds]
    assert any(r.missed for r in runs[0].rounds)
    reg = NodeRegistry()
    reg.register("x", "site_a", "vae")
    for _ in range(2):
        reg.missed("x")
    assert reg.status("x") == ("active", 2)
    reg.missed("x")
    assert reg.status("x") == ("degraded", 3)
    reg.succeeded("x")
    assert reg.status("x") == ("active", 0)


def test_exhausted_clients_stop_and_heterogeneous_epochs_are_drawn() -> None:
    pool = FakePool(["a1", "a2", "a3", "a4"], exhausted_after={"a2": 2})
    result = Server(pool, _cfg(local_epoch_choices=(1, 2, 3)), NodeRegistry()).run()
    assert result.rounds[2].declined == ["a2"]
    assert all("a2" not in r.participants for r in result.rounds[2:])
    assert result.registry["a2"]["status"] == "exhausted"
    assert set(pool.seen_epochs) <= {1, 2, 3} and len(set(pool.seen_epochs)) > 1
