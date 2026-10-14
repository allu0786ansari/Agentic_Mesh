"""ε continues across rounds; clients are recreated; a spent client declines (P2, F8)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("opacus")

from agmesh.common.config import Params
from agmesh.federated.backends import ClientSettings, InProcessPool
from agmesh.federated.registry import NodeRegistry
from agmesh.federated.server import Server, ServerConfig
from agmesh.federated.strategies import Strategy
from agmesh.privacy.accountant_store import AccountantStore, store_path


def _server(
    params: Params, root: Path, tmp: Path, *, rounds: int, plan_epochs: int, run_id: str
) -> tuple[Server, InProcessPool]:
    nodes = ["b1", "b2", "b3"]
    pool = InProcessPool(
        ClientSettings(
            params, "site_b", root, run_id, 3.0, 11, epochs=plan_epochs, privacy_root=tmp
        ),
        nodes,
    )
    cfg = ServerConfig("site_b", rounds, Strategy("fedavg"), 1, 2, 1e9, 11)
    return Server(pool, cfg, NodeRegistry()), pool


def test_epsilon_accumulates_across_rounds_and_clients_are_recreated(
    params: Params, site_b_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    root = site_b_run[1].parent
    server, pool = _server(params, root, tmp_path, rounds=3, plan_epochs=3, run_id="r1")
    result = server.run()
    assert all(r.aggregated for r in result.rounds)
    assert all(n >= 3 + 3 for n in pool.calls.values())  # a new client object per call
    for node in pool.nodes:
        path = store_path("r1", node, tmp_path)
        store = AccountantStore(
            path, node=node, delta=result.privacy[node]["delta"], target_epsilon=3.0
        )
        assert store.steps("normalisation") == 1
        assert store.steps("dpsgd") == result.privacy[node]["plan"]["steps"]
        assert store.epsilon() <= 3.0 + 1e-9 and store.epsilon() >= 2.99


def test_a_client_past_its_plan_declines(
    params: Params, site_b_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    root = site_b_run[1].parent
    server, _pool = _server(params, root, tmp_path, rounds=3, plan_epochs=1, run_id="r2")
    result = server.run()
    assert result.rounds[0].aggregated
    assert result.rounds[1].declined == ["b1", "b2", "b3"] and not result.rounds[1].aggregated
    assert all(v["status"] == "exhausted" for v in result.registry.values())
    assert all(p["epsilon"] <= 3.0 + 1e-9 for p in result.privacy.values())
    assert not math.isnan(result.rounds[2].val_loss)
