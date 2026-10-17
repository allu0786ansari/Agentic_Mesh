"""Pipeline-matched baselines, X2 output and H1c score files (ADR 0016), on the fixture."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("opacus")

from agmesh.common.config import Params
from agmesh.evaluation.detection import load_node_arrays
from agmesh.federated.client import FederatedClient
from agmesh.federated.run import federate, partition_suffix, record
from agmesh.privacy.units import n_training_units


def _quick(params: Params) -> Params:
    raw = params.model_dump()
    raw["federated"]["max_rounds"] = 2
    return Params.model_validate(raw)


def test_central_trains_one_pooled_client_and_every_node_evaluates(
    params: Params, site_a_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    root = site_a_run[1].parent
    p = _quick(params)
    cid, _rid, result = federate(
        p, "site_a", strategy="fedavg", epsilon=3.0, seed=11, partitions=root,
        privacy_root=tmp_path / "privacy", run_id="c1", mode="central",
    )  # fmt: skip
    assert cid == "site_a__central__vae__eps3"
    clients = list(p.data.partition.site_a.clients)
    assert list(result.privacy) == ["central"]
    assert 0 < result.privacy["central"]["epsilon"] <= 3.0
    assert sorted(result.final["per_client"]) == sorted(clients)
    assert all(r.participants == ["central"] for r in result.rounds)
    pooled = FederatedClient(
        p, "site_a", "central", partitions=root, run_id="x", epsilon=math.inf, seed=11,
        node_index=4, privacy_root=tmp_path / "privacy", train_nodes=clients,
    )  # fmt: skip
    from agmesh.data.access import load_partition

    per_node = sum(
        n_training_units("site_a", load_partition(p, "site_a", c, "train", root=root), p)
        for c in clients
    )
    assert pooled.n_windows == per_node
    with pytest.raises(RuntimeError, match="does not evaluate"):
        pooled.evaluate(None)  # type: ignore[arg-type]


def test_local_trains_each_node_alone(
    params: Params, site_b_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    root = site_b_run[1].parent
    p = _quick(params)
    cid, _rid, result = federate(
        p, "site_b", strategy="fedavg", epsilon=3.0, seed=11, partitions=root,
        privacy_root=tmp_path / "privacy", run_id="l1", mode="local",
    )  # fmt: skip
    clients = sorted(p.data.partition.site_b.clients)
    assert cid == "site_b__local__ae__eps3"
    assert sorted(result.privacy) == clients
    assert all(0 < v["epsilon"] <= 3.0 for v in result.privacy.values())
    assert sorted(result.final["per_client"]) == clients
    # each node trained alone: every round lists each node once, as its own federation
    assert all(sorted(r.participants) == clients for r in result.rounds)
    # a node alone gets the same model as a one-node federation with its own index
    alone = federate(
        p, "site_b", strategy="fedavg", epsilon=3.0, seed=11, partitions=root,
        privacy_root=tmp_path / "privacy2", run_id="l2", mode="local",
    )[2]  # fmt: skip
    for n in clients:
        assert alone.final["per_client"][n]["auroc"] == pytest.approx(
            result.final["per_client"][n]["auroc"], nan_ok=True
        )


def test_modes_refuse_what_they_do_not_support(
    params: Params, site_b_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    root = site_b_run[1].parent
    for kw in ({"h1c": True}, {"heterogeneous": True}, {"backend": "flower"}):
        with pytest.raises(ValueError, match="baselines"):
            federate(
                params, "site_b", strategy="fedavg", epsilon=3.0, seed=11, partitions=root,
                privacy_root=tmp_path, mode="central", **kw,
            )  # fmt: skip


def test_x2_artifacts_and_h1c_score_files(
    params: Params, site_a_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    root = site_a_run[1].parent
    p = _quick(params)
    cid, rid, result = federate(
        p, "site_a", strategy="fedprox", epsilon=10.0, seed=11, partitions=root,
        h1c=True, privacy_root=tmp_path / "privacy", run_id="h1",
    )  # fmt: skip
    path = record(
        p, "site_a", cid, rid, result, seed=11, strategy="fedprox", epsilon=10.0,
        partitions=root, results=tmp_path / "results", smoke=False, started_at="t",
        data_source="fixture", params_path=Path("configs/params.yaml"), experiment="X2",
    )  # fmt: skip
    assert path.parts[-3] == "X2"
    doc = json.loads(path.read_text())
    files = doc["h1c"]["scores_files"]
    assert sorted(files) == ["pooled", "r0_per_node"]
    for f in files.values():
        with np.load(path.with_name(f["path"]), allow_pickle=False) as data:
            arrays = load_node_arrays(data)
        assert sorted(arrays) == sorted(p.data.partition.site_a.clients)


def test_iid_partitions_get_their_own_config_id(tmp_path: Path) -> None:
    (tmp_path / "site_a").mkdir()
    (tmp_path / "site_a" / "manifest.json").write_text(json.dumps({"scheme": "iid"}))
    assert partition_suffix(tmp_path, "site_a") == "__iid"
    (tmp_path / "site_a" / "manifest.json").write_text(json.dumps({"scheme": "time"}))
    assert partition_suffix(tmp_path, "site_a") == ""
    assert partition_suffix(tmp_path, "site_b") == ""
