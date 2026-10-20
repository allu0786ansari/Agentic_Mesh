"""--final training runs (ADR 0019 §4) on the fixture: test at validation thresholds,
one access-log entry per client and run, validation compared with the committed one."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("opacus")

from agmesh.common.config import Params
from agmesh.data import access
from agmesh.evaluation import sealed
from agmesh.evaluation.results import compare_validation, result_path
from agmesh.federated.protocol import EvalInstruction
from agmesh.federated.run import federate, record_test


@pytest.fixture(autouse=True)
def _frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sealed, "frozen_problem", lambda *a, **k: None)
    monkeypatch.setattr(sealed, "frozen_sha", lambda *a, **k: "f" * 40)
    access._FINAL_READS.clear()


def _quick(params: Params) -> Params:
    raw = params.model_dump()
    raw["federated"]["max_rounds"] = 2
    return Params.model_validate(raw)


def _log(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_federated_h1c_final_reads_each_client_once_and_writes_a_test_artifact(
    params: Params, site_a_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    root = site_a_run[1].parent
    p = _quick(params)
    log = tmp_path / "test_access.log"
    cid, rid, result = federate(
        p, "site_a", strategy="fedprox", epsilon=10.0, seed=11, partitions=root, h1c=True,
        privacy_root=tmp_path / "privacy", run_id="f1", final=True, access_log=log,
    )  # fmt: skip
    entries = _log(log)
    clients = sorted(p.data.partition.site_a.clients)
    # three test evaluations (R3, pooled H1c, per-node R0), one read per client
    assert sorted(e["client"] for e in entries) == clients
    assert {e["config_id"] for e in entries} == {f"X1/{cid}"}
    assert all(e["run_id"] == "f1" and e["frozen_sha"] == "f" * 40 for e in entries)
    test = result.extras["test"]
    assert sorted(test["final"]["per_client"]) == clients
    assert test["h1c"]["threshold"] == result.h1c["threshold"]  # fixed on validation
    out = record_test(
        p, "site_a", cid, rid, result, seed=11, strategy="fedprox", epsilon=10.0,
        partitions=root, results=tmp_path / "results", started_at="t", data_source="fixture",
        params_path=Path("configs/params.yaml"), reference=tmp_path / "none",
    )  # fmt: skip
    assert out == result_path(tmp_path / "results" / "test", "X1", cid, 11)
    doc = json.loads(out.read_text())
    assert doc["split"] == "test" and doc["evaluation"]["evaluated_on"] == "test"
    assert doc["validation"]["metrics"] == result.final["macro"]
    assert doc["validation"]["comparison"]["reference"] is None
    assert set(doc["h1c"]["scores_files"]) == {"pooled", "r0_per_node"}
    assert (out.parent / doc["scores_file"]["path"]).is_file()


@pytest.mark.parametrize("mode", ["local", "central"])
def test_baseline_modes_evaluate_every_node_on_test_once(
    mode: str, params: Params, site_b_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    root = site_b_run[1].parent
    p = _quick(params)
    log = tmp_path / "log"
    _cid, _rid, result = federate(
        p, "site_b", strategy="fedavg", epsilon=3.0, seed=11, partitions=root,
        privacy_root=tmp_path / "privacy", run_id=f"{mode}1", mode=mode, final=True,
        access_log=log,
    )  # fmt: skip
    clients = sorted(p.data.partition.site_b.clients)
    assert sorted(result.extras["test"]["final"]["per_client"]) == clients
    assert sorted(e["client"] for e in _log(log)) == clients


def test_a_validation_run_cannot_evaluate_on_test(
    params: Params, site_b_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    from agmesh.federated.client import FederatedClient

    root = site_b_run[1].parent
    c = FederatedClient(
        params, "site_b", "b1", partitions=root, run_id="v", epsilon=float("inf"), seed=11,
        node_index=0, privacy_root=tmp_path,
    )  # fmt: skip
    spec = c.spec()
    from agmesh.federated.server import initial_weights, pooled_normalisation

    norm = pooled_normalisation([c.hello()])
    ins = EvalInstruction(initial_weights(spec, 11), norm, full=True, part="test")
    with pytest.raises(sealed.SealedTestError):
        c.evaluate(ins)


def test_compare_validation_uses_the_plan_tolerance(tmp_path: Path) -> None:
    ref = result_path(tmp_path, "X1", "c", 11)
    ref.parent.mkdir(parents=True)
    priv = {"clients": {"a1": {"epsilon": 3.0}}}
    ref.write_text(json.dumps({"metrics": {"auroc": 0.9, "window_fpr": 0.01},
                               "privacy": priv, "private": True}))  # fmt: skip
    same = compare_validation(tmp_path, "X1", "c", 11, {"auroc": 0.9, "window_fpr": 0.01}, priv)
    assert same["exact"] and same["accepted"]
    near = compare_validation(tmp_path, "X1", "c", 11, {"auroc": 0.903, "window_fpr": 0.01}, priv)
    assert not near["exact"] and near["accepted"]
    far = compare_validation(tmp_path, "X1", "c", 11, {"auroc": 0.91, "window_fpr": 0.01}, priv)
    assert not far["accepted"]
    other = compare_validation(tmp_path, "X1", "c", 11, {"auroc": 0.9}, {"clients": {}})
    assert not other["accepted"]
    assert compare_validation(tmp_path, "X1", "z", 11, {}, None)["reference"] is None


def test_baselines_final_write_test_artifacts_and_log_once_per_config_and_client(
    params: Params, site_b_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    from agmesh.experiments import x1_baselines as x1

    root = site_b_run[1].parent
    log = tmp_path / "log"
    rows = x1.run(
        params, ["site_b"], ["pca", "rule_new_peer"], root, tmp_path / "res",
        data_source="fixture", seeds=[11], final=True, access_log=log,
    )  # fmt: skip
    cids = sorted(r["config_id"] for r in rows)
    assert cids == ["site_b__central__pca", "site_b__local__pca", "site_b__local__rule_new_peer"]
    entries = _log(log)
    clients = sorted(params.data.partition.site_b.clients)
    for cid in cids:
        got = sorted(e["client"] for e in entries if e["config_id"] == f"X1/{cid}")
        assert got == clients
        doc = json.loads(result_path(tmp_path / "res" / "test", "X1", cid, 11).read_text())
        assert doc["split"] == "test" and "comparison" in doc["validation"]
    assert not (tmp_path / "res" / "X1").exists()  # no validation artifact is rewritten


def test_x3_test_run_reads_each_node_once_per_seed_and_reports_pt4(
    params: Params, site_b_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    from agmesh.edge.bundle import GlobalModel
    from agmesh.experiments import x3_fusion, x3_models

    root = site_b_run[1].parent
    p = _quick(params)
    _cid, _rid, result = federate(
        p, "site_b", strategy="fedprox", epsilon=3.0, seed=11, partitions=root,
        privacy_root=tmp_path / "privacy", run_id="m1",
    )  # fmt: skip
    models = tmp_path / "models"
    GlobalModel("site_b", result.spec, result.weights, result.norm, {"seed": 11}).save(
        x3_models.model_dir("site_b", 11, models)
    )
    log = tmp_path / "log"
    rep = x3_fusion.run_test(params, models, [11], root, log)
    entries = _log(log)
    assert sorted(e["client"] for e in entries) == ["b1", "b2", "b3"]
    assert {e["config_id"] for e in entries} == {"X3/site_b__fusion"}
    assert rep["split"] == "test" and "pt4_met" in rep["pt4"]
    assert rep["per_seed"][0]["tau"]["best_node"] in ("b1", "b2", "b3")
    text = x3_fusion.test_markdown(rep)
    assert "PT4" in text and "H2b" in text
