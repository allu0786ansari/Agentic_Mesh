"""Federated evaluation on the fixture: equals local evaluation; pooled AUROC bound holds."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("opacus")

from agmesh.common.config import Params
from agmesh.evaluation import metrics as m
from agmesh.evaluation.detection import load_node_arrays
from agmesh.federated import fed_eval
from agmesh.federated.run import federate, record
from tests.conftest import REPO_PARAMS


def _quick(params: Params) -> Params:
    raw = params.model_dump()
    raw["federated"]["max_rounds"] = 2
    return Params.model_validate(raw)


@pytest.fixture(scope="module")
def run_a(
    params: Params,
    site_a_run: tuple[dict[str, Any], Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Any:
    root = site_a_run[1].parent
    out = tmp_path_factory.mktemp("fed")
    p = _quick(params)
    cid, rid, result = federate(
        p,
        "site_a",
        strategy="fedprox",
        epsilon=3.0,
        seed=11,
        partitions=root,
        h1c=True,
        privacy_root=out / "privacy",
        run_id="t1",
    )
    return p, root, out, cid, rid, result


def test_federated_eval_matches_local_eval_of_the_same_model(run_a: Any) -> None:
    from agmesh.federated.client import FederatedClient
    from agmesh.federated.protocol import EvalInstruction

    p, root, out, _cid, _rid, result = run_a
    for i, reply in enumerate(result.final_replies):
        client = FederatedClient(
            p,
            "site_a",
            reply.node,
            partitions=root,
            run_id="other",
            epsilon=math.inf,
            seed=11,
            node_index=i,
            privacy_root=out / "privacy",
        )
        again = client.evaluate(EvalInstruction(result.weights, result.norm, full=True))
        for key in ("auroc", "auprc", "window_fpr", "event_recall"):
            assert again.metrics[key] == pytest.approx(reply.metrics[key], nan_ok=True)


@pytest.fixture(scope="module")
def run_b(
    params: Params,
    site_b_run: tuple[dict[str, Any], Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Any:
    out = tmp_path_factory.mktemp("fedb")
    return federate(
        _quick(params),
        "site_b",
        strategy="fedavg",
        epsilon=10.0,
        seed=11,
        partitions=site_b_run[1].parent,
        privacy_root=out / "privacy",
        run_id="tb",
    )[2]


def test_pooled_auroc_is_within_its_bin_error(run_b: Any) -> None:
    result = run_b
    scores, labels = [], []
    for reply in result.final_replies:
        arrays = load_node_arrays(reply.arrays)[reply.node]
        keep = arrays.scored | arrays.labels  # abstained attacks rank lowest
        ranked = np.where(arrays.scored, arrays.scores, -1e30)
        scores.append(ranked[keep])
        labels.append(arrays.labels[keep])
    y, s = np.concatenate(labels), np.concatenate(scores)
    assert 0 < y.sum() < y.size  # both classes present on the Site B fixture
    exact = m.auroc(y, s)
    est, bound = result.final["pooled_auroc_hist"], result.final["pooled_auroc_bin_error"]
    assert abs(exact - est) <= bound + 1e-9


def test_artifact_privacy_and_h1c(run_a: Any) -> None:
    import json

    p, root, out, cid, rid, result = run_a
    assert cid == "site_a__fedprox__eps3__h1c"
    for client in result.privacy.values():
        assert client["epsilon"] <= 3.0 + 1e-9
        names = [h["name"] for h in client["history"]]
        assert names == ["normalisation", "dpsgd", "threshold_histogram"]
    assert result.h1c is not None and result.h1c["released_by"] == ["a1", "a2", "a3", "a4"]
    path = record(
        p,
        "site_a",
        cid,
        rid,
        result,
        seed=11,
        strategy="fedprox",
        epsilon=3.0,
        partitions=root,
        results=out,
        smoke=False,
        started_at="2026-10-15T00:00:00Z",
        data_source="fixture",
        params_path=REPO_PARAMS,
    )
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["config_id"] == cid and doc["federation"]["rounds"] == 2
    assert path.with_name("seed_11.scores.npz").is_file()


def test_fed_eval_helpers() -> None:
    normal = np.zeros(fed_eval.GRID_BINS)
    attack = np.zeros(fed_eval.GRID_BINS)
    normal[100], attack[900] = 10, 5
    assert fed_eval.pooled_auroc(normal, attack) == (1.0, 0.0)
    attack[100] = 5
    auroc, bound = fed_eval.pooled_auroc(normal, attack)
    assert auroc == pytest.approx(0.75) and bound == pytest.approx(0.25)
    assert fed_eval.coarsen(np.ones(1000), 100).tolist() == [10.0] * 100
    assert fed_eval.bonferroni_cut(100) == pytest.approx(3.29, abs=0.01)
    scores = np.concatenate([np.linspace(0, 1, 990), np.linspace(5, 6, 10)])
    hist = fed_eval.coarsen(fed_eval.bin_scores(scores), 100)
    tau, _info = fed_eval.h1c_threshold(hist, noise_sd=0.0, target_fpr=0.01)
    assert tau >= 1.0 and (scores > tau).mean() <= 0.01
    assert fed_eval.inverse(float(fed_eval.transform(42.0))) == pytest.approx(42.0)


def test_same_seed_same_federation(
    params: Params, site_b_run: tuple[dict[str, Any], Path], tmp_path: Path, run_b: Any
) -> None:
    again = federate(
        _quick(params),
        "site_b",
        strategy="fedavg",
        epsilon=10.0,
        seed=11,
        partitions=site_b_run[1].parent,
        privacy_root=tmp_path / "privacy",
        run_id="tb2",
    )[2]
    for w1, w2 in zip(run_b.weights, again.weights, strict=True):
        np.testing.assert_array_equal(w1, w2)
    assert again.final["macro"] == run_b.final["macro"]
    assert [r.val_loss for r in again.rounds] == [r.val_loss for r in run_b.rounds]
