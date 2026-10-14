"""The Flower backend gives the same federation as the in-process backend (needs 'fl')."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("flwr")
pytest.importorskip("opacus")

from agmesh.common.config import Params
from agmesh.federated import flower
from agmesh.federated import protocol as pr
from agmesh.federated.run import federate
from tests.conftest import REPO_PARAMS

pytestmark = pytest.mark.fl


def test_wire_format_round_trips_without_pickle() -> None:
    reply = pr.EvalReply(
        "b1",
        0.25,
        {"auroc": 0.9, "ttd_median_s": float("nan"), "events": 3, "none": None},
        np.arange(5, dtype=np.float64),
        None,
        {"nodes": np.array(["b1"]), "b1__scores": np.ones(3, np.float32)},
    )
    back = flower.unpack(flower.pack(reply))
    assert back.node == "b1" and back.val_loss == 0.25 and back.metrics["events"] == 3
    assert math.isnan(back.metrics["ttd_median_s"]) and back.metrics["none"] is None
    np.testing.assert_array_equal(back.normal_hist, reply.normal_hist)
    assert back.arrays["b1__scores"].dtype == np.float32
    ins = pr.FitInstruction(
        2, [np.zeros((2, 2), np.float32)], pr.Normalisation(np.zeros(2), np.ones(2)), 1, 0.01, 5
    )
    again = flower.unpack(flower.pack(ins))
    assert again.round == 2 and again.norm.scale.tolist() == [1.0, 1.0]


def test_flower_equals_inprocess(
    params: Params, site_b_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    raw = params.model_dump()
    raw["federated"]["max_rounds"] = 1
    p = Params.model_validate(raw)
    root = site_b_run[1].parent
    import json

    import yaml

    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(json.loads(p.model_dump_json())), encoding="utf-8")
    runs = [
        federate(
            p,
            "site_b",
            strategy="fedavg",
            epsilon=3.0,
            seed=11,
            partitions=root,
            privacy_root=tmp_path / b,
            run_id="x",
            backend=b,
            params_path=params_file,
        )[2]
        for b in ("inprocess", "flower")
    ]
    for w1, w2 in zip(runs[0].weights, runs[1].weights, strict=True):
        np.testing.assert_array_equal(w1, w2)
    assert runs[0].final["macro"] == runs[1].final["macro"]
    assert {n: c["epsilon"] for n, c in runs[0].privacy.items()} == {
        n: c["epsilon"] for n, c in runs[1].privacy.items()
    }
    assert REPO_PARAMS.is_file()
