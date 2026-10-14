"""SecAgg+ (Flower) aggregates the uniform mean within one quantisation step (Plan W7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("flwr")
pytest.importorskip("opacus")

from agmesh.common.config import Params
from agmesh.experiments import x8_secagg

pytestmark = pytest.mark.fl


def _params_file(params: Params, tmp_path: Path) -> Path:
    import yaml

    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(json.loads(params.model_dump_json())), encoding="utf-8")
    return path


def test_secagg_matches_plaintext_and_survives_a_dropout(
    params: Params, site_a_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    root = site_a_run[1].parent
    pfile = _params_file(params, tmp_path)
    arms = {
        arm: x8_secagg.run_arm(
            params, pfile, root, arm=arm, rounds=2, seed=11, privacy_root=tmp_path / "privacy"
        )
        for arm in ("plaintext", "secagg", "secagg_drop")
    }
    step = arms["secagg"]["quantisation_step"]
    for row in arms["secagg"]["rounds"]:
        assert row["aggregated"] and row["updates"] == 4
        assert row["max_abs_error"] <= step and row["outside_clip_share"] == 0.0
    for row in arms["plaintext"]["rounds"]:
        assert row["max_abs_error"] < 1e-6  # plaintext FedAvg is the exact mean
    drop = arms["secagg_drop"]["rounds"][1]
    assert drop["aggregated"] and drop["updates"] == 3 and drop["max_abs_error"] <= step
    metrics = x8_secagg.summarise(arms)
    assert metrics["error_within_step"] == 1.0 and metrics["dropout_recovered"] == 1.0
    assert metrics["bytes_overhead_ratio"] and metrics["bytes_overhead_ratio"] > 1.0
