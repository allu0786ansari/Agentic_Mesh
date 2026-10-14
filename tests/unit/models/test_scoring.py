"""ADR 0003 scoring: error normalisation, top-k aggregation, causal smoothing, selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agmesh.common.config import Params
from agmesh.models.inputs import ModelInput
from agmesh.models.scoring import (
    GRID,
    ErrorReference,
    Scoring,
    aggregate,
    causal_smooth,
    window_scores,
)
from tests.conftest import REPO_PARAMS


def _input(n: int, features: int = 4, segments: int = 2) -> ModelInput:
    return ModelInput(
        x=np.zeros((n, features * segments), dtype=np.float32),
        window_start=np.arange(n, dtype=np.float64) * 10.0,
        window_length_s=10.0,
        stride_s=10.0,
        label=np.zeros(n, dtype=bool),
        base_features=tuple(f"f{i}" for i in range(features)),
        segments=segments,
    )


def test_aggregations_by_hand() -> None:
    inp = _input(1)
    errors = np.array(
        [[1, 1, 1, 9, 1, 1, 1, 9]], dtype=np.float32
    )  # feature f3 errs 9 in both segments
    ref = ErrorReference(np.ones(4), np.ones(4))
    assert aggregate(errors, inp, Scoring("mean"), ref).tolist() == [3.0]
    assert aggregate(errors, inp, Scoring("norm_mean"), ref).tolist() == [2.0]  # (0+0+0+8)/4
    assert aggregate(errors, inp, Scoring("norm_top3", top_k=1), ref).tolist() == [8.0]
    with pytest.raises(ValueError, match="ErrorReference"):
        aggregate(errors, inp, Scoring("norm_mean"), None)
    fitted = ErrorReference.fit(np.array([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]]))
    assert fitted.median.tolist() == [2.0, 1.0] and fitted.iqr[1] > 0


def test_causal_smoothing_never_crosses_gaps_or_looks_ahead() -> None:
    scores = np.array([0.0, 3.0, 6.0, 9.0, 100.0])
    starts = np.array([0.0, 10.0, 20.0, 30.0, 500.0])  # the last window follows a gap
    out = causal_smooth(scores, starts, 10.0, 3)
    assert out.tolist() == [0.0, 1.5, 3.0, 6.0, 100.0]
    shuffled = np.array([3, 0, 4, 1, 2])
    assert (
        causal_smooth(scores[shuffled], starts[shuffled], 10.0, 3).tolist()
        == out[shuffled].tolist()
    )
    assert causal_smooth(scores, starts, 10.0, 1).tolist() == scores.tolist()
    inp = _input(5)
    errs = np.ones((5, 8), dtype=np.float32)
    assert window_scores(errs, inp, Scoring("mean", 3), None).tolist() == [1.0] * 5


def test_grid_and_selection_rule() -> None:
    from agmesh.experiments.scoring_selection import select

    assert len(GRID) == 9 and len({s.name for s in GRID}) == 9
    rows = [
        {
            "variant": "mean__s1",
            "aggregation": "mean",
            "smooth": 1,
            "criterion": 0.900,
            "mean_fpr": 0.05,
        },
        {
            "variant": "norm_top3__s1",
            "aggregation": "norm_top3",
            "smooth": 1,
            "criterion": 0.903,
            "mean_fpr": 0.04,
        },
        {
            "variant": "norm_mean__s3",
            "aggregation": "norm_mean",
            "smooth": 3,
            "criterion": 0.850,
            "mean_fpr": 0.01,
        },
    ]
    assert select(rows)["variant"] == "norm_top3__s1"  # tie on AUPRC, lower FPR wins
    rows[1]["mean_fpr"] = 0.05
    assert select(rows)["variant"] == "mean__s1"  # full tie: the simpler variant wins


def test_selection_runner_on_fixture(
    params: Params, site_a_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from agmesh.experiments import scoring_selection

    raw = params.model_dump()
    raw["models"]["vae"]["max_epochs"] = 2
    raw["models"]["ae"]["max_epochs"] = 2
    out = tmp_path / "sel.json"
    report = scoring_selection.run(
        Params.model_validate(raw),
        ["site_a"],
        site_a_run[1].parent,
        out,
        data_source="fixture",
        params_path=REPO_PARAMS,
    )
    assert report["site_a"]["chosen"] in {s.name for s in GRID}
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["data_source"] == "fixture" and len(doc["sites"]["site_a"]["variants"]) == 9


def test_stability_check_on_fixture(
    params: Params, site_a_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from agmesh.experiments import scoring_selection

    raw = params.model_dump()
    raw["models"]["vae"]["max_epochs"] = 2
    out = tmp_path / "stability.json"
    report = scoring_selection.stability(
        Params.model_validate(raw),
        ["site_a"],
        site_a_run[1].parent,
        [11, 23],
        out,
        data_source="fixture",
        params_path=REPO_PARAMS,
    )
    site = report["site_a"]
    assert site["configured"] == "mean__s1" and site["seeds"] == 2
    assert 0 <= site["seeds_agreeing"] <= 2 and set(site["per_seed"]) == {"11", "23"}
    assert site["chosen_on_mean_criterion"] in {s.name for s in GRID}
    assert len(site["mean_over_seeds"]) == 9
    assert json.loads(out.read_text(encoding="utf-8"))["kind"] == "X1.scoring_stability"
