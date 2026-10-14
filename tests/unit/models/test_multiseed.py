"""Multi-seed X1 runs, the seed summary, and the ADR 0004 training-budget rule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agmesh.common.config import Params
from agmesh.experiments import summarise, training_budget
from tests.conftest import REPO_PARAMS


def _quick(params: Params, epochs: int = 2) -> Params:
    raw = params.model_dump()
    raw["models"]["vae"]["max_epochs"] = epochs
    raw["models"]["ae"]["max_epochs"] = epochs
    return Params.model_validate(raw)


def test_budget_rule() -> None:
    assert training_budget.capped({"best_epoch": 95}, 100, 10)
    assert not training_budget.capped({"best_epoch": 80}, 100, 10)
    rows = [
        {"max_epochs": 100, "capped_share": 0.30},
        {"max_epochs": 200, "capped_share": 0.04},
        {"max_epochs": 400, "capped_share": 0.0},
    ]
    assert training_budget.choose(rows) == 200
    assert training_budget.choose([{"max_epochs": 30, "capped_share": 0.5}]) == 30


def test_multi_seed_run_and_summary(
    params: Params, site_b_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from agmesh.experiments import x1_baselines

    results = tmp_path / "results"
    rows = x1_baselines.run(
        _quick(params),
        ["site_b"],
        ["ae", "pca"],
        site_b_run[1].parent,
        results,
        data_source="fixture",
        params_path=REPO_PARAMS,
        seeds=[11, 23, 37],
    )
    assert {r["seed"] for r in rows} == {11, 23, 37} and len(rows) == 12
    assert len(list((results / "X1" / "site_b__local__ae").glob("seed_*.json"))) == 3
    runs = summarise.load(results, "X1", "val")
    table = summarise.summarise(runs)
    ae = table["site_b__local__ae"]
    assert ae["seeds"] == [11, 23, 37] and len(ae["window_fpr"]["values"]) == 3
    assert ae["window_fpr"]["low"] <= ae["window_fpr"]["mean"] <= ae["window_fpr"]["high"]
    comparisons = summarise.compare(runs)
    assert [c["comparison"] for c in comparisons] == ["AE vs PCA FPR, Site B local"]
    assert "| site_b__local__ae | 3 |" in summarise.markdown(table, comparisons)
    assert summarise.main(["X1", "--results", str(results), "--params", str(REPO_PARAMS)]) == 0
    doc = json.loads((results / "X1" / "summary_val.json").read_text(encoding="utf-8"))
    assert doc["data_source"] == "fixture"


def test_training_budget_runner(
    params: Params,
    site_b_run: tuple[dict[str, Any], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setitem(training_budget.BUDGETS, "site_b", (2, 4))
    out = tmp_path / "budget.json"
    report = training_budget.run(
        params,
        ["site_b"],
        site_b_run[1].parent,
        [11],
        out,
        data_source="fixture",
        params_path=REPO_PARAMS,
    )
    rows = report["site_b"]["budgets"]
    assert [r["max_epochs"] for r in rows] == [2, 4] and len(
        rows[0]["runs"]
    ) == 4  # 3 local + 1 central
    assert report["site_b"]["chosen_max_epochs"] in (2, 4)
    assert json.loads(out.read_text(encoding="utf-8"))["kind"] == "X1.training_budget"
