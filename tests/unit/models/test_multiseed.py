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
    assert (results / "X1" / "site_b__local__ae" / "seed_11.scores.npz").is_file()
    args = ["X1", "--results", str(results), "--params", str(REPO_PARAMS), "--resamples", "200"]
    assert summarise.main(args) == 0
    doc = json.loads((results / "X1" / "summary_val.json").read_text(encoding="utf-8"))
    assert doc["data_source"] == "fixture"
    ev = doc["event_level"]
    assert ev["missing_score_files"] == [] and set(ev["configs"]) == set(runs)
    fpr = ev["configs"]["site_b__local__ae"]["window_fpr"]
    assert fpr["observed"] == pytest.approx(ae["window_fpr"]["mean"])
    (test,) = ev["comparisons"]
    assert (
        test["comparison"] == "AE vs PCA FPR, Site B local"
        and test["p_holm"] >= test["p_one_sided"]
    )


def test_shift_selection_runner(
    params: Params, site_a_run: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from agmesh.experiments import shift_selection

    out = tmp_path / "shift.json"
    report = shift_selection.run(
        _quick(params),
        ["site_a"],
        site_a_run[1].parent,
        [11],
        out,
        data_source="fixture",
        params_path=REPO_PARAMS,
    )
    site = report["site_a"]
    assert set(site["summary"]) == {"vae", "pca", "iforest"}
    assert set(site["summary"]["vae"]) == {"r0", "r1", "r2", "r3"}
    assert site["decision"]["chosen"] in {"r0", "r1", "r2", "r3"}
    assert site["summary"]["vae"]["r0"]["abstained_share"] == 0.0
    assert json.loads(out.read_text(encoding="utf-8"))["kind"] == "X1.shift_selection"


def test_shift_rule() -> None:
    from agmesh.experiments.shift_selection import select

    def row(fpr: float, recall: float = 1.0, auprc: float = 0.7) -> dict[str, float | None]:
        return {"window_fpr": fpr, "event_recall": recall, "auprc": auprc}

    base = {"r0": row(0.30), "r1": row(0.10), "r2": row(0.05, recall=0.9), "r3": row(0.095)}
    decision = select(base)
    assert decision["chosen"] == "r1"  # r2 loses events; r3 ties r1 within 0.01, r1 simpler
    base["r3"] = row(0.05, auprc=0.69)
    assert select(base)["chosen"] == "r3"
    base["r3"] = row(0.05, auprc=0.60)  # AUPRC guard
    assert select(base)["chosen"] == "r1"


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
