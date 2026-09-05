from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from federated.model_loader import GlobalModelLoader
from scripts.compare_week9_strategies import compare


def test_compare_saved_strategy_summaries() -> None:
    fedprox = {"run_id": "prox", "strategy": "fedprox", "rounds_completed": 3, "evaluation_loss": [[1, 0.4], [2, 0.3], [3, 0.2]], "fit_metrics": {}}
    fedavg = {"run_id": "avg", "strategy": "fedavg", "rounds_completed": 3, "evaluation_loss": [[1, 0.5], [2, 0.4], [3, 0.35]], "fit_metrics": {}}

    result = compare(fedprox, fedavg)

    assert result["delta"]["final_loss_fedprox_minus_fedavg"] == -0.15
    assert result["fedprox"]["best_global_loss"] == 0.2


def test_global_model_loader_reads_versioned_run(tmp_path: Path) -> None:
    run = tmp_path / "week9" / "run-1"
    run.mkdir(parents=True)
    np.savez_compressed(run / "global_model.npz", parameter_0=np.asarray([1.0, 2.0]), parameter_1=np.asarray([3.0]))
    (run / "summary.json").write_text(json.dumps({"run_id": "run-1"}), encoding="utf-8")

    parameters, summary = GlobalModelLoader(tmp_path).load("run-1")

    assert len(parameters) == 2
    assert np.array_equal(parameters[0], np.asarray([1.0, 2.0]))
    assert summary["run_id"] == "run-1"
