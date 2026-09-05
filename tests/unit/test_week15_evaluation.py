from __future__ import annotations

from pathlib import Path

from evaluation.chaos_test import ChaosTest
from evaluation.plot_metrics import generate_charts
from evaluation.ragas_eval import build_ragas_report


def test_chaos_test_passes_all_scenarios(tmp_path: Path) -> None:
    chaos = ChaosTest(database_path=tmp_path / "chaos.db")
    report = chaos.run_all(timeout_seconds=2.5)

    assert report["passed"] is True
    assert len(report["scenarios"]) == 4
    assert all(item["passed"] for item in report["scenarios"].values())


def test_ragas_report_reaches_quality_targets() -> None:
    report = build_ragas_report()

    assert report["mean_context_recall"] >= 0.85
    assert report["mean_context_precision"] >= 0.80
    assert report["n_queries"] >= 1


def test_metric_dashboards_are_generated(tmp_path: Path) -> None:
    output_dir = tmp_path / "architecture"
    outputs = generate_charts(output_dir=output_dir)

    assert len(outputs) == 4
    assert all(Path(path).exists() for path in outputs)
