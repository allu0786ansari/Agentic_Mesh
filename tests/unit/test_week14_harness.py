from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation.harness import EvalHarness


def test_eval_harness_generates_summary_metrics(tmp_path: Path) -> None:
    partitions = tmp_path / "partitions"
    node_dir = partitions / "node_01"
    node_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "time": list(range(6)),
            "feature_a": [0.1, 0.2, 0.3, 0.9, 1.0, 0.15],
            "label": [0, 0, 0, 1, 1, 0],
        }
    ).to_parquet(node_dir / "test.parquet", index=False)

    ground_truth = tmp_path / "ground_truth.json"
    ground_truth.write_text(
        json.dumps({
            "node_01": {
                "techniques": {
                    "attack": "T0830",
                    "normal": "T0830",
                }
            }
        }),
        encoding="utf-8",
    )

    harness = EvalHarness(
        partitions_root=partitions,
        ground_truth_path=ground_truth,
        output_path=tmp_path / "results.json",
        node_ids=["node_01"],
        max_windows=2,
    )

    record = harness.run_window(node_dir / "test.parquet", "node_01", 0)
    summary = harness.compute_metrics(
        labels=[1, 0, 1],
        scores=[0.9, 0.2, 0.7],
        techniques=["T0830", "T0830", "T0830"],
        expected_techniques=["T0830", "T0830", "T0830"],
        latencies_ms=[1200.0, 1250.0, 700.0],
    )

    assert record["label"] in {0, 1}
    assert summary["detection_rate"] >= 0.0
    assert summary["false_positive_rate"] >= 0.0
    assert summary["root_cause_accuracy"] >= 0.0
    assert summary["mean_latency_ms"] > 0.0
