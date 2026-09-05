from pathlib import Path

import numpy as np
import pandas as pd

from edge.models.isolation_forest import IsolationForestDetector
from edge.models.model_selector import ModelSelector
from edge.tabular_loader import TabularPartition


def make_partition(path: Path) -> None:
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, size=(80, 3))
    attack = rng.normal(6, 1, size=(20, 3))
    frame = pd.DataFrame(np.vstack([normal, attack]), columns=["a", "b", "c"])
    frame["label"] = [0] * 80 + [1] * 20
    frame.to_parquet(path)


def test_isolation_forest_scores_attacks_higher(tmp_path: Path) -> None:
    path = tmp_path / "partition.parquet"
    make_partition(path)
    partition = TabularPartition(path, normal_only=True)
    detector = IsolationForestDetector.fit(partition, seed=42, n_estimators=50)
    full = TabularPartition(path, normal_only=False, feature_columns=partition.feature_columns, scaler=detector.scaler)
    scores = detector.anomaly_scores(full.values)
    assert scores[80:].mean() > scores[:80].mean()


def test_model_selector_routes_isolation_forest(tmp_path: Path) -> None:
    node = tmp_path / "node_04"
    node.mkdir()
    (node / "meta.json").write_text('{"node_id":"node_04","model_type":"isolation_forest"}')
    selector = ModelSelector(tmp_path)
    assert selector.model_type("node_04") == "isolation_forest"
