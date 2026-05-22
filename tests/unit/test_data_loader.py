"""Unit tests for edge data loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edge.data_loader import WindowedDataset


ROOT = Path(__file__).resolve().parents[2]


def test_windowed_dataset_loads_node_partition() -> None:
    path = ROOT / "data" / "partitions" / "node_01" / "train.parquet"
    if not path.exists():
        pytest.skip("node_01 partition not present")

    dataset = WindowedDataset(path)
    meta = json.loads((path.parent / "meta.json").read_text())

    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == (meta["flat_feature_dim"],)
    assert int(y) in {0, 1}
