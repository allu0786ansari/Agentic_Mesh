from pathlib import Path

import pytest

from edge.data_loader import HAIWindowDataset


@pytest.mark.skipif(
    not Path("data/partitions/node_01/train.parquet").exists(),
    reason="real Week 2 partitions are not available",
)
def test_real_hai_loader_matches_metadata() -> None:
    dataset = HAIWindowDataset(
        "data/partitions/node_01/train.parquet",
        window_size=60,
        stride=10,
        normal_only=True,
        max_windows=3,
    )

    assert len(dataset) == 3
    assert dataset.input_dim == 60 * 38
    assert dataset[0].shape == (2280,)
