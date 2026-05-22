"""Dataset helpers for edge-node training and evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class WindowedDataset(Dataset[tuple[Tensor, Tensor]]):
    """Loads flattened time-series windows from a node parquet partition."""

    def __init__(self, parquet_path: str | Path) -> None:
        self.path = Path(parquet_path)
        df = pd.read_parquet(self.path)
        if "label" not in df.columns:
            raise ValueError(f"{self.path} must contain a label column")

        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        if not feature_cols:
            feature_cols = [col for col in df.columns if col != "label"]

        self.features = torch.tensor(df[feature_cols].to_numpy(dtype="float32"))
        self.labels = torch.tensor(df["label"].to_numpy(dtype="int64"))

    def __len__(self) -> int:
        """Return the number of available windows."""
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return one flattened window and its binary label."""
        return self.features[index], self.labels[index]

    @property
    def input_dim(self) -> int:
        """Return the flattened feature dimension."""
        return int(self.features.shape[1])


class TabularDataset(WindowedDataset):
    """Loads tabular feature rows from a node parquet partition."""
