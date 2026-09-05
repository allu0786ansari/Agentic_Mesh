from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from edge.preprocessing import MinMaxScalerArtifact


class HAIWindowDataset(Dataset[torch.Tensor]):
    """Lazy sliding-window dataset over one generated HAI node partition."""

    def __init__(
        self,
        path: str | Path,
        window_size: int = 60,
        stride: int = 10,
        normal_only: bool = True,
        max_windows: int | None = None,
        start_fraction: float = 0.0,
        end_fraction: float = 1.0,
        scaler_path: str | Path | None = None,
    ):
        if window_size < 1 or stride < 1:
            raise ValueError("window_size and stride must be positive")
        if not 0 <= start_fraction < end_fraction <= 1:
            raise ValueError("window fractions must satisfy 0 <= start < end <= 1")
        frame = pd.read_parquet(path)
        if "label" not in frame.columns:
            raise ValueError(f"HAI partition must contain a label column: {path}")
        feature_columns = [column for column in frame.columns if column not in {"time", "label"}]
        if not feature_columns:
            raise ValueError(f"HAI partition has no numeric feature columns: {path}")
        values = frame[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if scaler_path is not None:
            values = MinMaxScalerArtifact.load(scaler_path).transform(values, feature_columns)
        labels = frame["label"].to_numpy(dtype=np.int8)
        starts = np.arange(0, len(values) - window_size + 1, stride, dtype=np.int64)
        if normal_only:
            starts = starts[np.array([labels[start : start + window_size].max() == 0 for start in starts])]
        lower = int(len(starts) * start_fraction)
        upper = int(len(starts) * end_fraction)
        starts = starts[lower:upper]
        if max_windows is not None:
            if max_windows < 1:
                raise ValueError("max_windows must be positive when provided")
            starts = starts[:max_windows]
        if len(starts) == 0:
            raise ValueError(f"No usable windows found in {path}")
        self.values = values
        self.labels = labels
        self.starts = starts
        self.window_size = window_size
        self.feature_columns = feature_columns

    @property
    def input_dim(self) -> int:
        return self.window_size * len(self.feature_columns)

    def __len__(self) -> int:
        return len(self.starts)

    @property
    def window_labels(self) -> np.ndarray:
        """Return one anomaly label per window (1 if any row is anomalous)."""
        return np.asarray([self.labels[start : start + self.window_size].max() for start in self.starts], dtype=np.int8)

    @property
    def window_starts(self) -> np.ndarray:
        """Return source-row offsets for temporal and overlap diagnostics."""
        return self.starts.copy()

    def __getitem__(self, index: int) -> torch.Tensor:
        start = int(self.starts[index])
        window = self.values[start : start + self.window_size]
        return torch.from_numpy(window.reshape(-1).copy())

