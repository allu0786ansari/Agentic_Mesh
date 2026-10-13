from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from edge.preprocessing import MinMaxScalerArtifact, RobustScalerArtifact


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
        enable_spectral_features: bool = False,
        spectral_bands: int = 4,
    ):
        if window_size < 1 or stride < 1:
            raise ValueError("window_size and stride must be positive")
        if not 0 <= start_fraction < end_fraction <= 1:
            raise ValueError("window fractions must satisfy 0 <= start < end <= 1")
        if spectral_bands < 1:
            raise ValueError("spectral_bands must be positive")
        frame = pd.read_parquet(path)
        if "label" not in frame.columns:
            raise ValueError(f"HAI partition must contain a label column: {path}")
        feature_columns = [column for column in frame.columns if column not in {"time", "label"}]
        if not feature_columns:
            raise ValueError(f"HAI partition has no numeric feature columns: {path}")
        values = frame[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if scaler_path is not None:
            method = json.loads(Path(scaler_path).read_text(encoding="utf-8")).get("method", "minmax")
            scaler = RobustScalerArtifact.load(scaler_path) if method == "robust_normal_training_rows" else MinMaxScalerArtifact.load(scaler_path)
            values = scaler.transform(values, feature_columns)
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
        self.enable_spectral_features = enable_spectral_features
        self.spectral_bands = spectral_bands

    @property
    def input_dim(self) -> int:
        spectral_dim = self.spectral_bands * len(self.feature_columns) if self.enable_spectral_features else 0
        return self.window_size * len(self.feature_columns) + spectral_dim

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
        flattened = window.reshape(-1)
        if self.enable_spectral_features:
            spectrum = np.abs(np.fft.rfft(window, axis=0))
            edges = np.linspace(0, spectrum.shape[0], self.spectral_bands + 1, dtype=int)
            bands = [spectrum[edges[index] : max(edges[index + 1], edges[index] + 1)].mean(axis=0) for index in range(self.spectral_bands)]
            flattened = np.concatenate([flattened, np.concatenate(bands)])
        return torch.from_numpy(flattened.astype(np.float32, copy=True))

