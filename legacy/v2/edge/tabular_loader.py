from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TabularPartition:
    """Validated tabular partition with a normal-training-fitted scaler."""

    def __init__(
        self,
        path: str | Path,
        *,
        feature_columns: list[str] | None = None,
        scaler: StandardScaler | None = None,
        normal_only: bool = False,
    ):
        self.path = Path(path)
        frame = pd.read_parquet(self.path)
        if "label" not in frame.columns:
            raise ValueError(f"Partition must contain label: {self.path}")
        self.feature_columns = feature_columns or [c for c in frame.columns if c != "label"]
        if not self.feature_columns or any(c not in frame.columns for c in self.feature_columns):
            raise ValueError(f"Invalid feature columns for {self.path}")
        if normal_only:
            frame = frame.loc[frame["label"] == 0]
        values = frame[self.feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        labels = frame["label"].to_numpy(dtype=np.int8)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"Partition contains invalid numeric values: {self.path}")
        self.values = values
        self.labels = labels
        self.scaler = scaler

    def fit_scaler(self) -> StandardScaler:
        self.scaler = StandardScaler().fit(self.values)
        return self.scaler

    def transformed_values(self, scaler: StandardScaler | None = None) -> np.ndarray:
        active_scaler = scaler or self.scaler
        if active_scaler is None:
            raise ValueError("Fit or provide a scaler before transforming values")
        transformed = active_scaler.transform(self.values).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("Scaler produced non-finite values")
        return transformed

    def __len__(self) -> int:
        return len(self.values)
