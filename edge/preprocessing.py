from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class MinMaxScalerArtifact:
    """Serializable per-feature min-max scaler fitted on normal training rows."""

    def __init__(self, feature_columns: list[str], minimum: np.ndarray, maximum: np.ndarray):
        self.feature_columns = feature_columns
        self.minimum = minimum.astype(np.float64)
        self.maximum = maximum.astype(np.float64)
        self.scale = np.where(self.maximum > self.minimum, self.maximum - self.minimum, 1.0)

    @classmethod
    def fit(cls, parquet_path: str | Path, *, normal_only: bool = True) -> "MinMaxScalerArtifact":
        frame = pd.read_parquet(parquet_path)
        feature_columns = [column for column in frame.columns if column not in {"time", "label"}]
        if normal_only:
            frame = frame.loc[frame["label"] == 0]
        values = frame[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"Cannot fit scaler from invalid data: {parquet_path}")
        return cls(feature_columns, values.min(axis=0), values.max(axis=0))

    def transform(self, values: np.ndarray, feature_columns: list[str]) -> np.ndarray:
        if feature_columns != self.feature_columns:
            raise ValueError("Scaler feature order does not match the partition feature order")
        transformed = (values.astype(np.float64) - self.minimum) / self.scale
        return np.clip(transformed, 0.0, 1.0).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "minmax_normal_training_rows",
            "feature_columns": self.feature_columns,
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["sha256"] = hashlib.sha256(encoded).hexdigest()
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxScalerArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            list(payload["feature_columns"]),
            np.asarray(payload["minimum"], dtype=np.float64),
            np.asarray(payload["maximum"], dtype=np.float64),
        )


class RobustScalerArtifact:
    """Serializable median/IQR scaler fitted on normal training rows."""

    method = "robust_normal_training_rows"

    def __init__(self, feature_columns: list[str], median: np.ndarray, scale: np.ndarray):
        self.feature_columns = feature_columns
        self.median = median.astype(np.float64)
        self.scale = np.where(scale > 1e-9, scale, 1.0).astype(np.float64)

    @classmethod
    def fit(
        cls,
        parquet_path: str | Path,
        *,
        normal_only: bool = True,
        quantile_range: tuple[float, float] = (5.0, 95.0),
    ) -> "RobustScalerArtifact":
        frame = pd.read_parquet(parquet_path)
        feature_columns = [column for column in frame.columns if column not in {"time", "label"}]
        if normal_only:
            frame = frame.loc[frame["label"] == 0]
        values = frame[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"Cannot fit robust scaler from invalid data: {parquet_path}")
        lower, upper = np.percentile(values, quantile_range, axis=0)
        return cls(feature_columns, np.median(values, axis=0), upper - lower)

    def transform(self, values: np.ndarray, feature_columns: list[str]) -> np.ndarray:
        if feature_columns != self.feature_columns:
            raise ValueError("Scaler feature order does not match the partition feature order")
        return ((values.astype(np.float64) - self.median) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "feature_columns": self.feature_columns,
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RobustScalerArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            list(payload["feature_columns"]),
            np.asarray(payload["median"], dtype=np.float64),
            np.asarray(payload["scale"], dtype=np.float64),
        )
