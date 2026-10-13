from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from edge.tabular_loader import TabularPartition


class IsolationForestDetector:
    """Isolation Forest detector trained only on normal tabular rows."""

    def __init__(self, model: IsolationForest, scaler: StandardScaler, feature_columns: list[str]):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns

    @classmethod
    def fit(
        cls,
        partition: TabularPartition,
        *,
        seed: int = 42,
        n_estimators: int = 300,
        contamination: str | float = "auto",
        max_samples: str | int = "auto",
    ) -> "IsolationForestDetector":
        scaler = partition.fit_scaler()
        values = partition.transformed_values(scaler)
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(values)
        return cls(model, scaler, partition.feature_columns)

    def anomaly_scores(self, values: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(values).astype(np.float32)
        scores = -self.model.decision_function(transformed)
        if not np.isfinite(scores).all():
            raise ValueError("Isolation Forest produced non-finite anomaly scores")
        return scores.astype(np.float64)

    def predict(self, values: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        return (self.anomaly_scores(values) >= threshold).astype(np.int8)

    def save(self, path: str | Path, metadata: dict[str, Any]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "metadata": metadata,
            },
            destination,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> tuple["IsolationForestDetector", dict[str, Any]]:
        payload = joblib.load(path)
        detector = cls(payload["model"], payload["scaler"], payload["feature_columns"])
        return detector, payload.get("metadata", {})


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
