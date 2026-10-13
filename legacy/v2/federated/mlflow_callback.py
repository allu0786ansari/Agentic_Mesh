from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Mapping

import mlflow


class MLflowCallback:
    """Small, explicit MLflow adapter for federated round telemetry."""

    def __init__(self, tracking_uri: str | None = None, experiment: str = "agentic-mesh"):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, run_name: str = "week3-infrastructure") -> Iterator[mlflow.ActiveRun]:
        with mlflow.start_run(run_name=run_name) as active_run:
            yield active_run

    @staticmethod
    def log_round(round_number: int, metrics: Mapping[str, float]) -> None:
        if round_number < 1:
            raise ValueError("round_number must be at least 1")
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=round_number)

    @staticmethod
    def log_parameters(parameters: Mapping[str, object]) -> None:
        mlflow.log_params({key: str(value) for key, value in parameters.items()})
