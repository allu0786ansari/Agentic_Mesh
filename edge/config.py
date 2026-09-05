from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    node_id: str = "node_01"
    data_type: str = "timeseries"
    epsilon: float = 2.0
    delta: float = 1e-5
    seed: int = 42
    secure_rng: bool = False
    flower_num_clients: int = 6
    qdrant_path: str = "./qdrant_data"
    sqlite_path: str = "./agmesh.db"
    hai_window_size: int = 60
    hai_window_stride: int = 10
    latent_dim: int = 32
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    local_epochs: int = 50
    batch_size: int = 64
    max_grad_norm: float = 1.0
    checkpoint_root: str = "./checkpoints"
    metrics_log_path: str = "./logs/metrics.jsonl"
    run_manifest_path: str = "./results/week4_training.json"
    scaler_root: str = "./results/scalers"

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        return cls(
            node_id=os.getenv("NODE_ID", "node_01"),
            data_type=os.getenv("DATA_TYPE", "timeseries"),
            epsilon=float(os.getenv("EPSILON", "2.0")),
            delta=float(os.getenv("DELTA", "1e-5")),
            seed=int(os.getenv("RANDOM_SEED", "42")),
            secure_rng=os.getenv("SECURE_RNG", "false").lower() in {"1", "true", "yes"},
            flower_num_clients=int(os.getenv("FLOWER_NUM_CLIENTS", "6")),
            qdrant_path=os.getenv("QDRANT_PATH", "./qdrant_data"),
            sqlite_path=os.getenv("SQLITE_PATH", "./agmesh.db"),
            hai_window_size=int(os.getenv("HAI_WINDOW_SIZE", "60")),
            hai_window_stride=int(os.getenv("HAI_WINDOW_STRIDE", "10")),
            latent_dim=int(os.getenv("VAE_LATENT_DIM", "32")),
            hidden_dim=int(os.getenv("VAE_HIDDEN_DIM", "128")),
            learning_rate=float(os.getenv("LEARNING_RATE", "1e-3")),
            local_epochs=int(os.getenv("LOCAL_EPOCHS", "50")),
            batch_size=int(os.getenv("BATCH_SIZE", "64")),
            max_grad_norm=float(os.getenv("MAX_GRAD_NORM", "1.0")),
            checkpoint_root=os.getenv("CHECKPOINT_ROOT", "./checkpoints"),
            metrics_log_path=os.getenv("METRICS_LOG_PATH", "./logs/metrics.jsonl"),
            run_manifest_path=os.getenv("RUN_MANIFEST_PATH", "./results/week4_training.json"),
            scaler_root=os.getenv("SCALER_ROOT", "./results/scalers"),
        )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent
