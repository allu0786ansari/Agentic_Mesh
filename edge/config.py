from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Config:
    node_id: str = "node_01"
    data_type: str = "timeseries"
    epsilon: float = 2.0
    delta: float = 1e-5
    flower_num_clients: int = 5
    qdrant_path: str = "./qdrant_data"
    sqlite_path: str = "./agmesh.db"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            node_id=os.getenv("NODE_ID", "node_01"),
            data_type=os.getenv("DATA_TYPE", "timeseries"),
            epsilon=float(os.getenv("EPSILON", "2.0")),
            delta=float(os.getenv("DELTA", "1e-5")),
            flower_num_clients=int(os.getenv("FLOWER_NUM_CLIENTS", "5")),
            qdrant_path=os.getenv("QDRANT_PATH", "./qdrant_data"),
            sqlite_path=os.getenv("SQLITE_PATH", "./agmesh.db"),
        )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent
