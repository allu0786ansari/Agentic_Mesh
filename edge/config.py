"""
edge/config.py
Node-level configuration. Reads from environment variables and
the node's meta.json written by partition_data.py in Week 2.
All other edge modules import from here.
"""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

@dataclass
class NodeConfig:
    node_id: str
    data_type: str          # "timeseries" | "tabular"
    model_type: str         # "vae" | "isolation_forest"
    feature_count: int
    epsilon: float
    delta: float
    max_grad_norm: float
    flower_server_host: str
    nats_url: str
    kafka_bootstrap: str
    qdrant_host: str
    qdrant_port: int
    postgres_dsn: str
    mlflow_tracking_uri: str
    jwt_secret: str
    ollama_base_url: str
    phoenix_endpoint: str
    partition_dir: Path = field(init=False)
    train_path: Path = field(init=False)
    test_path: Path = field(init=False)
    meta_path: Path = field(init=False)

    def __post_init__(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.partition_dir = root / "data" / "partitions" / self.node_id
        self.train_path = self.partition_dir / "train.parquet"
        self.test_path  = self.partition_dir / "test.parquet"
        self.meta_path  = self.partition_dir / "meta.json"


def load_config() -> NodeConfig:
    required = [
        "NODE_ID", "DATA_TYPE", "FLOWER_SERVER_HOST",
        "NATS_URL", "KAFKA_BOOTSTRAP", "QDRANT_HOST", "QDRANT_PORT",
        "POSTGRES_DSN", "MLFLOW_TRACKING_URI",
        "JWT_SECRET", "OLLAMA_BASE_URL", "PHOENIX_ENDPOINT",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise ValueError(f"Missing required env vars: {missing}")

    node_id = os.environ["NODE_ID"]
    root = Path(__file__).resolve().parents[1]
    meta_path = root / "data" / "partitions" / node_id / "meta.json"

    model_type, feature_count = "vae", 0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        model_type    = meta.get("model_type", "vae")
        feature_count = meta.get("feature_count", 0)
        logger.info(f"Loaded meta.json: model={model_type}, features={feature_count}")
    else:
        logger.warning(f"meta.json not found at {meta_path}. Run partition_data.py first.")

    return NodeConfig(
        node_id             = node_id,
        data_type           = os.environ.get("DATA_TYPE", "timeseries"),
        model_type          = model_type,
        feature_count       = feature_count,
        epsilon             = float(os.environ.get("EPSILON", "2.0")),
        delta               = float(os.environ.get("DELTA", "1e-5")),
        max_grad_norm       = float(os.environ.get("MAX_GRAD_NORM", "1.0")),
        flower_server_host  = os.environ["FLOWER_SERVER_HOST"],
        nats_url            = os.environ["NATS_URL"],
        kafka_bootstrap     = os.environ["KAFKA_BOOTSTRAP"],
        qdrant_host         = os.environ["QDRANT_HOST"],
        qdrant_port         = int(os.environ.get("QDRANT_PORT", "6333")),
        postgres_dsn        = os.environ["POSTGRES_DSN"],
        mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"],
        jwt_secret          = os.environ["JWT_SECRET"],
        ollama_base_url     = os.environ["OLLAMA_BASE_URL"],
        phoenix_endpoint    = os.environ["PHOENIX_ENDPOINT"],
    )