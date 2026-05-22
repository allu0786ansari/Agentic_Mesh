r"""
scripts/seed_registry.py
Day 5 - Seed PostgreSQL node_registry from generated partition metadata.

Run: python scripts\seed_registry.py
Expected output: "Seeded 5 node_registry rows"
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
PART_DIR = ROOT / "data" / "partitions"
DSN = os.environ.get(
    "POSTGRES_DSN",
    "postgresql://agmesh:agmesh_dev@127.0.0.1:15432/agmesh",
)
DEFAULT_EPSILON_BUDGET = float(os.environ.get("TOTAL_EPSILON_BUDGET", "10.0"))


@dataclass(frozen=True)
class NodeRegistryRow:
    """A validated node_registry row derived from a partition meta.json file."""

    node_id: str
    data_type: str
    model_type: str
    feature_count: int
    epsilon_budget: float = DEFAULT_EPSILON_BUDGET
    epsilon_consumed: float = 0.0
    status: str = "ready"


def load_node_meta(meta_path: Path) -> NodeRegistryRow:
    """Load and validate one partition meta.json file."""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {meta_path}: {exc}") from exc

    required = {"node_id", "data_type", "model_type", "feature_count"}
    missing = required - set(meta)
    if missing:
        raise ValueError(f"{meta_path} missing required keys: {sorted(missing)}")

    node_id = str(meta["node_id"])
    data_type = str(meta["data_type"])
    model_type = str(meta["model_type"])
    feature_count = int(meta["feature_count"])

    if data_type not in {"timeseries", "tabular"}:
        raise ValueError(f"{meta_path}: unsupported data_type={data_type!r}")
    if model_type not in {"vae", "isolation_forest"}:
        raise ValueError(f"{meta_path}: unsupported model_type={model_type!r}")
    if feature_count <= 0:
        raise ValueError(f"{meta_path}: feature_count must be positive")

    return NodeRegistryRow(
        node_id=node_id,
        data_type=data_type,
        model_type=model_type,
        feature_count=feature_count,
    )


def discover_node_rows(partition_dir: Path = PART_DIR) -> list[NodeRegistryRow]:
    """Discover all node partition metadata files in deterministic order."""
    meta_paths = sorted(partition_dir.glob("node_*/meta.json"))
    if not meta_paths:
        raise FileNotFoundError(
            f"No node metadata found under {partition_dir}. Run scripts/partition_data.py first."
        )

    rows = [load_node_meta(path) for path in meta_paths]
    node_ids = [row.node_id for row in rows]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"Duplicate node IDs detected: {node_ids}")

    return rows


def ensure_node_registry_exists(cursor) -> None:
    """Fail clearly if init_db.py has not created the registry table."""
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'node_registry'
        )
        """
    )
    exists = bool(cursor.fetchone()[0])
    if not exists:
        raise RuntimeError("node_registry table not found. Run scripts/init_db.py first.")


def upsert_rows(rows: list[NodeRegistryRow]) -> int:
    """Upsert node rows into PostgreSQL and return the row count."""
    logger.info("Connecting to PostgreSQL: {}", DSN.split("@")[-1])
    conn = psycopg2.connect(DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                ensure_node_registry_exists(cur)
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO node_registry (
                            node_id,
                            data_type,
                            model_type,
                            feature_count,
                            epsilon_budget,
                            epsilon_consumed,
                            status,
                            last_seen
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (node_id) DO UPDATE SET
                            data_type = EXCLUDED.data_type,
                            model_type = EXCLUDED.model_type,
                            feature_count = EXCLUDED.feature_count,
                            epsilon_budget = EXCLUDED.epsilon_budget,
                            status = EXCLUDED.status,
                            last_seen = NOW()
                        """,
                        (
                            row.node_id,
                            row.data_type,
                            row.model_type,
                            row.feature_count,
                            row.epsilon_budget,
                            row.epsilon_consumed,
                            row.status,
                        ),
                    )
        return len(rows)
    finally:
        conn.close()


def run() -> None:
    """Seed PostgreSQL node_registry from partition metadata."""
    try:
        rows = discover_node_rows()
        count = upsert_rows(rows)
    except psycopg2.OperationalError as exc:
        logger.error("Cannot connect to PostgreSQL: {}", exc)
        logger.error(
            "Check POSTGRES_DSN and ensure the PostgreSQL port-forward or container is running."
        )
        sys.exit(1)
    except (psycopg2.Error, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Failed to seed node_registry: {}", exc)
        sys.exit(1)

    logger.success("Seeded {} node_registry rows", count)
    for row in rows:
        logger.info(
            "{}: data_type={} model_type={} feature_count={} status={}",
            row.node_id,
            row.data_type,
            row.model_type,
            row.feature_count,
            row.status,
        )


if __name__ == "__main__":
    run()
