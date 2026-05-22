r"""
tests/unit/test_week2_data.py
Week 2 gate — 12 tests confirming datasets are processed, partitioned correctly,
and the PostgreSQL schema is populated.

Run: pytest tests\unit\test_week2_data.py -v
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PART_DIR     = PROJECT_ROOT / "data" / "partitions"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
load_dotenv(PROJECT_ROOT / ".env")


# ── Processed files exist ────────────────────────────────────────────────────

def test_hai_combined_parquet_exists():
    p = PROC_DIR / "hai_combined.parquet"
    assert p.exists(), "data/processed/hai_combined.parquet missing — run process_datasets.py"

def test_modbus_combined_parquet_exists():
    p = PROC_DIR / "modbus2023_combined.parquet"
    assert p.exists(), "data/processed/modbus2023_combined.parquet missing — run process_datasets.py"


# ── Partition files exist ────────────────────────────────────────────────────

NODES = ["node_01", "node_02", "node_03", "node_04", "node_05"]

@pytest.mark.parametrize("node", NODES)
def test_node_train_parquet_exists(node):
    assert (PART_DIR / node / "train.parquet").exists(), f"{node}/train.parquet missing"

@pytest.mark.parametrize("node", NODES)
def test_node_test_parquet_exists(node):
    assert (PART_DIR / node / "test.parquet").exists(), f"{node}/test.parquet missing"

@pytest.mark.parametrize("node", NODES)
def test_node_meta_json_exists(node):
    assert (PART_DIR / node / "meta.json").exists(), f"{node}/meta.json missing"


# ── meta.json content ────────────────────────────────────────────────────────

def test_vae_nodes_meta_correct():
    """Nodes 01-03 must declare vae model type and timeseries data."""
    for node in ["node_01", "node_02", "node_03"]:
        meta = json.loads((PART_DIR / node / "meta.json").read_text())
        assert meta["model_type"]  == "vae",        f"{node}: expected model_type=vae"
        assert meta["data_type"]   == "timeseries", f"{node}: expected data_type=timeseries"
        assert meta["feature_count"] > 0,           f"{node}: feature_count must be > 0"

def test_hai_subsystem_prefixes_are_not_mixed():
    """HAI VAE nodes must map to physical subsystem prefixes."""
    expected = {
        "node_01": {"p1"},
        "node_02": {"p2"},
        "node_03": {"p3", "p4"},
    }
    for node, prefixes in expected.items():
        meta = json.loads((PART_DIR / node / "meta.json").read_text())
        actual = {col.split("_")[0] for col in meta["sensor_columns"]}
        assert actual == prefixes, f"{node}: expected {prefixes}, got {actual}"

def test_isoforest_nodes_meta_correct():
    """Nodes 04-05 must declare isolation_forest model type and tabular data."""
    for node in ["node_04", "node_05"]:
        meta = json.loads((PART_DIR / node / "meta.json").read_text())
        assert meta["model_type"] == "isolation_forest", f"{node}: expected model_type=isolation_forest"
        assert meta["data_type"]  == "tabular",          f"{node}: expected data_type=tabular"
        assert meta["feature_count"] > 0,                f"{node}: feature_count must be > 0"


# ── Parquet readability and shape ────────────────────────────────────────────

def test_all_parquet_files_readable():
    """All 10 parquet files (5 nodes × train+test) must load without error."""
    for node in NODES:
        for split in ["train", "test"]:
            path = PART_DIR / node / f"{split}.parquet"
            df = pd.read_parquet(path)
            assert len(df) > 0,      f"{node}/{split}.parquet is empty"
            assert "label" in df.columns, f"{node}/{split}.parquet missing 'label' column"


# ── PostgreSQL node_registry ─────────────────────────────────────────────────

def test_postgres_node_registry_populated():
    """
    Connects to PostgreSQL and verifies all 5 nodes are in node_registry.
    Skipped if PostgreSQL port-forward is not running (CI environment).
    """
    pytest.importorskip("psycopg2")
    import os
    import psycopg2

    dsn = os.environ.get("POSTGRES_DSN", "postgresql://agmesh:agmesh_dev@localhost:5432/agmesh")
    try:
        conn = psycopg2.connect(dsn, connect_timeout=3)
    except Exception:
        pytest.skip("PostgreSQL not reachable — ensure port-forward is running")

    cur = conn.cursor()
    cur.execute("SELECT node_id FROM node_registry ORDER BY node_id")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()

    assert len(rows) == 5, f"Expected 5 rows in node_registry, got {len(rows)}: {rows}"
    assert rows == sorted(NODES), f"Node IDs mismatch: {rows}"
