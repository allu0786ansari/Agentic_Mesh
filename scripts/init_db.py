"""
scripts/init_db.py
Day 4 — PostgreSQL schema migration.
Creates three tables: node_registry, alert_records, fl_rounds.
Also installs the pgvector extension for the alert_records.embedding column.

Run: python scripts\init_db.py
Expected output: "Schema created successfully — 3 tables"
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DSN = os.environ.get(
    "POSTGRES_DSN",
    "postgresql://agmesh:agmesh_dev@localhost:5432/agmesh"
)

SCHEMA_SQL = """
-- Enable pgvector for 128-dim alert embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- ── node_registry ────────────────────────────────────────────────
-- One row per edge node. Populated by seed_registry.py.
CREATE TABLE IF NOT EXISTS node_registry (
    node_id          VARCHAR(16)  PRIMARY KEY,
    data_type        VARCHAR(16)  NOT NULL,          -- timeseries | tabular
    model_type       VARCHAR(32)  NOT NULL,          -- vae | isolation_forest
    feature_count    INT          NOT NULL,
    epsilon_budget   FLOAT        NOT NULL DEFAULT 10.0,
    epsilon_consumed FLOAT        NOT NULL DEFAULT 0.0,
    status           VARCHAR(16)  NOT NULL DEFAULT 'inactive',
    last_seen        TIMESTAMP    WITH TIME ZONE,
    created_at       TIMESTAMP    WITH TIME ZONE DEFAULT NOW()
);

-- ── alert_records ────────────────────────────────────────────────
-- One row per anomaly alert raised by an edge node.
-- embedding column uses pgvector for future similarity search.
CREATE TABLE IF NOT EXISTS alert_records (
    alert_id             UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id              VARCHAR(16)  REFERENCES node_registry(node_id),
    timestamp            TIMESTAMP    WITH TIME ZONE NOT NULL,
    severity             FLOAT        NOT NULL,           -- reconstruction error / anomaly score
    embedding            vector(128),                     -- 128-dim Insight Embedding
    investigation_status VARCHAR(32)  NOT NULL DEFAULT 'pending',
    mitre_technique      VARCHAR(16),                     -- e.g. T0855
    mitre_tactic         VARCHAR(64),
    resolution_payload   JSONB,
    created_at           TIMESTAMP    WITH TIME ZONE DEFAULT NOW()
);

-- ── fl_rounds ────────────────────────────────────────────────────
-- One row per completed Flower FL round.
CREATE TABLE IF NOT EXISTS fl_rounds (
    round_id              SERIAL       PRIMARY KEY,
    started_at            TIMESTAMP    WITH TIME ZONE NOT NULL,
    completed_at          TIMESTAMP    WITH TIME ZONE,
    participating_nodes   INT,
    global_loss           FLOAT,
    global_auroc          FLOAT,
    epsilon_total         FLOAT,                          -- sum across all nodes
    model_version         VARCHAR(64),
    created_at            TIMESTAMP    WITH TIME ZONE DEFAULT NOW()
);

-- ── Indexes ──────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_alerts_node    ON alert_records(node_id);
CREATE INDEX IF NOT EXISTS idx_alerts_ts      ON alert_records(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_status  ON alert_records(investigation_status);
CREATE INDEX IF NOT EXISTS idx_fl_started     ON fl_rounds(started_at DESC);
"""


def run() -> None:
    logger.info(f"Connecting to PostgreSQL: {DSN.split('@')[-1]}")

    try:
        conn = psycopg2.connect(DSN)
        conn.autocommit = True
        cur = conn.cursor()

        logger.info("Running schema migration...")
        cur.execute(SCHEMA_SQL)

        # Verify tables created
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]
        expected = {"node_registry", "alert_records", "fl_rounds"}
        missing  = expected - set(tables)

        if missing:
            logger.error(f"Migration failed — missing tables: {missing}")
            sys.exit(1)

        logger.success(f"Schema created successfully — {len(tables)} tables: {', '.join(sorted(tables))}")
        cur.close()
        conn.close()

    except psycopg2.OperationalError as exc:
        logger.error(f"Cannot connect to PostgreSQL: {exc}")
        logger.error("Is the port-forward running? Run: Start-Job {{ kubectl port-forward svc/postgres-postgresql 5432:5432 -n agmesh }}")
        sys.exit(1)


if __name__ == "__main__":
    run()