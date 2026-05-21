"""
scripts/partition_data.py
Day 3 — Partition processed datasets into 5 Non-IID node allocations.

Node assignments (Non-IID — each node sees a genuinely different distribution):
  node_01: HAI 22.04 Boiler subsystem     → timeseries, VAE
  node_02: HAI 22.04 Turbine subsystem    → timeseries, VAE
  node_03: HAI 22.04 Water treatment      → timeseries, VAE
  node_04: CIC Modbus 2023 attack traffic → tabular, IsoForest
  node_05: CIC Modbus 2023 benign+mixed   → tabular, IsoForest

Each node partition gets: train.parquet, test.parquet, meta.json

Run: python scripts\partition_data.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

ROOT      = Path(__file__).resolve().parents[1]
PROC_DIR  = ROOT / "data" / "processed"
PART_DIR  = ROOT / "data" / "partitions"

WINDOW_SIZE  = 60   # seconds (1 Hz sensor data)
WINDOW_STRIDE = 10  # stride between windows
TEST_RATIO   = 0.2
RANDOM_SEED  = 42


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def make_windows(df: pd.DataFrame, sensor_cols: list[str],
                 window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a time-series DataFrame into sliding windows.
    Returns (X, y) where:
      X shape: (n_windows, window_size * n_sensors)  — flattened for Parquet storage
      y shape: (n_windows,)  — 1 if ANY row in window is an attack
    """
    data   = df[sensor_cols].values.astype(np.float32)
    labels = df["attack"].values.astype(int)
    n      = len(data)

    X_list, y_list = [], []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window = data[start:end]            # (window_size, n_sensors)
        label  = int(labels[start:end].any())
        X_list.append(window.flatten())    # flatten to 1D for Parquet
        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)


def save_partition(node_id: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   meta: dict) -> None:
    out_dir = PART_DIR / node_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build column names: feature_0 ... feature_N, label
    n_features = X_train.shape[1]
    feat_cols  = [f"feature_{i}" for i in range(n_features)]

    train_df = pd.DataFrame(X_train, columns=feat_cols)
    train_df["label"] = y_train
    test_df  = pd.DataFrame(X_test,  columns=feat_cols)
    test_df["label"]  = y_test

    train_df.to_parquet(out_dir / "train.parquet", index=False)
    test_df.to_parquet(out_dir  / "test.parquet",  index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    logger.success(
        f"{node_id}: train={train_df.shape}  test={test_df.shape}  "
        f"model={meta['model_type']}  features={meta['feature_count']}"
    )


# ══════════════════════════════════════════════════════════════════
# HAI 22.04 — VAE Nodes (01, 02, 03)
# ══════════════════════════════════════════════════════════════════

def partition_hai() -> None:
    logger.info("=== Partitioning HAI 22.04 → nodes 01, 02, 03 ===")

    combined_path = PROC_DIR / "hai_combined.parquet"
    if not combined_path.exists():
        logger.error("hai_combined.parquet not found — run process_datasets.py first")
        sys.exit(1)

    df = pd.read_parquet(combined_path)
    df.columns = df.columns.str.strip().str.lower()

    # Ensure attack column is binary int
    if "attack" not in df.columns:
        logger.error("No 'attack' column in HAI parquet — check process_datasets.py")
        sys.exit(1)
    df["attack"] = df["attack"].fillna(0).astype(int)

    # Identify sensor columns
    exclude = {"timestamp", "attack", "attack_p1", "attack_p2", "attack_p3",
               "_source_file", "label"}
    all_sensor_cols = [c for c in df.columns if c not in exclude]
    n_sensors = len(all_sensor_cols)
    logger.info(f"Total sensor columns: {n_sensors}")

    # ── HAI 22.04 subsystem split strategy ───────────────────────────
    # HAI 22.04 organises sensors as P1 (boiler), P2 (turbine), P3 (water treatment).
    # Column naming convention varies by HAI version — split evenly if not labelled.
    thirds = n_sensors // 3
    subsystems = {
        "node_01": all_sensor_cols[:thirds],           # Boiler
        "node_02": all_sensor_cols[thirds:2*thirds],   # Turbine
        "node_03": all_sensor_cols[2*thirds:],         # Water treatment
    }

    for node_id, sensor_cols in subsystems.items():
        logger.info(f"{node_id}: {len(sensor_cols)} sensors")
        X, y = make_windows(df, sensor_cols, WINDOW_SIZE, WINDOW_STRIDE)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y
        )
        attack_rate = float(y.mean())
        meta = {
            "node_id":          node_id,
            "data_type":        "timeseries",
            "model_type":       "vae",
            "feature_count":    len(sensor_cols),
            "window_size":      WINDOW_SIZE,
            "window_stride":    WINDOW_STRIDE,
            "flat_feature_dim": WINDOW_SIZE * len(sensor_cols),
            "sensor_columns":   sensor_cols,
            "attack_rate":      round(attack_rate, 4),
            "train_windows":    int(len(X_train)),
            "test_windows":     int(len(X_test)),
            "class_distribution": {
                "normal": int((y == 0).sum()),
                "attack": int((y == 1).sum()),
            },
        }
        save_partition(node_id, X_train, y_train, X_test, y_test, meta)


# ══════════════════════════════════════════════════════════════════
# CIC Modbus 2023 — IsoForest Nodes (04, 05)
# ══════════════════════════════════════════════════════════════════

def partition_modbus() -> None:
    logger.info("=== Partitioning CIC Modbus 2023 → nodes 04, 05 ===")

    modbus_path = PROC_DIR / "modbus2023_combined.parquet"
    if not modbus_path.exists():
        logger.error("modbus2023_combined.parquet not found — run process_datasets.py first")
        sys.exit(1)

    df = pd.read_parquet(modbus_path)
    df.columns = df.columns.str.strip().str.lower()

    # Find label column
    label_candidates = [c for c in df.columns if "label" in c or "class" in c]
    label_col = label_candidates[0] if label_candidates else "label"
    logger.info(f"Label column: '{label_col}'  values: {df[label_col].unique()[:8]}")

    # ── Node 04: Attack-heavy traffic ────────────────────────────────
    # reconnaissance, query flooding, false data injection
    attack_keywords = ["recon", "flood", "inject", "fdi", "query", "scan",
                       "reconnaissance", "false_data", "false data"]
    def is_attack_traffic(label: str) -> bool:
        label_l = str(label).lower()
        return any(kw in label_l for kw in attack_keywords)

    mask_04 = df[label_col].apply(is_attack_traffic)
    benign_mask = df[label_col].str.lower().str.contains("benign|normal|legitimate", na=False)

    # Node 04: attack traffic + some benign (for non-IID imbalance)
    df_04 = pd.concat([
        df[mask_04],
        df[benign_mask].sample(frac=0.3, random_state=RANDOM_SEED),
    ]).reset_index(drop=True)

    # Node 05: benign + remaining attack types (brute force, replay)
    remaining_mask = ~mask_04
    df_05 = df[remaining_mask].reset_index(drop=True)

    # If split produced empty nodes, fall back to random 50/50
    if len(df_04) < 100 or len(df_05) < 100:
        logger.warning("Attack keyword split produced small partitions — falling back to 50/50 split")
        mid = len(df) // 2
        df_04 = df.iloc[:mid].reset_index(drop=True)
        df_05 = df.iloc[mid:].reset_index(drop=True)

    logger.info(f"node_04 rows: {len(df_04):,}  |  node_05 rows: {len(df_05):,}")

    # Feature columns (all numeric except label)
    feature_cols = [c for c in df.columns
                    if c != label_col and c != "_source_file"
                    and df[c].dtype in [np.float32, np.float64, int, np.int64]]

    for node_id, node_df in [("node_04", df_04), ("node_05", df_05)]:
        X = node_df[feature_cols].values.astype(np.float32)
        y = (~node_df[label_col].str.lower().str.contains(
            "benign|normal|legitimate", na=False)).astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_RATIO, random_state=RANDOM_SEED,
            stratify=y if y.sum() > 1 else None
        )
        meta = {
            "node_id":          node_id,
            "data_type":        "tabular",
            "model_type":       "isolation_forest",
            "feature_count":    len(feature_cols),
            "feature_columns":  feature_cols,
            "label_column":     label_col,
            "attack_rate":      round(float(y.mean()), 4),
            "train_rows":       int(len(X_train)),
            "test_rows":        int(len(X_test)),
            "class_distribution": {
                "normal": int((y == 0).sum()),
                "attack": int((y == 1).sum()),
            },
        }
        save_partition(node_id, X_train, y_train, X_test, y_test, meta)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    partition_hai()
    partition_modbus()
    logger.success("=== Partitioning complete — 5 nodes ready ===")