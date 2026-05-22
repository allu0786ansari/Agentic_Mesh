r"""
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from sklearn.model_selection import train_test_split

ROOT      = Path(__file__).resolve().parents[1]
PROC_DIR  = ROOT / "data" / "processed"
PART_DIR  = ROOT / "data" / "partitions"

WINDOW_SIZE  = 60   # seconds (1 Hz sensor data)
WINDOW_STRIDE = 10  # stride between windows
TEST_RATIO   = 0.2
RANDOM_SEED  = 42
MODBUS_CLEAN_SHARD_DIR = PROC_DIR / "modbus2023_clean_shards"
MODBUS_TRAIN_BUCKET = int((1.0 - TEST_RATIO) * 10_000)


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


@dataclass
class StreamingPartitionWriter:
    """Append train/test batches for one tabular node without holding all rows."""

    node_id: str
    feature_cols: list[str]
    label_col: str = "label"
    train_rows: int = 0
    test_rows: int = 0
    class_distribution: dict[str, int] = field(
        default_factory=lambda: {"normal": 0, "attack": 0}
    )
    _train_writer: pq.ParquetWriter | None = field(default=None, init=False)
    _test_writer: pq.ParquetWriter | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.out_dir = PART_DIR / self.node_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for stale_file in ["train.parquet", "test.parquet", "meta.json"]:
            path = self.out_dir / stale_file
            if path.exists():
                path.unlink()
        self.feature_names = [f"feature_{i}" for i in range(len(self.feature_cols))]

    def write(self, df: pd.DataFrame, train_mask: pd.Series) -> None:
        """Write one routed batch into this node's train/test parquet files."""
        if df.empty:
            return
        features = df[self.feature_cols].astype(np.float32, copy=False)
        labels = to_binary_label(df[self.label_col])
        out = features.copy()
        out.columns = self.feature_names
        out["label"] = labels.astype(np.int8)

        attacks = int(out["label"].sum())
        self.class_distribution["attack"] += attacks
        self.class_distribution["normal"] += int(len(out) - attacks)

        self._write_split(out[train_mask.to_numpy()], "train")
        self._write_split(out[~train_mask.to_numpy()], "test")

    def _write_split(self, df: pd.DataFrame, split: str) -> None:
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if split == "train":
            if self._train_writer is None:
                self._train_writer = pq.ParquetWriter(
                    self.out_dir / "train.parquet",
                    table.schema,
                    compression="snappy",
                )
            self._train_writer.write_table(table)
            self.train_rows += len(df)
        else:
            if self._test_writer is None:
                self._test_writer = pq.ParquetWriter(
                    self.out_dir / "test.parquet",
                    table.schema,
                    compression="snappy",
                )
            self._test_writer.write_table(table)
            self.test_rows += len(df)

    def close(self) -> None:
        """Close any open Parquet writers."""
        if self._train_writer is not None:
            self._train_writer.close()
        if self._test_writer is not None:
            self._test_writer.close()

    def write_meta(self) -> None:
        """Persist node metadata used by model selection and training."""
        total_rows = self.train_rows + self.test_rows
        attack_rate = (
            self.class_distribution["attack"] / total_rows if total_rows else 0.0
        )
        meta = {
            "node_id": self.node_id,
            "data_type": "tabular",
            "model_type": "isolation_forest",
            "feature_count": len(self.feature_cols),
            "feature_columns": self.feature_cols,
            "label_column": self.label_col,
            "attack_rate": round(float(attack_rate), 4),
            "train_rows": int(self.train_rows),
            "test_rows": int(self.test_rows),
            "class_distribution": self.class_distribution,
        }
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.success(
            "{}: train=({:,}, {})  test=({:,}, {})  model=isolation_forest  features={}",
            self.node_id,
            self.train_rows,
            len(self.feature_cols) + 1,
            self.test_rows,
            len(self.feature_cols) + 1,
            len(self.feature_cols),
        )


def to_binary_label(labels: pd.Series) -> pd.Series:
    """Map Modbus label strings to 0=normal and 1=attack."""
    return (~labels.astype(str).str.lower().str.contains(
        "benign|normal|legitimate", na=False
    )).astype(np.int8)


def stable_bucket(parts: list[pd.Series], modulo: int = 10_000) -> pd.Series:
    """Create deterministic buckets for reproducible streaming splits."""
    key = parts[0].astype(str)
    for part in parts[1:]:
        key = key.str.cat(part.astype(str), sep="|")
    return (pd.util.hash_pandas_object(key, index=False) % modulo).astype(np.int64)


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

    # HAI 22.04 sensor names encode subsystem prefixes. Keep those physical
    # boundaries intact so each VAE node sees a genuinely different process area.
    prefix_groups = {
        "p1": [c for c in all_sensor_cols if c.startswith("p1_")],
        "p2": [c for c in all_sensor_cols if c.startswith("p2_")],
        "p3_p4": [
            c for c in all_sensor_cols if c.startswith("p3_") or c.startswith("p4_")
        ],
    }
    if any(not cols for cols in prefix_groups.values()):
        logger.warning(
            "HAI subsystem prefixes are incomplete; falling back to even sensor split"
        )
        thirds = n_sensors // 3
        subsystems = {
            "node_01": all_sensor_cols[:thirds],
            "node_02": all_sensor_cols[thirds : 2 * thirds],
            "node_03": all_sensor_cols[2 * thirds :],
        }
    else:
        subsystems = {
            "node_01": prefix_groups["p1"],
            "node_02": prefix_groups["p2"],
            "node_03": prefix_groups["p3_p4"],
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
            "subsystem":        {"node_01": "p1", "node_02": "p2", "node_03": "p3_p4"}[node_id],
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

    shard_paths = sorted(MODBUS_CLEAN_SHARD_DIR.glob("*.parquet"))
    if shard_paths:
        logger.info("Using {} cleaned Modbus shards for streaming partition", len(shard_paths))
    else:
        logger.warning(
            "Cleaned Modbus shards not found; falling back to combined parquet row groups. "
            "For best memory behavior, keep data/processed/modbus2023_clean_shards."
        )

    source_paths = shard_paths or [modbus_path]
    sample = pd.read_parquet(source_paths[0])
    sample.columns = sample.columns.str.strip().str.lower()

    label_candidates = [c for c in sample.columns if "label" in c or "class" in c]
    label_col = label_candidates[0] if label_candidates else "label"
    if label_col not in sample.columns:
        logger.error("No label column found in Modbus processed data")
        sys.exit(1)

    feature_cols = [
        c
        for c in sample.columns
        if c not in {label_col, "_source_file", "node", "ip.src", "ip.dst"}
        and pd.api.types.is_numeric_dtype(sample[c])
    ]
    if not feature_cols:
        logger.error("No numeric feature columns found in Modbus processed data")
        sys.exit(1)

    logger.info("Modbus feature columns: {}", feature_cols)
    logger.info(
        "Streaming Non-IID split: node_04 receives ~80% attack / ~20% benign; "
        "node_05 receives the complementary benign-heavy mix"
    )

    writers = {
        "node_04": StreamingPartitionWriter("node_04", feature_cols, label_col),
        "node_05": StreamingPartitionWriter("node_05", feature_cols, label_col),
    }

    try:
        for df in iter_modbus_batches(source_paths):
            df.columns = df.columns.str.strip().str.lower()
            labels = to_binary_label(df[label_col])
            row_number = pd.Series(np.arange(len(df)), index=df.index, dtype="int64")
            source = (
                df["_source_file"]
                if "_source_file" in df.columns
                else pd.Series("combined", index=df.index)
            )
            route_bucket = stable_bucket([source, row_number])

            # Non-IID but complete: every row is assigned exactly once.
            # Attacks mostly train node_04; benign traffic mostly trains node_05.
            node_04_mask = ((labels == 1) & (route_bucket < 8_000)) | (
                (labels == 0) & (route_bucket < 2_000)
            )
            node_05_mask = ~node_04_mask

            split_bucket = stable_bucket([source, row_number, pd.Series("split", index=df.index)])
            train_mask = split_bucket < MODBUS_TRAIN_BUCKET

            writers["node_04"].write(df[node_04_mask], train_mask[node_04_mask])
            writers["node_05"].write(df[node_05_mask], train_mask[node_05_mask])
    finally:
        for writer in writers.values():
            writer.close()

    for writer in writers.values():
        if writer.train_rows == 0 or writer.test_rows == 0:
            logger.error("{} produced an empty train/test split", writer.node_id)
            sys.exit(1)
        writer.write_meta()


def iter_modbus_batches(paths: list[Path]):
    """Yield Modbus DataFrames from shards or row groups with bounded memory."""
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for row_group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group)
            yield table.to_pandas()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    partition_hai()
    partition_modbus()
    logger.success("=== Partitioning complete — 5 nodes ready ===")
