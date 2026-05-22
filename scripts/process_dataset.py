"""
scripts/process_datasets.py
Day 2 — Clean both datasets and save as processed Parquet files.

HAI 22.04:
  - Loads all 5 CSV files, concatenates train and test separately
  - Forward-fills missing sensor values
  - Normalises sensors to [0,1] using scaler fitted on normal-only data
  - Saves hai_normal.parquet and hai_attacks.parquet

CIC Modbus 2023:
  - Loads all CSV files, concatenates
  - Drops columns >50% NaN, fills remaining NaN with median
  - Drops Inf values, clips to 99th percentile per column
  - Normalises numeric columns
  - Saves modbus2023_combined.parquet

Run: python scripts/process_dataset.py
"""
from __future__ import annotations
import os
import sys
import csv
import io
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

ROOT       = Path(__file__).resolve().parents[1]
HAI_VERSION = os.environ.get("HAI_VERSION", "hai-22.04")
HAI_DIR    = ROOT / "data" / "raw" / "hai" / HAI_VERSION
MODBUS_DIR = ROOT / "data" / "raw" / "modbus2023"
MODBUS_ATTACK_DIR = MODBUS_DIR / "attack"
MODBUS_BENIGN_DIR = MODBUS_DIR / "benign"
PROC_DIR   = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            first = fh.readline()
        return first.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


TSHARK_PATH = None  # Cache the tshark path after first search


def find_tshark() -> str | None:
    """Find tshark executable, caching result for performance."""
    global TSHARK_PATH
    
    if TSHARK_PATH is not None:
        return TSHARK_PATH
    
    # Try 1: shutil.which() with current environment
    if shutil.which("tshark") is not None:
        TSHARK_PATH = "tshark"
        return TSHARK_PATH
    
    # Try 2: Use 'where' command on Windows to find tshark
    try:
        result = subprocess.run(
            ["where", "tshark"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            TSHARK_PATH = result.stdout.strip().split("\n")[0]
            logger.debug(f"Found tshark at: {TSHARK_PATH}")
            return TSHARK_PATH
    except Exception as exc:
        logger.debug(f"'where tshark' failed: {exc}")
    
    # Try 3: Common Windows Wireshark paths
    common_paths = [
        r"C:\Program Files\Wireshark\tshark.exe",
        r"C:\Program Files (x86)\Wireshark\tshark.exe",
    ]
    for path in common_paths:
        if Path(path).exists():
            TSHARK_PATH = path
            logger.debug(f"Found tshark at common path: {TSHARK_PATH}")
            return TSHARK_PATH
    
    return None


def tshark_available() -> bool:
    return find_tshark() is not None


def parse_pcap_with_tshark(path: Path) -> pd.DataFrame:
    tshark_exe = find_tshark()
    if tshark_exe is None:
        raise RuntimeError("tshark not found on system")
    
    # Use basic network fields that should exist in any PCAP
    # (not Modbus-specific, since these PCAPs may not contain Modbus protocol data)
    fields = [
        "frame.time_epoch",
        "frame.len",
        "ip.src",
        "ip.dst",
        "ip.proto",
        "tcp.srcport",
        "tcp.dstport",
        "tcp.flags",
        "tcp.window_size_value",
        "udp.srcport",
        "udp.dstport",
        "udp.length",
    ]
    
    tshark_cmd = [
        tshark_exe,
        "-r",
        str(path),
        "-T",
        "fields",
        "-E",
        "separator=,",
        "-E",
        "quote=d",
        "-E",
        "occurrence=f",
    ]
    for field in fields:
        tshark_cmd.extend(["-e", field])

    result = subprocess.run(
        tshark_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"tshark failed for {path}: {result.stderr.strip() or result.stdout.strip()}"
        )

    rows = []
    reader = csv.reader(io.StringIO(result.stdout), delimiter=",")
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=fields)

    df = pd.DataFrame(rows, columns=fields)
    
    # Convert numeric columns
    numeric_cols = [
        "frame.time_epoch",
        "frame.len",
        "ip.proto",
        "tcp.srcport",
        "tcp.dstport",
        "tcp.flags",
        "tcp.window_size_value",
        "udp.srcport",
        "udp.dstport",
        "udp.length",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        logger.error("Unable to write Parquet file %s: %s", path, exc)
        logger.error("Install pyarrow or fastparquet: pip install pyarrow")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# HAI 22.04 Processing
# ══════════════════════════════════════════════════════════════════

def process_hai() -> None:
    logger.info(f"=== Processing HAI {HAI_VERSION} ===")

    if not HAI_DIR.exists():
        logger.error(f"HAI dataset folder not found: {HAI_DIR}")
        sys.exit(1)

    train_files = sorted(HAI_DIR.glob("*train*.csv"))
    test_files  = sorted(HAI_DIR.glob("*test*.csv"))

    if not train_files:
        logger.error(f"No HAI train files found in {HAI_DIR}")
        sys.exit(1)

    if not test_files:
        logger.warning(f"No HAI test files found in {HAI_DIR}; processing only train files")

    # ── Load ────────────────────────────────────────────────────────
    logger.info(f"Loading {len(train_files)} train files + {len(test_files)} test files from {HAI_DIR.name}")

    for f in train_files + test_files:
        if is_git_lfs_pointer(f):
            logger.error(f"Found Git LFS pointer file in HAI data: {f}. Run: git lfs pull")
            sys.exit(1)

    train_dfs = [pd.read_csv(f) for f in train_files]
    test_dfs  = [pd.read_csv(f) for f in test_files]

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df  = pd.concat(test_dfs,  ignore_index=True)
    all_df   = pd.concat([train_df, test_df], ignore_index=True)

    logger.info(f"Total rows: {len(all_df):,}  |  Columns: {list(all_df.columns[:5])}...")

    # ── Normalise column names ───────────────────────────────────────
    all_df.columns = all_df.columns.str.strip().str.lower()

    # ── Identify sensor columns (everything except timestamp and attack)
    exclude_cols = {"timestamp", "attack", "attack_p1", "attack_p2", "attack_p3"}
    sensor_cols = [c for c in all_df.columns if c not in exclude_cols]
    logger.info(f"Sensor columns: {len(sensor_cols)}")

    # ── Parse timestamp ─────────────────────────────────────────────
    if "timestamp" in all_df.columns:
        all_df["timestamp"] = pd.to_datetime(all_df["timestamp"], errors="coerce")
        all_df = all_df.sort_values("timestamp").reset_index(drop=True)

    # ── Forward-fill missing sensor values ──────────────────────────
    missing_pct = all_df[sensor_cols].isna().mean().mean() * 100
    logger.info(f"Missing sensor values before fill: {missing_pct:.3f}%")
    all_df[sensor_cols] = all_df[sensor_cols].ffill().bfill()

    # ── Ensure attack column is integer binary ───────────────────────
    # HAI 22.04 may encode attack as float or multi-column — normalise to binary
    if "attack" not in all_df.columns:
        # Some HAI versions use attack_p1/p2/p3 — OR them together
        atk_cols = [c for c in all_df.columns if c.startswith("attack")]
        all_df["attack"] = (all_df[atk_cols].sum(axis=1) > 0).astype(int)
        logger.info(f"Synthesised attack column from: {atk_cols}")
    else:
        all_df["attack"] = all_df["attack"].fillna(0).astype(int)

    attack_rate = all_df["attack"].mean() * 100
    logger.info(f"Attack rate: {attack_rate:.2f}%  (normal: {100-attack_rate:.2f}%)")

    # ── Fit MinMaxScaler on NORMAL data only — no data leakage ──────
    normal_df = all_df[all_df["attack"] == 0]
    scaler = MinMaxScaler()
    scaler.fit(normal_df[sensor_cols].values.astype(np.float32))

    all_df[sensor_cols] = scaler.transform(
        all_df[sensor_cols].values.astype(np.float32)
    )

    # ── Split and save ───────────────────────────────────────────────
    normal_df  = all_df[all_df["attack"] == 0].reset_index(drop=True)
    attacks_df = all_df[all_df["attack"] == 1].reset_index(drop=True)

    normal_path  = PROC_DIR / "hai_normal.parquet"
    attacks_path = PROC_DIR / "hai_attacks.parquet"
    combined_path = PROC_DIR / "hai_combined.parquet"

    save_parquet(normal_df, normal_path)
    save_parquet(attacks_df, attacks_path)
    save_parquet(all_df, combined_path)

    logger.success(f"Saved: {normal_path}  ({len(normal_df):,} rows)")
    logger.success(f"Saved: {attacks_path} ({len(attacks_df):,} rows)")
    logger.success(f"Saved: {combined_path} ({len(all_df):,} rows)")

    # ── Persist scaler column list for partition_data.py ────────────
    import json
    scaler_meta = {
        "sensor_cols": sensor_cols,
        "n_sensors": len(sensor_cols),
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
    }
    (PROC_DIR / "hai_scaler_meta.json").write_text(json.dumps(scaler_meta, indent=2))
    logger.info("Saved scaler metadata to hai_scaler_meta.json")


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        logger.error("Unable to write Parquet file %s: %s", path, exc)
        logger.error("Install pyarrow or fastparquet: pip install pyarrow")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# CIC Modbus 2023 Processing
# ══════════════════════════════════════════════════════════════════

def process_modbus() -> None:
    logger.info("=== Processing CIC Modbus 2023 ===")

    csv_files = sorted(MODBUS_DIR.glob("*.csv"))
    if csv_files:
        logger.info(f"Loading {len(csv_files)} CSV files")
        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                df["_source_file"] = f.stem
                dfs.append(df)
                logger.info(f"  {f.name}: {len(df):,} rows, {len(df.columns)} cols")
            except Exception as exc:
                logger.warning(f"  Skipping {f.name}: {exc}")
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined: {len(combined):,} rows, {len(combined.columns)} columns")
    else:
        if not (MODBUS_ATTACK_DIR.exists() and MODBUS_BENIGN_DIR.exists()):
            logger.error(
                f"No CSV files in {MODBUS_DIR} and no PCAP folders found. "
                "Download the CIC Modbus 2023 dataset and place attack/ and benign/ subfolders under data/raw/modbus2023"
            )
            sys.exit(1)

        attack_files = sorted(MODBUS_ATTACK_DIR.rglob("*.pcap"))
        benign_files = sorted(MODBUS_BENIGN_DIR.rglob("*.pcap"))

        if not attack_files and not benign_files:
            logger.error(
                f"No PCAP files found under {MODBUS_ATTACK_DIR} or {MODBUS_BENIGN_DIR}."
            )
            sys.exit(1)

        if not tshark_available():
            logger.error(
                "tshark is required to parse Modbus PCAP files but was not found. "
                "Install Wireshark/tshark and ensure it is on your PATH."
            )
            sys.exit(1)

        logger.info(
            f"Parsing {len(attack_files)} attack and {len(benign_files)} benign PCAP files"
        )
        rows = []
        for f in attack_files:
            try:
                df = parse_pcap_with_tshark(f)
                df["_source_file"] = f.stem
                df["label"] = "attack"
                df["node"] = f.parent.name
                rows.append(df)
                logger.info(f"  parsed {f.name}: {len(df):,} rows")
            except Exception as exc:
                logger.warning(f"  Skipping attack PCAP {f.name}: {exc}")
        for f in benign_files:
            try:
                df = parse_pcap_with_tshark(f)
                df["_source_file"] = f.stem
                df["label"] = "benign"
                df["node"] = f.parent.name
                rows.append(df)
                logger.info(f"  parsed {f.name}: {len(df):,} rows")
            except Exception as exc:
                logger.warning(f"  Skipping benign PCAP {f.name}: {exc}")
        if not rows:
            logger.error("No Modbus packet rows were parsed from PCAP files")
            sys.exit(1)
        combined = pd.concat(rows, ignore_index=True)
        logger.info(f"Combined PCAP rows: {len(combined):,} rows, {len(combined.columns)} columns")

    # ── Normalise column names ───────────────────────────────────────
    combined.columns = combined.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── Identify label column ────────────────────────────────────────
    label_candidates = [c for c in combined.columns if "label" in c or "class" in c or "attack" in c]
    label_col = label_candidates[0] if label_candidates else None
    if label_col:
        logger.info(f"Label column detected: '{label_col}'  unique values: {combined[label_col].unique()[:10]}")
    else:
        logger.warning("No label column found — adding 'label' = BENIGN as placeholder")
        combined["label"] = "BENIGN"
        label_col = "label"

    # ── Drop columns with >50% NaN ───────────────────────────────────
    nan_frac = combined.isnull().mean()
    drop_cols = nan_frac[nan_frac > 0.5].index.tolist()
    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} columns with >50% NaN: {drop_cols[:5]}...")
        combined = combined.drop(columns=drop_cols)

    # ── Fill remaining NaN with column median ────────────────────────
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    combined[numeric_cols] = combined[numeric_cols].fillna(
        combined[numeric_cols].median()
    )

    # ── Replace Inf values ───────────────────────────────────────────
    combined[numeric_cols] = combined[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(combined[numeric_cols].median())

    # ── Clip to 99th percentile to remove extreme outliers ───────────
    for col in numeric_cols:
        p99 = combined[col].quantile(0.99)
        p01 = combined[col].quantile(0.01)
        combined[col] = combined[col].clip(lower=p01, upper=p99)

    # ── Normalise numeric columns ────────────────────────────────────
    scaler = MinMaxScaler()
    combined[numeric_cols] = scaler.fit_transform(
        combined[numeric_cols].values.astype(np.float32)
    )

    # ── Save ─────────────────────────────────────────────────────────
    out_path = PROC_DIR / "modbus2023_combined.parquet"
    save_parquet(combined, out_path)
    logger.success(f"Saved: {out_path}  ({len(combined):,} rows, {len(combined.columns)} cols)")

    dist = combined[label_col].value_counts()
    logger.info(f"Label distribution:\n{dist.to_string()}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    process_hai()
    process_modbus()
    logger.success("=== Processing complete ===")