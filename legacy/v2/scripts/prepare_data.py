from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from scripts.feature_selection import screen_features
from scripts.synthetic_anomaly_injection import inject_anomalies
from storage.sqlite_schema import seed_registry

HAI_LABEL_COLUMNS = {"attack", "attack_p1", "attack_p2", "attack_p3"}
HAI_ROLLING_WINDOW = 15
HAI_ROLLING_WINDOWS = (5, 15, 60)
PACKET_COLUMNS = (
    "timestamp_epoch",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "frame_len",
    "tcp_flags",
    "modbus_func",
    "modbus_unit",
    "tcp_stream",
)


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    source: str
    model_type: str
    data_type: str
    feature_prefix: str | None = None
    device_directory: str | None = None


NODE_SPECS = (
    NodeSpec("node_01", "hai_21.03_p1", "vae", "timeseries", feature_prefix="P1_"),
    NodeSpec("node_02", "hai_21.03_p2", "vae", "timeseries", feature_prefix="P2_"),
    NodeSpec("node_03", "hai_21.03_p3", "vae", "timeseries", feature_prefix="P3_"),
    NodeSpec("node_04", "cic_modbus_2023_ied1a", "isolation_forest", "tabular", device_directory="ied1a"),
    NodeSpec("node_05", "cic_modbus_2023_ied1b", "isolation_forest", "tabular", device_directory="ied1b"),
    NodeSpec("node_06", "cic_modbus_2023_scada_hmi", "isolation_forest", "tabular", device_directory="scada-hmi"),
)


def _digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(frame), 100_000):
        chunk = frame.iloc[start : start + 100_000]
        digest.update(chunk.to_json(orient="split", date_format="iso", double_precision=15).encode())
    return digest.hexdigest()


def _write_node(
    node: NodeSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_root: Path,
    *,
    seed: int,
    metadata_extra: dict[str, object] | None = None,
) -> Path:
    if train.empty or test.empty:
        raise ValueError(f"{node.node_id} requires non-empty train and test frames")
    node_dir = output_root / node.node_id
    node_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(node_dir / "train.parquet", index=False, compression="snappy")
    test.to_parquet(node_dir / "test.parquet", index=False, compression="snappy")

    label_columns = [column for column in train.columns if column.lower() in HAI_LABEL_COLUMNS or column == "label"]
    timestamp_column = next((column for column in train.columns if column.lower() in {"time", "timestamp"}), None)
    excluded = set(label_columns)
    if timestamp_column:
        excluded.add(timestamp_column)
    feature_columns = [column for column in train.columns if column not in excluded]
    metadata: dict[str, object] = {
        "node_id": node.node_id,
        "source": node.source,
        "model_type": node.model_type,
        "data_type": node.data_type,
        "label_column": "label" if "label" in train.columns else (label_columns[0] if label_columns else None),
        "timestamp_column": timestamp_column,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "row_count": int(len(train) + len(test)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "seed": seed,
        "content_sha256": _digest(pd.concat([train, test], ignore_index=True)),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    (node_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return node_dir / "meta.json"


def _hai_columns(path: Path, prefix: str) -> list[str]:
    columns = list(pd.read_csv(path, compression="infer", nrows=0).columns)
    selected = [column for column in columns if column.lower() in HAI_LABEL_COLUMNS or column.lower() == "time" or column.startswith(prefix)]
    if not selected:
        raise ValueError(f"No {prefix} columns found in HAI file {path}")
    return selected


def _load_hai_split(root: Path, split: str, prefix: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"{split}*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No HAI {split}*.csv.gz files found under {root}")
    selected = _hai_columns(paths[0], prefix)
    frames = [pd.read_csv(path, compression="infer", usecols=selected) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    feature_columns = [column for column in frame.columns if column.startswith(prefix)]
    frame[feature_columns] = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    frame[feature_columns] = frame[feature_columns].ffill().bfill()
    attack_columns = [column for column in frame.columns if column.lower() in HAI_LABEL_COLUMNS]
    frame["label"] = frame[attack_columns].fillna(0).max(axis=1).astype("int8")
    return frame.drop(columns=attack_columns)


def _engineer_hai_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    rolling_window: int = HAI_ROLLING_WINDOW,
    rolling_windows: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Add temporal features while selecting base sensors from normal training rows."""
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")
    feature_columns = [column for column in train.columns if column.startswith(("P1_", "P2_", "P3_"))]
    rolling_windows = tuple(sorted(set(rolling_windows or (rolling_window,))))
    if not rolling_windows or any(window < 2 for window in rolling_windows):
        raise ValueError("rolling_windows must contain positive windows of at least two rows")

    def infer_pairs(columns: list[str]) -> list[tuple[str, str]]:
        stems: dict[str, dict[str, str]] = {}
        for column in columns:
            if column.endswith(("D", "Z")) and len(column) > 1:
                stems.setdefault(column[:-1], {})[column[-1]] = column
        return [(values["D"], values["Z"]) for values in stems.values() if "D" in values and "Z" in values]

    normal = train.loc[train["label"] == 0, feature_columns]
    if normal.empty:
        raise ValueError("HAI training split contains no normal rows for feature selection")

    retained_columns = [
        column
        for column in feature_columns
        if normal[column].nunique(dropna=True) > 1
        and float(normal[column].std(ddof=0)) > 1e-12
    ]
    removed_columns = [column for column in feature_columns if column not in retained_columns]
    binary_columns = [column for column in retained_columns if normal[column].nunique(dropna=True) <= 2]
    control_loop_pairs = []
    for command, measured in infer_pairs(retained_columns):
        normal_command = normal[[command, measured]]
        correlation = normal_command[command].corr(normal_command[measured])
        if pd.notna(correlation) and float(correlation) > 0.6:
            control_loop_pairs.append((command, measured))

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame[retained_columns].apply(pd.to_numeric, errors="raise")
        blocks: dict[str, pd.Series] = {}
        for command, measured in control_loop_pairs:
            residual = numeric[command] - numeric[measured]
            blocks[f"{command}__tracking_error"] = residual
            blocks[f"{command}__tracking_error_rate"] = residual.diff().fillna(0.0)
        for column in retained_columns:
            series = numeric[column]
            blocks[column] = series
            if column not in binary_columns:
                for window in rolling_windows:
                    rolling = series.rolling(window, min_periods=1)
                    blocks[f"{column}__rolling_std_w{window}"] = rolling.std(ddof=0).fillna(0.0)
                    blocks[f"{column}__local_deviation_w{window}"] = series - rolling.mean()
            else:
                blocks[f"{column}__transition"] = series.ne(series.shift()).astype(np.float32)
        engineered = pd.DataFrame(blocks, index=frame.index)
        return pd.concat([frame[["time", "label"]], engineered], axis=1)

    engineered_train = transform(train)
    engineered_test = transform(test)
    if list(engineered_train.columns) != list(engineered_test.columns):
        raise ValueError("HAI feature engineering produced different train and test columns")

    engineered_feature_columns = [
        column for column in engineered_train.columns if column not in {"time", "label"}
    ]
    base_feature_set = set(retained_columns)
    candidate_columns = [column for column in engineered_feature_columns if column not in base_feature_set]
    selected_engineered_columns = candidate_columns
    screening_report: list[dict[str, object]] = []
    screening_skipped_reason: str | None = None
    validation_rows = min(max(10 * rolling_windows[-1], 1), len(engineered_train))
    if candidate_columns and validation_rows >= 60:
        validation_frame = engineered_train.iloc[-validation_rows:].drop(columns="time").reset_index(drop=True)
        validation_frame, synthetic_labels = inject_anomalies(
            validation_frame.drop(columns="label"),
            candidate_columns,
            seed=42,
            event_count=16,
            event_length=min(30, max(5, validation_rows // 20)),
        )
        validation_frame["label"] = synthetic_labels
        report = screen_features(validation_frame, candidate_columns, min_auc=0.55)
        screening_report = report.to_dict(orient="records")
        selected_engineered_columns = report.loc[report["selected"], "feature"].tolist()
        if not selected_engineered_columns:
            raise ValueError("Synthetic validation screening rejected every engineered HAI feature")
        keep_columns = retained_columns + selected_engineered_columns
        engineered_train = engineered_train[["time", "label", *keep_columns]]
        engineered_test = engineered_test[["time", "label", *keep_columns]]
    elif candidate_columns:
        screening_skipped_reason = "insufficient training rows for synthetic validation"
    engineered_values = engineered_train.drop(columns=["time", "label"]).to_numpy(dtype=np.float64)
    if not np.isfinite(engineered_values).all():
        raise ValueError("HAI feature engineering produced non-finite training values")
    metadata = {
        "rolling_window_rows": rolling_window,
        "base_feature_count": len(feature_columns),
        "retained_base_features": retained_columns,
        "removed_constant_features": removed_columns,
        "binary_base_features": binary_columns,
        "control_loop_pairs": control_loop_pairs,
        "rolling_windows": list(rolling_windows),
        "candidate_engineered_features": candidate_columns,
        "selected_engineered_features": selected_engineered_columns,
        "feature_screening": screening_report,
        "feature_screening_skipped_reason": screening_skipped_reason,
        "engineered_feature_count": len(retained_columns)
        + len(selected_engineered_columns),
    }
    return engineered_train, engineered_test, metadata


def _scale_hai(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    method: str = "minmax",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method not in {"minmax", "robust"}:
        raise ValueError("method must be minmax or robust")
    feature_columns = [column for column in train.columns if column not in {"time", "label"}]
    normal = train.loc[train["label"] == 0, feature_columns]
    if normal.empty:
        raise ValueError("HAI training split contains no normal rows for scaler fitting")
    minimum = normal.min()
    scaled_train = train.copy()
    scaled_test = test.copy()
    if method == "minmax":
        span = (normal.max() - minimum).replace(0, 1.0)
        scaled_train[feature_columns] = (train[feature_columns] - minimum) / span
        scaled_test[feature_columns] = (test[feature_columns] - minimum) / span
    else:
        median = normal.median()
        scale = (normal.quantile(0.95) - normal.quantile(0.05)).replace(0, 1.0)
        scaled_train[feature_columns] = (train[feature_columns] - median) / scale
        scaled_test[feature_columns] = (test[feature_columns] - median) / scale
    return scaled_train, scaled_test


def prepare_hai(
    root: Path,
    output_root: Path,
    seed: int,
    *,
    partition_scaling: str = "none",
) -> list[Path]:
    if partition_scaling not in {"none", "minmax", "robust"}:
        raise ValueError("partition_scaling must be none, minmax, or robust")
    metadata_paths: list[Path] = []
    for node in NODE_SPECS[:3]:
        train = _load_hai_split(root, "train", node.feature_prefix or "")
        test = _load_hai_split(root, "test", node.feature_prefix or "")
        train, test, feature_metadata = _engineer_hai_features(
            train,
            test,
            rolling_windows=HAI_ROLLING_WINDOWS,
        )
        if partition_scaling != "none":
            train, test = _scale_hai(train, test, method=partition_scaling)
        metadata_paths.append(
            _write_node(
                node,
                train,
                test,
                output_root,
                seed=seed,
                metadata_extra={
                    "window_size_rows": 60,
                    "window_stride_rows": 10,
                    "normal_only_scaler": True,
                    "partition_scaling": partition_scaling,
                    "feature_engineering": feature_metadata,
                },
            )
        )
    return metadata_paths


def _parse_int(value: object) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(int(text, 0))
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return 0.0


def _aggregate_packets(frame: pd.DataFrame, label: int, window_seconds: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["timestamp_epoch"], errors="coerce"), unit="s", utc=True)
    frame = frame.dropna(subset=["timestamp"])
    if frame.empty:
        return pd.DataFrame()
    frame["window_start"] = frame["timestamp"].dt.floor(f"{window_seconds}s")
    frame["frame_len"] = pd.to_numeric(frame["frame_len"], errors="coerce").fillna(0)
    frame["tcp_flags_int"] = frame["tcp_flags"].map(_parse_int)
    frame["modbus_func_int"] = frame["modbus_func"].map(_parse_int)
    grouped = frame.groupby("window_start", sort=True)
    result = grouped.agg(
        packet_count=("frame_len", "size"),
        byte_count=("frame_len", "sum"),
        mean_frame_len=("frame_len", "mean"),
        std_frame_len=("frame_len", "std"),
        min_frame_len=("frame_len", "min"),
        max_frame_len=("frame_len", "max"),
        unique_sources=("src_ip", "nunique"),
        unique_destinations=("dst_ip", "nunique"),
        unique_tcp_streams=("tcp_stream", "nunique"),
        syn_count=("tcp_flags_int", lambda values: int(sum(int(value) & 0x0002 != 0 for value in values))),
        rst_count=("tcp_flags_int", lambda values: int(sum(int(value) & 0x0004 != 0 for value in values))),
        modbus_function_count=("modbus_func_int", lambda values: int(sum(value > 0 for value in values))),
        modbus_read_count=("modbus_func_int", lambda values: int(sum(int(value) in {1, 2, 3, 4} for value in values))),
        modbus_write_count=("modbus_func_int", lambda values: int(sum(int(value) in {5, 6, 15, 16} for value in values))),
    ).reset_index()
    result["std_frame_len"] = result["std_frame_len"].fillna(0.0)
    result["packets_per_second"] = result["packet_count"] / window_seconds
    result["request_response_ratio"] = result["modbus_read_count"] / result["modbus_write_count"].replace(0, 1)
    result["label"] = np.int8(label)
    return result


def _tshark_rows(path: Path, tshark: str, chunk_size: int) -> Iterator[pd.DataFrame]:
    command = [
        tshark,
        "-n",
        "-r",
        str(path),
        "-T",
        "fields",
        "-E",
        "header=n",
        "-E",
        "separator=,",
        "-E",
        "quote=d",
        *[item for column in PACKET_COLUMNS for item in ("-e", _tshark_field(column))],
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout is not None
    try:
        for chunk in pd.read_csv(process.stdout, names=list(PACKET_COLUMNS), chunksize=chunk_size, dtype=str, keep_default_na=False):
            yield chunk
    finally:
        if process.stdout:
            process.stdout.close()
        stderr = process.stderr.read() if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"TShark failed for {path}: {stderr[-1000:]}")


def _tshark_field(column: str) -> str:
    return {
        "timestamp_epoch": "frame.time_epoch",
        "src_ip": "ip.src",
        "dst_ip": "ip.dst",
        "src_port": "tcp.srcport",
        "dst_port": "tcp.dstport",
        "frame_len": "frame.len",
        "tcp_flags": "tcp.flags",
        "modbus_func": "modbus.func_code",
        "modbus_unit": "mbtcp.unit_id",
        "tcp_stream": "tcp.stream",
    }[column]


def extract_capture(path: Path, tshark: str, label: int, window_seconds: int, chunk_size: int = 100_000) -> pd.DataFrame:
    aggregated = [_aggregate_packets(chunk, label, window_seconds) for chunk in _tshark_rows(path, tshark, chunk_size)]
    aggregated = [frame for frame in aggregated if not frame.empty]
    if not aggregated:
        return pd.DataFrame()
    combined = pd.concat(aggregated, ignore_index=True)
    numeric = [column for column in combined.columns if column not in {"window_start", "label"}]
    combined[numeric] = combined[numeric].apply(pd.to_numeric, errors="coerce").fillna(0)
    combined["label"] = combined["label"].astype("int8")
    return combined.groupby("window_start", as_index=False).agg({**{column: "sum" for column in numeric}, "label": "max"})


def _split_capture_paths(root: Path, device_directory: str, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    benign = sorted((root / "benign" / device_directory).glob("*.pcap"))
    attack = sorted((root / "attack" / device_directory).glob("*.pcap"))
    if len(benign) < 2:
        raise ValueError(f"At least two benign captures are required for {device_directory}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(benign))
    train_count = max(1, int(len(benign) * 0.8))
    train = [benign[index] for index in order[:train_count]]
    held_out = [benign[index] for index in order[train_count:]]
    return train, held_out, attack


def prepare_modbus(root: Path, output_root: Path, tshark: str, seed: int, window_seconds: int) -> list[Path]:
    metadata_paths: list[Path] = []
    for offset, node in enumerate(NODE_SPECS[3:]):
        device_directory = node.device_directory or ""
        benign_paths, held_out_paths, attack_paths = _split_capture_paths(root, device_directory, seed + offset)
        train_frames = [extract_capture(path, tshark, 0, window_seconds) for path in benign_paths]
        test_frames = [extract_capture(path, tshark, 0, window_seconds) for path in held_out_paths]
        test_frames.extend(extract_capture(path, tshark, 1, window_seconds) for path in attack_paths)
        train = pd.concat([frame for frame in train_frames if not frame.empty], ignore_index=True)
        test = pd.concat([frame for frame in test_frames if not frame.empty], ignore_index=True)
        metadata_paths.append(
            _write_node(
                node,
                train,
                test,
                output_root,
                seed=seed + offset,
                metadata_extra={
                    "window_seconds": window_seconds,
                    "label_granularity": "capture",
                    "train_capture_count": len(benign_paths),
                    "held_out_benign_capture_count": len(held_out_paths),
                    "attack_capture_count": len(attack_paths),
                },
            )
        )
    return metadata_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the real HAI and CIC Modbus Week 2 datasets")
    parser.add_argument("--hai-root", type=Path, required=True)
    parser.add_argument("--modbus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/partitions"))
    parser.add_argument("--database", type=Path, default=Path("agmesh.db"))
    parser.add_argument("--tshark", default=shutil.which("tshark") or "tshark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument(
        "--hai-partition-scaling",
        choices=("none", "minmax", "robust"),
        default="none",
        help="Keep upgrade partitions raw so the reusable scaler is the only scaling step.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.hai_root.exists():
        raise FileNotFoundError(args.hai_root)
    if not args.modbus_root.exists():
        raise FileNotFoundError(args.modbus_root)
    metadata_paths = prepare_hai(
        args.hai_root,
        args.output,
        args.seed,
        partition_scaling=args.hai_partition_scaling,
    )
    metadata_paths.extend(prepare_modbus(args.modbus_root, args.output, args.tshark, args.seed, args.window_seconds))
    count = seed_registry(args.database, metadata_paths)
    print(json.dumps({"nodes": count, "output": str(args.output), "database": str(args.database)}))


if __name__ == "__main__":
    main()
