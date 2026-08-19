from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

LABEL_COLUMNS = ("label", "target", "attack", "is_attack", "class")
TIMESTAMP_COLUMNS = ("timestamp", "time", "datetime", "date")


def load_table(path: Path) -> pd.DataFrame:
    """Load a supported tabular source and return a copied dataframe."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path).copy()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path).copy()
    if suffix == ".json":
        return pd.read_json(path).copy()
    raise ValueError(f"Unsupported input format: {path.suffix}")


def infer_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _stable_digest(frame: pd.DataFrame) -> str:
    payload = frame.to_json(orient="split", date_format="iso", double_precision=15).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalise_metadata(
    frame: pd.DataFrame,
    *,
    node_id: str,
    source_name: str,
    model_type: str,
    data_type: str,
    seed: int,
) -> dict[str, object]:
    label_column = infer_column(frame.columns, LABEL_COLUMNS)
    timestamp_column = infer_column(frame.columns, TIMESTAMP_COLUMNS)
    excluded_columns = {column for column in (label_column, timestamp_column) if column is not None}
    feature_columns = [column for column in frame.columns if column not in excluded_columns]
    return {
        "node_id": node_id,
        "source": source_name,
        "model_type": model_type,
        "data_type": data_type,
        "label_column": label_column,
        "timestamp_column": timestamp_column,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "row_count": int(len(frame)),
        "seed": seed,
        "content_sha256": _stable_digest(frame),
    }


def partition_frame(frame: pd.DataFrame, node_count: int, seed: int) -> list[pd.DataFrame]:
    if node_count < 1:
        raise ValueError("node_count must be at least 1")
    if frame.empty:
        raise ValueError("Cannot partition an empty dataframe")

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return [part.reset_index(drop=True) for part in __split_balanced(shuffled, node_count)]


def __split_balanced(frame: pd.DataFrame, node_count: int) -> list[pd.DataFrame]:
    base_size, remainder = divmod(len(frame), node_count)
    partitions: list[pd.DataFrame] = []
    start = 0
    for index in range(node_count):
        size = base_size + (1 if index < remainder else 0)
        partitions.append(frame.iloc[start : start + size])
        start += size
    return partitions


def write_partitions(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    source_name: str,
    node_count: int,
    seed: int,
    model_type: str,
    data_type: str,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    partitions = partition_frame(frame, node_count, seed)
    metadata: list[dict[str, object]] = []

    for index, partition in enumerate(partitions, start=1):
        node_id = f"node_{index:02d}"
        node_dir = output_dir / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        train_size = max(1, int(len(partition) * 0.8)) if len(partition) > 1 else len(partition)
        train = partition.iloc[:train_size].reset_index(drop=True)
        test = partition.iloc[train_size:].reset_index(drop=True)
        if test.empty:
            test = train.copy()

        train.to_parquet(node_dir / "train.parquet", index=False)
        test.to_parquet(node_dir / "test.parquet", index=False)
        node_metadata = _normalise_metadata(
            partition,
            node_id=node_id,
            source_name=source_name,
            model_type=model_type,
            data_type=data_type,
            seed=seed,
        )
        node_metadata.update({"train_rows": len(train), "test_rows": len(test)})
        (node_dir / "meta.json").write_text(json.dumps(node_metadata, indent=2), encoding="utf-8")
        metadata.append(node_metadata)

    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create deterministic Agentic Mesh node partitions")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/partitions"))
    parser.add_argument("--nodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-name", default="unknown")
    parser.add_argument("--model-type", default="vae", choices=("vae", "isolation_forest"))
    parser.add_argument("--data-type", default="timeseries", choices=("timeseries", "tabular"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    frame = load_table(args.input)
    metadata = write_partitions(
        frame,
        args.output,
        source_name=args.source_name,
        node_count=args.nodes,
        seed=args.seed,
        model_type=args.model_type,
        data_type=args.data_type,
    )
    print(json.dumps({"nodes": len(metadata), "rows": len(frame), "output": str(args.output)}))


if __name__ == "__main__":
    main()
