"""Load HAI 21.03 files and decide which features are modelled (HAI-1, HAI-2).

Each file becomes a frame with a ``time`` column (datetime64, treated as UTC), the
sensor and actuator features as float32, and the label columns as int8. The label
columns are kept for splitting and evaluation only; model code receives a
label-free view (Week 3 leakage guard).
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agmesh.common.config import HaiConfig

TIME_COLUMN = "time"
_FILE = re.compile(r"^(train|test)(\d+)\.csv(\.gz)?$")


@dataclass(frozen=True)
class HaiFile:
    name: str  # e.g. "train1"
    split: str  # "train" or "test"
    path: Path


@dataclass
class LoadReport:
    rows_read: int = 0
    rows_dropped_missing: int = 0
    values_forward_filled: int = 0


@dataclass(frozen=True)
class FeatureSpec:
    """Which HAI columns the models see, and why the others are excluded."""

    features: list[str]
    constant: dict[str, float] = field(default_factory=dict)  # training-constant -> value
    dropped: list[str] = field(default_factory=list)  # explicitly dropped (HAI-2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": len(self.features),
            "features": self.features,
            "training_constant": self.constant,
            "dropped": self.dropped,
        }


def find_files(root: Path) -> list[HaiFile]:
    """``train*`` and ``test*`` CSV files in ``root``, in natural order."""
    found: list[tuple[str, int, HaiFile]] = []
    for path in root.iterdir() if root.is_dir() else []:
        match = _FILE.match(path.name)
        if match and path.is_file():
            split, number = match.group(1), int(match.group(2))
            found.append((split, number, HaiFile(f"{split}{number}", split, path)))
    found.sort(key=lambda item: (item[0] != "train", item[1]))
    return [item[2] for item in found]


def _separator(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    return ";" if header.count(";") > header.count(",") else ","


def label_columns(cfg: HaiConfig) -> list[str]:
    return [cfg.label_column, *cfg.process_label_columns]


def read_file(path: Path, cfg: HaiConfig, report: LoadReport | None = None) -> pd.DataFrame:
    """Read one HAI CSV: parse time, cast types, forward-fill short gaps, drop the rest."""
    frame = pd.read_csv(path, sep=_separator(path))
    frame.columns = [str(c).strip() for c in frame.columns]
    time_col = next((c for c in frame.columns if c.lower() in ("time", "timestamp")), None)
    if time_col is None:
        raise ValueError(f"{path.name}: no time column")
    frame = frame.rename(columns={time_col: TIME_COLUMN})
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], errors="coerce")
    labels = label_columns(cfg)
    missing_labels = [c for c in labels if c not in frame.columns]
    if missing_labels:
        raise ValueError(f"{path.name}: missing label columns {missing_labels}")
    features = [c for c in frame.columns if c != TIME_COLUMN and c not in labels]
    frame[features] = frame[features].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    frame[labels] = frame[labels].fillna(0).astype(np.int8)
    frame = frame.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)

    rows = len(frame)
    before = int(frame[features].isna().sum().sum())
    if before and cfg.max_ffill_s > 0:
        frame[features] = frame[features].ffill(limit=cfg.max_ffill_s)
    after = int(frame[features].isna().sum().sum())
    keep = frame[TIME_COLUMN].notna() & frame[features].notna().all(axis=1)
    frame = frame.loc[keep].reset_index(drop=True)
    if report is not None:
        report.rows_read += rows
        report.values_forward_filled += before - after
        report.rows_dropped_missing += rows - len(frame)
    return frame[[TIME_COLUMN, *features, *labels]]


def load(root: Path, cfg: HaiConfig) -> tuple[dict[str, pd.DataFrame], dict[str, Path], LoadReport]:
    """Read every HAI file in ``root``. Returns frames and paths keyed by file name."""
    files = find_files(root)
    if not any(f.split == "train" for f in files) or not any(f.split == "test" for f in files):
        raise FileNotFoundError(f"no HAI train and test files found in {root}")
    report = LoadReport()
    frames: dict[str, pd.DataFrame] = {}
    for item in files:
        print(f"  HAI {item.name}", flush=True)
        frames[item.name] = read_file(item.path, cfg, report)
    columns = {name: list(frame.columns) for name, frame in frames.items()}
    reference = next(iter(columns.values()))
    inconsistent = [name for name, cols in columns.items() if cols != reference]
    if inconsistent:
        raise ValueError(f"HAI files with a different column list: {inconsistent}")
    return frames, {f.name: f.path for f in files}, report


def feature_columns(frame: pd.DataFrame, cfg: HaiConfig) -> list[str]:
    labels = set(label_columns(cfg))
    return [c for c in frame.columns if c != TIME_COLUMN and c not in labels]


def build_feature_spec(train_frames: list[pd.DataFrame], cfg: HaiConfig) -> FeatureSpec:
    """Model every feature that varies in training, minus ``cfg.drop_features`` (HAI-2)."""
    if not train_frames:
        raise ValueError("no training frames")
    candidates = feature_columns(train_frames[0], cfg)
    stacked = pd.concat([f[candidates] for f in train_frames], ignore_index=True)
    lo = stacked.min()
    hi = stacked.max()
    constant = {c: float(lo[c]) for c in candidates if lo[c] == hi[c]}
    dropped = [c for c in cfg.drop_features if c in candidates and c not in constant]
    excluded = set(dropped) | (set(constant) if cfg.drop_train_constant_features else set())
    features = [c for c in candidates if c not in excluded]
    if not cfg.drop_train_constant_features:
        constant = {}
    return FeatureSpec(features=features, constant=constant, dropped=dropped)


def process_combination(frame: pd.DataFrame, cfg: HaiConfig, start: int, stop: int) -> str:
    """Which processes an event targeted, e.g. "P1+P3", from the process label columns."""
    targeted = [
        col.removeprefix("attack_")
        for col in cfg.process_label_columns
        if bool(frame[col].iloc[start:stop].to_numpy().any())
    ]
    return "+".join(targeted) if targeted else "unspecified"
