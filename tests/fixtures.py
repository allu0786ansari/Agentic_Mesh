"""Builders for small synthetic inputs used by the unit tests.

Every file produced here is synthetic. Nothing in the test suite reads the real
datasets, so CI never needs raw industrial data.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pandas as pd


def write_pcap(path: Path, timestamps: list[float], *, nanoseconds: bool = False) -> Path:
    """Write a minimal little-endian pcap with one 4-byte packet per timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    magic = 0xA1B23C4D if nanoseconds else 0xA1B2C3D4
    scale = 1e9 if nanoseconds else 1e6
    out = bytearray(struct.pack("<IHHiIII", magic, 2, 4, 0, 0, 65535, 1))
    for ts in timestamps:
        seconds = int(ts)
        fraction = round((ts - seconds) * scale)
        out += struct.pack("<IIII", seconds, fraction, 4, 4) + b"\x00\x01\x02\x03"
    path.write_bytes(bytes(out))
    return path


def _pcapng_block(block_type: int, body: bytes) -> bytes:
    padding = (4 - len(body) % 4) % 4
    total = 12 + len(body) + padding
    return (
        struct.pack("<II", block_type, total) + body + b"\x00" * padding + struct.pack("<I", total)
    )


def write_pcapng(path: Path, timestamps: list[float], *, tsresol: int | None = None) -> Path:
    """Write a minimal pcapng (SHB, one IDB, one EPB per timestamp)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    shb = _pcapng_block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    options = b""
    resolution = 1e-6
    if tsresol is not None:
        options = (
            struct.pack("<HH", 9, 1) + bytes([tsresol]) + b"\x00" * 3 + struct.pack("<HH", 0, 0)
        )
        resolution = 10.0**-tsresol
    idb = _pcapng_block(0x00000001, struct.pack("<HHI", 1, 0, 65535) + options)
    out = bytearray(shb + idb)
    for ts in timestamps:
        ticks = round(ts / resolution)
        body = struct.pack("<IIIII", 0, ticks >> 32, ticks & 0xFFFFFFFF, 4, 4) + b"\x00\x01\x02\x03"
        out += _pcapng_block(0x00000006, body)
    path.write_bytes(bytes(out))
    return path


def write_hai_csv(
    path: Path,
    *,
    hours: float,
    attack_windows: list[tuple[int, int]] | None = None,
    start: str = "2021-07-11 10:00:00",
    n_features: int = 5,
    seed: int = 0,
    level_shift_every_s: int | None = None,
) -> Path:
    """Write a gzipped HAI-like CSV at 1 Hz with ``attack``/``attack_P1`` label columns.

    ``attack_windows`` holds (start_row, end_row) pairs, end exclusive.
    ``level_shift_every_s`` adds a step change to the first feature, imitating
    operating regimes for the regime analysis.
    """
    rng = np.random.default_rng(seed)
    n = int(hours * 3600)
    frame = pd.DataFrame({"time": pd.date_range(start, periods=n, freq="s").astype(str)})
    for i in range(n_features):
        values = rng.normal(0.0, 1.0, n)
        if i == 0 and level_shift_every_s:
            values += 10.0 * ((np.arange(n) // level_shift_every_s) % 2)
        frame[f"P1_F{i:02d}"] = values
    label = np.zeros(n, dtype=int)
    for begin, end in attack_windows or []:
        label[begin:end] = 1
    frame["attack"] = label
    frame["attack_P1"] = label
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False)
    return path


def write_lfs_pointer(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 123456\n",
        encoding="utf-8",
    )
    return path
