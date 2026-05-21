"""
notebooks/01_data_exploration.py
Day 1 — Quick data exploration script (run as a plain .py or convert to Jupyter).
Documents dataset sizes, class distributions, and sensor statistics.

Run: python notebooks\01_data_exploration.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT     = Path(__file__).resolve().parents[1]
HAI_DIR  = ROOT / "data" / "raw" / "hai"
MOD_DIR  = ROOT / "data" / "raw" / "modbus2023"

print("=" * 60)
print("HAI 22.04 — Dataset Summary")
print("=" * 60)

hai_files = sorted(HAI_DIR.glob("*.csv"))
for f in hai_files:
    df = pd.read_csv(f, nrows=10_000)
    df.columns = df.columns.str.strip().str.lower()
    n_sensors = len([c for c in df.columns if c not in {"timestamp","attack","attack_p1","attack_p2","attack_p3"}])
    has_attack = "attack" in df.columns
    print(f"  {f.name:30s}  size={f.stat().st_size//1024:6d} KB  sensors={n_sensors}  has_attack={has_attack}")

print()
print("=" * 60)
print("CIC Modbus 2023 — Dataset Summary")
print("=" * 60)

mod_files = sorted(MOD_DIR.glob("*.csv"))
if mod_files:
    for f in mod_files:
        df = pd.read_csv(f, nrows=1_000)
        df.columns = df.columns.str.strip().str.lower()
        label_cols = [c for c in df.columns if "label" in c or "class" in c]
        print(f"  {f.name:40s}  size={f.stat().st_size//1024:6d} KB  label_cols={label_cols}")
        if label_cols:
            print(f"    Label distribution (sample): {df[label_cols[0]].value_counts().to_dict()}")
else:
    print("  No CSV files found — download from https://www.unb.ca/cic/datasets/modbus-2023.html")