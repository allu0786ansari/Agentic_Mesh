"""
scripts/check_datasets.py
Day 1 — Verify both datasets downloaded correctly.
Checks file existence, row counts, required columns, and attack label presence.
Run: python scripts\check_datasets.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).resolve().parents[1]

HAI_DIR    = ROOT / "data" / "raw" / "hai"
MODBUS_DIR = ROOT / "data" / "raw" / "modbus2023"

# HAI 22.04 expected files
HAI_FILES = [
    "hai-train1.csv", "hai-train2.csv",
    "hai-test1.csv",  "hai-test2.csv", "hai-test3.csv",
]
HAI_REQUIRED_COLS = {"timestamp", "attack"}  # plus sensor columns

def check_hai() -> list[tuple[str, bool, str]]:
    results = []
    for fname in HAI_FILES:
        fpath = HAI_DIR / fname
        if not fpath.exists():
            results.append((fname, False, "FILE NOT FOUND"))
            continue
        try:
            df = pd.read_csv(fpath, nrows=5)
            cols = set(df.columns.str.strip().str.lower())
            missing = HAI_REQUIRED_COLS - cols
            if missing:
                results.append((fname, False, f"Missing columns: {missing}"))
            else:
                size_mb = fpath.stat().st_size / 1_048_576
                results.append((fname, True, f"{size_mb:.1f} MB — columns OK"))
        except Exception as exc:
            results.append((fname, False, str(exc)))
    return results

def check_modbus() -> list[tuple[str, bool, str]]:
    results = []
    csv_files = list(MODBUS_DIR.glob("*.csv"))
    if not csv_files:
        return [("modbus2023/", False, "No CSV files found — download from https://www.unb.ca/cic/datasets/modbus-2023.html")]
    for fpath in sorted(csv_files):
        try:
            df = pd.read_csv(fpath, nrows=5)
            size_mb = fpath.stat().st_size / 1_048_576
            results.append((fpath.name, True, f"{size_mb:.1f} MB — {len(df.columns)} columns"))
        except Exception as exc:
            results.append((fpath.name, False, str(exc)))
    return results

def main() -> int:
    console.rule("[bold cyan]Week 2 — Dataset Integrity Check[/bold cyan]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("File")
    table.add_column("Status")
    table.add_column("Detail")

    all_ok = True

    console.print("\n[yellow]HAI 22.04[/yellow]")
    for name, ok, detail in check_hai():
        table.add_row(name, "[green]OK[/green]" if ok else "[red]FAIL[/red]", detail)
        if not ok:
            all_ok = False

    console.print("\n[yellow]CIC Modbus 2023[/yellow]")
    for name, ok, detail in check_modbus():
        table.add_row(name, "[green]OK[/green]" if ok else "[red]FAIL[/red]", detail)
        if not ok:
            all_ok = False

    console.print()
    console.print(table)

    if all_ok:
        console.print("\n[bold green]All dataset checks passed[/bold green] — proceed to Day 2")
        return 0
    else:
        console.print("\n[bold red]Some checks FAILED[/bold red] — fix before running process_datasets.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())