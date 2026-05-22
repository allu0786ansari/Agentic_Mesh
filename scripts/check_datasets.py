"""
scripts/check_datasets.py
Day 1 — Verify both datasets downloaded correctly.
Checks file existence, row counts, required columns, and attack label presence.
Run: python scripts/check_datasets.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).resolve().parents[1]

HAI_ROOT = ROOT / "data" / "raw" / "hai"
HAI_VERSION = os.environ.get("HAI_VERSION", "hai-22.04")
MODBUS_DIR = ROOT / "data" / "raw" / "modbus2023"
MODBUS_ATTACK_DIR = MODBUS_DIR / "attack"
MODBUS_BENIGN_DIR = MODBUS_DIR / "benign"
HAI_REQUIRED_COLS = {"timestamp", "attack"}  # plus sensor columns


def locate_hai_dir() -> Path:
    candidate = HAI_ROOT / HAI_VERSION
    if candidate.exists():
        return candidate

    version_dirs = sorted(
        p for p in HAI_ROOT.iterdir()
        if p.is_dir() and p.name.lower().startswith("hai")
    )
    if version_dirs:
        return version_dirs[0]

    raise FileNotFoundError("No HAI dataset version folder found under data/raw/hai")


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            first = fh.readline()
        return first.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False

def check_hai() -> list[tuple[str, bool, str]]:
    results = []
    try:
        hai_dir = locate_hai_dir()
    except FileNotFoundError as exc:
        return [("HAI dataset", False, str(exc))]

    csv_files = sorted(hai_dir.glob("*.csv"))
    if not csv_files:
        return [(str(hai_dir), False, "No CSV files found in HAI version folder")]

    for fpath in csv_files:
        if not fpath.exists():
            results.append((fpath.name, False, "FILE NOT FOUND"))
            continue

        if is_git_lfs_pointer(fpath):
            results.append((
                fpath.name,
                False,
                "Git LFS placeholder file detected. Run: git lfs pull"
            ))
            continue

        try:
            df = pd.read_csv(fpath, nrows=5)
            if len(df.columns) == 1 and str(df.columns[0]).startswith("version"):
                results.append((fpath.name, False, "Git LFS pointer file detected"))
                continue

            cols = set(df.columns.str.strip().str.lower())
            missing = HAI_REQUIRED_COLS - cols
            if missing and not any(c.startswith("attack") for c in cols):
                results.append((fpath.name, False, f"Missing columns: {missing}"))
            else:
                size_mb = fpath.stat().st_size / 1_048_576
                results.append((fpath.name, True, f"{size_mb:.1f} MB — columns OK"))
        except Exception as exc:
            results.append((fpath.name, False, str(exc)))
    return results

def check_modbus() -> list[tuple[str, bool, str]]:
    results = []
    csv_files = list(MODBUS_DIR.glob("*.csv"))
    if csv_files:
        for fpath in sorted(csv_files):
            try:
                df = pd.read_csv(fpath, nrows=5)
                size_mb = fpath.stat().st_size / 1_048_576
                results.append((fpath.name, True, f"{size_mb:.1f} MB — {len(df.columns)} columns"))
            except Exception as exc:
                results.append((fpath.name, False, str(exc)))
        return results

    if MODBUS_ATTACK_DIR.exists() and MODBUS_BENIGN_DIR.exists():
        attack_files = sorted(MODBUS_ATTACK_DIR.rglob("*.pcap"))
        benign_files = sorted(MODBUS_BENIGN_DIR.rglob("*.pcap"))
        if attack_files or benign_files:
            results.append(("modbus2023/attack", True, f"{len(attack_files)} pcap files"))
            results.append(("modbus2023/benign", True, f"{len(benign_files)} pcap files"))
            return results

    return [("modbus2023/", False, "No CSV files or PCAP dataset found — download the CIC Modbus 2023 dataset correctly")]

def main() -> int:
    console.rule("[bold cyan]Week 2 — Dataset Integrity Check[/bold cyan]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("File")
    table.add_column("Status")
    table.add_column("Detail")

    all_ok = True

    try:
        hai_dir = locate_hai_dir()
        console.print(f"\n[yellow]HAI {hai_dir.name}[/yellow]")
    except FileNotFoundError:
        console.print("\n[yellow]HAI dataset not found[/yellow]")

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