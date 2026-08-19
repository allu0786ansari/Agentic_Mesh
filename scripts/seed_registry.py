from __future__ import annotations

import argparse
from pathlib import Path

from storage.sqlite_schema import seed_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the SQLite node registry from partition metadata")
    parser.add_argument("--partitions", type=Path, default=Path("data/partitions"))
    parser.add_argument("--database", type=Path, default=Path("agmesh.db"))
    args = parser.parse_args()

    metadata_paths = sorted(args.partitions.glob("node_*/meta.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No node metadata found under {args.partitions}")
    count = seed_registry(args.database, metadata_paths)
    print(f"Seeded {count} node registry records into {args.database}")


if __name__ == "__main__":
    main()
