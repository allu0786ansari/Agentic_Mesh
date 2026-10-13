"""Record or verify SHA-256 checksums of everything under data/raw.

uv run python scripts/verify_checksums.py --record   # write data/raw_checksums.json
uv run python scripts/verify_checksums.py --verify   # compare against it (exit 1 on mismatch)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agmesh.common.checksums import record, verify
from agmesh.common.config import DEFAULT_PARAMS_PATH, load_params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--root", type=Path, default=None, help="default: paths.raw_root")
    parser.add_argument("--manifest", type=Path, default=None, help="default: paths.checksums_file")
    args = parser.parse_args(argv)

    params = load_params(args.params)
    root = args.root or params.paths.raw_root
    manifest = args.manifest or params.paths.checksums_file
    if not root.is_dir():
        print(f"ERROR: {root} does not exist. Copy the raw datasets there first.", file=sys.stderr)
        return 1

    if args.record:
        payload = record(root, manifest, progress=True)
        print(
            f"Recorded {payload['n_files']} files ({payload['total_bytes'] / 2**30:.2f} GiB) "
            f"to {manifest}"
        )
        if payload["lfs_pointers"]:
            print("WARNING: these files are Git LFS pointers, not data:")
            for path in payload["lfs_pointers"]:
                print(f"  {path}")
        return 0

    if not manifest.is_file():
        print(f"ERROR: {manifest} not found. Run with --record first.", file=sys.stderr)
        return 1
    report = verify(root, manifest, progress=True)
    for label, items in (
        ("missing", report.missing),
        ("changed", report.changed),
        ("unexpected", report.unexpected),
    ):
        for item in items:
            print(f"  {label}: {item}")
    print("Checksums match." if report.ok else "Checksum verification FAILED.")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
