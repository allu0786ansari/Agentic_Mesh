"""Phase 5 data (ADR 0021): verify the new downloads without reading their contents.

    python -m agmesh.data.phase5 verify     # HAI 22.04 / 23.05 against the LFS checksums
    python -m agmesh.data.phase5 record     # checksums of the CIC compromised scenarios

Both commands only hash bytes; neither parses a file. That matters for the sealed
final data (``hai-23.05``, ``compromised-scada``): it must not be looked at before
``g5-frozen``. Hashing it is how we know later that the file we evaluate on is the
file we downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml

from agmesh.common.artifacts import write_artifact

CONFIG = Path("configs/datasets/phase5.yaml")
RECORD = Path("data/raw_checksums_phase5.json")


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_hai(cfg: dict[str, Any], root: Path = Path()) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rel, spec in cfg["hai"]["releases"].items():
        base = root / spec["dir"]
        for name, want in spec["files"].items():
            p = base / name
            if not p.is_file():
                out[f"{rel}/{name}"] = {"ok": False, "problem": "missing"}
                continue
            size = p.stat().st_size
            got = sha256(p) if size == want["bytes"] else None
            ok = size == want["bytes"] and got == want["sha256"]
            row: dict[str, Any] = {"ok": ok, "role": spec["role"], "bytes": size}
            if not ok:
                row["problem"] = "size or sha256 differ"
            out[f"{rel}/{name}"] = row
    return out


def record_cic(cfg: dict[str, Any], root: Path = Path()) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in cfg["modbus2023"]["scenarios"].items():
        if name == "external":
            continue
        base = root / spec["dir"]
        files = sorted(p for p in base.rglob("*") if p.is_file()) if base.is_dir() else []
        out[name] = {
            "role": spec["role"],
            "dir": spec["dir"],
            "files": {
                p.relative_to(base).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
                for p in files
            },
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 data checks (ADR 0021)")
    parser.add_argument("command", choices=("verify", "record"))
    args = parser.parse_args(argv)
    cfg = load_config()
    if args.command == "verify":
        rep = verify_hai(cfg)
        bad = {k: v for k, v in rep.items() if not v["ok"]}
        for k, v in rep.items():
            print(f"{'OK ' if v['ok'] else 'BAD'} {k} {v.get('problem', '')}")
        print(f"{len(rep) - len(bad)}/{len(rep)} HAI files verified")
        return 1 if bad else 0
    rep = record_cic(cfg)
    for name, v in rep.items():
        print(f"{name} ({v['role']}): {len(v['files'])} files")
    write_artifact(RECORD, kind="data.checksums.phase5", data_source="runtime", payload=rep)
    print(RECORD)
    return 0 if all(v["files"] for v in rep.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
