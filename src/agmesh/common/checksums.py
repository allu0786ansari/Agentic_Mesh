"""Record and verify SHA-256 checksums of the raw datasets.

The raw data never enters Git (it is DVC-tracked), but its checksum manifest does,
so anyone can confirm they are running on exactly the same files (CC7). Git LFS
pointer files — small text stubs left behind when an LFS download fails — are
flagged, because they look like data files but contain none.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agmesh.common.artifacts import read_artifact, write_artifact
from agmesh.common.provenance import sha256_file

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"
_SKIP_NAMES = {".gitignore", ".DS_Store", "Thumbs.db", "desktop.ini"}


def is_lfs_pointer(path: Path) -> bool:
    """True if ``path`` is a Git LFS pointer stub rather than real content."""
    try:
        if path.stat().st_size > 1024:
            return False
        with path.open("rb") as handle:
            return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


@dataclass(frozen=True)
class FileEntry:
    path: str
    size_bytes: int
    sha256: str
    lfs_pointer: bool


def iter_data_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.name not in _SKIP_NAMES and not p.name.endswith(".dvc")
    )


def scan(root: Path, *, progress: bool = False) -> list[FileEntry]:
    files = iter_data_files(root)
    entries: list[FileEntry] = []
    for i, path in enumerate(files, start=1):
        if progress:
            print(f"  [{i}/{len(files)}] {path.relative_to(root).as_posix()}", flush=True)
        entries.append(
            FileEntry(
                path=path.relative_to(root).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                lfs_pointer=is_lfs_pointer(path),
            )
        )
    return entries


def record(root: Path, manifest: Path, *, progress: bool = False) -> dict[str, Any]:
    """Hash every file under ``root`` and write the manifest artifact."""
    entries = scan(root, progress=progress)
    payload = {
        "root": root.as_posix(),
        "n_files": len(entries),
        "total_bytes": sum(e.size_bytes for e in entries),
        "lfs_pointers": [e.path for e in entries if e.lfs_pointer],
        "files": [asdict(e) for e in entries],
    }
    write_artifact(manifest, kind="data.raw_checksums", data_source="runtime", payload=payload)
    return payload


@dataclass
class VerifyReport:
    missing: list[str]
    changed: list[str]
    unexpected: list[str]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.changed or self.unexpected)


def verify(root: Path, manifest: Path, *, progress: bool = False) -> VerifyReport:
    """Compare the files under ``root`` with a previously recorded manifest."""
    recorded = {f["path"]: f for f in read_artifact(manifest)["files"]}
    current = {e.path: e for e in scan(root, progress=progress)}
    return VerifyReport(
        missing=sorted(set(recorded) - set(current)),
        changed=sorted(
            p for p in set(recorded) & set(current) if recorded[p]["sha256"] != current[p].sha256
        ),
        unexpected=sorted(set(current) - set(recorded)),
    )
