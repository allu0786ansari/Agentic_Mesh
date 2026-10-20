"""Provenance helpers: hashes, git revision and environment description.

Every artifact the project writes carries this information so that each reported
number can be traced to the exact code, data, parameters and machine that produced
it (Proposal §9.5, completion criterion CC3).
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

_CHUNK = 1 << 20  # 1 MiB


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path | str) -> str:
    """SHA-256 of a file, streamed in 1 MiB chunks so large captures fit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


#: Paths that hold outputs, not inputs. Rewriting them (for example re-running an
#: experiment whose artifacts are committed) does not make a run irreproducible.
OUTPUT_PATHS = ("results",)


def git_state(cwd: Path | None = None) -> dict[str, Any]:
    """Return the current commit, branch and whether code, config or inputs have local changes.

    Tracked files under :data:`OUTPUT_PATHS` are ignored: an artifact must not count as
    "uncommitted" because an earlier artifact of the same run was just rewritten.
    ``git_dirty_paths`` lists up to ten changed files, so a rejection can be explained.
    """
    sha = _git("rev-parse", "HEAD", cwd=cwd)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    excludes = [f":(exclude){p}" for p in OUTPUT_PATHS]
    status = _git("status", "--porcelain", "--untracked-files=no", "--", ".", *excludes, cwd=cwd)
    paths = (
        [line.split(None, 1)[-1] for line in status.splitlines() if line.strip()] if status else []
    )
    return {
        "git_sha": sha or "unknown",
        "git_branch": branch or "unknown",
        "git_dirty": bool(paths) if status is not None else None,
        "git_dirty_paths": paths[:10],
    }


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


#: Packages whose installed version can change numerical results (ADR 0018 §1). The
#: lock file's hash says what should be installed; these say what was.
NUMERIC_PACKAGES = ("numpy", "scipy", "torch", "opacus", "scikit-learn", "pandas", "flwr")


def installed_versions(names: tuple[str, ...] = NUMERIC_PACKAGES) -> dict[str, str | None]:
    """The installed version of each package, or None if it is not installed."""
    from importlib import metadata

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment(lockfile: Path | str = "uv.lock") -> dict[str, Any]:
    """Describe the Python runtime and the machine the code is running on."""
    lock = Path(lockfile)
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "wsl": _is_wsl(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_mb": round(memory.total / 2**20),
        "swap_total_mb": round(swap.total / 2**20),
        "lockfile_sha256": sha256_file(lock) if lock.is_file() else None,
        "packages": installed_versions(),
        "hostname_hash": sha256_bytes(platform.node().encode())[:12],
        "pid": os.getpid(),
    }


def build_provenance(
    *, params_path: Path | str | None = None, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Assemble the provenance block written into every artifact."""
    record: dict[str, Any] = {"created_at": utc_now_iso(), **git_state()}
    if params_path is not None and Path(params_path).is_file():
        record["params_sha256"] = sha256_file(params_path)
    record["environment"] = environment()
    if extra:
        record.update(extra)
    return record
