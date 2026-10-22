"""The sealed test set (WR-2, Proposal §10.1).

Test partitions can be read only with a :class:`FinalAccess`, which entry points
create only when run with ``--final``. Every access is appended to
``results/test_access.log`` (one JSON object per line) with the commit, the
configuration and the seed. By default a final access also requires the
``g4-frozen`` Git tag, so final runs cannot happen before Gate G4 freezes the
configurations. The checker rejects test-split results without a matching log entry.

**Preconditions (ADR 0019 §3).** With ``require_frozen``, :func:`frozen_problem`
must find nothing wrong:

* the tag exists;
* ``HEAD`` is the tagged commit, or a descendant that differs from it only under
  ``results/``;
* the tree is clean outside ``results/`` (appending to the access log is allowed).

The tagged commit is recorded in every log entry as ``frozen_sha``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agmesh.common.provenance import OUTPUT_PATHS, git_state, sha256_file, utc_now_iso

TEST_ACCESS_LOG = Path("results/test_access.log")
FROZEN_TAG = "g4-frozen"
#: Phase 5 (ADR 0021): the final data may be read only after this tag.
FINAL_TAG = "g5-frozen"
PHASE5_CONFIG = Path("configs/datasets/phase5.yaml")


class SealedTestError(PermissionError):
    """Test data was requested without an authorised final run."""


def tag_exists(tag: str = FROZEN_TAG) -> bool:
    try:
        out = subprocess.run(
            ["git", "tag", "--list", tag], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and out.stdout.strip() == tag


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def frozen_sha(tag: str = FROZEN_TAG, cwd: Path | None = None) -> str | None:
    return _git("rev-parse", "--verify", "-q", f"{tag}^{{commit}}", cwd=cwd) or None


def frozen_problem(tag: str = FROZEN_TAG, cwd: Path | None = None) -> str | None:
    """Why a final access is not allowed here, or None (ADR 0019 §3)."""
    tagged = frozen_sha(tag, cwd)
    if tagged is None:
        return (
            f"--final needs the Git tag {tag!r}, created at Gate G4 when the "
            "configurations are frozen"
        )
    head = _git("rev-parse", "HEAD", cwd=cwd)
    if head != tagged:
        if _git("merge-base", "--is-ancestor", tagged, "HEAD", cwd=cwd) is None:
            return f"HEAD {head} does not descend from {tag} ({tagged})"
        changed = (_git("diff", "--name-only", tagged, "HEAD", cwd=cwd) or "").splitlines()
        code = [c for c in changed if not any(c.startswith(f"{o}/") for o in OUTPUT_PATHS)]
        if code:
            return f"files changed since {tag} outside results/: {code[:10]}"
    state = git_state(cwd)
    if state["git_dirty"]:
        return f"uncommitted changes outside results/: {state['git_dirty_paths']}"
    return None


@dataclass(frozen=True)
class FinalAccess:
    """Permission to read test data for one frozen configuration and seed."""

    config_id: str
    seed: int
    log_path: Path = TEST_ACCESS_LOG
    params_path: Path = Path("configs/params.yaml")
    require_frozen: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def authorise(access: FinalAccess | None, *, site: str, client: str) -> None:
    """Raise unless test access is allowed; log the access when it is."""
    if access is None:
        raise SealedTestError(
            f"{site}/{client}: the test split is sealed; it can only be read by a --final "
            "run after Gate G4 (WR-2)"
        )
    frozen: str | None = None
    if access.require_frozen:
        problem = frozen_problem()
        if problem is not None:
            raise SealedTestError(problem)
        frozen = frozen_sha()
    record = {
        "ts": utc_now_iso(),
        **git_state(),
        "config_id": access.config_id,
        "seed": access.seed,
        "site": site,
        "client": client,
        "params_sha256": sha256_file(access.params_path) if access.params_path.is_file() else None,
        "frozen_sha": frozen,
        **access.extra,
    }
    access.log_path.parent.mkdir(parents=True, exist_ok=True)
    with access.log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def sealed_paths(config: Path = PHASE5_CONFIG) -> list[Path]:
    """The phase 5 final data (ADR 0021 §1), from ``configs/datasets/phase5.yaml``."""
    import yaml

    if not config.is_file():
        return []
    return [Path(p) for p in (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get(
        "sealed", [])]  # fmt: skip


def assert_not_sealed(
    path: Path | str, access: FinalAccess | None = None, *, config: Path = PHASE5_CONFIG
) -> None:
    """Refuse to read a sealed phase 5 path unless the final run is authorised.

    A path is sealed when it is inside one listed in the phase 5 config. It may be
    read only with a :class:`FinalAccess` after the ``g5-frozen`` tag, with the same
    commit rules as G4; the read is logged.
    """
    target = Path(path).resolve()
    for sealed in sealed_paths(config):
        if target == sealed.resolve() or sealed.resolve() in target.parents:
            if access is None:
                raise SealedTestError(
                    f"{path} is sealed final data (ADR 0021): it can only be read by the "
                    f"frozen pipeline after {FINAL_TAG!r}"
                )
            problem = frozen_problem(FINAL_TAG) if access.require_frozen else None
            if problem is not None:
                raise SealedTestError(problem)
            authorise_record(access, site=str(sealed), client=target.name)
            return


def authorise_record(access: FinalAccess, *, site: str, client: str) -> None:
    """Append one access entry (the checks are the caller's)."""
    record = {
        "ts": utc_now_iso(),
        **git_state(),
        "config_id": access.config_id,
        "seed": access.seed,
        "site": site,
        "client": client,
        "frozen_sha": frozen_sha(FINAL_TAG),
        **access.extra,
    }
    access.log_path.parent.mkdir(parents=True, exist_ok=True)
    with access.log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_log(path: Path = TEST_ACCESS_LOG) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def add_final_argument(parser: argparse.ArgumentParser) -> None:
    """The ``--final`` flag shared by every entry point that may evaluate on test data."""
    parser.add_argument(
        "--final",
        action="store_true",
        help="evaluate on the sealed test split (logged; needs the g4-frozen tag)",
    )
