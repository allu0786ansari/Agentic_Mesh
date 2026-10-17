"""Check result artifacts before any number is reported (Implementation Plan §6.5).

    python -m agmesh.evaluation.checker [--results results] [--experiments X1 X2]

For every ``results/X*/<config_id>/seed_<n>.json`` the checker verifies that:

1. it matches the :class:`~agmesh.evaluation.results.ResultArtifact` schema;
2. ``data_source`` is ``runtime`` (fixture and synthetic results are rejected);
3. the provenance fields are present and non-empty, including the lock-file hash;
4. detection experiments (X1-X3) report every metric in ``DETECTION_METRICS`` and
   have their per-window score file, with the recorded hash (ADR 0005);
5. a test-split result has a matching entry (config, seed, commit) in
   ``results/test_access.log``.

Success-criteria checks (CC1-CC8, PT1-PT7) are added in Weeks 12-14 on top of this.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agmesh.common.provenance import sha256_file
from agmesh.evaluation.results import (
    DETECTION_METRICS,
    REQUIRED_PROVENANCE,
    ResultArtifact,
    scores_path,
)
from agmesh.evaluation.sealed import TEST_ACCESS_LOG, read_log

DETECTION_EXPERIMENTS = ("X1", "X2", "X3")


@dataclass
class Finding:
    path: str
    ok: bool
    problems: list[str] = field(default_factory=list)


def _score_file_problems(path: Path, record: Any) -> list[str]:
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        return ["no per-window score file recorded (needed for event-level inference)"]
    target = path.with_name(str(record["path"]))
    if target != scores_path(path):
        return ["score file name does not match the artifact"]
    if not target.is_file():
        return [f"score file {target.name} is missing"]
    if sha256_file(target) != record["sha256"]:
        return [f"score file {target.name} does not match its recorded hash"]
    return []


def check_file(path: Path, accesses: list[dict[str, Any]]) -> Finding:
    finding = Finding(path.as_posix(), ok=True)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        artifact = ResultArtifact.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        return Finding(path.as_posix(), ok=False, problems=[f"invalid artifact: {exc}"])
    problems = finding.problems
    if artifact.data_source != "runtime":
        problems.append(f"data_source is {artifact.data_source!r}; only runtime results count")
    prov = artifact.provenance
    for key in REQUIRED_PROVENANCE:
        if not prov.get(key):
            problems.append(f"provenance.{key} is missing")
    if not (prov.get("environment") or {}).get("lockfile_sha256"):
        problems.append("provenance.environment.lockfile_sha256 is missing")
    if prov.get("git_dirty"):
        changed = ", ".join(prov.get("git_dirty_paths") or []) or "unknown files"
        problems.append(f"produced from a working tree with uncommitted changes ({changed})")
    if artifact.experiment in DETECTION_EXPERIMENTS:
        missing = [m for m in DETECTION_METRICS if m not in artifact.metrics]
        if missing:
            problems.append(f"metrics missing: {missing}")
        problems.extend(_score_file_problems(path, raw.get("scores_file")))
    if path.parent.name != artifact.config_id or path.name != f"seed_{artifact.seed}.json":
        problems.append("file path does not match config_id and seed")
    if artifact.split == "test":
        logged = any(
            a.get("config_id") == artifact.config_id
            and a.get("seed") == artifact.seed
            and a.get("git_sha") == prov.get("git_sha")
            for a in accesses
        )
        if not logged:
            problems.append("test-split result has no matching entry in the test access log")
    finding.ok = not problems
    return finding


def check(
    results_root: Path, access_log: Path, experiments: list[str] | None = None
) -> list[Finding]:
    accesses = read_log(access_log)
    selected = experiments or sorted(p.name for p in results_root.glob("X*") if p.is_dir())
    findings: list[Finding] = []
    for experiment in selected:
        for path in sorted((results_root / experiment).glob("*/seed_*.json")):
            findings.append(check_file(path, accesses))
    return findings


def check_traces(run_dir: Path, traces_root: Path = Path("results/traces")) -> list[str]:
    """Every session of a mesh run has a trace with its ``session.id`` (ADR 0014 §7, EA3)."""
    from agmesh.observability.tracing import check_run

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    run_id = summary["run_id"]
    sessions = sorted(
        {
            str(e["payload"]["session_id"])
            for e in map(json.loads, (run_dir / "events.jsonl").read_text().splitlines())
            if e.get("event") == "session" and e["payload"].get("session_id")
        }
    )
    if not sessions:
        return [f"{run_dir}: no sessions"]
    return check_run(traces_root / run_id, sessions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate result artifacts")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--access-log", type=Path, default=TEST_ACCESS_LOG)
    parser.add_argument("--experiments", nargs="*", default=None, type=lambda v: v.strip(","))
    parser.add_argument(
        "--traces",
        nargs="+",
        type=Path,
        default=None,
        metavar="MESH_RUN",
        help="check that every session of these mesh runs has a trace (EA3)",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/experiments/x1_x2.yaml"),
        help="X1/X2 run matrix: report missing runs and the CC4 privacy rule (ADR 0016)",
    )
    args = parser.parse_args(argv)
    grid_exps = sorted(set(args.experiments or []) & {"X1", "X2"})
    if grid_exps and args.matrix.is_file():
        from agmesh.common.config import load_params
        from agmesh.experiments.grid import completeness, privacy_violations

        parts = load_params().paths.partitions_dir
        missing = completeness(args.results, parts, grid_exps, args.matrix)
        over = privacy_violations(args.results, grid_exps)
        print(f"matrix {args.matrix}: {len(missing)} expected run(s) missing for {grid_exps}")
        for m in missing[:50]:
            print(f"  - missing {m}")
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")
        print(f"privacy rule (CC4): {len(over)} violation(s)")
        for o in over:
            print(f"  - {o}")
        if missing or over:
            findings = check(args.results, args.access_log, args.experiments)
            bad = [f for f in findings if not f.ok]
            print(f"{len(findings) - len(bad)}/{len(findings)} result artifacts accepted")
            return 1
    if args.traces:
        failed = 0
        for run in args.traces:
            problems = check_traces(run, args.results / "traces")
            failed += bool(problems)
            print(f"{run}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
            for p in problems[:20]:
                print(f"  - {p}")
        return 1 if failed else 0
    findings = check(args.results, args.access_log, args.experiments)
    bad = [f for f in findings if not f.ok]
    for f in bad:
        print(f"REJECTED {f.path}")
        for problem in f.problems:
            print(f"  - {problem}")
    print(f"{len(findings) - len(bad)}/{len(findings)} result artifacts accepted")
    if not findings:
        print("no result artifacts found", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
