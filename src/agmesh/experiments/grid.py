"""Run the X1 and X2 matrix (ADR 0016), resumably, one run per process.

    python -m agmesh.experiments.grid --dry-run           # what remains, with an ETA
    python -m agmesh.experiments.grid --hours 9           # start runs for at most 9 h
    python -m agmesh.experiments.grid --only X1-fed --sites site_a
    make grid HOURS=9

**The matrix.** ``configs/experiments/x1_x2.yaml`` lists groups. Each group expands
to every combination of site, strategy, ε, μ and seed, in the file's order.

**One run per process.** Each run calls ``agmesh.federated.run`` in its own process
with one seed. Its memory is returned when it ends, and Ctrl-C stops only the
current run.

**Skipping and the time budget.**

* A run whose result artifact already exists is skipped.
* With ``--hours``, no new run starts once the budget would be exceeded by the
  run's estimate.

**The log.** ``results/grid/log.jsonl`` records every start and finish: command,
seconds and exit code. A failed run is not retried automatically. It runs again,
with the same seed, the next time the grid is started (ADR 0016 §5: seeds are never
replaced).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agmesh.common.config import DEFAULT_PARAMS_PATH, load_params
from agmesh.evaluation.results import result_path
from agmesh.federated.run import config_id, eps_label, partition_suffix

MATRIX = Path("configs/experiments/x1_x2.yaml")
LOG = Path("results/grid/log.jsonl")


@dataclass(frozen=True)
class Run:
    group: str
    experiment: str
    site: str
    mode: str
    strategy: str
    epsilon: float
    seed: int
    h1c: bool
    heterogeneous: bool
    dropout: float
    mu: float | None
    partitions: Path | None
    minutes: float

    def config_id(self, default_partitions: Path) -> str:
        parts = self.partitions or default_partitions
        cid = config_id(
            self.site,
            self.strategy,
            self.epsilon,
            h1c=self.h1c,
            heterogeneous=self.heterogeneous,
            dropout=self.dropout,
            mode=self.mode,
        )
        manifest = parts / self.site / "manifest.json"
        if self.partitions is not None and not manifest.is_file():
            tag = "__iid"  # not built yet (make partitions-iid); the matrix's only variant
        else:
            tag = partition_suffix(parts, self.site)
        return cid + tag + (f"__mu{self.mu:g}" if self.mu is not None else "")

    def artifact(self, results: Path, default_partitions: Path) -> Path:
        return result_path(results, self.experiment, self.config_id(default_partitions), self.seed)

    def argv(self, results: Path, params: Path) -> list[str]:
        args = [
            "--params", str(params), "--site", self.site, "--strategy", self.strategy,
            "--epsilon", "inf" if math.isinf(self.epsilon) else f"{self.epsilon:g}",
            "--seeds", str(self.seed), "--mode", self.mode, "--experiment", self.experiment,
            "--results", str(results),
        ]  # fmt: skip
        if self.h1c:
            args.append("--h1c")
        if self.heterogeneous:
            args.append("--heterogeneous")
        if self.dropout:
            args += ["--dropout", f"{self.dropout:g}"]
        if self.mu is not None:
            args += ["--mu", f"{self.mu:g}"]
        if self.partitions is not None:
            args += ["--partitions", str(self.partitions)]
        return args


def load_matrix(path: Path = MATRIX) -> tuple[list[int], list[dict[str, Any]]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [int(s) for s in data["seeds"]], list(data["groups"])


def expand(path: Path = MATRIX) -> list[Run]:
    seeds, groups = load_matrix(path)
    runs: list[Run] = []
    for g in groups:
        mus = g.get("mus") or [None]
        for site, strategy, eps, mu, seed in itertools.product(
            g["sites"], g["strategies"], g["epsilons"], mus, seeds
        ):
            runs.append(
                Run(
                    group=str(g["id"]),
                    experiment=str(g["experiment"]),
                    site=str(site),
                    mode=str(g.get("mode", "federated")),
                    strategy=str(strategy),
                    epsilon=float(eps),
                    seed=int(seed),
                    h1c=bool(g.get("h1c", False)),
                    heterogeneous=bool(g.get("heterogeneous", False)),
                    dropout=float(g.get("dropout", 0.0)),
                    mu=None if mu is None else float(mu),
                    partitions=Path(g["partitions"]) if g.get("partitions") else None,
                    minutes=float(g["minutes"][site]),
                )
            )
    return runs


def remaining(runs: list[Run], results: Path, default_partitions: Path) -> list[Run]:
    return [r for r in runs if not r.artifact(results, default_partitions).is_file()]


def completeness(
    results: Path, default_partitions: Path, experiments: list[str], matrix: Path = MATRIX
) -> list[str]:
    """Expected (experiment, config, seed) with no artifact (Plan Week 12 check)."""
    return [
        f"{r.experiment}/{r.config_id(default_partitions)}/seed_{r.seed}"
        for r in expand(matrix)
        if r.experiment in experiments and not r.artifact(results, default_partitions).is_file()
    ]


def privacy_violations(results: Path, experiments: list[str]) -> list[str]:
    """CC4: in every private run, every client's final ε is at most the target."""
    bad = []
    for exp in experiments:
        for path in sorted((results / exp).glob("*/seed_*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            target = doc.get("target_epsilon")
            clients = (doc.get("privacy") or {}).get("clients") or {}
            if target in (None, "inf"):
                continue
            limit = float(target) * (1 + 1e-9)
            over = {
                c: v.get("epsilon")
                for c, v in clients.items()
                if v.get("epsilon") is None or float(v["epsilon"]) > limit
            }
            if over:
                bad.append(f"{path}: client ε above target {target}: {over}")
    return bad


def _log(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def describe(r: Run) -> str:
    extra = "".join(
        [
            " h1c" if r.h1c else "",
            " het" if r.heterogeneous else "",
            f" drop{r.dropout:g}" if r.dropout else "",
            f" mu{r.mu:g}" if r.mu is not None else "",
            " iid" if r.partitions else "",
        ]
    )
    return (
        f"{r.group} {r.site} {r.mode} {r.strategy} eps{eps_label(r.epsilon)}{extra} seed {r.seed}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the X1/X2 matrix (ADR 0016)")
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--only", nargs="*", default=None, help="group ids")
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--hours", type=float, default=None, help="time budget for this batch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    params = load_params(args.params)
    default_parts = params.paths.partitions_dir
    runs = [
        r
        for r in expand(args.matrix)
        if (not args.only or r.group in args.only) and (not args.sites or r.site in args.sites)
    ]
    todo = remaining(runs, args.results, default_parts)
    est_h = sum(r.minutes for r in todo) / 60
    print(f"{len(runs)} runs in scope, {len(runs) - len(todo)} done, {len(todo)} to go "
          f"(about {est_h:.1f} h)")  # fmt: skip
    missing_parts = sorted(
        {str(r.partitions) for r in todo if r.partitions and not r.partitions.is_dir()}
    )
    if missing_parts:
        print(f"note: partitions {missing_parts} do not exist yet (make partitions-iid)")
    if args.dry_run:
        by_group: dict[str, list[Run]] = {}
        for r in todo:
            by_group.setdefault(r.group, []).append(r)
        for g, rs in by_group.items():
            print(f"  {g}: {len(rs)} runs, about {sum(r.minutes for r in rs) / 60:.1f} h")
        return 0
    started = time.monotonic()
    budget_s = None if args.hours is None else args.hours * 3600
    for i, r in enumerate(todo, 1):
        if r.partitions and not r.partitions.is_dir():
            print(f"skip (no partitions {r.partitions}): {describe(r)}")
            continue
        elapsed = time.monotonic() - started
        if budget_s is not None and elapsed + r.minutes * 60 > budget_s:
            print(f"time budget reached after {elapsed / 3600:.1f} h; {len(todo) - i + 1} left")
            break
        cmd = [sys.executable, "-m", "agmesh.federated.run", *r.argv(args.results, args.params)]
        print(f"[{i}/{len(todo)}] {describe(r)} (about {r.minutes:.0f} min)", flush=True)
        _log(LOG, {"event": "start", "run": describe(r), "cmd": cmd[2:], "at": time.time()})
        t0 = time.monotonic()
        code = subprocess.call(cmd)
        secs = time.monotonic() - t0
        ok = code == 0 and r.artifact(args.results, default_parts).is_file()
        _log(LOG, {"event": "end", "run": describe(r), "seconds": round(secs, 1),
                   "returncode": code, "ok": ok, "at": time.time()})  # fmt: skip
        if not ok:
            print(f"  FAILED (exit {code}); it will run again next time", flush=True)
    left = remaining(runs, args.results, default_parts)
    print(f"{len(runs) - len(left)}/{len(runs)} runs done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
