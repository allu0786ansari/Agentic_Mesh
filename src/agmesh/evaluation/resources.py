"""X7: peak memory per process and stage on the 8 GB laptop (ADR 0019 §1; CC8).

    python -m agmesh.evaluation.resources run --stage training -- \
        python -m agmesh.federated.run --site site_a --strategy fedprox --epsilon 3 \
        --seeds 11 --results results/x7_scratch
    python -m agmesh.evaluation.resources watch --stage operations     # Ctrl-C to stop
    python -m agmesh.evaluation.resources report                       # results/X7/

**Sampling.** Every 0.5 s, for each sampled process: RSS and PSS (``psutil``
``memory_full_info``, from ``/proc/<pid>/smaps_rollup``). For the whole system:
used memory (total minus available) and used swap.

**Which processes.**

* ``run``: the command's process tree. Its exit code is recorded; a kill by
  signal 9 is reported as a likely out-of-memory kill.
* ``watch``: every process whose command line matches an operations role (node
  API, approval API, mesh, edge runtime, Ollama or llama-server, Phoenix).

**Output.** Samples go to ``results/X7/samples/<stage>__<time>.jsonl``.
``report`` takes the latest file of each stage and writes
``results/X7/resources.json`` with, per stage:

* the peak PSS per role;
* the peak of the stage's summed PSS;
* the peak system used memory and the peak swap;
* the budget check: peak system used at most 6.0 GB, and no out-of-memory exit.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import psutil

from agmesh.common.artifacts import write_artifact

OUT = Path("results/X7")
INTERVAL_S = 0.5
BUDGET_MB = 6000.0  # Plan §3.3: ≤ 6.0 GB inside WSL2
MB = 2**20

#: Operations roles, matched against the command line (first match wins).
ROLES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("node_api", re.compile(r"agmesh\.nodeapi\.serve")),
    ("approval_api", re.compile(r"agmesh\.mesh\.approvals_api")),
    ("mesh", re.compile(r"agmesh\.mesh\.runner")),
    ("edge", re.compile(r"agmesh\.edge\.runtime")),
    ("llm", re.compile(r"(^|/)(ollama|llama-server)(\s|$)|ollama runner")),
    ("phoenix", re.compile(r"phoenix")),
)


def role_of(cmdline: Iterable[str]) -> str | None:
    text = " ".join(cmdline)
    for role, pattern in ROLES:
        if pattern.search(text):
            return role
    return None


def _proc_sample(p: psutil.Process, role: str) -> dict[str, Any] | None:
    try:
        with p.oneshot():
            info = p.memory_full_info()
            return {
                "pid": p.pid,
                "role": role,
                "name": p.name(),
                "rss_mb": info.rss / MB,
                "pss_mb": getattr(info, "pss", info.rss) / MB,
            }
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def system_sample() -> dict[str, float]:
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    return {
        "total_mb": vm.total / MB,
        "used_mb": (vm.total - vm.available) / MB,
        "available_mb": vm.available / MB,
        "swap_used_mb": sw.used / MB,
    }


def tree(root: psutil.Process) -> list[psutil.Process]:
    try:
        return [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def sample(procs: list[tuple[psutil.Process, str]]) -> dict[str, Any]:
    rows = [r for p, role in procs if (r := _proc_sample(p, role)) is not None]
    return {"t": time.time(), "system": system_sample(), "processes": rows}


def by_role() -> list[tuple[psutil.Process, str]]:
    out = []
    for p in psutil.process_iter(["cmdline"]):
        role = role_of(p.info.get("cmdline") or [])
        if role is not None:
            out.append((p, role))
    return out


def _writer(stage: str, out: Path) -> tuple[Path, Any]:
    path = out / "samples" / f"{stage}__{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, path.open("w", encoding="utf-8")


def run_command(stage: str, cmd: list[str], out: Path = OUT) -> tuple[Path, int]:
    """Run ``cmd`` and sample its process tree until it exits."""
    path, fh = _writer(stage, out)
    proc = subprocess.Popen(cmd)
    root = psutil.Process(proc.pid)
    fh.write(json.dumps({"meta": {"stage": stage, "mode": "run", "cmd": cmd}}) + "\n")
    try:
        while proc.poll() is None:
            fh.write(json.dumps(sample([(p, "training") for p in tree(root)])) + "\n")
            fh.flush()
            time.sleep(INTERVAL_S)
    finally:
        code = proc.wait()
        fh.write(json.dumps({"exit": {"returncode": code}}) + "\n")
        fh.close()
    return path, code


def watch(stage: str, seconds: float | None = None, out: Path = OUT) -> Path:
    """Sample every operations-role process until Ctrl-C (or ``seconds``)."""
    path, fh = _writer(stage, out)
    fh.write(json.dumps({"meta": {"stage": stage, "mode": "watch"}}) + "\n")
    stop = time.monotonic() + seconds if seconds else None
    stopped = False

    def _stop(*_: Any) -> None:
        nonlocal stopped
        stopped = True

    old = signal.signal(signal.SIGINT, _stop)
    try:
        while not stopped and (stop is None or time.monotonic() < stop):
            fh.write(json.dumps(sample(by_role())) + "\n")
            fh.flush()
            time.sleep(INTERVAL_S)
    finally:
        signal.signal(signal.SIGINT, old)
        fh.write(json.dumps({"exit": {"returncode": 0, "stopped_by": "user"}}) + "\n")
        fh.close()
    return path


def summarise(lines: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Peaks of one stage's samples."""
    meta: dict[str, Any] = {}
    exit_: dict[str, Any] = {}
    per_role: dict[str, float] = {}
    stage_pss = used = swap = 0.0
    total = 0.0
    n = 0
    t0 = t1 = None
    for line in lines:
        if "meta" in line:
            meta = line["meta"]
            continue
        if "exit" in line:
            exit_ = line["exit"]
            continue
        n += 1
        t0 = line["t"] if t0 is None else t0
        t1 = line["t"]
        sys_ = line["system"]
        used, swap = max(used, sys_["used_mb"]), max(swap, sys_["swap_used_mb"])
        total = sys_["total_mb"]
        sums: dict[str, float] = {}
        for p in line["processes"]:
            sums[p["role"]] = sums.get(p["role"], 0.0) + p["pss_mb"]
        for role, v in sums.items():
            per_role[role] = max(per_role.get(role, 0.0), v)
        stage_pss = max(stage_pss, sum(sums.values()))
    code = exit_.get("returncode")
    oom = code in (-9, 137)
    return {
        "mode": meta.get("mode"),
        "command": meta.get("cmd"),
        "samples": n,
        "seconds": round((t1 or 0) - (t0 or 0), 1),
        "peak_pss_mb_by_role": {k: round(v, 1) for k, v in sorted(per_role.items())},
        "peak_stage_pss_mb": round(stage_pss, 1),
        "peak_system_used_mb": round(used, 1),
        "peak_swap_used_mb": round(swap, 1),
        "system_total_mb": round(total, 1),
        "returncode": code,
        "likely_oom_kill": oom,
        "within_budget": bool(n and used <= BUDGET_MB and not oom and code in (0, None)),
    }


def report(out: Path = OUT) -> Path:
    stages: dict[str, Any] = {}
    for path in sorted((out / "samples").glob("*.jsonl")):
        stage = path.name.split("__")[0]
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        stages[stage] = {"file": path.as_posix(), **summarise(lines)}  # latest wins
    payload = {
        "protocol": "docs/adr/0019-g4-final-test-protocol-week14.md §1",
        "budget_mb": BUDGET_MB,
        "stages": stages,
        "passed": bool(stages) and all(s["within_budget"] for s in stages.values()),
        "missing_stages": [s for s in ("training", "operations") if s not in stages],
    }
    payload["passed"] = payload["passed"] and not payload["missing_stages"]
    return write_artifact(out / "resources.json", kind="X7.resources", data_source="runtime",
                          payload=payload)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="X7 memory per stage (ADR 0019 §1)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--stage", default="training")
    p_run.add_argument("command", nargs=argparse.REMAINDER)
    p_watch = sub.add_parser("watch")
    p_watch.add_argument("--stage", default="operations")
    p_watch.add_argument("--seconds", type=float, default=None)
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not cmd:
            parser.error("give the command after --")
        path, code = run_command(args.stage, cmd)
        print(f"{args.stage}: exit {code}; samples -> {path}")
        return code
    if args.cmd == "watch":
        print("sampling operations processes; press Ctrl-C to stop", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            path = watch(args.stage, args.seconds)
        print(f"samples -> {path}")
        return 0
    path = report()
    doc = json.loads(path.read_text())
    for stage, s in doc["stages"].items():
        verdict = "OK" if s["within_budget"] else "OVER"
        print(
            f"{stage}: peak used {s['peak_system_used_mb']:.0f} MB, swap "
            f"{s['peak_swap_used_mb']:.0f} MB, stage PSS {s['peak_stage_pss_mb']:.0f} MB "
            f"{s['peak_pss_mb_by_role']} -> {verdict}"
        )
    print(
        f"X7 {'passed' if doc['passed'] else 'FAILED'}; missing {doc['missing_stages']} -> {path}"
    )
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
