"""Start and stop one site's node APIs as separate processes (ADR 0014 §1).

    python -m agmesh.stack up --site site_b        # make stack-up SITE=site_b
    python -m agmesh.stack down                    # make stack-down

**``up``.**

* It starts one ``agmesh.nodeapi.serve`` process per node of the site, each serving
  the latest validation edge run, or ``--edge-run``.
* Each process logs to ``results/stack/<node>.log``.
* It waits until every node answers ``/healthz`` over mTLS with the mesh's
  certificate and the node's own server name.
* Process IDs go to ``results/stack/pids.json``.

**``down``.** It stops those processes: SIGTERM first, then SIGKILL after 10 s.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from agmesh.mesh.investigator import SITE_NODES
from agmesh.mesh.runner import latest_edge_run
from agmesh.nodeapi.serve import STACK, load_stack
from agmesh.security.redaction import install as install_redaction
from agmesh.security.tls import Pki, client_context

STATE = Path("results/stack")


def healthy(host: str, port: int, node: str, pki: Pki, timeout_s: float = 30.0) -> bool:
    ctx = client_context(pki.cert("mesh"), pki.key("mesh"), pki.cert("ca-server"))
    deadline = time.monotonic() + timeout_s
    with httpx.Client(verify=ctx, timeout=2.0) as http:
        while time.monotonic() < deadline:
            try:
                r = http.get(
                    f"https://{host}:{port}/healthz",
                    extensions={"sni_hostname": f"node-{node}.agmesh"},
                )
                if r.status_code == 200 and r.json().get("ok"):
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
    return False


def up(
    site: str,
    edge_run: Path,
    stack_path: Path,
    state: Path = STATE,
    fault_delay_s: dict[str, float] | None = None,
) -> dict[str, int]:
    """``fault_delay_s`` (node -> seconds) slows chosen nodes: X6 only (ADR 0017 §6)."""
    stack = load_stack(stack_path)
    pki = Pki(Path(stack["pki"]))
    pki.require("mesh", *(f"node-{n}" for n in SITE_NODES[site]))
    state.mkdir(parents=True, exist_ok=True)
    if (state / "pids.json").is_file():
        raise SystemExit(f"a stack is already recorded in {state}/pids.json: make stack-down")
    pids: dict[str, int] = {}
    for node in SITE_NODES[site]:
        log = (state / f"{node}.log").open("ab")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agmesh.nodeapi.serve",
                "--node",
                node,
                "--edge-run",
                str(edge_run),
                "--stack",
                str(stack_path),
                "--fault-delay-s",
                str((fault_delay_s or {}).get(node, 0.0)),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pids[node] = proc.pid
    (state / "pids.json").write_text(json.dumps(pids, indent=2) + "\n", encoding="utf-8")
    for node in SITE_NODES[site]:
        if not healthy(stack["host"], int(stack["nodes"][node]), node, pki):
            down(state)
            raise SystemExit(f"node API {node} did not become healthy; see {state}/{node}.log")
        print(
            f"node-api {node} up on https://{stack['host']}:{stack['nodes'][node]}"
            f" (pid {pids[node]})"
        )
    return pids


def down(state: Path = STATE) -> list[str]:
    path = state / "pids.json"
    if not path.is_file():
        return []
    pids: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
    for pid in pids.values():
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    for pid in pids.values():
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    path.unlink()
    return sorted(pids)


def main(argv: list[str] | None = None) -> int:
    install_redaction()
    parser = argparse.ArgumentParser(description="Node API processes for one site (ADR 0014)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_up = sub.add_parser("up")
    p_up.add_argument("--site", choices=tuple(SITE_NODES), required=True)
    p_up.add_argument("--edge-run", type=Path, default=None)
    p_up.add_argument("--edge-root", type=Path, default=Path("results/edge"))
    p_up.add_argument("--stack", type=Path, default=STACK)
    p_up.add_argument("--split", choices=("val", "test"), default="val")
    sub.add_parser("down")
    args = parser.parse_args(argv)
    if args.cmd == "up":
        edge = args.edge_run or latest_edge_run(args.edge_root, args.site, args.split)
        up(args.site, edge, args.stack)
        print(f"stack up for {args.site} over {edge}; stop it with: make stack-down")
    else:
        stopped = down()
        print(f"stopped {stopped}" if stopped else "no stack recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
