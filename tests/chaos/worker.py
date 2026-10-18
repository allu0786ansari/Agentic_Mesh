"""A mesh process for the kill-and-restart scenario (ADR 0017 §6, case 4).

    python -m tests.chaos.worker <storage.db> <stack.yaml>

It consumes ``alerts.site_b`` from the bus in ``storage.db`` with 1 s leases, and
reaches the node API processes over mTLS. It is killed from outside mid-run, then
started again on the same database.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from agmesh.mesh.runner import consume
from agmesh.mesh.triage import Triage
from agmesh.security.tls import Pki
from tests.chaos.conftest import Stack, make_rig


def main(db: str, stack_yaml: str) -> int:
    cfg = yaml.safe_load(Path(stack_yaml).read_text())
    stack = Stack(Path(), Pki(Path(cfg["pki"])), Path(), Path(stack_yaml), cfg["nodes"])
    rig = make_rig(stack, Path(db))
    consume(
        rig.mesh, rig.bus, Triage(60.0), "site_b", "retrieval_only", "chaos", lease_s=1.0, batch=1
    )
    rig.client.close()
    print("worker done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
