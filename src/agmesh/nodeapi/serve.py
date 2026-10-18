"""Run one node API process (ADR 0014 §1).

    python -m agmesh.nodeapi.serve --node b1 --edge-run results/edge/site_b__val__<time>

* **Listener:** only ``configs/stack.yaml``'s host and port for the node, over TLS 1.3.
* **Client certificates:** only those from ``ca-mesh`` are accepted.
* **Tokens:** verified with the mesh's public key (``jwt-mesh.pub``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from agmesh.common.config import DEFAULT_PARAMS_PATH, load_params
from agmesh.security.ca import NODES
from agmesh.security.jwt import TokenVerifier
from agmesh.security.redaction import install as install_redaction
from agmesh.security.tls import Pki, https_server, server_context

STACK = Path("configs/stack.yaml")
SITE_OF = {"a": "site_a", "b": "site_b"}


def load_stack(path: Path = STACK) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def build(
    node: str,
    edge_run: Path,
    pki: Pki,
    stack: dict[str, Any],
    ttl_s: int,
    fault_delay_s: float = 0.0,
) -> Any:
    from agmesh.nodeapi.app import create_app

    pki.require(f"node-{node}")
    site = SITE_OF[node[0]]
    verifier = TokenVerifier((pki.root / "jwt-mesh.pub").read_bytes(), node, site, max_ttl_s=ttl_s)
    db = edge_run / f"evidence_{node}.db"
    if not db.is_file():
        raise SystemExit(f"{node}: no evidence store {db}")
    app = create_app(node, site, db, verifier, fault_delay_s=fault_delay_s)
    ctx = server_context(pki.cert(f"node-{node}"), pki.key(f"node-{node}"), pki.cert("ca-mesh"))
    return https_server(app, stack["host"], int(stack["nodes"][node]), ctx)


def main(argv: list[str] | None = None) -> int:
    install_redaction()
    parser = argparse.ArgumentParser(description="One node API process (ADR 0014)")
    parser.add_argument("--node", choices=NODES, required=True)
    parser.add_argument("--edge-run", type=Path, required=True)
    parser.add_argument("--stack", type=Path, default=STACK)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument(
        "--fault-delay-s", type=float, default=0.0, help="chaos tests only (ADR 0017 §6)"
    )
    args = parser.parse_args(argv)
    stack = load_stack(args.stack)
    params = load_params(args.params)
    server = build(
        args.node,
        args.edge_run,
        Pki(Path(stack["pki"])),
        stack,
        params.security.jwt.ttl_s,
        args.fault_delay_s,
    )
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
