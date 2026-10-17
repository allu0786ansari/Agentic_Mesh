"""Approval API: operator decisions on one mesh run's pending actions (ADR 0014 §8).

    python -m agmesh.mesh.approvals_api --run results/mesh/<run>

* **Listener:** HTTPS only on ``configs/stack.yaml``'s ``approval_api`` port.
* **Client certificates:** only those issued by ``ca-operator``.

**Calls.**

* ``GET /v1/actions?status=pending_approval`` lists the actions with that status.
* ``POST /v1/approvals`` takes ``{action_id, decision, approver, expires_at, token}``.
  It calls :func:`agmesh.mesh.approvals.decide`, which holds every check (HMAC with
  ``MESH_APPROVAL_KEY``, expiry, action and decision). A refused decision returns
  403.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from agmesh.bus.sqlite_bus import SqliteBus
from agmesh.common.env import load_env
from agmesh.mesh import approvals
from agmesh.mesh.sessions import SessionStore


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: str
    decision: str
    approver: str
    expires_at: int
    token: str


def create_app(storage: Path, *, key: bytes | None = None) -> FastAPI:
    store, bus = SessionStore(storage), SqliteBus(storage)
    app = FastAPI(title="approval-api", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/v1/actions")
    def actions(status: str = "pending_approval") -> list[dict[str, Any]]:
        return [a for a in store.actions() if a["status"] == status]

    @app.post("/v1/approvals")
    def decide(body: Decision) -> dict[str, str]:
        try:
            new = approvals.decide(
                store,
                bus,
                body.action_id,
                body.decision,
                body.approver,
                body.expires_at,
                body.token,
                key=key,
            )
        except approvals.ApprovalError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return {"action_id": body.action_id, "status": new}

    return app


def main(argv: list[str] | None = None) -> int:
    from agmesh.nodeapi.serve import STACK, load_stack
    from agmesh.security.redaction import install as install_redaction
    from agmesh.security.tls import Pki, https_server, server_context

    load_env()
    install_redaction()
    parser = argparse.ArgumentParser(description="Approval API for one mesh run (ADR 0014 §8)")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stack", type=Path, default=STACK)
    args = parser.parse_args(argv)
    stack = load_stack(args.stack)
    pki = Pki(Path(stack["pki"]))
    pki.require("approval-api")
    storage = args.run / "storage.db"
    if not storage.is_file():
        raise SystemExit(f"no storage.db in {args.run}")
    approvals._key(None)  # refuse to start without MESH_APPROVAL_KEY
    ctx = server_context(pki.cert("approval-api"), pki.key("approval-api"), pki.cert("ca-operator"))
    https_server(create_app(storage), stack["host"], int(stack["approval_api"]), ctx).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
