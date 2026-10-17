"""The node API application (Proposal §9.2; ADR 0014 §1, §4).

    GET /healthz                          mTLS only
    GET /v1/evidence?start=<s>&end=<s>    mTLS and a bearer JWT

Evidence is the node's minimised records whose window ends in ``[start, end]``
(epoch seconds), validated by :class:`EvidenceRecord` on the way out (CC5). Nothing
else is served.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, Query

from agmesh.common.schemas import EvidenceRecord
from agmesh.edge.evidence import iso
from agmesh.edge.evidence_store import EvidenceStore
from agmesh.nodeapi.deps import authorise, bearer, window
from agmesh.security.jwt import TokenVerifier

log = logging.getLogger("agmesh.nodeapi")


def create_app(node: str, site: str, evidence_db: Path, verifier: TokenVerifier) -> FastAPI:
    app = FastAPI(title=f"node-api {node}", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"node": node, "site": site, "ok": evidence_db.is_file()}

    @app.get("/v1/evidence", response_model=list[EvidenceRecord])
    def evidence(
        start: float = Query(...),
        end: float = Query(...),
        token: str = Depends(bearer),
    ) -> list[EvidenceRecord]:
        lo, hi = window(start, end)
        claims = authorise(verifier, token, lo, hi)
        store = EvidenceStore(evidence_db)
        try:
            records = store.between(node, iso(lo), iso(hi))
        finally:
            store.close()
        log.info("evidence %s session=%s n=%d", node, claims.get("sub"), len(records))
        return records

    return app
