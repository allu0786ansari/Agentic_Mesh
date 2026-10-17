"""Request checks for the node API: bearer token and window (ADR 0014 §4)."""

from __future__ import annotations

import math

from fastapi import HTTPException, Request

from agmesh.security.jwt import TokenError, TokenVerifier


def bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="bearer token required")
    return token.strip()


def window(start: float, end: float) -> tuple[float, float]:
    if not (math.isfinite(start) and math.isfinite(end)) or start > end:
        raise HTTPException(status_code=422, detail="start and end must be finite, start <= end")
    return start, end


def authorise(verifier: TokenVerifier, token: str, start: float, end: float) -> dict[str, object]:
    try:
        return verifier.verify(token, start, end)
    except TokenError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.reason) from exc
