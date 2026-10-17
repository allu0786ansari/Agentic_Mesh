"""Per-request evidence tokens: ES256, one session, one node, one window (ADR 0014 §4).

* :class:`TokenMinter` (in the mesh) signs one token per evidence request with the
  mesh's private key.
* :class:`TokenVerifier` (in each node API) checks every claim with only the public
  key, then records the ``jti`` in a :class:`ReplayCache`.

**Errors.** A failed check raises :class:`TokenError`, whose ``status`` is:

* 401 for an invalid token (signature, algorithm, issuer, audience, time);
* 403 for a valid token that is not allowed this request (scope, site, window,
  replay);
* 503 when the replay cache is full.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization

ALGORITHM = "ES256"
ISSUER = "mesh"
SCOPE = "evidence:read"
LEEWAY_S = 1.0


class TokenError(Exception):
    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status, self.reason = status, reason


class TokenMinter:
    def __init__(
        self,
        private_key_pem: bytes,
        *,
        issuer: str = ISSUER,
        ttl_s: int = 120,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.key = serialization.load_pem_private_key(private_key_pem, password=None)
        self.issuer, self.ttl_s, self.clock = issuer, ttl_s, clock

    @classmethod
    def from_file(cls, path: Path, **kw: Any) -> TokenMinter:
        return cls(path.read_bytes(), **kw)

    def mint(
        self, node: str, site: str, session_id: str, window_start: float, window_end: float
    ) -> str:
        now = int(self.clock())
        claims = {
            "iss": self.issuer,
            "aud": node,
            "sub": session_id,
            "scope": SCOPE,
            "site": site,
            "window_start": float(window_start),
            "window_end": float(window_end),
            "iat": now,
            "exp": now + self.ttl_s,
            "jti": secrets.token_hex(16),
        }
        return jwt.encode(claims, self.key, algorithm=ALGORITHM)  # type: ignore[arg-type]


class ReplayCache:
    """``jti`` → expiry, kept until expiry; bounded, and refuses when full."""

    def __init__(self, max_entries: int = 100_000, clock: Callable[[], float] = time.time):
        self.max_entries, self.clock = max_entries, clock
        self.seen: dict[str, float] = {}
        self.lock = threading.Lock()

    def add(self, jti: str, exp: float) -> None:
        with self.lock:
            now = self.clock()
            if jti in self.seen and self.seen[jti] + LEEWAY_S >= now:
                raise TokenError(403, "token replayed")
            if len(self.seen) >= self.max_entries:
                self.seen = {j: e for j, e in self.seen.items() if e + LEEWAY_S >= now}
                if len(self.seen) >= self.max_entries:
                    raise TokenError(503, "replay cache full")
            self.seen[jti] = exp


class TokenVerifier:
    def __init__(
        self,
        public_key_pem: bytes,
        node: str,
        site: str,
        *,
        issuer: str = ISSUER,
        max_ttl_s: int = 120,
        replay: ReplayCache | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.key = serialization.load_pem_public_key(public_key_pem)
        self.node, self.site, self.issuer, self.max_ttl_s = node, site, issuer, max_ttl_s
        self.clock = clock
        self.replay = replay if replay is not None else ReplayCache(clock=clock)

    def verify(self, token: str, start_s: float, end_s: float) -> dict[str, Any]:
        """Check every claim for a request of ``[start_s, end_s]``; returns the claims."""
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise TokenError(401, f"malformed token: {exc}") from exc
        if header.get("alg") != ALGORITHM:
            raise TokenError(401, f"algorithm {header.get('alg')!r} is not {ALGORITHM}")
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                self.key,  # type: ignore[arg-type]
                algorithms=[ALGORITHM],
                audience=self.node,
                issuer=self.issuer,
                options={
                    "require": ["iss", "aud", "sub", "exp", "iat", "jti"],
                    # time claims are checked below against this verifier's clock
                    "verify_exp": False,
                    "verify_iat": False,
                    "verify_nbf": False,
                },
            )
        except jwt.InvalidAudienceError as exc:
            raise TokenError(401, "wrong audience") from exc
        except jwt.InvalidIssuerError as exc:
            raise TokenError(401, "wrong issuer") from exc
        except jwt.PyJWTError as exc:
            raise TokenError(401, f"invalid token: {exc}") from exc
        try:
            iat, exp = float(claims["iat"]), float(claims["exp"])
        except (TypeError, ValueError) as exc:
            raise TokenError(401, "iat and exp must be numbers") from exc
        now = self.clock()
        if exp + LEEWAY_S < now:
            raise TokenError(401, "token expired")
        if iat > now + LEEWAY_S:
            raise TokenError(401, "token issued in the future")
        if exp - iat > self.max_ttl_s:
            raise TokenError(401, "token lifetime exceeds the maximum")
        if claims.get("scope") != SCOPE:
            raise TokenError(403, "scope does not allow evidence:read")
        if claims.get("site") != self.site:
            raise TokenError(403, "token is for another site")
        try:
            lo, hi = float(claims["window_start"]), float(claims["window_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TokenError(403, "token has no window") from exc
        if not (lo <= start_s <= end_s <= hi):
            raise TokenError(403, "requested window is outside the token's window")
        self.replay.add(str(claims["jti"]), exp)
        return claims
