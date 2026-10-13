from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt


class JWTAuthManager:
    """Issue and validate short-lived JWT session tokens."""

    def __init__(self, secret_key: str | None = None, *, algorithm: str = "HS256", ttl_seconds: int = 900) -> None:
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY")
        if not self.secret_key:
            raise ValueError("secret_key or JWT_SECRET_KEY is required")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        self.algorithm = algorithm
        self.ttl_seconds = ttl_seconds

    def create_token(self, subject: str, claims: dict[str, Any] | None = None) -> str:
        if not subject.strip():
            raise ValueError("subject must not be empty")
        now = datetime.now(timezone.utc)
        payload: dict[str, Any] = {"sub": subject, "iat": now, "exp": now + timedelta(seconds=self.ttl_seconds), "jti": secrets.token_urlsafe(12)}
        if claims:
            payload.update(claims)
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def validate_token(self, token: str) -> dict[str, Any]:
        if not token.strip():
            raise ValueError("token must not be empty")
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except JWTError as exc:
            raise ValueError("invalid or expired token") from exc
        if not payload.get("sub"):
            raise ValueError("token subject is missing")
        return payload


__all__ = ["JWTAuthManager"]
