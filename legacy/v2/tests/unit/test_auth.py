from __future__ import annotations

import time

import pytest

from agents.auth import JWTAuthManager


def test_jwt_round_trip() -> None:
    manager = JWTAuthManager("test-secret", ttl_seconds=30)
    token = manager.create_token("session-1", {"role": "node"})
    payload = manager.validate_token(token)
    assert payload["sub"] == "session-1"
    assert payload["role"] == "node"


def test_jwt_expiry() -> None:
    manager = JWTAuthManager("test-secret", ttl_seconds=1)
    token = manager.create_token("session-1")
    time.sleep(2.1)
    with pytest.raises(ValueError, match="invalid or expired"):
        manager.validate_token(token)
