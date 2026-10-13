"""Tests for the LLM egress allowlist."""

from __future__ import annotations

import pytest

from agmesh.security.egress import EgressDeniedError, is_allowlisted, require_allowlisted

ALLOW = ["127.0.0.1", "::1", "localhost"]


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:11434", "http://localhost:11434/api", "http://[::1]:11434"],
)
def test_loopback_is_allowed(url: str) -> None:
    assert is_allowlisted(url, ALLOW)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com/v1",
        "http://192.168.1.20:11434",
        "http://localhost.evil.example:11434",
        "not a url",
    ],
)
def test_other_hosts_are_denied(url: str) -> None:
    assert not is_allowlisted(url, ALLOW)
    with pytest.raises(EgressDeniedError):
        require_allowlisted(url, ALLOW)
