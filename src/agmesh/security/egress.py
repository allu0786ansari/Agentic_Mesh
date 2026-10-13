"""Egress allowlist for outbound service calls (Proposal §5.2, adversary ADV-5).

Investigation content must never reach a third-party LLM API, so the mesh (and the
Week 1 LLM benchmark) refuse to talk to any endpoint whose host is not on the
allowlist from ``configs/params.yaml``.
"""

from __future__ import annotations

from urllib.parse import urlsplit


class EgressDeniedError(ValueError):
    """Raised when a URL's host is not on the egress allowlist."""


def host_of(url: str) -> str:
    host = urlsplit(url).hostname
    if not host:
        raise EgressDeniedError(f"URL has no host: {url!r}")
    return host.lower()


def is_allowlisted(url: str, allowlist: list[str]) -> bool:
    try:
        host = host_of(url)
    except EgressDeniedError:
        return False
    return host in {entry.lower().strip("[]") for entry in allowlist}


def require_allowlisted(url: str, allowlist: list[str]) -> None:
    if not is_allowlisted(url, allowlist):
        raise EgressDeniedError(
            f"Refusing to call {url!r}: host is not in the egress allowlist {allowlist}"
        )
