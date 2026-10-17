"""Per-request JWT checks (ADR 0014 §4; EA2): expired, audience, scope, window, replay..."""

from __future__ import annotations

import time
from typing import Any

import httpx
import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization

from agmesh.security.jwt import ReplayCache, TokenError, TokenMinter, TokenVerifier
from agmesh.security.tls import Pki
from tests.security.conftest import T0, NodeServer

SNI = {"sni_hostname": "node-b1.agmesh"}


def verifier(pki: Pki, now: float = T0, **kw: Any) -> TokenVerifier:
    return TokenVerifier(
        (pki.root / "jwt-mesh.pub").read_bytes(), "b1", "site_b", clock=lambda: now, **kw
    )


def minter(pki: Pki, now: float = T0, key: str = "jwt-mesh.key", **kw: Any) -> TokenMinter:
    return TokenMinter.from_file(pki.root / key, clock=lambda: now, **kw)


def claims(pki: Pki, **over: Any) -> str:
    base = {
        "iss": "mesh", "aud": "b1", "sub": "s1", "scope": "evidence:read", "site": "site_b",
        "window_start": T0, "window_end": T0 + 60, "iat": int(T0), "exp": int(T0) + 120,
        "jti": "j-" + str(over.pop("jti_suffix", "1")),
    }  # fmt: skip
    base.update(over)
    key = serialization.load_pem_private_key(
        (pki.root / "jwt-mesh.key").read_bytes(), password=None
    )
    return pyjwt.encode({k: v for k, v in base.items() if v is not None}, key, algorithm="ES256")  # type: ignore[arg-type]


def status(v: TokenVerifier, token: str, lo: float = T0, hi: float = T0 + 10) -> int:
    try:
        v.verify(token, lo, hi)
    except TokenError as exc:
        return exc.status
    return 200


def test_a_good_token_passes_once(pki: Pki) -> None:
    v, token = verifier(pki), minter(pki).mint("b1", "site_b", "s1", T0, T0 + 60)
    assert status(v, token) == 200
    assert status(v, token) == 403  # replayed


@pytest.mark.parametrize(
    ("make", "code"),
    [
        (lambda p: minter(p, now=T0 - 200).mint("b1", "site_b", "s", T0, T0 + 60), 401),  # expired
        (lambda p: minter(p).mint("b2", "site_b", "s", T0, T0 + 60), 401),  # wrong audience
        (lambda p: minter(p, issuer="intruder").mint("b1", "site_b", "s", T0, T0 + 60), 401),
        (
            lambda p: minter(p, key="foreign/jwt-mesh.key").mint("b1", "site_b", "s", T0, T0 + 60),
            401,
        ),
        (lambda p: minter(p, ttl_s=3600).mint("b1", "site_b", "s", T0, T0 + 60), 401),  # too long
        (
            lambda p: minter(p, now=T0 + 30).mint("b1", "site_b", "s", T0, T0 + 60),
            401,
        ),  # future iat
        (lambda p: claims(p, scope="evidence:write"), 403),  # out of scope
        (lambda p: claims(p, scope=None), 403),
        (lambda p: claims(p, site="site_a"), 403),  # another site
        (lambda p: claims(p, window_start=T0 + 5), 403),  # request starts before the window
        (lambda p: claims(p, window_end=T0 + 5), 403),  # request ends after the window
        (lambda p: claims(p, window_start=None), 403),
        (lambda p: claims(p, jti=None), 401),  # every token needs a one-time ID
    ],
)
def test_refused_tokens(pki: Pki, make: Any, code: int) -> None:
    assert status(verifier(pki), make(pki)) == code


def test_algorithm_confusion_is_refused(pki: Pki) -> None:
    body = {"iss": "mesh", "aud": "b1", "sub": "s", "iat": int(T0), "exp": int(T0) + 60, "jti": "x"}
    unsigned = pyjwt.encode(body, None, algorithm="none")  # type: ignore[arg-type]
    hmac = pyjwt.encode(body, "a-shared-secret-of-enough-length!", algorithm="HS256")
    assert status(verifier(pki), unsigned) == 401
    assert status(verifier(pki), hmac) == 401
    assert status(verifier(pki), "not-a-token") == 401


def test_the_replay_cache_forgets_expired_ids_and_refuses_when_full() -> None:
    now = [T0]
    cache = ReplayCache(max_entries=2, clock=lambda: now[0])
    cache.add("a", T0 + 10)
    cache.add("b", T0 + 10)
    with pytest.raises(TokenError) as err:
        cache.add("c", T0 + 10)
    assert err.value.status == 503
    now[0] = T0 + 20  # a and b have expired
    cache.add("c", T0 + 30)
    cache.add("a", T0 + 30)  # an expired ID may come back; it is a new token's window


def test_the_node_api_maps_token_failures_to_401_and_403(node_b1: NodeServer) -> None:
    url = f"{node_b1.url}/v1/evidence"
    params = {"start": T0, "end": T0 + 10}
    good = node_b1.minter().mint("b1", "site_b", "s", T0, T0 + 60)
    other_site = node_b1.minter().mint("b1", "site_a", "s", T0, T0 + 60)
    with httpx.Client(verify=node_b1.mesh_ctx(), timeout=3.0) as http:

        def call(token: str | None) -> int:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            r = http.get(url, params=params, headers=headers, extensions=SNI)
            return r.status_code

        assert call(None) == 401
        assert call(good) == 200
        assert call(good) == 403  # replay
        assert call(other_site) == 403
        expired = TokenMinter.from_file(
            node_b1.pki.root / "jwt-mesh.key", clock=lambda: time.time() - 600
        )
        assert call(expired.mint("b1", "site_b", "s", T0, T0 + 60)) == 401
