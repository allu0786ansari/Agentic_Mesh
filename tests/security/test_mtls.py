"""mTLS between processes (ADR 0014 §2-3; EA2): only the mesh, only TLS 1.3, only the right node."""

from __future__ import annotations

import ssl

import httpx
import pytest

from agmesh.mesh.nodeclient import HttpsNodeClient, NodeError, TransientNodeError
from agmesh.security.jwt import TokenMinter
from agmesh.security.tls import Pki, client_context
from tests.security.conftest import T0, NodeServer

SNI = {"sni_hostname": "node-b1.agmesh"}


def _get(url: str, ctx: ssl.SSLContext | bool, **kw: object) -> httpx.Response:
    with httpx.Client(verify=ctx, timeout=3.0) as http:
        return http.get(url, **kw)  # type: ignore[arg-type]


def test_the_mesh_gets_evidence_over_mtls_with_a_token(node_b1: NodeServer) -> None:
    client = HttpsNodeClient({"b1": node_b1.url}, node_b1.mesh_ctx(), node_b1.minter())
    try:
        got = client.evidence("b1", T0 - 60, T0 + 60, 2.0, session_id="s1")
    finally:
        client.close()
    assert [r.node_id for r in got] == ["b1"] * 3


def test_plaintext_is_rejected(node_b1: NodeServer) -> None:
    with pytest.raises(httpx.TransportError):
        _get(f"http://127.0.0.1:{node_b1.port}/healthz", False)


def test_a_client_without_a_certificate_is_rejected(node_b1: NodeServer) -> None:
    p = node_b1.pki
    ctx = client_context(None, None, p.cert("ca-server"))
    with pytest.raises(httpx.TransportError):
        _get(f"{node_b1.url}/healthz", ctx, extensions=SNI)


@pytest.mark.parametrize(
    ("cert", "why"),
    [
        ("foreign/mesh", "a mesh certificate from another CA"),
        ("node-b2", "a node's server certificate (ca-server, serverAuth)"),
        ("operator", "the operator certificate (ca-operator)"),
    ],
)
def test_a_wrong_client_certificate_is_rejected(node_b1: NodeServer, cert: str, why: str) -> None:
    p = node_b1.pki
    ctx = client_context(p.cert(cert), p.key(cert), p.cert("ca-server"))
    with pytest.raises(httpx.TransportError):
        _get(f"{node_b1.url}/healthz", ctx, extensions=SNI)


def test_tls_1_2_is_rejected(node_b1: NodeServer) -> None:
    p = node_b1.pki
    ctx = client_context(p.cert("mesh"), p.key("mesh"), p.cert("ca-server"))
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    with pytest.raises(httpx.TransportError):
        _get(f"{node_b1.url}/healthz", ctx, extensions=SNI)


def test_a_node_cannot_answer_for_another_node(node_b1: NodeServer) -> None:
    # The mesh asks for b2 but reaches b1's process: the server name does not match.
    with pytest.raises(httpx.ConnectError):
        _get(
            f"{node_b1.url}/healthz",
            node_b1.mesh_ctx(),
            extensions={"sni_hostname": "node-b2.agmesh"},
        )
    client = HttpsNodeClient({"b2": node_b1.url}, node_b1.mesh_ctx(), node_b1.minter())
    with pytest.raises(TransientNodeError):
        client.evidence("b2", T0, T0 + 10, 2.0)
    client.close()


def test_a_server_from_a_foreign_ca_is_not_trusted(node_b1: NodeServer, pki: Pki) -> None:
    ctx = client_context(pki.cert("mesh"), pki.key("mesh"), pki.cert("foreign/ca-server"))
    with pytest.raises(httpx.ConnectError):
        _get(f"{node_b1.url}/healthz", ctx, extensions=SNI)


def test_health_answers_over_mtls(node_b1: NodeServer) -> None:
    r = _get(f"{node_b1.url}/healthz", node_b1.mesh_ctx(), extensions=SNI)
    assert r.status_code == 200 and r.json() == {"node": "b1", "site": "site_b", "ok": True}


def test_a_refused_token_is_not_retried(node_b1: NodeServer) -> None:
    token = node_b1.minter().mint("b1", "site_b", "s", T0, T0 + 10)

    class Captured(TokenMinter):  # an attacker replaying one captured token
        def mint(self, *_a: object, **_k: object) -> str:
            return token

    captured = Captured((node_b1.pki.root / "jwt-mesh.key").read_bytes())
    client = HttpsNodeClient({"b1": node_b1.url}, node_b1.mesh_ctx(), captured)
    client.evidence("b1", T0, T0 + 10, 2.0)
    with pytest.raises(NodeError) as err:  # the same jti again: 403, non-transient
        client.evidence("b1", T0, T0 + 10, 2.0)
    assert not isinstance(err.value, TransientNodeError) and "403" in str(err.value)
    client.close()
