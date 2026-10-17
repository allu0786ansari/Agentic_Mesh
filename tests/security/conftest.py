"""A development PKI, an evidence store and node APIs on free ports (ADR 0014)."""

from __future__ import annotations

import socket
import ssl
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agmesh.edge.evidence_store import EvidenceStore
from agmesh.nodeapi.app import create_app
from agmesh.security import ca
from agmesh.security.jwt import TokenMinter, TokenVerifier
from agmesh.security.tls import Pki, ServerThread, client_context, https_server, server_context
from tests.mesh_doubles import record

T0 = 1_675_242_222.0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="session")
def pki(tmp_path_factory: pytest.TempPathFactory) -> Pki:
    root = tmp_path_factory.mktemp("pki")
    ca.build(root)
    # a second, foreign PKI: certificates the stack must never accept
    foreign = root / "foreign"
    ca.build(foreign)
    return Pki(root)


@pytest.fixture(scope="session")
def evidence_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("edge") / "evidence_b1.db"
    store = EvidenceStore(path)
    for i in range(3):
        store.put(record("b1", T0 + 10 * i, [("req_write", "up", "high")]))
    store.close()
    return path


@dataclass
class NodeServer:
    url: str
    port: int
    pki: Pki
    verifier: TokenVerifier

    def minter(self, **kw: Any) -> TokenMinter:
        return TokenMinter.from_file(self.pki.root / "jwt-mesh.key", **kw)

    def mesh_ctx(self) -> ssl.SSLContext:
        p = self.pki
        return client_context(p.cert("mesh"), p.key("mesh"), p.cert("ca-server"))


@pytest.fixture
def node_b1(pki: Pki, evidence_db: Path) -> Iterator[NodeServer]:
    port = free_port()
    verifier = TokenVerifier((pki.root / "jwt-mesh.pub").read_bytes(), "b1", "site_b")
    app = create_app("b1", "site_b", evidence_db, verifier)
    ctx = server_context(pki.cert("node-b1"), pki.key("node-b1"), pki.cert("ca-mesh"))
    with ServerThread(https_server(app, "127.0.0.1", port, ctx)):
        yield NodeServer(f"https://127.0.0.1:{port}", port, pki, verifier)
