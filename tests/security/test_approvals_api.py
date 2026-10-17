"""The approval API: operator certificate only, and every decision through the HMAC gate."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from agmesh.bus.sqlite_bus import SqliteBus
from agmesh.mesh import approvals
from agmesh.mesh.approvals_api import create_app
from agmesh.mesh.attribution import Attribution
from agmesh.mesh.dispatcher import dispatch
from agmesh.mesh.sessions import SessionStore
from agmesh.security.tls import Pki, ServerThread, client_context, https_server, server_context
from tests.security.conftest import free_port

KEY = b"0123456789abcdef-test-key"
SNI = {"sni_hostname": "approval-api.agmesh"}


@pytest.fixture
def api(pki: Pki, tmp_path: Path) -> Iterator[tuple[str, str]]:
    db = tmp_path / "storage.db"
    store, bus = SessionStore(db), SqliteBus(db)
    sid = store.open("site_b", ["a"], "r")[0].session_id
    att = Attribution("full", "full", "T0806", ["x"], 0.9, ["T0806"], [], "text")
    high = next(
        a for a in dispatch(store, bus, sid, "site_b", att, "high") if a["impact"] == "high"
    )
    port = free_port()
    ctx = server_context(pki.cert("approval-api"), pki.key("approval-api"), pki.cert("ca-operator"))
    with ServerThread(https_server(create_app(db, key=KEY), "127.0.0.1", port, ctx)):
        yield f"https://127.0.0.1:{port}", high["action_id"]


def _client(pki: Pki, who: str) -> httpx.Client:
    return httpx.Client(
        verify=client_context(pki.cert(who), pki.key(who), pki.cert("ca-server")), timeout=3.0
    )


def test_operator_approves_with_a_valid_token(pki: Pki, api: tuple[str, str]) -> None:
    url, action = api
    exp = 2_000_000_000
    with _client(pki, "operator") as http:
        pending = http.get(f"{url}/v1/actions", extensions=SNI).json()
        assert [a["action_id"] for a in pending] == [action]
        body = {"action_id": action, "decision": "approved", "approver": "op1", "expires_at": exp}
        bad = http.post(f"{url}/v1/approvals", json={**body, "token": "0" * 64}, extensions=SNI)
        assert bad.status_code == 403
        token = approvals.sign(action, "approved", "op1", exp, KEY)
        ok = http.post(f"{url}/v1/approvals", json={**body, "token": token}, extensions=SNI)
        assert ok.status_code == 200 and ok.json()["status"] == "approved"
        again = http.post(f"{url}/v1/approvals", json={**body, "token": token}, extensions=SNI)
        assert again.status_code == 403  # no longer pending


def test_only_the_operator_certificate_is_accepted(pki: Pki, api: tuple[str, str]) -> None:
    url, _ = api
    with _client(pki, "mesh") as http, pytest.raises(httpx.TransportError):
        http.get(f"{url}/v1/actions", extensions=SNI)
