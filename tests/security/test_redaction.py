"""No secrets in logs (ADR 0014 §6)."""

from __future__ import annotations

import logging

import pytest

from agmesh.security.redaction import MARK, install, redact

# Built at run time so that the repository's secret scan never sees a literal token or key.
JWT = ".".join(["eyJ" + "hbGciOiJFUzI1NiJ9", "eyJ" + "pc3MiOiJtZXNoIn0", "c2lnbmF0dXJlLWJ5dGVz"])
PEM = "\n".join(
    [
        "-----BEGIN " + "PRIVATE KEY-----",
        "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg",
        "-----END " + "PRIVATE KEY-----",
    ]
)
HF = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3"


@pytest.mark.parametrize(
    "secret_line",
    ids=["header", "dict", "bare-jwt", "pem", "hf"],
    argvalues=[
        f"GET /v1/evidence Authorization: Bearer {JWT}",
        f"headers={{'authorization': 'Bearer {JWT}'}}",
        f"token {JWT} refused",
        f"loaded key {PEM}",
        f"hub token {HF}",
    ],
)
def test_redact_removes_secrets(secret_line: str) -> None:
    out = redact(secret_line)
    assert MARK in out
    assert JWT not in out and "MIGHAgEA" not in out and HF not in out


def test_env_secret_values_are_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MESH_APPROVAL_KEY", "approval-key-value-123")
    assert "approval-key-value-123" not in redact("key=approval-key-value-123 used")


def test_every_log_record_is_redacted(caplog: pytest.LogCaptureFixture) -> None:
    install()
    install()  # idempotent
    log = logging.getLogger("agmesh.test.redaction")
    with caplog.at_level(logging.INFO):
        log.info("request %s with %s", "/v1/evidence", f"Bearer {JWT}")
        log.warning("Authorization: Bearer %s", JWT)
        log.info(  # uvicorn's access-log shape: its formatter unpacks record.args
            '%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "GET", "/healthz", "1.1", 200
        )
    assert JWT not in caplog.text and MARK in caplog.text
    access = caplog.records[-1]
    assert isinstance(access.args, tuple) and access.args[1] == "GET"  # formatter-safe
