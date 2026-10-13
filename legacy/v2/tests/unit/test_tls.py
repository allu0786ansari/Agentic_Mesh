from __future__ import annotations

from pathlib import Path

import pytest

from api.tls import TLSConfig
import agents.tracing as tracing
from agents.tracing import configure_phoenix, get_tracer
from scripts.generate_dev_mtls import generate


def test_tls_config_produces_uvicorn_settings(tmp_path: Path) -> None:
    cert, key = generate(tmp_path, "localhost")
    config = TLSConfig(cert, key)
    assert config.uvicorn_kwargs() == {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}


def test_tls_config_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TLSConfig(tmp_path / "missing-cert.pem", tmp_path / "missing-key.pem").uvicorn_kwargs()


def test_phoenix_export_configuration_is_required() -> None:
    configure_phoenix("http://127.0.0.1:6006/v1/traces")
    with get_tracer("week10-strict-phoenix").span("strict-span") as span:
        span.set_attribute("required", True)


def test_phoenix_span_rejects_unconfigured_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "_PHOENIX_CONFIGURED", False)
    with pytest.raises(RuntimeError, match="Phoenix tracing is not configured"):
        with get_tracer("week10-strict-required").span("must-export"):
            pass
