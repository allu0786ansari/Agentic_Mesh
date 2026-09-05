from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from agents.auth import JWTAuthManager
from api.node_api import create_app
from api.tls import TLSConfig
from scripts.generate_dev_mtls import generate


def test_protected_embedding_routes_require_valid_jwt(tmp_path: Path) -> None:
    manager = JWTAuthManager("test-secret")
    client = TestClient(create_app(tmp_path / "qdrant", "week10-secured", manager))
    payload = {
        "point_id": "point-1",
        "embedding": [0.0] * 128,
        "payload": {
            "run_id": "week10-test",
            "node_id": "node_01",
            "model_type": "vae",
            "score": 0.9,
            "threshold": 0.5,
            "index": 0,
        },
    }

    assert client.post("/embeddings", json=payload).status_code == 401
    token = manager.create_token("node_01")
    response = client.post("/embeddings", json=payload, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_mtls_configuration_requires_ca_for_client_auth(tmp_path: Path) -> None:
    cert, key = generate(tmp_path, "localhost")
    config = TLSConfig(cert, key, cert, require_client_certificate=True)
    settings = config.uvicorn_kwargs()
    assert settings["ssl_cert_reqs"] == 2
    assert settings["ssl_ca_certs"] == str(cert)
