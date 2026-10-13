from __future__ import annotations

from pathlib import Path

from cryptography import x509

from scripts.generate_dev_mtls import generate


def test_generate_dev_certificate_and_key(tmp_path: Path) -> None:
    certificate_path, key_path = generate(tmp_path, "localhost")
    certificate = x509.load_pem_x509_certificate(certificate_path.read_bytes())
    assert certificate.subject.rfc4514_string() == "CN=localhost"
    assert key_path.read_bytes().startswith(b"-----BEGIN PRIVATE KEY-----")
