from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from ipaddress import IPv4Address
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def generate(output_dir: Path, common_name: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(common_name), x509.IPAddress(IPv4Address("127.0.0.1"))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path = output_dir / "dev-key.pem"
    cert_path = output_dir / "dev-cert.pem"
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a self-signed development mTLS certificate")
    parser.add_argument("--output", type=Path, default=Path("certs/dev"))
    parser.add_argument("--common-name", default="localhost")
    args = parser.parse_args()
    cert, key = generate(args.output, args.common_name)
    print(f"certificate={cert}")
    print(f"private_key={key}")


if __name__ == "__main__":
    main()
