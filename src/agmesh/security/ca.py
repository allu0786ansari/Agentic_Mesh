"""Development PKI for the multi-process stack (ADR 0014 §2).

    python -m agmesh.security.ca --out certs/dev
    make certs

**What it writes.** All keys are ECDSA P-256 (``.key``, mode 0600, never committed):

* three CAs, one per role:
  * ``ca-server`` signs one server certificate per node API and one for the
    approval API;
  * ``ca-mesh`` signs the mesh client certificate;
  * ``ca-operator`` signs the operator client certificate;
* ``jwt-mesh.key`` and ``jwt-mesh.pub``, the ES256 key pair the mesh signs node
  tokens with.

**Certificates.**

* Each server certificate carries its DNS name (``node-<id>.agmesh``,
  ``approval-api.agmesh``), ``localhost`` and ``127.0.0.1``, with the extended key
  usage ``serverAuth`` only.
* Client certificates carry ``clientAuth`` only.
* Leaf certificates are valid for 30 days and CAs for 90.
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

NODES = ("a1", "a2", "a3", "a4", "b1", "b2", "b3")
LEAF_DAYS, CA_DAYS = 30, 90


def server_name(node: str) -> str:
    return f"node-{node}.agmesh"


APPROVAL_NAME = "approval-api.agmesh"


@dataclass(frozen=True)
class Issued:
    cert: x509.Certificate
    key: ec.EllipticCurvePrivateKey


def _name(cn: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "agmesh-dev"),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def make_ca(cn: str, days: int = CA_DAYS) -> Issued:
    key = ec.generate_private_key(ec.SECP256R1())
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(_name(cn))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    return Issued(cert, key)


def make_leaf(
    ca: Issued, cn: str, *, server: bool, dns: tuple[str, ...] = (), days: int = LEAF_DAYS
) -> Issued:
    key = ec.generate_private_key(ec.SECP256R1())
    now = _now()
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(ca.cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=True,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [ExtendedKeyUsageOID.SERVER_AUTH if server else ExtendedKeyUsageOID.CLIENT_AUTH]
            ),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca.key.public_key()),
            critical=False,
        )
    )
    if server:
        sans: list[x509.GeneralName] = [x509.DNSName(d) for d in (*dns, "localhost")]
        sans.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))
        builder = builder.add_extension(x509.SubjectAlternativeName(sans), critical=False)
    cert = builder.sign(ca.key, hashes.SHA256())
    return Issued(cert, key)


def _write_key(path: Path, key: ec.EllipticCurvePrivateKey) -> None:
    data = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    path.chmod(0o600)


def _write(out: Path, name: str, issued: Issued) -> None:
    (out / f"{name}.pem").write_bytes(issued.cert.public_bytes(serialization.Encoding.PEM))
    _write_key(out / f"{name}.key", issued.key)


def build(out: Path) -> list[str]:
    """Write the whole development PKI to ``out``; returns the file names written."""
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    ca_server, ca_mesh, ca_operator = (
        make_ca("agmesh-dev ca-server"),
        make_ca("agmesh-dev ca-mesh"),
        make_ca("agmesh-dev ca-operator"),
    )
    _write(out, "ca-server", ca_server)
    _write(out, "ca-mesh", ca_mesh)
    _write(out, "ca-operator", ca_operator)
    for node in NODES:
        name = server_name(node)
        _write(out, f"node-{node}", make_leaf(ca_server, name, server=True, dns=(name,)))
    _write(
        out,
        "approval-api",
        make_leaf(ca_server, APPROVAL_NAME, server=True, dns=(APPROVAL_NAME,)),
    )
    _write(out, "mesh", make_leaf(ca_mesh, "mesh", server=False))
    _write(out, "operator", make_leaf(ca_operator, "operator", server=False))
    jwt_key = ec.generate_private_key(ec.SECP256R1())
    _write_key(out / "jwt-mesh.key", jwt_key)
    (out / "jwt-mesh.pub").write_bytes(
        jwt_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    return sorted(p.name for p in out.iterdir())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Development PKI (ADR 0014 §2)")
    parser.add_argument("--out", type=Path, default=Path("certs/dev"))
    args = parser.parse_args(argv)
    names = build(args.out)
    print(f"{len(names)} files -> {args.out} (private keys mode 0600; never commit them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
