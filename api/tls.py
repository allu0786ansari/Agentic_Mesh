from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TLSConfig:
    """Uvicorn TLS settings for the development node API."""

    certfile: Path
    keyfile: Path
    cafile: Path | None = None
    require_client_certificate: bool = False

    def validate(self) -> None:
        if not self.certfile.is_file():
            raise FileNotFoundError(f"TLS certificate not found: {self.certfile}")
        if not self.keyfile.is_file():
            raise FileNotFoundError(f"TLS private key not found: {self.keyfile}")
        if self.cafile is not None and not self.cafile.is_file():
            raise FileNotFoundError(f"TLS CA certificate not found: {self.cafile}")

    def uvicorn_kwargs(self) -> dict[str, object]:
        self.validate()
        settings: dict[str, object] = {
            "ssl_certfile": str(self.certfile),
            "ssl_keyfile": str(self.keyfile),
        }
        if self.cafile is not None:
            settings["ssl_ca_certs"] = str(self.cafile)
        if self.require_client_certificate:
            if self.cafile is None:
                raise ValueError("cafile is required when client certificates are mandatory")
            settings["ssl_cert_reqs"] = 2
        return settings


__all__ = ["TLSConfig"]
