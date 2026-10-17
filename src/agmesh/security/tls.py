"""mTLS contexts and an HTTPS-only uvicorn server (ADR 0014 §3).

* **Version:** TLS 1.3 only, at both ends.
* **Servers** require a client certificate issued by the CA of the role they serve
  (:func:`server_context`).
* **Clients** present their own certificate and check the server's name against the
  name they ask for. With httpx, the name is the ``sni_hostname`` request extension.
* :func:`https_server` builds a uvicorn server whose only listener uses that
  context, so no plaintext port exists.
"""

from __future__ import annotations

import ssl
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PKI_DIR = Path("certs/dev")


@dataclass(frozen=True)
class Pki:
    root: Path = PKI_DIR

    def cert(self, name: str) -> Path:
        return self.root / f"{name}.pem"

    def key(self, name: str) -> Path:
        return self.root / f"{name}.key"

    def require(self, *names: str) -> None:
        missing = [
            p.as_posix() for n in names for p in (self.cert(n), self.key(n)) if not p.is_file()
        ]
        if missing:
            raise SystemExit(f"missing certificates {missing}: run make certs")


def server_context(cert: Path, key: Path, client_ca: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(cert, key)
    ctx.load_verify_locations(cafile=client_ca)
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def client_context(cert: Path | None, key: Path | None, server_ca: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_verify_locations(cafile=server_ca)
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    if cert is not None and key is not None:
        ctx.load_cert_chain(cert, key)
    return ctx


def https_server(app: Any, host: str, port: int, ctx: ssl.SSLContext, **kw: Any) -> Any:
    """A uvicorn server whose only listener is TLS with ``ctx`` (uvicorn >= 0.49)."""
    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        ssl_context_factory=lambda _config, _default: ctx,
        log_level=kw.pop("log_level", "warning"),
        ws="none",  # no websocket endpoints; also avoids loading the websockets stack
        **kw,
    )
    return uvicorn.Server(config)


class ServerThread:
    """Run a uvicorn server in a background thread (tests and the in-process stack)."""

    def __init__(self, server: Any) -> None:
        self.server = server
        self.thread = threading.Thread(target=server.run, daemon=True)

    def __enter__(self) -> ServerThread:
        self.thread.start()
        deadline = time.monotonic() + 10.0
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.01)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10.0)
