from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from api.node_api import create_app
from api.tls import TLSConfig
from agents.auth import JWTAuthManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agentic Mesh node API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--qdrant-path", type=Path, default=Path("qdrant_data"))
    parser.add_argument("--collection", default="insight_embeddings")
    parser.add_argument("--certfile", type=Path, default=None)
    parser.add_argument("--keyfile", type=Path, default=None)
    parser.add_argument("--ca-file", type=Path, default=None)
    parser.add_argument("--require-client-certificate", action="store_true")
    parser.add_argument("--jwt-secret", default=None)
    args = parser.parse_args()
    auth_manager = JWTAuthManager(args.jwt_secret) if args.jwt_secret else None
    application = create_app(args.qdrant_path, args.collection, auth_manager)
    kwargs: dict[str, object] = {"host": args.host, "port": args.port, "app": application}
    if (args.certfile is None) != (args.keyfile is None):
        raise ValueError("--certfile and --keyfile must be provided together")
    if args.certfile is not None and args.keyfile is not None:
        kwargs.update(TLSConfig(args.certfile, args.keyfile, args.ca_file, args.require_client_certificate).uvicorn_kwargs())
    uvicorn.run(**kwargs)


if __name__ == "__main__":
    main()
