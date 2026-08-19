from __future__ import annotations

import argparse
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flower simulation scaffold for Agentic Mesh")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--stub", action="store_true", help="Run stub clients only")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    print(f"Flower simulation scaffold starting with {args.rounds} round(s).")
    print(f"Stub mode: {args.stub}")
    print("Week 1 scaffold is operational; no training yet.")


if __name__ == "__main__":
    main()
