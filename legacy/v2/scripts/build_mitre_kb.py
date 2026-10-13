from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.security.kb_builder import MITREKnowledgeBaseBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Index MITRE ATT&CK for ICS and NIST guidance in embedded Qdrant")
    parser.add_argument("--stix", type=Path, required=True)
    parser.add_argument("--nist", type=Path, default=None)
    parser.add_argument("--qdrant-path", type=Path, default=Path("qdrant_data_week12"))
    parser.add_argument("--collection", default="mitre_kb")
    args = parser.parse_args()
    builder = MITREKnowledgeBaseBuilder(args.qdrant_path, collection=args.collection)
    indexed = builder.build(args.stix, args.nist)
    print(json.dumps({"indexed": indexed, "collection": args.collection, "count": builder.count()}, indent=2))


if __name__ == "__main__":
    main()
