"""Build the knowledge base (Plan Week 9 task 3; ADR 0010 §1-2).

    python -m agmesh.kb.build                    # reference encoder, embedded Qdrant
    make kb

The build:

1. **Fetches and verifies the pinned ATT&CK for ICS bundle.** It parses every active
   technique and writes ``data/kb/techniques.json`` (names, tactics, analytics,
   mitigations; Week 10's dispatcher reads the mitigations from it).
2. **Embeds the passages** with the reference encoder into the ``mitre_ics``
   collection.
3. **Embeds NIST SP 800-82 Rev. 3** into ``nist_80082`` when the PDF is present
   (``make kb-sources``).
4. **Creates ``incident_memory`` empty.**
5. **Writes ``results/kb/manifest.json``.** It records the bundle pin, the
   encoder's identity, and the technique, passage and dimension counts per
   collection.

The first reference build records the encoder snapshot in
``configs/kb/encoder.lock.json``, and NIST's SHA-256 in
``configs/kb/sources.lock.json``. Later builds refuse anything different.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from agmesh.common.artifacts import write_artifact
from agmesh.common.env import load_env
from agmesh.evaluation.attack_ics import PIN
from agmesh.kb import chunking, stix_parser
from agmesh.kb.embed import Encoder, HashingEncoder, SentenceEncoder
from agmesh.kb.store import COLLECTIONS, QdrantStore, Store

NIST_URL = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-82r3.pdf"
NIST_FILE = "NIST.SP.800-82r3.pdf"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) agentic-mesh-kb/1.0"
ENCODER_LOCK = Path("configs/kb/encoder.lock.json")
SOURCES_LOCK = Path("configs/kb/sources.lock.json")


def _lock(path: Path, key: str, value: dict[str, Any]) -> None:
    """Record ``value`` under ``key`` on first use; afterwards it must match."""
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if key in current:
        if current[key] != value:
            raise SystemExit(f"{path}: {key} is pinned to {current[key]}, this build has {value}")
        return
    current[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_nist(dest_dir: Path) -> Path:
    dest = dest_dir / NIST_FILE
    if not dest.is_file():
        dest_dir.mkdir(parents=True, exist_ok=True)
        # nvlpubs.nist.gov answers 403 to Python's default User-Agent.
        request = urllib.request.Request(NIST_URL, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
                dest.write_bytes(response.read())
        except urllib.error.HTTPError as exc:
            raise SystemExit(
                f"NIST refused the download ({exc.code}). Fetch it by hand, then run `make kb`:\n"
                f'  curl -L -A "{USER_AGENT}" -o {dest} {NIST_URL}\n'
                "The knowledge base builds without it; PT6 does not depend on NIST (ADR 0010 §1)."
            ) from exc
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    _lock(SOURCES_LOCK, "nist_sp_800_82r3", {"url": NIST_URL, "sha256": digest})
    return dest


def build(
    techniques: list[stix_parser.Technique],
    encoder: Encoder,
    store: Store,
    *,
    nist_text: str | None = None,
) -> dict[str, Any]:
    passages = chunking.technique_passages(techniques)
    store.create("mitre_ics", encoder.dim)
    vectors = encoder.encode([p.text for p in passages])
    store.upsert(
        "mitre_ics",
        vectors,
        [
            {
                "passage_id": p.passage_id,
                "kind": p.kind,
                "technique_id": p.technique_id,
                "text": p.text,
            }
            for p in passages
        ],
    )
    store.create("nist_80082", encoder.dim)
    nist = chunking.nist_passages(nist_text) if nist_text else []
    if nist:
        store.upsert(
            "nist_80082",
            encoder.encode([p.text for p in nist]),
            [{"passage_id": p.passage_id, "section": p.section, "text": p.text} for p in nist],
        )
    store.create("incident_memory", encoder.dim)
    per_tech = collections.Counter(p.technique_id for p in passages)
    return {
        "attack_pin": PIN,
        "techniques": len(techniques),
        "techniques_indexed": len(per_tech),
        "passages": {
            "summary": sum(p.kind == "summary" for p in passages),
            "detection": sum(p.kind == "detection" for p in passages),
        },
        "passages_per_technique": {"min": min(per_tech.values()), "max": max(per_tech.values())},
        "collections": {c: {"points": store.count(c), "dim": store.dim(c)} for c in COLLECTIONS},
        "nist": {"passages": len(nist), "present": bool(nist)},
        "encoder": encoder.identity(),
    }


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(description="Build the ATT&CK for ICS knowledge base")
    parser.add_argument("--out", type=Path, default=Path("data/kb"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--encoder", choices=("reference", "hashing"), default="reference")
    parser.add_argument("--fetch-nist", action="store_true", help="download NIST SP 800-82r3 first")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)
    bundle = stix_parser.fetch(args.out / Path(PIN["path"]).name)
    techniques = stix_parser.load(bundle)
    (args.out / "techniques.json").write_text(
        json.dumps([t.to_dict() for t in techniques], indent=1) + "\n", encoding="utf-8"
    )
    encoder: Encoder = SentenceEncoder() if args.encoder == "reference" else HashingEncoder()
    identity = encoder.identity()
    if identity.get("reference"):
        _lock(
            ENCODER_LOCK,
            identity["name"],
            {k: identity[k] for k in ("revision", "weights_sha256", "dim")},
        )
    nist_pdf = fetch_nist(args.out) if args.fetch_nist else args.out / NIST_FILE
    nist_text = None
    if nist_pdf.is_file():
        digest = hashlib.sha256(nist_pdf.read_bytes()).hexdigest()
        _lock(SOURCES_LOCK, "nist_sp_800_82r3", {"url": NIST_URL, "sha256": digest})
        nist_text = chunking.pdf_text(nist_pdf)
    else:
        print(f"note: {nist_pdf} not found; nist_80082 stays empty (ADR 0010 §1)")
    if args.encoder != "reference" and not args.fixture:
        raise SystemExit("the hashing encoder is for tests only: pass --fixture (ADR 0010 §5)")
    store = QdrantStore(args.out / "qdrant")
    manifest = build(techniques, encoder, store, nist_text=nist_text)
    manifest["store"] = (args.out / "qdrant").as_posix()
    path = write_artifact(
        args.results / "kb" / "manifest.json",
        kind="kb.manifest",
        data_source="fixture" if args.fixture or args.encoder != "reference" else "runtime",
        payload=manifest,
    )
    store.close()
    print(
        f"{manifest['techniques']} techniques, {sum(manifest['passages'].values())} passages, "
        f"NIST {manifest['nist']['passages']} passages, dim {encoder.dim} -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
