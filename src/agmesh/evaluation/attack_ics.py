"""The pinned MITRE ATT&CK for ICS release (mapping protocol §1 rule 5, v1.1).

    python -m agmesh.evaluation.attack_ics extract <ics-attack-20.0.json>   # rebuild
    python -m agmesh.evaluation.attack_ics check                            # verify truth

The ground truth names techniques of **ATT&CK for ICS v20.0**. The STIX bundle is
``ics-attack/ics-attack-20.0.json`` in ``mitre-attack/attack-stix-data`` at the commit
in :data:`PIN`. Only a compact extract is committed: per technique, its name,
tactics, and status. A revoked technique also records MITRE's ``revoked-by`` target.
``extract`` rebuilds the extract from the bundle and refuses a bundle whose SHA-256
differs from the pin.

**Revoked IDs.** v19.0 (28 April 2026) revoked T0855 Unauthorized Command Message and
T0856 Spoof Reporting Message. They are replaced by T1692.001 Command Message and
T1692.002 Reporting Message, sub-techniques of T1692 Unauthorized Message. Models
trained before then name the old IDs. :func:`canonical` follows MITRE's own
``revoked-by`` relationships, so a predicted T0855 is scored as T1692.001. The
mapping is declared before any attribution output exists (protocol v1.1, §5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PIN = {
    "release": "20.0",
    "repository": "https://github.com/mitre-attack/attack-stix-data",
    "path": "ics-attack/ics-attack-20.0.json",
    "commit": "c2494b29dd268c46a5d0ea0990553e4327cd2253",
    "sha256": "c9ca0ce523a1b5e512060a390415e8aefe8f6d32c5f82d7de81264960c1e1183",
}
EXTRACT = Path("evaluation/attack/ics_attack_20.0_techniques.json")


def _tid(obj: dict[str, Any]) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return str(ref["external_id"])
    return None


def extract(bundle_path: Path) -> dict[str, Any]:
    """The technique table of a pinned STIX bundle (refuses any other bundle)."""
    raw = bundle_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PIN["sha256"]:
        raise ValueError(f"{bundle_path}: sha256 {digest} is not the pinned {PIN['sha256']}")
    objects = json.loads(raw)["objects"]
    by_id = {o["id"]: o for o in objects}
    tactic = {o["x_mitre_shortname"]: o["name"] for o in objects if o["type"] == "x-mitre-tactic"}
    revoked_by = {
        o["source_ref"]: o["target_ref"]
        for o in objects
        if o["type"] == "relationship" and o["relationship_type"] == "revoked-by"
    }
    techniques: dict[str, Any] = {}
    for o in objects:
        if o["type"] != "attack-pattern":
            continue
        tid = _tid(o)
        if tid is None:
            continue
        entry: dict[str, Any] = {
            "name": o["name"],
            "tactics": sorted(tactic[p["phase_name"]] for p in o.get("kill_chain_phases", [])),
            "status": "revoked"
            if o.get("revoked")
            else "deprecated"
            if o.get("x_mitre_deprecated")
            else "active",
        }
        if o["id"] in revoked_by:
            entry["revoked_by"] = _tid(by_id[revoked_by[o["id"]]])
        techniques[tid] = entry
    return {"pin": PIN, "techniques": dict(sorted(techniques.items()))}


@lru_cache(maxsize=1)
def table(path: Path = EXTRACT) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8"))["techniques"])


def canonical(tid: str) -> str:
    """The active technique an ID stands for in the pinned release.

    Follows ``revoked-by`` (T0855 -> T1692.001). An unknown or deprecated ID is
    returned unchanged; it then matches nothing in the ground truth.
    """
    techniques = table()
    seen = set()
    while tid in techniques and techniques[tid]["status"] == "revoked" and tid not in seen:
        seen.add(tid)
        tid = techniques[tid].get("revoked_by") or tid
    return tid


def tactics(tid: str) -> list[str]:
    return list(table()[canonical(tid)]["tactics"])


def check_ids(ids: set[str]) -> list[str]:
    """Problems with the IDs a ground truth uses: each must be active in the pin."""
    techniques = table()
    problems = []
    for tid in sorted(ids):
        entry = techniques.get(tid)
        if entry is None:
            problems.append(f"{tid}: not in ATT&CK for ICS {PIN['release']}")
        elif entry["status"] != "active":
            problems.append(f"{tid}: {entry['status']} in ATT&CK for ICS {PIN['release']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pinned ATT&CK for ICS release")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("bundle", type=Path)
    ex.add_argument("--out", type=Path, default=EXTRACT)
    sub.add_parser("check")
    args = parser.parse_args(argv)
    if args.cmd == "extract":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(extract(args.bundle), indent=1) + "\n", encoding="utf-8")
        print(args.out)
        return 0
    from agmesh.evaluation.ground_truth import OUT_DIR, TACTICS

    ids = set(TACTICS)
    for site in ("a", "b"):
        doc = json.loads((OUT_DIR / f"mapping_site_{site}.json").read_text(encoding="utf-8"))
        for event in doc["events"].values():
            ids |= {event["primary"], *event["acceptable"]} - {None}
    problems = check_ids(ids)
    problems += [
        f"{t}: tactics {TACTICS[t]} differ from the pin {table()[t]['tactics']}"
        for t in TACTICS
        if t in table() and sorted(TACTICS[t]) != table()[t]["tactics"]
    ]
    print("\n".join(problems) or f"{len(ids)} technique IDs active in v{PIN['release']}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
