"""Parse the pinned ATT&CK for ICS STIX bundle (Plan Week 9 task 1; ADR 0010 §1-2).

The bundle is ``ics-attack-20.0.json`` at the commit in
:data:`agmesh.evaluation.attack_ics.PIN`. :func:`fetch` downloads it from GitHub
and :func:`load` refuses any other SHA-256. For every **active** technique the
parser returns:

* its name, tactics and description;
* the analytics of its detection strategies (``detects`` relationships, ATT&CK v18
  and later);
* its mitigations (``mitigates``);
* its parent, if it is a sub-technique.

Text is cleaned of citation markers, markdown links and HTML tags.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agmesh.evaluation.attack_ics import PIN

RAW_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/{commit}/{path}"
DEFAULT_PATH = Path("data/kb/ics-attack-20.0.json")


@dataclass(frozen=True)
class Analytic:
    name: str
    description: str


@dataclass(frozen=True)
class Technique:
    technique_id: str
    name: str
    tactics: tuple[str, ...]
    description: str
    is_subtechnique: bool
    parent: str | None
    analytics: tuple[Analytic, ...] = ()
    mitigations: tuple[tuple[str, str], ...] = ()  # (M-ID, name)
    platforms: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CITATION = re.compile(r"\(Citation:[^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|/)[^)]*\)")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def clean(text: str) -> str:
    text = _CITATION.sub("", text or "")
    text = _LINK.sub(r"\1", text)
    text = _TAG.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([.,;:])", r"\1", text)


def _tid(obj: dict[str, Any]) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return str(ref["external_id"])
    return None


def fetch(dest: Path = DEFAULT_PATH) -> Path:
    """Download the pinned bundle (GitHub raw at the pinned commit) and verify it."""
    if dest.is_file():
        verify(dest)
        return dest
    url = RAW_URL.format(commit=PIN["commit"], path=PIN["path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (fixed https URL)
        tmp.write_bytes(response.read())
    verify(tmp)
    tmp.replace(dest)
    return dest


def verify(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PIN["sha256"]:
        raise ValueError(f"{path}: SHA-256 {digest} is not the pinned {PIN['sha256']}")


def parse(bundle: dict[str, Any]) -> list[Technique]:
    """Active techniques of a STIX bundle, sorted by ID."""
    objects = bundle["objects"]
    by_id = {o["id"]: o for o in objects}
    tactic = {o["x_mitre_shortname"]: o["name"] for o in objects if o["type"] == "x-mitre-tactic"}
    detects: dict[str, list[str]] = {}
    mitigates: dict[str, list[str]] = {}
    parent_of: dict[str, str] = {}
    for o in objects:
        if o["type"] != "relationship" or o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        kind, src, dst = o["relationship_type"], o["source_ref"], o["target_ref"]
        if kind == "detects":
            detects.setdefault(dst, []).append(src)
        elif kind == "mitigates":
            mitigates.setdefault(dst, []).append(src)
        elif kind == "subtechnique-of":
            parent_of[src] = dst
    out = []
    for o in objects:
        if o["type"] != "attack-pattern" or o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        tid = _tid(o)
        if tid is None:
            continue
        analytics: list[Analytic] = []
        for sid in sorted(detects.get(o["id"], [])):
            strategy = by_id.get(sid, {})
            for aid in strategy.get("x_mitre_analytic_refs", []):
                a = by_id.get(aid)
                if a is None or a.get("revoked") or a.get("x_mitre_deprecated"):
                    continue
                text = clean(a.get("description", ""))
                if text:
                    analytics.append(Analytic(clean(a.get("name", "")), text))
        mits = sorted(
            {
                (_tid(by_id[m]) or "", clean(by_id[m]["name"]))
                for m in mitigates.get(o["id"], [])
                if m in by_id
                and not by_id[m].get("revoked")
                and not by_id[m].get("x_mitre_deprecated")
            }
        )
        parent = parent_of.get(o["id"])
        out.append(
            Technique(
                technique_id=tid,
                name=clean(o["name"]),
                tactics=tuple(
                    sorted(tactic[p["phase_name"]] for p in o.get("kill_chain_phases", []))
                ),
                description=clean(o.get("description", "")),
                is_subtechnique=bool(o.get("x_mitre_is_subtechnique")),
                parent=_tid(by_id[parent]) if parent in by_id else None,
                analytics=tuple(analytics),
                mitigations=tuple(mits),
                platforms=tuple(o.get("x_mitre_platforms", [])),
            )
        )
    return sorted(out, key=lambda t: t.technique_id)


def load(path: Path = DEFAULT_PATH, *, check: bool = True) -> list[Technique]:
    if check:
        verify(path)
    return parse(json.loads(path.read_text(encoding="utf-8")))
