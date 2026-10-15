"""Passages for the vector store (Plan Week 9 task 2; ADR 0010 §2).

**ATT&CK** (``mitre_ics``). Each technique gets:

* one summary passage: ``"<name>. Tactics: <tactics>. <description>"``;
* one detection passage per analytic: ``"<name>. Detection: <analytic>"``.

A technique's score is the maximum over its passages (``retriever``). The encoder
reads at most 256 word-pieces, so a single combined chunk would lose the
detection text.

**NIST SP 800-82 Rev. 3** (``nist_80082``). The text is split at numbered section
headings, then into pieces of at most :data:`NIST_MAX_WORDS` words. These are
supplementary context only, never technique candidates.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from agmesh.kb.stix_parser import Technique

NIST_MAX_WORDS = 180


@dataclass(frozen=True)
class Passage:
    passage_id: str
    kind: str  # "summary" | "detection" | "nist"
    text: str
    technique_id: str | None = None
    section: str | None = None


def technique_passages(techniques: Iterable[Technique]) -> list[Passage]:
    out: list[Passage] = []
    for t in techniques:
        tactics = ", ".join(t.tactics)
        out.append(
            Passage(
                f"{t.technique_id}/summary",
                "summary",
                f"{t.name}. Tactics: {tactics}. {t.description}",
                t.technique_id,
            )
        )
        for i, a in enumerate(t.analytics):
            out.append(
                Passage(
                    f"{t.technique_id}/detection/{i}",
                    "detection",
                    f"{t.name}. Detection: {a.description}",
                    t.technique_id,
                )
            )
    return out


_HEADING = re.compile(
    r"^(?P<num>(?:\d{1,2}|[A-H])(?:\.\d{1,2}){0,3})\.?\s+"
    r"(?P<title>[A-Z][A-Za-z0-9 ,:;()/&'\-]{3,90})$"
)


def split_sections(text: str) -> list[tuple[str, str]]:
    """``(section, body)`` pairs at numbered headings (text before the first: ``front``)."""
    sections: list[tuple[str, list[str]]] = [("front", [])]
    for raw in text.splitlines():
        line = raw.strip()
        m = _HEADING.match(line)
        if m and not line.endswith("."):
            sections.append((f"{m.group('num')} {m.group('title').strip()}", []))
        elif line:
            sections[-1][1].append(line)
    return [(name, " ".join(body)) for name, body in sections if body]


def nist_passages(text: str, max_words: int = NIST_MAX_WORDS) -> list[Passage]:
    out: list[Passage] = []
    for name, body in split_sections(text):
        words = body.split()
        for i in range(0, len(words), max_words):
            piece = " ".join(words[i : i + max_words])
            out.append(
                Passage(
                    f"nist/{name.split()[0]}/{i // max_words}",
                    "nist",
                    f"{name}. {piece}",
                    None,
                    name,
                )
            )
    return out


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
