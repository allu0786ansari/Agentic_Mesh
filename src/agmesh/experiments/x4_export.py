"""Export each mesh run's sessions for X4 scoring (ADR 0017 §4).

    python -m agmesh.experiments.x4_export              # make x4-export
    python -m agmesh.experiments.x4_export --runs results/mesh/<run> ...

``storage.db`` stays on the machine that ran the mesh. This writes
``sessions.jsonl`` next to each run's ``summary.json``. It is small and committed,
and it holds per session:

* **The session:** ID, status, the incident (start, end, nodes, severity, alert
  count) and the mode used.
* **The attribution:**
  * the technique, its tactics, the top-3 of its ranked list and the confidence;
  * the number of candidates;
  * the LLM's schema validity, grounding, error class and seconds.

It holds no evidence text and no prompt. Those stay in ``storage.db`` with the run.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def _step(con: sqlite3.Connection, sid: str, step: str) -> dict[str, Any] | None:
    row = con.execute(
        "SELECT payload FROM session_steps WHERE session_id = ? AND step = ?", (sid, step)
    ).fetchone()
    return json.loads(row[0]) if row else None


def export(run: Path) -> Path:
    db = run / "storage.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        sessions = con.execute(
            "SELECT session_id, status, site_id, opened_at, attribution_mode FROM sessions "
            "ORDER BY opened_at, session_id"
        ).fetchall()
        out = run / "sessions.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for sid, status, site, opened, mode in sessions:
                inc = _step(con, sid, "triage") or {}
                att = _step(con, sid, "attribute") or {}
                llm = att.get("llm") or {}
                validation = llm.get("validation") or {}
                record = {
                    "session_id": sid,
                    "site": site,
                    "status": status,
                    "opened_at": opened,
                    "incident": {
                        "start_s": inc.get("start_s"),
                        "end_s": inc.get("end_s"),
                        "nodes": inc.get("nodes"),
                        "severity": inc.get("severity"),
                        "alerts": len(inc.get("alert_ids") or []),
                    },
                    "attribution_mode": att.get("attribution_mode", mode),
                    "technique_id": att.get("technique_id"),
                    "tactics": att.get("tactics") or [],
                    "ranked_top3": (att.get("ranked") or [])[:3],
                    "confidence": att.get("confidence"),
                    "candidates_n": len(att.get("candidates") or []),
                    "llm": {
                        "called": bool(llm),
                        "error_kind": llm.get("kind"),
                        "seconds": llm.get("seconds"),
                        "schema_valid": validation.get("schema_valid"),
                        "grounded": validation.get("grounded"),
                        "valid": validation.get("valid"),
                        "raw_technique": validation.get("technique_id"),
                    },
                }
                fh.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        con.close()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export mesh sessions for X4 (ADR 0017 §4)")
    parser.add_argument("--runs", nargs="*", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path("results/mesh"))
    args = parser.parse_args(argv)
    runs = args.runs or sorted(p.parent for p in args.root.glob("*/storage.db"))
    for run in runs:
        out = export(run)
        n = sum(1 for _ in out.open(encoding="utf-8"))
        print(f"{run.name}: {n} sessions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
