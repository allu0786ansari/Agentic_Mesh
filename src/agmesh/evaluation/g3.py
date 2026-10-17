"""Gate G3: the ADR 0014 §9 decision rule applied to the validation mesh runs.

    python -m agmesh.evaluation.g3 --runs results/mesh/<run> ...   # make g3

**Which runs.** The rule is applied to the `full` runs over mTLS (`nodes = https`),
one per site. The `retrieval_only` runs over mTLS, and any `full` runs in-process,
are reported alongside but do not decide.

**The rule.**

* **Default mode stays ``full``** when, on both sites:
  * JSON validity is at least 95%;
  * `full` incident-to-dispatch p95 is at most L_LLM (24 s);
  * investigation p95 is at most 2 s.
* **Otherwise it becomes ``retrieval_only``.**
* **L_LLM stays as preregistered.**

Every number comes from the runs' ``summary.json`` files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import DEFAULT_PARAMS_PATH, load_params

VALIDITY_MIN = 0.95
INVESTIGATION_P95_MS = 2000.0


def row(run: Path) -> dict[str, Any]:
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    llm = s.get("llm") or {}
    lat = s["latency_ms"]
    return {
        "run": run.as_posix(),
        "site": s["site"],
        "mode": s["mode"],
        "nodes": s.get("nodes") or "local",
        "sessions": s["sessions"],
        "by_status": s["by_status"],
        "llm_answered": llm.get("answered"),
        "llm_calls": llm.get("calls"),
        "validity": llm.get("valid_rate"),
        "grounding": llm.get("raw_grounding_rate"),
        "failures": llm.get("failures"),
        "investigation_p95_ms": lat["investigation"]["p95"],
        "incident_to_dispatch_p95_ms": lat["incident_to_dispatch"]["p95"],
        "traces_problems": (s.get("traces") or {}).get("problem_count"),
    }


def decide(rows: list[dict[str, Any]], l_llm_s: float) -> dict[str, Any]:
    deciding = {r["site"]: r for r in rows if r["mode"] == "full" and r["nodes"] == "https"}
    missing = sorted({"site_a", "site_b"} - set(deciding))
    if missing:
        raise SystemExit(f"G3 needs a full run over https for {missing}")
    checks = {}
    for site, r in sorted(deciding.items()):
        validity = r["validity"] or 0.0
        checks[site] = {
            "validity": validity,
            "validity_ok": validity >= VALIDITY_MIN,
            "incident_to_dispatch_p95_s": r["incident_to_dispatch_p95_ms"] / 1000.0,
            "latency_ok": r["incident_to_dispatch_p95_ms"] <= l_llm_s * 1000.0,
            "investigation_p95_ms": r["investigation_p95_ms"],
            "investigation_ok": r["investigation_p95_ms"] <= INVESTIGATION_P95_MS,
        }
    passed = all(
        c["validity_ok"] and c["latency_ok"] and c["investigation_ok"] for c in checks.values()
    )
    return {
        "rule": "docs/adr/0014-process-boundaries-security-tracing-week11.md §9",
        "thresholds": {
            "validity_min": VALIDITY_MIN,
            "incident_to_dispatch_p95_s_max": l_llm_s,
            "investigation_p95_ms_max": INVESTIGATION_P95_MS,
        },
        "checks": checks,
        "default_mode": "full" if passed else "retrieval_only",
        "l_llm_s": l_llm_s,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate G3 from validation mesh runs (ADR 0014 §9)")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--out", type=Path, default=Path("results/G3/g3_decision.json"))
    args = parser.parse_args(argv)
    params = load_params(args.params)
    rows = [row(r) for r in args.runs]
    if any(r["run"].find("__test__") >= 0 for r in rows):
        raise SystemExit("G3 uses validation runs only")
    result = decide(rows, params.mesh.attribution.l_llm_s)
    path = write_artifact(
        args.out,
        kind="gate.g3",
        data_source="runtime",
        payload={**result, "runs": rows, "split": "val"},
        params_path=args.params,
    )
    for site, c in result["checks"].items():
        print(
            f"{site}: validity {c['validity']:.3f} (>= {VALIDITY_MIN}: {c['validity_ok']}), "
            f"full incident-to-dispatch p95 {c['incident_to_dispatch_p95_s']:.1f} s "
            f"(<= {result['l_llm_s']}: {c['latency_ok']}), investigation p95 "
            f"{c['investigation_p95_ms']:.0f} ms (<= 2000: {c['investigation_ok']})"
        )
    print(f"default mode: {result['default_mode']}; L_LLM {result['l_llm_s']} s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
