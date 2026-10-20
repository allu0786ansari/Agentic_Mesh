"""X4: attribution in the four modes on validation incidents (ADR 0017 §4; RQ3).

    python -m agmesh.experiments.x4_score                 # make x4
    python -m agmesh.experiments.x4_score --runs results/mesh/<run> ...

It reads each mesh run's ``sessions.jsonl`` (written by ``make x4-export``), the
frozen mappings in ``evaluation/ground_truth/`` and the validation partitions, which
give the events' labelled spans. It writes ``results/X4/attribution_val.{json,md}``.

**Runs.** For each site and mode, the latest validation run with a
``sessions.jsonl``. For ``full`` it is the latest run over mTLS (``nodes = https``,
the Week 11 runs).

**Site A, 15 events.**

* **Matching:** an incident matches an event when its window ``[start, end]`` meets
  the event's labelled span ``[start, end + 60 s)``. That is the same tolerance
  convention as event recall (:mod:`agmesh.evaluation.events`).
* **The event's attribution:** that of its earliest matching incident.
* **Scores:** top-1 and top-3 are exact matches of the canonical ID against the
  acceptable set (revoked-by followed; a parent never matches a sub-technique).
  * Their primary denominator is all 15 events, so a missed event or an abstention
    counts as wrong.
  * The rate over the events that got an answer is also given.
* **Tactic accuracy:** any shared tactic.
* **Grounding:** the raw rate over every incident whose LLM answered.

**Site B, a case study.** Validation episodes are spans across the three nodes.

* **Per episode and mode:** the technique of the earliest matching incident, with
  the acceptable set beside it, and no accuracy.
* **Per mode:** JSON validity and raw grounding.

**Checks.**

* Every mode is present.
* Every dispatched technique is an active candidate technique (grounding after
  validation 100%).
* A valid LLM answer is the technique dispatched.
* Each export has the session count of its run.

A failure is reported and the exit status is 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import Params, load_params
from agmesh.data.access import load_partition
from agmesh.data.manifest import read_manifest
from agmesh.evaluation import attack_ics
from agmesh.evaluation.stats import bootstrap_ci, paired_bootstrap
from agmesh.kb.retriever import excluded
from agmesh.mesh.attribution import MODES

GT = Path("evaluation/ground_truth")
RESAMPLES = 10_000
SEED = 2026


@dataclass(frozen=True)
class Event:
    key: str
    nodes: tuple[str, ...]
    start_s: float
    end_s: float
    primary: str
    acceptable: tuple[str, ...]
    tactics: tuple[str, ...]


@dataclass
class Run:
    site: str
    mode: str
    path: Path
    summary: dict[str, Any]
    sessions: list[dict[str, Any]] = field(default_factory=list)


# ------------------------------------------------------------------------- inputs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def site_a_events(params: Params, root: Path, mapping: dict[str, Any]) -> list[Event]:
    """The validation events, with their labelled spans from the partitions."""
    wanted = {k: v for k, v in mapping["events"].items() if v["partition"] == "val_attack"}
    out: list[Event] = []
    for client in sorted(read_manifest(root / "site_a")["nodes"]):
        labels = load_partition(params, "site_a", client, "val_attack", root=root).labels
        hit = labels.target > 0
        for key in np.unique(labels.group[hit]):
            entry = wanted.get(str(key))
            if entry is None or entry["client"] != client:
                raise ValueError(f"site_a {client} {key}: labelled but not in the mapping")
            t = labels.time_s[(labels.group == key) & hit]
            out.append(
                Event(
                    str(key),
                    (client,),
                    float(t.min()),
                    float(t.max()) + 1.0,  # 1 Hz rows
                    entry["primary"],
                    tuple(entry["acceptable"]),
                    tuple(entry["tactics"]),
                )
            )
    if {e.key for e in out} != set(wanted):
        raise ValueError("site_a: validation events in the mapping and the partitions differ")
    return sorted(out, key=lambda e: e.start_s)


def site_b_episodes(params: Params, root: Path, mapping: dict[str, Any]) -> list[Event]:
    """The validation episodes: each one's labelled windows on any node."""
    length = float(params.data.windows.site_b.length_s)
    spans: dict[str, list[float]] = {}
    nodes: dict[str, set[str]] = {}
    for client in sorted(read_manifest(root / "site_b")["nodes"]):
        labels = load_partition(params, "site_b", client, "val_attack", root=root).labels
        hit = (labels.target > 0) & (labels.group != "-1")
        for key in np.unique(labels.group[hit]):
            t = labels.time_s[(labels.group == key) & hit]
            s = spans.setdefault(str(key), [np.inf, -np.inf])
            s[0], s[1] = min(s[0], float(t.min())), max(s[1], float(t.max()) + length)
            nodes.setdefault(str(key), set()).add(client)
    out = []
    for key, (a, b) in spans.items():
        entry = mapping["events"][key]
        out.append(
            Event(
                key,
                tuple(sorted(nodes[key])),
                a,
                b,
                entry["primary"],
                tuple(entry["acceptable"]),
                tuple(entry["tactics"]),
            )
        )
    return sorted(out, key=lambda e: e.start_s)


def _run_of(path: Path) -> Run:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    lines = (path / "sessions.jsonl").read_text(encoding="utf-8").splitlines()
    return Run(summary["site"], summary["mode"], path, summary, [json.loads(x) for x in lines if x])


def select_runs(root: Path, split: str = "val") -> list[Run]:
    """The latest run per site and mode with an export; ``full`` over mTLS."""
    runs: list[Run] = []
    for site in ("site_a", "site_b"):
        for mode in MODES:
            found = sorted(
                p.parent
                for p in root.glob(f"{site}__{mode}__{split}__*/sessions.jsonl")
                if (p.parent / "summary.json").is_file()
            )
            if mode == "full":
                found = [
                    p
                    for p in found
                    if json.loads((p / "summary.json").read_text()).get("nodes") == "https"
                ]
            if found:
                runs.append(_run_of(found[-1]))
    return runs


# ------------------------------------------------------------------------ scoring


def matches(incident: dict[str, Any], event: Event, tolerance_s: float) -> bool:
    a, b = float(incident["start_s"]), float(incident["end_s"])
    return a < event.end_s + tolerance_s and b >= event.start_s


def earliest(
    sessions: list[dict[str, Any]], event: Event, tolerance_s: float
) -> tuple[dict[str, Any] | None, int]:
    hits = [s for s in sessions if matches(s["incident"], event, tolerance_s)]
    hits.sort(key=lambda s: (float(s["incident"]["start_s"]), s["opened_at"], s["session_id"]))
    return (hits[0] if hits else None), len(hits)


def _canon(tid: str | None) -> str | None:
    return attack_ics.canonical(tid) if tid else None


def per_event(
    sessions: list[dict[str, Any]], events: list[Event], tolerance_s: float
) -> list[dict[str, Any]]:
    rows = []
    for e in events:
        s, n = earliest(sessions, e, tolerance_s)
        tid = _canon(s["technique_id"]) if s else None
        top3 = [_canon(t) for t in (s["ranked_top3"] if s else [])]
        tactics = set(attack_ics.tactics(tid)) if tid else set()
        rows.append(
            {
                "event": e.key,
                "nodes": list(e.nodes),
                "acceptable": list(e.acceptable),
                "matched": s is not None,
                "matching_incidents": n,
                "session_id": s["session_id"] if s else None,
                "incident_nodes": s["incident"]["nodes"] if s else None,
                "attribution_mode": s["attribution_mode"] if s else None,
                "technique": tid,
                "top3": top3,
                "abstained": s is not None and tid is None,
                "top1": tid in e.acceptable,
                "top3_hit": any(t in e.acceptable for t in top3),
                "tactic": bool(tactics & set(e.tactics)),
            }
        )
    return rows


def _ci(values: list[bool]) -> dict[str, float]:
    iv = bootstrap_ci(np.asarray(values, float), resamples=RESAMPLES, seed=SEED)
    return {"estimate": iv.estimate, "low": iv.low, "high": iv.high}


def llm_rates(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [s["llm"] for s in sessions if s["llm"]["called"] and s["llm"]["error_kind"] is None]
    called = [s for s in sessions if s["llm"]["called"]]
    if not called:
        return {"calls": 0}
    return {
        "calls": len(called),
        "answered": len(answered),
        "schema_valid_rate": float(np.mean([bool(x["schema_valid"]) for x in answered]))
        if answered
        else None,
        "raw_grounding_rate": float(np.mean([bool(x["grounded"]) for x in answered]))
        if answered
        else None,
        "valid_rate": float(np.mean([bool(x["valid"]) for x in answered])) if answered else None,
        "errors": dict(Counter(s["llm"]["error_kind"] for s in called if s["llm"]["error_kind"])),
    }


def diversity(techniques: list[str | None]) -> dict[str, Any]:
    """How varied a mode's answers are (ADR 0018 §3); abstentions are left out.

    A mode whose every answer is the same technique is ``constant``: its accuracy is
    then the share of events whose acceptable set contains that technique, which is
    a constant-answer baseline, not attribution.
    """
    answers = [t for t in techniques if t]
    counts = Counter(answers)
    modal, n = counts.most_common(1)[0] if counts else (None, 0)
    return {
        "answered": len(answers),
        "distinct": len(counts),
        "modal": modal,
        "modal_share": n / len(answers) if answers else None,
        "constant": len(answers) > 1 and len(counts) == 1,
    }


def site_a_mode(
    sessions: list[dict[str, Any]], events: list[Event], tolerance_s: float
) -> dict[str, Any]:
    rows = per_event(sessions, events, tolerance_s)
    answered = [r for r in rows if r["technique"]]
    unmatched = [
        s for s in sessions if not any(matches(s["incident"], e, tolerance_s) for e in events)
    ]
    return {
        "events": len(rows),
        "matched": sum(r["matched"] for r in rows),
        "missed": sum(not r["matched"] for r in rows),
        "abstained": sum(r["abstained"] for r in rows),
        "top1": _ci([r["top1"] for r in rows]),
        "top3": _ci([r["top3_hit"] for r in rows]),
        "top1_answered": float(np.mean([r["top1"] for r in answered])) if answered else None,
        "top3_answered": float(np.mean([r["top3_hit"] for r in answered])) if answered else None,
        "answered": len(answered),
        "tactic_accuracy_answered": float(np.mean([r["tactic"] for r in answered]))
        if answered
        else None,
        "incidents": len(sessions),
        "false_positive_incidents": len(unmatched),
        "llm": llm_rates(sessions),
        "answers": diversity([r["technique"] for r in rows]),
        "per_event": rows,
    }


def baselines(events: list[Event], mapping: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(t for e in events for t in e.acceptable)
    best = max(sorted(counts), key=lambda t: counts[t])
    constant = {
        t: {
            "all_50": v,
            "these_events": float(np.mean([t in e.acceptable for e in events])),
        }
        for t, v in sorted(mapping["constant_answer_baseline"].items())
    }
    return {
        "majority_class": {
            "technique": best,
            "top1": _ci([best in e.acceptable for e in events]),
        },
        "constant_answer": constant,
    }


def site_b_mode(
    sessions: list[dict[str, Any]], episodes: list[Event], tolerance_s: float
) -> dict[str, Any]:
    rows = per_event(sessions, episodes, tolerance_s)
    for r in rows:  # a case study: the flag is shown per episode, never aggregated
        r["in_acceptable"] = r.pop("top1")
        r.pop("top3_hit")
        r.pop("tactic")
    return {
        "incidents": len(sessions),
        "by_status": dict(Counter(s["status"] for s in sessions)),
        "by_attribution_mode": dict(Counter(s["attribution_mode"] for s in sessions)),
        "techniques": dict(Counter(_canon(s["technique_id"]) for s in sessions)),
        "answers": diversity([_canon(s["technique_id"]) for s in sessions]),
        "llm": llm_rates(sessions),
        "per_episode": rows,
    }


# ------------------------------------------------------------------------- checks


def candidate_universe() -> set[str]:
    gone = excluded(("Impact",))
    return {t for t, v in attack_ics.table().items() if v["status"] == "active" and t not in gone}


def checks(runs: list[Run]) -> list[str]:
    problems: list[str] = []
    universe = candidate_universe()
    for site in ("site_a", "site_b"):
        have = {r.mode for r in runs if r.site == site}
        problems += [f"{site}: no {m} run with sessions.jsonl" for m in MODES if m not in have]
    for r in runs:
        name = r.path.name
        if len(r.sessions) != r.summary["sessions"]:
            problems.append(f"{name}: {len(r.sessions)} exported, {r.summary['sessions']} in run")
        for s in r.sessions:
            tid = _canon(s["technique_id"])
            if tid is not None and tid not in universe:
                problems.append(f"{name} {s['session_id']}: {tid} is not a candidate technique")
            valid = s["llm"]["valid"] and s["attribution_mode"] == r.mode
            if valid and _canon(s["llm"]["raw_technique"]) != tid:
                problems.append(f"{name} {s['session_id']}: valid answer is not dispatched")
    return problems


# ------------------------------------------------------------------------ driver


def run(
    params: Params, runs: list[Run], partitions: Path, gt: Path = GT
) -> tuple[dict[str, Any], list[str]]:
    tol = float(params.evaluation.event_tolerance_s)
    map_a, map_b = gt / "mapping_site_a.json", gt / "mapping_site_b.json"
    mapping_a = json.loads(map_a.read_text())
    mapping_b = json.loads(map_b.read_text())
    out: dict[str, Any] = {
        "protocol": "docs/adr/0017-x3-x6-week13.md §4",
        "split": "val",
        "tolerance_s": tol,
        "resamples": RESAMPLES,
        "seed": SEED,
        "mappings": {"site_a": _sha(map_a), "site_b": _sha(map_b)},
        "runs": {f"{r.site}/{r.mode}": r.path.name for r in runs},
        "development_evidence": True,
    }
    by = {(r.site, r.mode): r for r in runs}
    if any(r.site == "site_a" for r in runs):
        events = site_a_events(params, partitions, mapping_a)
        a: dict[str, Any] = {
            "events": [
                {"event": e.key, "node": e.nodes[0], "start_s": e.start_s, "end_s": e.end_s}
                for e in events
            ],
            "baselines": baselines(events, mapping_a),
            "modes": {},
        }
        for mode in MODES:
            if ("site_a", mode) in by:
                a["modes"][mode] = site_a_mode(by["site_a", mode].sessions, events, tol)
        if {"full", "retrieval_only"} <= set(a["modes"]):
            f = [r["top3_hit"] for r in a["modes"]["full"]["per_event"]]
            g = [r["top3_hit"] for r in a["modes"]["retrieval_only"]["per_event"]]
            pb = paired_bootstrap(
                np.asarray(f, float), np.asarray(g, float), resamples=RESAMPLES, seed=SEED
            )
            a["full_vs_retrieval_only_top3"] = {
                "difference": pb.mean_difference,
                "low": pb.ci.low,
                "high": pb.ci.high,
                "p_two_sided": pb.p_value,
            }
        out["site_a"] = a
    if any(r.site == "site_b" for r in runs):
        episodes = site_b_episodes(params, partitions, mapping_b)
        b: dict[str, Any] = {
            "episodes": [
                {"episode": e.key, "nodes": list(e.nodes), "start_s": e.start_s, "end_s": e.end_s}
                for e in episodes
            ],
            "modes": {},
        }
        for mode in MODES:
            if ("site_b", mode) in by:
                b["modes"][mode] = site_b_mode(by["site_b", mode].sessions, episodes, tol)
        out["site_b"] = b
    problems = checks(runs)
    out["checks"] = {"passed": not problems, "problems": problems}
    return out, problems


def _answers(d: dict[str, Any]) -> str:
    if not d["answered"]:
        return "—"
    tag = " **constant**" if d["constant"] else ""
    return f"{d['distinct']} ({d['modal']} {_pct(d['modal_share'])}){tag}"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def markdown(rep: dict[str, Any]) -> str:
    lines = ["# X4 — attribution on validation incidents (development evidence)", ""]
    lines += [
        f"Protocol: {rep['protocol']}. Tolerance {rep['tolerance_s']:.0f} s; "
        f"bootstrap B = {rep['resamples']}, seed {rep['seed']}.",
        "",
    ]
    if "site_a" in rep:
        a = rep["site_a"]
        lines += [
            "## Site A (15 events)",
            "",
            "| Mode | Run | Matched | Abstained | Top-1 (95% CI) | Top-3 (95% CI) | "
            "Top-1 answered | Tactic answered | FP incidents | Raw grounding | "
            "Distinct answers (modal share) |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for mode, m in a["modes"].items():
            t1, t3 = m["top1"], m["top3"]
            lines.append(
                f"| {mode} | {rep['runs']['site_a/' + mode]} | {m['matched']}/{m['events']} | "
                f"{m['abstained']} | {_pct(t1['estimate'])} [{_pct(t1['low'])}, {_pct(t1['high'])}]"
                f" | {_pct(t3['estimate'])} [{_pct(t3['low'])}, {_pct(t3['high'])}] | "
                f"{_pct(m['top1_answered'])} | {_pct(m['tactic_accuracy_answered'])} | "
                f"{m['false_positive_incidents']}/{m['incidents']} | "
                f"{_pct(m['llm'].get('raw_grounding_rate'))} | {_answers(m['answers'])} |"
            )
        bl = a["baselines"]
        mc = bl["majority_class"]
        lines += [
            "",
            f"Majority-class baseline ({mc['technique']}): "
            f"{_pct(mc['top1']['estimate'])} [{_pct(mc['top1']['low'])}, "
            f"{_pct(mc['top1']['high'])}]. Constant answers (all 50 events / these 15): "
            + "; ".join(
                f"{t} {_pct(v['all_50'])} / {_pct(v['these_events'])}"
                for t, v in bl["constant_answer"].items()
            )
            + ".",
        ]
        if "full_vs_retrieval_only_top3" in a:
            d = a["full_vs_retrieval_only_top3"]
            lines += [
                "",
                f"`full` - `retrieval_only`, top-3 (paired event bootstrap): "
                f"{d['difference']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}], "
                f"p = {d['p_two_sided']:.3f}.",
            ]
    if "site_b" in rep:
        b = rep["site_b"]
        modes = list(b["modes"])
        lines += [
            "",
            "## Site B (case study, no accuracy)",
            "",
            "| Episode | Acceptable | " + " | ".join(modes) + " |",
            "|---|---|" + "---|" * len(modes),
        ]
        for i, ep in enumerate(b["episodes"]):
            cells = []
            for mode in modes:
                r = b["modes"][mode]["per_episode"][i]
                cells.append("no incident" if not r["matched"] else (r["technique"] or "abstained"))
            acc = ", ".join(b["modes"][modes[0]]["per_episode"][i]["acceptable"]) if modes else ""
            lines.append(f"| {ep['episode']} | {acc} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "| Mode | Incidents | LLM calls | Schema valid | Raw grounding | "
            "Distinct answers over incidents (modal share) |",
            "|---|---|---|---|---|---|",
        ]
        for mode, m in b["modes"].items():
            llm = m["llm"]
            lines.append(
                f"| {mode} | {m['incidents']} | {llm.get('calls', 0)} | "
                f"{_pct(llm.get('schema_valid_rate'))} | {_pct(llm.get('raw_grounding_rate'))} | "
                f"{_answers(m['answers'])} |"
            )
    c = rep["checks"]
    lines += ["", "## Checks", "", "passed" if c["passed"] else "FAILED"]
    lines += [f"- {p}" for p in c["problems"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="X4 attribution scoring (ADR 0017 §4)")
    parser.add_argument("--runs", nargs="*", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path("results/mesh"))
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/X4"))
    args = parser.parse_args(argv)
    params = load_params()
    runs = [_run_of(p) for p in args.runs] if args.runs else select_runs(args.root)
    rep, problems = run(params, runs, args.partitions or params.paths.partitions_dir)
    path = write_artifact(
        args.out / "attribution_val.json",
        kind="X4.attribution",
        data_source="runtime",
        payload=rep,
    )
    (args.out / "attribution_val.md").write_text(markdown(rep), encoding="utf-8")
    print(markdown(rep))
    print(path)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
