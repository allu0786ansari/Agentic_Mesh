"""The test-split summary: every preregistered target in one place (ADR 0020).

    python -m agmesh.experiments.final_summary     # -> results/test/summary_test.{json,md}

It reads only committed test artifacts:

* ``X1/analysis_test.json`` and ``X2/analysis_test.json``;
* the headline score files;
* ``X3/fusion_test.json`` and ``X4/attribution_test.json``;
* ``retrieval/test__pt6_v2.json`` and ``X5/latency_test.json``.

No partition is read. The verdicts follow the preregistration (§3, §3a, §3b), with
the rules of ADRs 0016-0019.

It also recomputes each final run's validation comparison with the corrected
privacy check (ADR 0020 §2), from the artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.common.artifacts import write_artifact
from agmesh.evaluation.detection import load_node_arrays
from agmesh.evaluation.results import compare_validation

TEST = Path("results/test")


def _j(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _test(analysis: dict[str, Any], label: str) -> dict[str, Any]:
    for t in analysis["tests"]:
        if t["label"] == label:
            return dict(t)
    raise KeyError(label)


def _verdict(t: dict[str, Any], alpha: float = 0.05) -> bool:
    p = t.get("p_holm")
    return bool((p if p is not None else t["p_one_sided"]) < alpha)


def pt3(root: Path = TEST) -> dict[str, Any]:
    from agmesh.experiments.x1x2_analysis import Analysis

    cid = "site_a__fedprox__eps3"
    an = Analysis(root, "test", 10_000)
    pooled = []
    for f in sorted((root / "X1" / cid).glob("seed_*.scores.npz")):
        arrays = load_node_arrays(np.load(f))
        det = sum(int(np.asarray(a.event_detected, bool).sum()) for a in arrays.values())
        tot = sum(int(np.asarray(a.event_detected).size) for a in arrays.values())
        pooled.append({"seed": int(f.name.split("_")[1].split(".")[0]), "detected": det,
                       "events": tot})  # fmt: skip
    recall = an.ci(("X1", cid), "event_recall")
    fpr = an.ci(("X1", cid), "window_fpr")
    assert recall is not None and fpr is not None
    pooled_mean = float(np.mean([p["detected"] / p["events"] for p in pooled]))
    return {
        "config": f"X1/{cid}",
        "event_recall_node_mean": recall,
        "event_recall_pooled_35": {"per_seed": pooled, "mean": pooled_mean},
        "window_fpr": fpr,
        "recall_target": 0.70,
        "fpr_target": 0.02,
        "met": bool(recall["observed"] >= 0.70 and pooled_mean >= 0.70 and fpr["observed"] <= 0.02),
    }


def validation_reproduction(root: Path = TEST) -> dict[str, Any]:
    rows = []
    for f in sorted(root.glob("X[12]/*/seed_*.json")):
        d = _j(f)
        v = d.get("validation") or {}
        if not v:
            continue
        cmp = compare_validation(
            Path("results"), d["experiment"], d["config_id"], d["seed"], v["metrics"],
            d.get("privacy"),
        )  # fmt: skip
        rows.append({"artifact": f.as_posix(), **cmp})
    bad = [r for r in rows if not r["accepted"]]
    return {
        "runs": len(rows),
        "accepted": len(rows) - len(bad),
        "exact": sum(1 for r in rows if r["exact"]),
        "not_accepted": [
            {"artifact": r["artifact"], "auroc_abs_difference": r["auroc_abs_difference"]}
            for r in bad
        ],
        "max_auroc_abs_difference": max(r["auroc_abs_difference"] or 0.0 for r in rows),
    }


def summarise(root: Path = TEST) -> dict[str, Any]:
    x1 = _j(root / "X1" / "analysis_test.json")
    x2 = _j(root / "X2" / "analysis_test.json")
    x3 = _j(root / "X3" / "fusion_test.json")
    x4 = _j(root / "X4" / "attribution_test.json")
    pt6 = _j(root / "retrieval" / "test__pt6_v2.json")
    x5 = _j(root / "X5" / "latency_test.json")
    targets: dict[str, Any] = {}
    for site in ("site_a", "site_b"):
        for pt in ("PT1", "PT2"):
            t = _test(x1, f"{pt} {site}")
            targets[f"{pt} {site}"] = {
                "difference": t["observed_difference"], "ci": t["ci"],
                "p_holm": t["p_holm"], "met": _verdict(t),
            }  # fmt: skip
    targets["PT3 site_a"] = pt3(root)
    p4 = x3["pt4"]["reduction_2of3"]
    targets["PT4 site_b"] = {**p4, "met": x3["pt4"]["pt4_met"]}
    targets["PT5 site_a"] = x4["pt5"]
    v2 = pt6["versions"]["v2"]["all"]
    missed = len(pt6["missed_incidents"])
    targets["PT6-v2"] = {
        "recall@10": v2["recall@10"], "queries": v2["queries"], "missed_incidents": missed,
        "recall@10_missed_as_failures": v2["recall@10"] * v2["queries"] / (v2["queries"] + missed),
        "recall@5": v2["recall@5"], "mrr": v2["mrr"], "v1_recall@10":
        pt6["versions"]["v1"]["all"]["recall@10"], "met": pt6["pt6_v2_met"],
    }  # fmt: skip
    targets["PT7"] = {
        "segments": {f"{s}/{seg}": {"p95_ms": v["p95_ms"], "target_ms": v["target_p95_ms"],
                                    "met": v["meets_target"]}
                     for s, row in x5["sites"].items() for seg, v in row.items()},
    }  # fmt: skip
    targets["PT7"]["met"] = all(v["met"] for v in targets["PT7"]["segments"].values())
    hypotheses = {
        "H1b": [
            {"test": t["label"], "difference": t["observed_difference"], "p_holm": t["p_holm"]}
            for t in x2["tests"] if t.get("family") == "H1b"
        ],
        "H1c": [
            {"test": t["label"], "difference": t["observed_difference"], "ci": t["ci"],
             "p_one_sided": t["p_one_sided"], "p_holm": t.get("p_holm")}
            for t in x1["tests"] if "H1c" in t["label"]
        ],
        "H2b": x3["h2b"],
    }  # fmt: skip
    return {
        "protocol": "docs/adr/0020-g4-test-results.md",
        "split": "test",
        "targets": targets,
        "hypotheses": hypotheses,
        "validation_reproduction": validation_reproduction(root),
    }


def _f(x: Any, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def markdown(rep: dict[str, Any]) -> str:
    t = rep["targets"]
    yes = {True: "met", False: "**not met**"}
    lines = ["# Test-split summary (after g4-frozen)", "", "| Target | Test result | Verdict |",
             "|---|---|---|"]  # fmt: skip
    for k in ("PT1 site_a", "PT2 site_a", "PT1 site_b", "PT2 site_b"):
        v = t[k]
        lines.append(f"| {k} | {v['difference']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], "
                     f"Holm p {_f(v['p_holm'], 4)} | {yes[v['met']]} |")  # fmt: skip
    p3 = t["PT3 site_a"]
    r = p3["event_recall_node_mean"]
    lines.append(f"| PT3 site_a | recall {r['observed']:.3f} [{r['low']:.3f}, {r['high']:.3f}] "
                 f"(pooled over 35: {p3['event_recall_pooled_35']['mean']:.3f}); window FPR "
                 f"{p3['window_fpr']['observed']:.4f} | {yes[p3['met']]} |")  # fmt: skip
    p4 = t["PT4 site_b"]
    p4p = p4["p_one_sided_H0_le_0.30"]
    lines.append(f"| PT4 site_b | reduction {p4['estimate']:.3f} [{p4['low']:.3f}, "
                 f"{p4['high']:.3f}], p {p4p:.4f} | {yes[p4['met']]} |")  # fmt: skip
    p5 = t["PT5 site_a"]
    lines.append(f"| PT5 site_a (`full`) | top-3 {p5['top3']['estimate']:.3f} "
                 f"[{p5['top3']['low']:.3f}, {p5['top3']['high']:.3f}]; majority on these events "
                 f"{p5['majority_baseline_these_events']:.3f} | {yes[p5['met']]} |")  # fmt: skip
    p6 = t["PT6-v2"]
    lines.append(f"| PT6-v2 | Recall@10 {p6['recall@10']:.3f} on {p6['queries']} queries "
                 f"({p6['missed_incidents']} incidents without a query; "
                 f"{p6['recall@10_missed_as_failures']:.3f} if counted as misses) | "
                 f"{yes[p6['met']]} |")  # fmt: skip
    lines.append(f"| PT7 | every segment within its p95 target | {yes[t['PT7']['met']]} |")
    vr = rep["validation_reproduction"]
    lines += ["", f"Validation re-check inside the final runs: {vr['accepted']}/{vr['runs']} "
              f"within the Plan's tolerance ({vr['exact']} exact); largest |dAUROC| "
              f"{vr['max_auroc_abs_difference']:.4f}.", ""]  # fmt: skip
    for b in vr["not_accepted"]:
        lines.append(f"- not within tolerance: {b['artifact']} (|dAUROC| "
                     f"{b['auroc_abs_difference']:.4f})")  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test-split summary (ADR 0020)")
    parser.add_argument("--root", type=Path, default=TEST)
    args = parser.parse_args(argv)
    rep = summarise(args.root)
    write_artifact(args.root / "summary_test.json", kind="summary.test", data_source="runtime",
                   payload=rep)  # fmt: skip
    text = markdown(rep)
    (args.root / "summary_test.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
