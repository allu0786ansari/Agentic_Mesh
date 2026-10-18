"""X1 and X2 analysis, exactly as fixed in ADR 0016 §4.

    python -m agmesh.evaluation.stats --experiment X1     # make stats
    python -m agmesh.evaluation.stats --experiment X2

**Methods.**

* **Primary:** the event-level two-stage block bootstrap (ADR 0005). It is paired
  on one resampling plan per site, with seeds averaged inside each draw.
* **Secondary:** a seed-level paired bootstrap and the exact Wilcoxon test.

**Families.** Holm correction is applied within each:

* **RQ1:** PT1 and PT2 on both sites (X1);
* **H1b:** FedProx against FedAvg in the homogeneous and heterogeneous arms, both
  sites (X2);
* **H1c:** the DP pooled threshold against two comparators (X1).

Everything outside these families is reported, with no adjustment.

**How a claim is tested.** Each claim has a one-sided p-value.

* **Non-inferiority (PT1):** H1: A - B > -margin.
* **"At or above" (PT2):** H1: A - B ≥ 0, so a draw counts against it only when it
  is strictly below zero.
* **Improvement (H1b, H1c):** H1: A - B > 0.

**Unpaired comparisons.** IID partitions have their own normal-validation windows
(ADR 0016 §3), so an IID configuration cannot share the time-scheme plan. IID
against non-IID is compared on independent draws and labelled ``unpaired``.

**What it reads.** Everything comes from result artifacts and their score files.
Validation results are development evidence; the claims are made on the test split
after G4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.evaluation import inference as inf
from agmesh.evaluation.detection import NodeArrays, load_node_arrays
from agmesh.evaluation.results import scores_path
from agmesh.evaluation.stats import paired_bootstrap, wilcoxon_signed_rank

Records = dict[int, dict[str, NodeArrays]]
MODEL = {"site_a": "vae", "site_b": "ae"}


@dataclass(frozen=True)
class Test:
    family: str | None  # None: reported, no adjustment
    label: str
    a: tuple[str, str]  # (experiment, config); a config may carry "#h1c_<name>"
    b: tuple[str, str]
    metric: str
    claim: str  # "noninferior", "at_or_above", "greater", "less"
    margin: float = 0.0
    unpaired: bool = False


def x1_tests() -> list[Test]:
    out: list[Test] = []
    for site in ("site_a", "site_b"):
        head = ("X1", f"{site}__fedprox__eps3")
        m = MODEL[site]
        out += [
            Test(
                "RQ1",
                f"PT1 {site}",
                head,
                ("X1", f"{site}__central__{m}"),
                "auroc",
                "noninferior",
                0.05,
            ),
            Test("RQ1", f"PT2 {site}", head, ("X1", f"{site}__local__{m}"), "auroc", "at_or_above"),
            # secondary: ε = 10 and FedAvg against the same references
            Test(
                None,
                f"PT1 {site}, eps10 (secondary)",
                ("X1", f"{site}__fedprox__eps10"),
                ("X1", f"{site}__central__{m}"),
                "auroc",
                "noninferior",
                0.05,
            ),
            Test(
                None,
                f"PT2 {site}, FedAvg eps3 (secondary)",
                ("X1", f"{site}__fedavg__eps3"),
                ("X1", f"{site}__local__{m}"),
                "auroc",
                "at_or_above",
            ),
            # cost of DP and of federation, through the same pipeline
            Test(
                None,
                f"cost of DP, central, {site}",
                ("X1", f"{site}__central__{m}__epsinf"),
                ("X1", f"{site}__central__{m}__eps3"),
                "auroc",
                "greater",
            ),
            Test(
                None,
                f"cost of federation at eps3, {site}",
                ("X1", f"{site}__central__{m}__eps3"),
                head,
                "auroc",
                "greater",
            ),
            Test(
                None,
                f"cost of federation at eps inf, {site}",
                ("X1", f"{site}__central__{m}__epsinf"),
                ("X1", f"{site}__fedprox__epsinf"),
                "auroc",
                "greater",
            ),
            Test(
                None,
                f"federation vs local-only at eps3, {site}",
                head,
                ("X1", f"{site}__local__{m}__eps3"),
                "auroc",
                "greater",
            ),
        ]
    h1c = "site_a__fedprox__eps10__h1c"
    for comp, name in (("r0_per_node", "per-node R0"), ("r3", "per-node R3")):
        b = ("X1", f"{h1c}#h1c_{comp}") if comp != "r3" else ("X1", h1c)
        out.append(
            Test(
                "H1c",
                f"H1c FPR: pooled vs {name}",
                ("X1", f"{h1c}#h1c_pooled"),
                b,
                "window_fpr",
                "less",
            )
        )
        out.append(
            Test(
                None,
                f"H1c recall: pooled vs {name} (within 0.05)",
                ("X1", f"{h1c}#h1c_pooled"),
                b,
                "event_recall",
                "noninferior",
                0.05,
            )
        )
    return out


def x2_tests() -> list[Test]:
    out: list[Test] = []
    for site in ("site_a", "site_b"):
        out.append(
            Test(
                "H1b",
                f"FedProx vs FedAvg, homogeneous, {site}",
                ("X1", f"{site}__fedprox__eps3"),
                ("X1", f"{site}__fedavg__eps3"),
                "auroc",
                "greater",
            )
        )
        out.append(
            Test(
                "H1b",
                f"FedProx vs FedAvg, heterogeneous, {site}",
                ("X2", f"{site}__fedprox__eps3__het__drop0.2"),
                ("X2", f"{site}__fedavg__eps3__het__drop0.2"),
                "auroc",
                "greater",
            )
        )
        for strat in ("fedavg", "fedprox"):
            out.append(
                Test(
                    None,
                    f"heterogeneous vs homogeneous, {strat}, {site}",
                    ("X2", f"{site}__{strat}__eps3__het__drop0.2"),
                    ("X1", f"{site}__{strat}__eps3"),
                    "auroc",
                    "greater",
                )
            )
    for strat in ("fedavg", "fedprox"):
        out.append(
            Test(
                None,
                f"IID vs non-IID, {strat}, site_a",
                ("X2", f"site_a__{strat}__eps3__iid"),
                ("X1", f"site_a__{strat}__eps3"),
                "auroc",
                "greater",
                unpaired=True,
            )
        )
    for mu in ("0.001", "0.1"):
        out.append(
            Test(
                None,
                f"mu {mu} vs 0.01, homogeneous, site_a",
                ("X2", f"site_a__fedprox__eps3__mu{mu}"),
                ("X1", "site_a__fedprox__eps3"),
                "auroc",
                "greater",
            )
        )
        out.append(
            Test(
                None,
                f"mu {mu} vs 0.01, heterogeneous, site_a",
                ("X2", f"site_a__fedprox__eps3__het__drop0.2__mu{mu}"),
                ("X2", "site_a__fedprox__eps3__het__drop0.2"),
                "auroc",
                "greater",
            )
        )
    return out


# ------------------------------------------------------------------------ loading
def _docs(results: Path, exp: str, cid: str, split: str) -> dict[int, dict[str, Any]]:
    out = {}
    for path in sorted((results / exp / cid).glob("seed_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("split") == split:
            out[int(doc["seed"])] = doc | {"_path": path.as_posix()}
    return out


def _npz(path: Path) -> dict[str, NodeArrays] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return load_node_arrays(data)


def load_records(results: Path, key: tuple[str, str], split: str) -> tuple[Records, dict[int, Any]]:
    exp, name = key
    cid, _, variant = name.partition("#")
    docs = _docs(results, exp, cid, split)
    recs: Records = {}
    for seed, doc in docs.items():
        path = Path(doc["_path"])
        if variant:
            which = variant.removeprefix("h1c_")
            f = (doc.get("h1c") or {}).get("scores_files", {}).get(which)
            arrays = _npz(path.with_name(f["path"])) if f else None
        else:
            arrays = _npz(scores_path(path))
        if arrays is not None:
            recs[seed] = arrays
    return recs, docs


def _seed_metric(doc: dict[str, Any], variant: str, metric: str) -> float | None:
    if not variant:
        v = doc["metrics"].get(metric)
    elif variant == "h1c_pooled":
        v = (doc.get("h1c") or {}).get("macro", {}).get(metric)
    else:
        v = (doc.get("h1c") or {}).get("r0_per_node_thresholds", {}).get("macro", {}).get(metric)
    return None if v is None or not np.isfinite(v) else float(v)


# ------------------------------------------------------------------------ testing
def _p(diff: np.ndarray, claim: str, margin: float) -> float:
    if claim == "noninferior":  # H1: diff > -margin
        bad = np.sum(diff <= -margin)
    elif claim == "at_or_above":  # H1: diff >= 0
        bad = np.sum(diff < 0)
    elif claim == "greater":  # H1: diff > 0
        bad = np.sum(diff <= 0)
    elif claim == "less":  # H1: diff < 0
        bad = np.sum(diff >= 0)
    else:
        raise ValueError(claim)
    return float((1 + bad) / (diff.size + 1))


class Analysis:
    def __init__(self, results: Path, split: str = "val", resamples: int = 10_000) -> None:
        self.results, self.split, self.b = results, split, resamples
        self.plans: dict[str, inf.EventBootstrap] = {}
        self.cache: dict[tuple[str, str], tuple[dict[str, Any], dict[str, float]] | None] = {}
        self.docs: dict[tuple[str, str], dict[int, Any]] = {}

    def _plan_key(self, key: tuple[str, str], site: str) -> str:
        return f"{site}__iid" if "__iid" in key[1] else site

    def draws(self, key: tuple[str, str]) -> tuple[dict[str, Any], dict[str, float]] | None:
        if key in self.cache:
            return self.cache[key]
        recs, docs = load_records(self.results, key, self.split)
        self.docs[key] = docs
        if not recs or len(recs) != len(docs):
            self.cache[key] = None
            return None
        site = str(next(iter(docs.values()))["site"])
        pk = self._plan_key(key, site)
        if pk not in self.plans:
            self.plans[pk] = inf.EventBootstrap(
                next(iter(recs.values())), block_s=inf.BLOCK_S[site], b=self.b, seed=0
            )
        plan = self.plans[pk]
        self.cache[key] = (plan.draws(recs), plan.observed(recs))
        return self.cache[key]

    def ci(self, key: tuple[str, str], metric: str) -> dict[str, Any] | None:
        d = self.draws(key)
        if d is None:
            return None
        values = d[0][metric]
        finite = values[np.isfinite(values)]
        return {
            "observed": d[1][metric],
            "low": float(np.quantile(finite, 0.025)) if finite.size else None,
            "high": float(np.quantile(finite, 0.975)) if finite.size else None,
            "seeds": sorted(self.docs.get(key, {})),
        }

    def test(self, t: Test) -> dict[str, Any]:
        da, db = self.draws(t.a), self.draws(t.b)
        out: dict[str, Any] = {
            "family": t.family,
            "label": t.label,
            "a": "/".join(t.a),
            "b": "/".join(t.b),
            "metric": t.metric,
            "claim": t.claim,
            "margin": t.margin,
            "pairing": "unpaired" if t.unpaired else "paired",
        }
        if da is None or db is None:
            return out | {"missing": [k for k, d in ((t.a, da), (t.b, db)) if d is None]}
        diff = da[0][t.metric] - db[0][t.metric]
        diff = diff[np.isfinite(diff)]
        observed = da[1][t.metric] - db[1][t.metric]
        out |= {
            "observed_a": da[1][t.metric],
            "observed_b": db[1][t.metric],
            "observed_difference": float(observed),
            "ci": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]
            if diff.size
            else [None, None],
            "p_one_sided": _p(diff, t.claim, t.margin) if diff.size else None,
            "resamples": int(diff.size),
        }
        # secondary: seeds
        va, vb = self.docs[t.a], self.docs[t.b]
        seeds = sorted(set(va) & set(vb))
        xa = [_seed_metric(va[s], t.a[1].partition("#")[2], t.metric) for s in seeds]
        xb = [_seed_metric(vb[s], t.b[1].partition("#")[2], t.metric) for s in seeds]
        pairs = [(x, y) for x, y in zip(xa, xb, strict=True) if x is not None and y is not None]
        if len(pairs) >= 2:
            pa, pb = np.array(pairs).T
            boot = paired_bootstrap(pa, pb, seed=0)
            out["seed_level"] = {
                "seeds": len(pairs),
                "mean_difference": boot.mean_difference,
                "ci": [boot.ci.low, boot.ci.high],
                "p_two_sided_bootstrap": boot.p_value,
                "p_wilcoxon": wilcoxon_signed_rank(pa, pb).p_value,
            }
        return out


def curve(experiment: str) -> set[str]:
    """Every configuration of the privacy-utility curve (ADR 0016 §4), for its CIs."""
    if experiment != "X1":
        return set()
    return {
        f"X1/{site}__{strat}__eps{e}"
        for site in ("site_a", "site_b")
        for strat in ("fedavg", "fedprox")
        for e in ("1", "3", "10", "inf")
    }


def _key(c: str) -> tuple[str, str]:
    exp, cid = c.split("/", 1)
    return exp, cid


def run(
    results: Path, experiment: str, split: str = "val", resamples: int = 10_000
) -> dict[str, Any]:
    tests = x1_tests() if experiment == "X1" else x2_tests()
    an = Analysis(results, split, resamples)
    rows = [an.test(t) for t in tests]
    for fam in sorted({r["family"] for r in rows if r["family"]}):
        members = [r for r in rows if r["family"] == fam]
        adjusted = inf.holm([r.get("p_one_sided") or float("nan") for r in members])
        for r, p in zip(members, adjusted, strict=True):
            r["p_holm"] = p
    configs = sorted({c for t in tests for c in ("/".join(t.a), "/".join(t.b))} | curve(experiment))
    metrics = ("auroc", "auprc", "event_recall", "window_fpr")
    cis = {c: {m: an.ci(_key(c), m) for m in metrics} for c in configs}
    return {
        "experiment": experiment,
        "split": split,
        "protocol": "docs/adr/0016-x1-x2-matrix-week12.md §4",
        "method": "two-stage stratified block bootstrap (ADR 0005), paired per site",
        "resamples": resamples,
        "status": "development evidence (validation)" if split == "val" else "confirmatory (test)",
        "tests": rows,
        "configs": cis,
    }


def _f(v: Any, digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def markdown(result: dict[str, Any]) -> str:
    lines = [
        f"### {result['experiment']} ({result['status']})",
        "",
        "| Family | Test | Metric | A | B | A - B [95% CI] | p (one-sided) | p Holm |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["tests"]:
        if "missing" in r:
            lines.append(
                f"| {r['family'] or '—'} | {r['label']} | {r['metric']} | missing | | | | |"
            )
            continue
        ci = r["ci"]
        tag = " (unpaired)" if r["pairing"] == "unpaired" else ""
        lines.append(
            f"| {r['family'] or '—'} | {r['label']}{tag} "
            f"| {r['metric']} | {_f(r['observed_a'])} | {_f(r['observed_b'])} "
            f"| {r['observed_difference']:+.3f} [{_f(ci[0])}, {_f(ci[1])}] "
            f"| {_f(r['p_one_sided'], 4)} | {_f(r.get('p_holm'), 4) if r['family'] else '—'} |"
        )
    return "\n".join(lines)
