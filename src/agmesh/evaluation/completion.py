"""The completion tier, CC1-CC8 (ADR 0019 §6; Plan Week 14 validation).

    python -m agmesh.evaluation.checker --tier completion

Each criterion gets a list of problems; it passes when the list is empty.

* **CC1, CC5:** the contract tests (``tests/contract``) pass.
* **CC2:**
  * every test artifact has its access entries, per client, for its own run and
    commit;
  * every final run's validation comparison is recorded;
  * the entries of final runs carry the tagged commit;
  * out-of-band entries are listed.
* **CC3:** every test result artifact passes the result checks (runtime data,
  provenance, score files).
* **CC4:** the privacy rule on every private test artifact.
* **CC6:** the frozen matrix, the baselines, X3, X4 and PT6-v2 are complete on test.
* **CC7:** ``results/reproduce/reproduction.json`` passed.
* **CC8:** ``results/X7/resources.json`` passed.

Seeds whose validation comparison is not accepted are reported under CC2 as notes,
not failures (ADR 0019 §4).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import load_params
from agmesh.evaluation.sealed import TEST_ACCESS_LOG, frozen_sha, read_log

NON_FINAL = {"ground_truth_site_a", "ground_truth_review_site_a"}


def _doc(path: Path) -> dict[str, Any] | None:
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None


def contract_tests() -> list[str]:
    code = subprocess.call(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/contract"]
    )
    return [] if code == 0 else [f"tests/contract failed (exit {code})"]


def baseline_ids() -> list[str]:
    from agmesh.experiments.x1_baselines import SITE_MODELS

    out = []
    for site, kinds in SITE_MODELS.items():
        for kind in kinds:
            modes = ["local"] if kind in ("vae_flat", "rule_new_peer") else ["local", "central"]
            out += [f"{site}__{m}__{kind}" for m in modes]
    return out


def completion(
    results: Path = Path("results"),
    access_log: Path = TEST_ACCESS_LOG,
    matrix: Path = Path("configs/experiments/x1_x2.yaml"),
    *,
    run_contract_tests: bool = True,
) -> dict[str, Any]:
    from agmesh.evaluation.checker import check
    from agmesh.experiments.grid import completeness, privacy_violations

    test = results / "test"
    params = load_params()
    seeds = list(params.project.seeds)
    accesses = read_log(access_log)
    cc: dict[str, list[str]] = {f"CC{i}": [] for i in range(1, 9)}
    notes: dict[str, list[str]] = {"CC2": []}

    if run_contract_tests:
        cc["CC1"] = cc["CC5"] = contract_tests()

    # CC3 and the per-artifact part of CC2: the result checks on results/test
    findings = check(test, access_log, ["X1", "X2"]) if test.is_dir() else []
    for f in findings:
        for p in f.problems:
            (cc["CC2"] if "access" in p or "validation comparison" in p else cc["CC3"]).append(
                f"{f.path}: {p}"
            )
        doc = _doc(Path(f.path)) or {}
        cmp = (doc.get("validation") or {}).get("comparison") or {}
        if cmp and not cmp.get("accepted"):
            notes["CC2"].append(f"{f.path}: validation not within tolerance ({cmp})")

    # CC2: the other test runs, and the tagged commit on every final entry
    tagged = frozen_sha()
    for a in accesses:
        cid = str(a.get("config_id"))
        if a.get("out_of_band"):
            notes["CC2"].append(f"out-of-band access (disclosed): {cid} {a.get('client')}")
            continue
        if cid in NON_FINAL:
            continue
        if tagged and a.get("frozen_sha") != tagged:
            cc["CC2"].append(f"access {cid} seed {a.get('seed')} {a.get('client')}: "
                             f"frozen_sha {a.get('frozen_sha')} is not {tagged}")  # fmt: skip

    def logged(cid: str, seed: int, clients: list[str]) -> list[str]:
        have = {
            a.get("client") for a in accesses if a.get("config_id") == cid and a.get("seed") == seed
        }
        return [c for c in clients if c not in have]

    site_b = list(params.data.partition.site_b.clients)
    site_a = list(params.data.partition.site_a.clients)
    x3 = test / "X3" / "fusion_test.json"
    if not x3.is_file():
        cc["CC6"].append(f"missing {x3}")
    else:
        for seed in (_doc(x3) or {}).get("seeds", []):
            if miss := logged("X3/site_b__fusion", seed, site_b):
                cc["CC2"].append(f"X3 seed {seed}: no access entry for {miss}")
    x4 = test / "X4" / "attribution_test.json"
    if not x4.is_file():
        cc["CC6"].append(f"missing {x4}")
    elif miss := logged("X4/attribution", 0, site_a + site_b):
        cc["CC2"].append(f"X4: no access entry for {miss}")
    pt6 = test / "retrieval" / "test__pt6_v2.json"
    if not pt6.is_file():
        cc["CC6"].append(f"missing {pt6}")
    for site in ("site_a", "site_b"):
        if not any(a.get("config_id") == f"edge__{site}" for a in accesses):
            cc["CC6"].append(f"no test edge run logged for {site}")

    # CC4 and CC6 on the matrix and the baselines
    if test.is_dir():
        cc["CC4"] += privacy_violations(test, ["X1", "X2"])
    cc["CC6"] += [f"missing {m}" for m in completeness(test, params.paths.partitions_dir,
                                                       ["X1", "X2"], matrix)]  # fmt: skip
    for cid in baseline_ids():
        for seed in seeds:
            if not (test / "X1" / cid / f"seed_{seed}.json").is_file():
                cc["CC6"].append(f"missing X1/{cid}/seed_{seed}")

    # CC7 and CC8; a criterion the author waived (ADR 0019 §2a) is reported as waived
    waived: dict[str, Any] = {}
    waiver = _doc(results / "reproduce" / "waiver.json")
    for key, path in (("CC7", results / "reproduce" / "reproduction.json"),
                      ("CC8", results / "X7" / "resources.json")):  # fmt: skip
        art = _doc(path)
        if art is None and key == "CC7" and waiver and waiver.get("criterion") == "CC7":
            waived[key] = waiver
        elif art is None:
            cc[key].append(f"missing {path}")
        elif not art.get("passed"):
            cc[key].append(f"{path}: not passed")

    status = {k: not v for k, v in cc.items() if k not in waived}
    return {
        "protocol": "docs/adr/0019-g4-final-test-protocol-week14.md §6",
        "frozen_sha": tagged,
        "passed": all(status.values()),
        "passed_note": "with waived criteria listed under 'waived'" if waived else None,
        "waived": waived,
        "status": status,
        "problems": cc,
        "notes": notes,
        "contract_tests_run": run_contract_tests,
    }


def main_completion(results: Path, access_log: Path, matrix: Path) -> int:
    rep = completion(results, access_log, matrix)
    path = write_artifact(results / "test" / "completion_check.json", kind="completion",
                          data_source="runtime", payload=rep)  # fmt: skip
    for k, w in rep["waived"].items():
        print(f"{k}: WAIVED ({w.get('decided_by')}, {w.get('date')}: {w.get('reason')})")
    for k, ok in rep["status"].items():
        probs = rep["problems"][k]
        print(f"{k}: {'pass' if ok else f'FAIL ({len(probs)})'}")
        for p in probs[:15]:
            print(f"  - {p}")
        if len(probs) > 15:
            print(f"  ... and {len(probs) - 15} more")
    for p in rep["notes"]["CC2"][:30]:
        print(f"  note: {p}")
    tail = f" (waived: {sorted(rep['waived'])})" if rep["waived"] else ""
    print(f"completion tier {'PASSED' if rep['passed'] else 'FAILED'}{tail} -> {path}")
    return 0 if rep["passed"] else 1
