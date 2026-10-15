"""The retrieval query set (Plan Week 9 validation; ADR 0010 §4-5).

* Paraphrases share no 8-gram with any ATT&CK passage of the pinned release.
* Every paraphrase points to a validation incident of the frozen ground truth.
* Ranking and metrics follow the protocol: canonical IDs, no parent match.

The overlap test needs the pinned bundle. It uses ``data/kb`` when the bundle is
there and otherwise fetches it (verified by SHA-256). It is skipped only when the
network is unavailable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from agmesh.evaluation import retrieval_eval as re_eval
from agmesh.evaluation.ground_truth import OUT_DIR
from agmesh.kb import chunking, stix_parser

N = 8


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _grams(text: str, n: int = N) -> set[tuple[str, ...]]:
    w = _tokens(text)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def _paraphrases() -> list[dict[str, str]]:
    items: list[dict[str, str]] = yaml.safe_load(re_eval.PARAPHRASES.read_text(encoding="utf-8"))[
        "paraphrases"
    ]
    return items


@pytest.fixture(scope="module")
def passages(tmp_path_factory: pytest.TempPathFactory) -> list[chunking.Passage]:
    local = stix_parser.DEFAULT_PATH
    try:
        path = stix_parser.fetch(
            local if local.is_file() else tmp_path_factory.mktemp("kb") / local.name
        )
    except OSError as exc:  # offline
        pytest.skip(f"pinned ATT&CK bundle unavailable: {exc}")
    techs = stix_parser.load(path)
    assert len(techs) == sum(
        v["status"] == "active"
        for v in json.loads(
            Path("evaluation/attack/ics_attack_20.0_techniques.json").read_text(encoding="utf-8")
        )["techniques"].values()
    )
    return chunking.technique_passages(techs)


def test_paraphrases_share_no_8gram_with_any_passage(passages: list[chunking.Passage]) -> None:
    kb: set[tuple[str, ...]] = set()
    for p in passages:
        kb |= _grams(p.text)
    overlaps = {item["id"]: sorted(_grams(item["text"]) & kb) for item in _paraphrases()}
    assert {k: v for k, v in overlaps.items() if v} == {}


def test_paraphrases_point_to_validation_incidents() -> None:
    items = _paraphrases()
    assert len(items) == 30 and len({i["id"] for i in items}) == 30
    gt_a = json.loads((OUT_DIR / "mapping_site_a.json").read_text(encoding="utf-8"))["events"]
    valid = {f"site_a/{k}" for k, e in gt_a.items() if e["partition"] == "val_attack"}
    valid |= {f"site_b/{e}" for e in ("0", "1", "2", "3")}
    assert {i["incident"] for i in items} <= valid


def test_rank_and_metrics_follow_the_protocol() -> None:
    assert (
        re_eval.ranks(["T0801", "T0855", "T0836"], ("T1692.001",)) == 2
    )  # revoked ID -> canonical
    assert re_eval.ranks(["T1692", "T0801"], ("T1692.001",)) is None  # parent is not a match
    m = re_eval.metrics([1, 3, None, 6])
    assert m["queries"] == 4 and m["recall@1"] == 0.25 and m["recall@5"] == 0.5
    assert abs(m["mrr"] - (1 + 1 / 3 + 0 + 1 / 6) / 4) < 1e-12
