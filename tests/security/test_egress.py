"""The mesh refuses a non-allowlisted LLM endpoint before doing anything (ADR 0014 §5; ADV-5)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agmesh.mesh import llm_calibrate, runner

REPO_PARAMS = Path(__file__).resolve().parents[2] / "configs" / "params.yaml"


def _params(tmp: Path, url: str) -> Path:
    data = yaml.safe_load(REPO_PARAMS.read_text())
    data["llm"]["base_url"] = url
    out = tmp / "params.yaml"
    out.write_text(yaml.safe_dump(data))
    return out


@pytest.mark.parametrize("mode", ["full", "llm_no_retrieval"])
def test_the_runner_refuses_a_third_party_llm(tmp_path: Path, mode: str) -> None:
    params = _params(tmp_path, "https://api.example-llm.com")
    with pytest.raises(SystemExit, match="egress allowlist"):
        runner.main(
            ["--params", str(params), "--site", "site_b", "--mode", mode, "--out", str(tmp_path)]
        )
    assert not any(tmp_path.glob("site_b__*"))  # refused before a run directory exists


def test_the_calibration_refuses_a_third_party_llm(tmp_path: Path) -> None:
    params = _params(tmp_path, "http://10.0.0.7:11434")
    with pytest.raises(SystemExit, match="egress allowlist"):
        llm_calibrate.main(
            ["--params", str(params), "--mode", "full", "--lock", str(tmp_path / "l.json")]
        )


def test_llm_free_modes_do_not_need_the_llm_endpoint(tmp_path: Path) -> None:
    params = _params(tmp_path, "https://api.example-llm.com")
    with pytest.raises(SystemExit) as err:  # fails later, for the missing edge run
        runner.main(
            ["--params", str(params), "--site", "site_b", "--mode", "rule_only",
             "--edge-root", str(tmp_path / "none"), "--out", str(tmp_path)]
        )  # fmt: skip
    assert "egress" not in str(err.value)
