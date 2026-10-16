"""``.env`` loading: secrets from a git-ignored file, never overriding the shell."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agmesh.common.env import load_env, parse


def test_parse_accepts_the_usual_forms() -> None:
    text = """
    # comment
    HF_TOKEN=hf_abc123
    export MESH_APPROVAL_KEY="placeholder"
    QUOTED='a # not a comment'
    TRAILING=value # a comment
    EMPTY=
    not a line
    """
    assert parse(text) == {
        "HF_TOKEN": "hf_abc123",
        "MESH_APPROVAL_KEY": "placeholder",
        "QUOTED": "a # not a comment",
        "TRAILING": "value",
        "EMPTY": "",
    }


def test_load_env_never_overrides_and_skips_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text("AGMESH_T_NEW=x\nAGMESH_T_SET=from_file\nAGMESH_T_EMPTY=\n")
    monkeypatch.setenv("AGMESH_T_SET", "from_shell")
    monkeypatch.delenv("AGMESH_T_NEW", raising=False)
    monkeypatch.delenv("AGMESH_T_EMPTY", raising=False)
    assert load_env(env) == ["AGMESH_T_NEW"]
    assert os.environ["AGMESH_T_NEW"] == "x" and os.environ["AGMESH_T_SET"] == "from_shell"
    assert "AGMESH_T_EMPTY" not in os.environ
    monkeypatch.delenv("AGMESH_T_NEW")
    assert load_env(tmp_path / "missing") == []


def test_example_is_committed_and_dot_env_is_ignored() -> None:
    example = Path(".env.example")
    keys = parse(example.read_text())
    assert set(keys) == {"HF_TOKEN", "MESH_APPROVAL_KEY"} and not any(keys.values())
    ignore = Path(".gitignore").read_text().splitlines()
    assert ".env" in ignore and "!.env.example" in ignore
