"""ADR 0019 §3: a final access needs the tagged code and a tree clean outside results/."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agmesh.evaluation import sealed


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "results").mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.org")
    _git(repo, "config", "user.name", "t")
    (repo / "src" / "a.py").write_text("x = 1\n")
    (repo / "results" / "test_access.log").write_text("")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    return repo


def test_no_tag_is_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert "needs the Git tag" in (sealed.frozen_problem(cwd=repo) or "")


def test_the_tagged_commit_and_results_only_changes_are_allowed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "tag", sealed.FROZEN_TAG)
    assert sealed.frozen_problem(cwd=repo) is None
    (repo / "results" / "test_access.log").write_text('{"a": 1}\n')  # an append: allowed
    assert sealed.frozen_problem(cwd=repo) is None
    _git(repo, "commit", "-qam", "results")
    assert sealed.frozen_problem(cwd=repo) is None


def test_code_changes_after_the_tag_are_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "tag", sealed.FROZEN_TAG)
    (repo / "src" / "a.py").write_text("x = 2\n")
    assert "uncommitted changes" in (sealed.frozen_problem(cwd=repo) or "")
    _git(repo, "commit", "-qam", "code")
    assert "outside results/" in (sealed.frozen_problem(cwd=repo) or "")


def test_a_head_that_does_not_descend_from_the_tag_is_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "other")
    (repo / "src" / "a.py").write_text("x = 3\n")
    _git(repo, "commit", "-qam", "other")
    _git(repo, "tag", sealed.FROZEN_TAG)
    _git(repo, "checkout", "-q", "-")
    assert "does not descend" in (sealed.frozen_problem(cwd=repo) or "")
