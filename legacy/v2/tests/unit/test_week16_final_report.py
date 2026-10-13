from __future__ import annotations

import json
from pathlib import Path

from evaluation.check_success_criteria import check
from evaluation.final_report import build_final_report


def test_final_report_consolidates_week14_and_week15(tmp_path: Path) -> None:
    report_path = tmp_path / "final_report_data.json"
    report = build_final_report(output_path=report_path)
    criteria = check(report)

    assert report_path.exists()
    assert report["weeks"]["week14"]["detection"]["detection_rate"] >= 0.9
    assert report["weeks"]["week15"]["retrieval"]["mean_context_recall"] >= 0.85
    assert any(item["status"] == "NOT_MEASURED" for item in criteria)
    assert json.loads(report_path.read_text(encoding="utf-8"))["project"]
