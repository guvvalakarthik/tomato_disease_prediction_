from __future__ import annotations

from pathlib import Path

from api.feedback import FeedbackStore
from scripts.export_user_study import aggregate_study


PAYLOAD = {
    "participant_role": "agriculture_student",
    "task_completed": True,
    "interpretation_without_help": True,
    "uncertainty_understood": True,
    "expert_confirmation_intended": True,
    "usefulness": 4,
    "clarity": 5,
    "issue_tags": ["attention_map_unclear"],
}


def test_study_export_waits_for_ten_participants(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    for _ in range(9):
        store.record(PAYLOAD, "1.0.0")
    report = aggregate_study(store.path)
    assert report["status"] == "insufficient_participants"
    assert report["publishable"] is False


def test_study_export_contains_aggregates_only(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    for _ in range(10):
        store.record(PAYLOAD, "1.0.0")
    report = aggregate_study(store.path)
    assert report["status"] == "complete"
    assert report["participant_count"] == 10
    assert report["unassisted_interpretation_rate"] == 1.0
    assert report["mean_clarity"] == 5.0
    assert report["issue_tag_counts"] == {"attention_map_unclear": 10}
    assert report["contains_row_level_or_personal_data"] is False
    assert "feedback_id" not in report
