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


def study_payload(success: bool) -> dict[str, object]:
    return {
        **PAYLOAD,
        "task_completed": success,
        "interpretation_without_help": success,
        "uncertainty_understood": success,
        "usefulness": 5 if success else 2,
        "clarity": 5 if success else 2,
    }


def test_version_comparison_waits_for_both_cohorts(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    for _ in range(10):
        store.record(study_payload(False), "baseline")
    for _ in range(5):
        store.record(study_payload(True), "candidate")
    report = aggregate_study(
        store.path,
        baseline_version="baseline",
        candidate_version="candidate",
        minimum_per_version=10,
        bootstrap_iterations=200,
    )
    comparison = report["impact_comparison"]
    assert comparison["publishable"] is False
    assert comparison["demonstrated_improvement"] is False
    assert report["impact_publishable"] is False


def test_version_comparison_demonstrates_preregistered_improvement(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    for _ in range(12):
        store.record(study_payload(False), "baseline")
        store.record(study_payload(True), "candidate")
    report = aggregate_study(
        store.path,
        baseline_version="baseline",
        candidate_version="candidate",
        minimum_per_version=10,
        primary_metric="task_completed",
        bootstrap_iterations=200,
        seed=17,
    )
    comparison = report["impact_comparison"]
    primary = comparison["candidate_minus_baseline_95_ci"]["task_completed"]
    assert comparison["unit"] == "consented_response"
    assert comparison["publishable"] is True
    assert primary == {
        "candidate_minus_baseline": 1.0,
        "lower_95": 1.0,
        "upper_95": 1.0,
    }
    assert report["demonstrated_improvement"] is True
    assert report["contains_row_level_or_personal_data"] is False


def test_same_outcomes_do_not_claim_improvement(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    for _ in range(10):
        store.record(study_payload(True), "baseline")
        store.record(study_payload(True), "candidate")
    report = aggregate_study(
        store.path,
        baseline_version="baseline",
        candidate_version="candidate",
        bootstrap_iterations=200,
    )
    assert report["impact_publishable"] is True
    assert report["demonstrated_improvement"] is False
