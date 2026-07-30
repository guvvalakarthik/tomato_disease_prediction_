from __future__ import annotations

from scripts.generate_results_report import generate_report


def test_missing_evidence_stays_pending_without_fake_zeroes() -> None:
    report = generate_report(None, None, None, None, None, None, {})
    assert "| Model metadata | Pending |" in report
    assert "| Expert field evaluation | Pending |" in report
    assert "0.00%" not in report
    assert "No evidence files supplied" in report


def test_non_comparable_before_after_suppresses_numbers() -> None:
    report = generate_report(
        None,
        None,
        None,
        {"comparable": False, "baseline": 0.84, "candidate": 0.95},
        None,
        None,
        {},
    )
    assert "not evaluated on the same locked samples" in report
    assert "84.00%" not in report
    assert "95.00%" not in report


def test_comparable_results_and_evidence_hashes_are_published() -> None:
    clean = {
        "macro_f1": 0.91,
        "ece_15_bin": 0.04,
        "selective": {"acceptance_coverage": 0.82},
        "ood_test": {"false_acceptance_rate": 0.08},
        "per_class": {
            "tomato_healthy": {"precision": 0.95, "recall": 0.96, "f1": 0.955}
        },
        "classification_bootstrap_95_ci": {
            "macro_f1": {"lower_95": 0.89, "upper_95": 0.93},
            "per_class": {
                "tomato_healthy": {
                    "f1": {"lower_95": 0.92, "upper_95": 0.98}
                }
            },
        },
    }
    comparison = {
        "comparable": True,
        "metric": "macro_f1",
        "baseline": 0.84,
        "candidate": 0.91,
        "evaluation_id": "locked-clean-v1",
    }
    report = generate_report(
        {"version": "1.0.0", "model_sha256": "a" * 64},
        clean,
        None,
        comparison,
        {"p50_ms": 100, "p95_ms": 220, "model_size_mb": 8.2, "device": "CPU"},
        None,
        {"clean_metrics": "b" * 64},
    )
    assert "Macro F1: 91.00%" in report
    assert "Baseline macro_f1: 84.00%" in report
    assert "89.00% to 93.00%" in report
    assert "92.00% to 98.00%" in report
    assert "Candidate macro_f1: 91.00%" in report
    assert "locked-clean-v1" in report
    assert "`clean_metrics`: `" + "b" * 64 + "`" in report


def test_unpublishable_user_study_remains_pending() -> None:
    report = generate_report(
        None,
        None,
        None,
        None,
        None,
        {"publishable": False, "participant_count": 4},
        {},
    )
    assert "| Aggregate user study | Pending |" in report
    assert "Participants: 4" not in report


def test_user_impact_comparison_is_rendered_only_when_publishable() -> None:
    user_study = {
        "publishable": True,
        "impact_publishable": True,
        "participant_count": 20,
        "task_completion_rate": 0.8,
        "unassisted_interpretation_rate": 0.75,
        "uncertainty_understanding_rate": 0.9,
        "impact_comparison": {
            "publishable": True,
            "unit": "consented_response",
            "baseline_version": "0.1.0-legacy",
            "candidate_version": "1.0.0",
            "primary_metric": "task_completed",
            "candidate_minus_baseline_95_ci": {
                "task_completed": {
                    "candidate_minus_baseline": 0.2,
                    "lower_95": 0.05,
                    "upper_95": 0.35,
                }
            },
            "demonstrated_improvement": True,
        },
    }
    report = generate_report(None, None, None, None, None, user_study, {})
    assert "User-study version impact" in report
    assert "Consented responses: 20" in report
    assert "Bootstrap 95% CI: +0.0500 to +0.3500" in report
    assert "Demonstrated improvement: True" in report
