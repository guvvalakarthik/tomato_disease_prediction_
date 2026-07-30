from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
import numpy as np


OUTCOME_COLUMNS = (
    "task_completed",
    "interpretation_without_help",
    "uncertainty_understood",
    "expert_confirmation_intended",
    "usefulness",
    "clarity",
)


def load_rows(database: Path | str):
    location = str(database)
    if location.startswith(("postgresql://", "postgres://")):
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(location, row_factory=dict_row) as connection:
            return connection.execute(
                "SELECT * FROM feedback ORDER BY created_at"
            ).fetchall()
    with sqlite3.connect(location) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            "SELECT * FROM feedback ORDER BY created_at"
        ).fetchall()


def bootstrap_version_difference(
    baseline_rows,
    candidate_rows,
    column: str,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    baseline = np.asarray([float(row[column]) for row in baseline_rows])
    candidate = np.asarray([float(row[column]) for row in candidate_rows])
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)
    for index in range(iterations):
        baseline_sample = rng.choice(baseline, size=len(baseline), replace=True)
        candidate_sample = rng.choice(candidate, size=len(candidate), replace=True)
        differences[index] = candidate_sample.mean() - baseline_sample.mean()
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return {
        "candidate_minus_baseline": float(candidate.mean() - baseline.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def compare_versions(
    rows,
    baseline_version: str,
    candidate_version: str,
    minimum_per_version: int,
    primary_metric: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    if primary_metric not in OUTCOME_COLUMNS:
        raise ValueError(f"Unknown primary user-study metric: {primary_metric}")
    baseline = [row for row in rows if row["model_version"] == baseline_version]
    candidate = [row for row in rows if row["model_version"] == candidate_version]
    counts = {baseline_version: len(baseline), candidate_version: len(candidate)}
    if min(counts.values()) < minimum_per_version:
        return {
            "status": "insufficient_version_cohorts",
            "publishable": False,
            "version_response_counts": counts,
            "minimum_responses_per_version": minimum_per_version,
            "demonstrated_improvement": False,
        }
    intervals = {
        column: bootstrap_version_difference(
            baseline, candidate, column, iterations=iterations, seed=seed
        )
        for column in OUTCOME_COLUMNS
    }
    primary = intervals[primary_metric]
    return {
        "status": "complete",
        "publishable": True,
        "unit": "consented_response",
        "baseline_version": baseline_version,
        "candidate_version": candidate_version,
        "version_response_counts": counts,
        "primary_metric": primary_metric,
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "candidate_minus_baseline_95_ci": intervals,
        "demonstrated_improvement": primary["lower_95"] > 0.0,
    }

def aggregate_study(
    database: Path | str,
    minimum_participants: int = 10,
    baseline_version: str | None = None,
    candidate_version: str | None = None,
    minimum_per_version: int = 10,
    primary_metric: str = "task_completed",
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    rows = load_rows(database)
    count = len(rows)
    if count < minimum_participants:
        return {
            "status": "insufficient_participants",
            "participant_count": count,
            "minimum_participants": minimum_participants,
            "publishable": False,
            "unit": "consented_response",
            "demonstrated_improvement": False,
        }

    def rate(column: str) -> float:
        return sum(int(row[column]) for row in rows) / count

    roles = Counter(row["participant_role"] for row in rows)
    models = Counter(row["model_version"] for row in rows)
    issues: Counter[str] = Counter()
    for row in rows:
        issues.update(json.loads(row["issue_tags"]))
    report: dict[str, object] = {
        "status": "complete",
        "publishable": True,
        "participant_count": count,
        "date_range": {"from": rows[0]["created_at"], "to": rows[-1]["created_at"]},
        "participant_roles": dict(sorted(roles.items())),
        "model_versions": dict(sorted(models.items())),
        "task_completion_rate": rate("task_completed"),
        "unassisted_interpretation_rate": rate("interpretation_without_help"),
        "uncertainty_understanding_rate": rate("uncertainty_understood"),
        "expert_confirmation_intent_rate": rate("expert_confirmation_intended"),
        "mean_usefulness": sum(row["usefulness"] for row in rows) / count,
        "mean_clarity": sum(row["clarity"] for row in rows) / count,
        "issue_tag_counts": dict(sorted(issues.items())),
        "contains_row_level_or_personal_data": False,
    }
    report["unit"] = "consented_response"
    if bool(baseline_version) != bool(candidate_version):
        raise ValueError("baseline_version and candidate_version must be supplied together")
    if baseline_version and candidate_version:
        comparison = compare_versions(
            rows,
            baseline_version,
            candidate_version,
            minimum_per_version,
            primary_metric,
            bootstrap_iterations,
            seed,
        )
        report["impact_comparison"] = comparison
        report["impact_publishable"] = comparison["publishable"]
        report["demonstrated_improvement"] = comparison[
            "demonstrated_improvement"
        ]
    else:
        report["impact_publishable"] = False
        report["demonstrated_improvement"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export anonymized aggregate user-study results")
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-participants", type=int, default=10)
    parser.add_argument("--baseline-version")
    parser.add_argument("--candidate-version")
    parser.add_argument("--minimum-per-version", type=int, default=10)
    parser.add_argument("--primary-metric", choices=OUTCOME_COLUMNS, default="task_completed")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = aggregate_study(
        args.database,
        args.minimum_participants,
        args.baseline_version,
        args.candidate_version,
        args.minimum_per_version,
        args.primary_metric,
        args.bootstrap_iterations,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
