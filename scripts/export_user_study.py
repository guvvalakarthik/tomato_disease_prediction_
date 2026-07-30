from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def aggregate_study(database: Path, minimum_participants: int = 10) -> dict[str, object]:
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM feedback ORDER BY created_at").fetchall()
    count = len(rows)
    if count < minimum_participants:
        return {
            "status": "insufficient_participants",
            "participant_count": count,
            "minimum_participants": minimum_participants,
            "publishable": False,
        }

    def rate(column: str) -> float:
        return sum(int(row[column]) for row in rows) / count

    roles = Counter(row["participant_role"] for row in rows)
    models = Counter(row["model_version"] for row in rows)
    issues: Counter[str] = Counter()
    for row in rows:
        issues.update(json.loads(row["issue_tags"]))
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export anonymized aggregate user-study results")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-participants", type=int, default=10)
    args = parser.parse_args()
    report = aggregate_study(args.database, args.minimum_participants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
