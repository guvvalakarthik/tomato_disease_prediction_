from __future__ import annotations

from pathlib import Path

import pytest

from api.config import Settings
from api.feedback import PostgresFeedbackStore, create_feedback_store


class FakeConnection:
    def __init__(self, statements: list[tuple[str, object]]):
        self.statements = statements

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement: str, values=None):
        self.statements.append((statement, values))


def payload() -> dict[str, object]:
    return {
        "participant_role": "farmer",
        "task_completed": True,
        "interpretation_without_help": True,
        "uncertainty_understood": True,
        "expert_confirmation_intended": True,
        "usefulness": 4,
        "clarity": 5,
        "issue_tags": ["result_unclear", "result_unclear"],
    }


def test_postgres_store_initialises_and_records_parameterised_feedback() -> None:
    statements: list[tuple[str, object]] = []
    store = PostgresFeedbackStore(
        "postgresql://unused", lambda: FakeConnection(statements)
    )
    feedback_id = store.record(payload(), "1.0.0")
    assert store.durable is True
    assert feedback_id
    assert "CREATE TABLE IF NOT EXISTS feedback" in statements[0][0]
    insert, values = statements[1]
    assert "VALUES (%s, %s, %s" in insert
    assert values[2] == "1.0.0"
    assert values[-1] == '["result_unclear"]'


def test_required_durable_storage_fails_without_database_url(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Durable feedback is required"):
        create_feedback_store(None, tmp_path / "feedback.sqlite3", True)


def test_settings_read_durable_feedback_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "TOMATOGUARD_FEEDBACK_DATABASE_URL", "postgresql://db.example/tomatoguard"
    )
    monkeypatch.setenv("TOMATOGUARD_REQUIRE_DURABLE_FEEDBACK", "true")
    settings = Settings.from_env()
    assert settings.feedback_database_url == "postgresql://db.example/tomatoguard"
    assert settings.require_durable_feedback is True
