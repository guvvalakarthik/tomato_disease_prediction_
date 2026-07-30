from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path


class FeedbackStore:
    durable = False

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    participant_role TEXT NOT NULL,
                    task_completed INTEGER NOT NULL,
                    interpretation_without_help INTEGER NOT NULL,
                    uncertainty_understood INTEGER NOT NULL,
                    expert_confirmation_intended INTEGER NOT NULL,
                    usefulness INTEGER NOT NULL,
                    clarity INTEGER NOT NULL,
                    issue_tags TEXT NOT NULL
                )
                """
            )

    def record(self, payload: dict[str, object], model_version: str) -> str:
        feedback_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        values = (
            feedback_id,
            created_at,
            model_version,
            payload["participant_role"],
            int(bool(payload["task_completed"])),
            int(bool(payload["interpretation_without_help"])),
            int(bool(payload["uncertainty_understood"])),
            int(bool(payload["expert_confirmation_intended"])),
            int(payload["usefulness"]),
            int(payload["clarity"]),
            json.dumps(sorted(set(payload.get("issue_tags", [])))),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values
            )
        return feedback_id

class PostgresFeedbackStore:
    durable = True

    def __init__(self, database_url: str, connect_factory=None) -> None:
        if connect_factory is None:
            import psycopg

            connect_factory = lambda: psycopg.connect(database_url)
        self._connect = connect_factory
        self._initialise()

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id UUID PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    model_version TEXT NOT NULL,
                    participant_role TEXT NOT NULL,
                    task_completed BOOLEAN NOT NULL,
                    interpretation_without_help BOOLEAN NOT NULL,
                    uncertainty_understood BOOLEAN NOT NULL,
                    expert_confirmation_intended BOOLEAN NOT NULL,
                    usefulness SMALLINT NOT NULL CHECK (usefulness BETWEEN 1 AND 5),
                    clarity SMALLINT NOT NULL CHECK (clarity BETWEEN 1 AND 5),
                    issue_tags JSONB NOT NULL
                )
                """
            )

    def record(self, payload: dict[str, object], model_version: str) -> str:
        feedback_id = str(uuid.uuid4())
        values = (
            feedback_id,
            datetime.now(UTC),
            model_version,
            payload["participant_role"],
            bool(payload["task_completed"]),
            bool(payload["interpretation_without_help"]),
            bool(payload["uncertainty_understood"]),
            bool(payload["expert_confirmation_intended"]),
            int(payload["usefulness"]),
            int(payload["clarity"]),
            json.dumps(sorted(set(payload.get("issue_tags", [])))),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback (
                    feedback_id, created_at, model_version, participant_role,
                    task_completed, interpretation_without_help,
                    uncertainty_understood, expert_confirmation_intended,
                    usefulness, clarity, issue_tags
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                values,
            )
        return feedback_id


def create_feedback_store(
    database_url: str | None,
    sqlite_path: Path,
    require_durable: bool = False,
):
    if database_url:
        return PostgresFeedbackStore(database_url)
    if require_durable:
        raise RuntimeError(
            "Durable feedback is required but TOMATOGUARD_FEEDBACK_DATABASE_URL is unset"
        )
    return FeedbackStore(sqlite_path)
