"""Tests for automated backup verification and restore drills (#883).

Validates:
- SQLite online backup creation, sha256 checksums, and least-privilege permissions (0600)
- Restore drill into disposable directory with schema compatibility verification
- Detection of corrupted backup files triggering integrity failure alerts
- Detection of incompatible database schemas (missing columns or indexes)
- Credential and secret redaction in connection URLs
- Full drill execution and RTO/RPO target measurement
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from scripts.backup_stores import backup_sqlite, calculate_sha256, redact_url
from scripts.restore_drill import (
    RestoreDrillError,
    execute_restore_drill,
    verify_sqlite_restore,
)

SAMPLE_FEEDBACK_SCHEMA = """
CREATE TABLE feedback (
    feedback_id       TEXT NOT NULL,
    chat_id           TEXT NOT NULL,
    message_id        TEXT NOT NULL,
    rating            TEXT NOT NULL,
    categories        TEXT NOT NULL DEFAULT '[]',
    comment           TEXT,
    prompt            TEXT,
    answer            TEXT,
    model_name        TEXT,
    generation_config TEXT,
    created_at        TEXT NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);
CREATE INDEX idx_feedback_rating    ON feedback(rating);
CREATE INDEX idx_feedback_created   ON feedback(created_at);
CREATE INDEX idx_feedback_model     ON feedback(model_name);
"""


@pytest.fixture
def sample_sqlite_db(tmp_path: Path) -> Path:
    """Create a populated, schema-compliant sample SQLite feedback database."""
    db_path = tmp_path / "test_feedback.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SAMPLE_FEEDBACK_SCHEMA)

    # Insert sample records
    records = [
        (
            "fb-1",
            "chat-101",
            "msg-1",
            "up",
            json.dumps(["accurate_source"]),
            "Great answer on wudu",
            "How to perform wudu?",
            "Start with intention...",
            "gemini-2.5-flash",
            json.dumps({"temperature": 0.2}),
            str(time.time()),
        ),
        (
            "fb-2",
            "chat-102",
            "msg-2",
            "down",
            json.dumps(["missing_citation"]),
            "Needs Bukhari reference",
            "Hadith about intentions",
            "Actions are judged by intentions...",
            "gemini-2.5-flash",
            None,
            str(time.time()),
        ),
    ]

    conn.executemany(
        """
        INSERT INTO feedback
            (feedback_id, chat_id, message_id, rating, categories,
             comment, prompt, answer, model_name, generation_config, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        records,
    )
    conn.commit()
    conn.close()
    return db_path


def test_sqlite_backup_creation_and_integrity(sample_sqlite_db: Path, tmp_path: Path) -> None:
    """Verify that backup_sqlite produces an ACID copy with 0600 permissions and valid checksum."""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    report = backup_sqlite(sample_sqlite_db, backup_dir)

    assert report["status"] == "success"
    assert report["store"] == "sqlite_feedback"
    assert report["integrity"] == "ok"
    assert report["row_count"] == 2
    assert report["size_bytes"] > 0
    assert report["age_seconds"] >= 0

    backup_file = Path(report["backup_path"])
    assert backup_file.exists()

    # Permissions must be least-privilege (0600)
    mode = backup_file.stat().st_mode & 0o777
    assert mode == 0o600

    # Verify sha256 checksum
    computed_sha256 = calculate_sha256(backup_file)
    assert report["sha256"] == computed_sha256


def test_restore_drill_verifies_schema_and_records(sample_sqlite_db: Path, tmp_path: Path) -> None:
    """Verify that the restore drill restores into disposable storage and validates schema/records."""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    backup_report = backup_sqlite(sample_sqlite_db, backup_dir)
    backup_path = Path(backup_report["backup_path"])

    with tempfile.TemporaryDirectory() as disposable_tmp:
        disposable_dir = Path(disposable_tmp)
        result = verify_sqlite_restore(backup_path, disposable_dir)

        assert result["status"] == "passed"
        assert result["schema_verified"] is True
        assert result["records_sampled"] == 2
        assert result["total_records"] == 2
        assert result["rto_target_met"] is True
        assert result["rpo_target_met"] is True
        assert result["rto_seconds"] >= 0.0


def test_corrupted_backup_fails_integrity_check(tmp_path: Path) -> None:
    """Verify that a corrupted backup file is caught and raises an actionable error."""
    corrupted_file = tmp_path / "sqlite_feedback_corrupt.db"
    # Write invalid / header-corrupted sqlite file
    corrupted_file.write_bytes(b"SQLite format 3\x00" + b"\xff" * 4096)
    corrupted_file.chmod(0o600)

    with tempfile.TemporaryDirectory() as disposable_tmp:
        disposable_dir = Path(disposable_tmp)
        with pytest.raises(RestoreDrillError) as exc_info:
            verify_sqlite_restore(corrupted_file, disposable_dir)

        assert (
            "integrity check failed" in str(exc_info.value).lower()
            or "database disk image is malformed" in str(exc_info.value).lower()
        )


def test_incompatible_schema_missing_column_fails_drill(tmp_path: Path) -> None:
    """Verify that a database missing a required column fails the schema compatibility check."""
    bad_db = tmp_path / "bad_schema.db"
    conn = sqlite3.connect(str(bad_db))
    # Create feedback table missing generation_config
    conn.executescript("""
    CREATE TABLE feedback (
        feedback_id TEXT NOT NULL,
        chat_id TEXT NOT NULL,
        message_id TEXT NOT NULL,
        rating TEXT NOT NULL,
        categories TEXT NOT NULL DEFAULT '[]',
        comment TEXT,
        prompt TEXT,
        answer TEXT,
        model_name TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (chat_id, message_id)
    );
    CREATE INDEX idx_feedback_rating ON feedback(rating);
    CREATE INDEX idx_feedback_created ON feedback(created_at);
    CREATE INDEX idx_feedback_model ON feedback(model_name);
    """)
    conn.commit()
    conn.close()

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    backup_report = backup_sqlite(bad_db, backup_dir)
    backup_path = Path(backup_report["backup_path"])

    with tempfile.TemporaryDirectory() as disposable_tmp:
        disposable_dir = Path(disposable_tmp)
        with pytest.raises(RestoreDrillError) as exc_info:
            verify_sqlite_restore(backup_path, disposable_dir)

        assert "missing required columns" in str(exc_info.value)
        assert "generation_config" in str(exc_info.value)


def test_incompatible_schema_missing_index_fails_drill(tmp_path: Path) -> None:
    """Verify that a database missing a required index fails the schema compatibility check."""
    bad_db = tmp_path / "bad_index.db"
    conn = sqlite3.connect(str(bad_db))
    # Full table but missing indexes
    conn.executescript("""
    CREATE TABLE feedback (
        feedback_id TEXT NOT NULL,
        chat_id TEXT NOT NULL,
        message_id TEXT NOT NULL,
        rating TEXT NOT NULL,
        categories TEXT NOT NULL DEFAULT '[]',
        comment TEXT,
        prompt TEXT,
        answer TEXT,
        model_name TEXT,
        generation_config TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (chat_id, message_id)
    );
    """)
    conn.commit()
    conn.close()

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    backup_report = backup_sqlite(bad_db, backup_dir)
    backup_path = Path(backup_report["backup_path"])

    with tempfile.TemporaryDirectory() as disposable_tmp:
        disposable_dir = Path(disposable_tmp)
        with pytest.raises(RestoreDrillError) as exc_info:
            verify_sqlite_restore(backup_path, disposable_dir)

        assert "missing required indexes" in str(exc_info.value)


def test_redact_url_hides_secrets() -> None:
    """Verify that credentials in connection strings are strictly redacted."""
    secret_url = "redis://admin:SUPER_SECRET_PASS_12345@prod-cluster.internal:6379/0"
    clean = redact_url(secret_url)
    assert "SUPER_SECRET_PASS_12345" not in clean
    assert clean == "redis://admin:***@prod-cluster.internal:6379/0"

    assert redact_url("") == ""
    assert redact_url("redis://localhost:6379/0") == "redis://localhost:6379/0"


def test_full_drill_execution_flow(sample_sqlite_db: Path, tmp_path: Path) -> None:
    """Verify end-to-end execute_restore_drill with fresh backup generation."""
    backup_dir = tmp_path / "drill_backups"

    report = execute_restore_drill(
        backup_dir=backup_dir,
        run_backup_first=True,
        sqlite_path=str(sample_sqlite_db),
    )

    assert report["status"] == "passed"
    assert report["total_duration_seconds"] > 0.0
    assert report["failures"] == []
    assert len(report["stores_verified"]) >= 1

    sqlite_res = next(s for s in report["stores_verified"] if s["store"] == "sqlite_feedback")
    assert sqlite_res["status"] == "passed"
    assert sqlite_res["integrity_check"] == "ok"
