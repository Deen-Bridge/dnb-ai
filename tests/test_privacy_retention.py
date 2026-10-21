"""Tests for privacy redaction, retention windows, and provider disclosure (#528)."""

from __future__ import annotations

from datetime import UTC, datetime

from privacy_retention import RetentionPolicy, is_record_expired, redact_log_payload


def test_redacts_tokens_prompts_uploads_and_inline_pii() -> None:
    raw = {
        "request_id": "req-1",
        "prompt": "My personal question",
        "raw_upload": b"audio-bytes",
        "Authorization": "Bearer secret-jwt-token",
        "note": "Contact me at user@example.com or +234 801 234 5678",
    }
    clean = redact_log_payload(raw)
    assert clean["request_id"] == "req-1"
    assert clean["prompt"] == "[REDACTED]"
    assert clean["raw_upload"] == "[REDACTED]"
    assert clean["Authorization"] == "[REDACTED]"
    assert "user@example.com" not in clean["note"]


def test_retention_expiration_and_disclosure_defaults() -> None:
    policy = RetentionPolicy()
    assert policy.routine_logs_retain_prompts is False
    created = datetime(2026, 10, 1, tzinfo=UTC)
    now = datetime(2026, 10, 10, tzinfo=UTC)
    assert is_record_expired(created, policy.trace_retention_days, now=now) is True
