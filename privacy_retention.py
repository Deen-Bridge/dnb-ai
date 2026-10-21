"""Field-level redaction, retention windows, and deletion policy for AI prompts, uploads, and traces (#528)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "raw_upload",
        "voice_transcript",
        "authorization",
        "api_key",
        "token",
        "email",
        "phone",
        "password",
    }
)

PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+\b", re.IGNORECASE),
    re.compile(r"\b(?:\+?\d[\d\s-]{7,}\d)\b"),
)


@dataclass(frozen=True)
class RetentionPolicy:
    routine_logs_retain_prompts: bool = False
    trace_retention_days: int = 7
    upload_retention_hours: int = 24
    third_party_provider_disclosure: str = (
        "Sanitized prompts are sent to configured model providers solely for inference; "
        "raw uploads and personal identifiers are redacted from routine telemetry."
    )


def redact_log_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip sensitive fields and inline PII before writing routine logs or traces."""
    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in SENSITIVE_KEYS:
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, str):
            text = value
            for pat in PII_PATTERNS:
                text = pat.sub("[REDACTED]", text)
            sanitized[key] = text
        elif isinstance(value, dict):
            sanitized[key] = redact_log_payload(value)
        else:
            sanitized[key] = value
    return sanitized


def is_record_expired(created_at: datetime, retention_days: int, now: datetime | None = None) -> bool:
    """Determine deterministically whether a stored trace or conversation record must be purged."""
    current = now or datetime.now(UTC)
    return current >= created_at + timedelta(days=retention_days)
