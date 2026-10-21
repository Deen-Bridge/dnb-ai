"""Field-level redaction, retention windows, and deletion policy for AI prompts, uploads, and traces (#528, #586)."""

from __future__ import annotations

import hashlib
import hmac
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
        "secret",
        "secret_key",
        "seed_phrase",
        "private_key",
        "memo",
        "memo_text",
        "user_text",
        "input_text",
        "user_input",
    }
)

ONCHAIN_IDENTIFIER_KEYS: frozenset[str] = frozenset(
    {
        "account_id",
        "public_key",
        "tx_hash",
        "transaction_hash",
        "source_account",
        "destination",
        "address",
    }
)

PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+\b", re.IGNORECASE),
    re.compile(r"\b(?:\+?\d[\d\s-]{7,}\d)\b"),
)

STELLAR_SECRET_PATTERN: re.Pattern[str] = re.compile(r"\bS[A-Z2-7]{55}\b")
STELLAR_PUBLIC_PATTERN: re.Pattern[str] = re.compile(r"\bG[A-Z2-7]{55}\b")
STELLAR_HASH_PATTERN: re.Pattern[str] = re.compile(r"\b[0-9a-fA-F]{64}\b")

DEFAULT_CORRELATION_SALT: str = "dnb-chain-privacy-v1"


def generate_correlation_token(identifier: str, salt: str = DEFAULT_CORRELATION_SALT) -> str:
    """Generate a deterministic, one-way keyed pseudonymous correlation token for diagnostic logging."""
    cleaned = identifier.strip()
    if not cleaned:
        return "[EMPTY]"
    digest = hmac.new(salt.encode("utf-8"), cleaned.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"corr_{digest[:16]}"


@dataclass(frozen=True)
class RetentionPolicy:
    routine_logs_retain_prompts: bool = False
    trace_retention_days: int = 7
    upload_retention_hours: int = 24
    diagnostic_trace_access_boundary: str = (
        "Diagnostic traces and logs are restricted to authorized site reliability engineering (SRE) "
        "and security operations roles. Access requires internal role-based authentication or IP "
        "allowlisting, and all on-chain addresses, transaction hashes, and user memo contents are "
        "redacted or pseudonymized with keyed correlation tokens prior to recording."
    )
    third_party_provider_disclosure: str = (
        "Sanitized prompts are sent to configured model providers solely for inference; "
        "raw uploads and personal identifiers are redacted from routine telemetry."
    )


def redact_log_payload(
    payload: dict[str, Any],
    *,
    salt: str = DEFAULT_CORRELATION_SALT,
    tokenize_identifiers: bool = True,
) -> dict[str, Any]:
    """Strip sensitive fields, raw on-chain identifiers, memos, and inline PII before logging."""
    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        key_lower = key.lower()
        if key_lower in SENSITIVE_KEYS:
            sanitized[key] = "[REDACTED]"
        elif key_lower in ONCHAIN_IDENTIFIER_KEYS:
            if isinstance(value, str):
                sanitized[key] = generate_correlation_token(value, salt=salt) if tokenize_identifiers else "[REDACTED]"
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [
                    (generate_correlation_token(item, salt=salt) if tokenize_identifiers else "[REDACTED]")
                    if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = "[REDACTED]"
        elif isinstance(value, str):
            text = value
            # 1. Never allow Stellar secret keys in logs
            text = STELLAR_SECRET_PATTERN.sub("[REDACTED_SECRET_KEY]", text)
            # 2. PII patterns
            for pat in PII_PATTERNS:
                text = pat.sub("[REDACTED]", text)
            # 3. Public account addresses
            if tokenize_identifiers:
                text = STELLAR_PUBLIC_PATTERN.sub(lambda m: generate_correlation_token(m.group(0), salt=salt), text)
                text = STELLAR_HASH_PATTERN.sub(lambda m: generate_correlation_token(m.group(0), salt=salt), text)
            else:
                text = STELLAR_PUBLIC_PATTERN.sub("[REDACTED_ADDRESS]", text)
                text = STELLAR_HASH_PATTERN.sub("[REDACTED_HASH]", text)
            sanitized[key] = text
        elif isinstance(value, dict):
            sanitized[key] = redact_log_payload(value, salt=salt, tokenize_identifiers=tokenize_identifiers)
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [
                redact_log_payload(item, salt=salt, tokenize_identifiers=tokenize_identifiers)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def sanitize_diagnostic_trace(trace_payload: dict[str, Any], salt: str = DEFAULT_CORRELATION_SALT) -> dict[str, Any]:
    """Prepare a diagnostic trace record: preserves trace timings/status while redacting raw chain IDs and memos."""
    return redact_log_payload(trace_payload, salt=salt, tokenize_identifiers=True)


def is_record_expired(created_at: datetime, retention_days: int, now: datetime | None = None) -> bool:
    """Determine deterministically whether a stored trace or conversation record must be purged."""
    current = now or datetime.now(UTC)
    return current >= created_at + timedelta(days=retention_days)
