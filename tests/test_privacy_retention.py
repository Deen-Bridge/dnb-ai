"""Tests for privacy redaction, retention windows, and provider disclosure (#528, #586)."""

from __future__ import annotations

from datetime import UTC, datetime

from privacy_retention import (
    RetentionPolicy,
    generate_correlation_token,
    is_record_expired,
    redact_log_payload,
    sanitize_diagnostic_trace,
)


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
    assert policy.trace_retention_days == 7
    assert policy.upload_retention_hours == 24
    assert "restricted to authorized site reliability engineering" in policy.diagnostic_trace_access_boundary.lower()
    created = datetime(2026, 10, 1, tzinfo=UTC)
    now = datetime(2026, 10, 10, tzinfo=UTC)
    assert is_record_expired(created, policy.trace_retention_days, now=now) is True


def test_redact_stellar_identifiers_and_memos_in_diagnostic_logs() -> None:
    account_id = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
    tx_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    secret_key = "SBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
    memo_text = "Private donation from Ahmed"

    diagnostic_payload = {
        "request_id": "req-stellar-42",
        "account_id": account_id,
        "tx_hash": tx_hash,
        "memo": memo_text,
        "memo_text": "Zakat for local masjid",
        "nested_details": {
            "source_account": account_id,
            "secret_key": secret_key,
            "message": f"Funded account {account_id} with tx {tx_hash} including secret {secret_key}",
        },
    }

    sanitized = redact_log_payload(diagnostic_payload)

    # 1. Raw account_id must not appear
    assert account_id not in str(sanitized)
    assert sanitized["account_id"].startswith("corr_")

    # 2. Raw tx_hash must not appear
    assert tx_hash not in str(sanitized)
    assert sanitized["tx_hash"].startswith("corr_")

    # 3. Secret key must be completely redacted
    assert secret_key not in str(sanitized)
    assert sanitized["nested_details"]["secret_key"] == "[REDACTED]"
    assert "[REDACTED_SECRET_KEY]" in sanitized["nested_details"]["message"]

    # 4. Memos must be redacted
    assert sanitized["memo"] == "[REDACTED]"
    assert sanitized["memo_text"] == "[REDACTED]"
    assert memo_text not in str(sanitized)


def test_keyed_correlation_tokens_are_deterministic_and_one_way() -> None:
    account1 = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
    account2 = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"

    tok1_a = generate_correlation_token(account1, salt="salt-test-1")
    tok1_b = generate_correlation_token(account1, salt="salt-test-1")
    tok2 = generate_correlation_token(account2, salt="salt-test-1")
    tok1_different_salt = generate_correlation_token(account1, salt="salt-test-2")

    assert tok1_a == tok1_b
    assert tok1_a != tok2
    assert tok1_a != tok1_different_salt
    assert tok1_a.startswith("corr_")
    assert len(tok1_a) == 21  # "corr_" (5) + 16 chars


def test_response_path_retains_identifiers_while_traces_redact_them() -> None:
    account_id = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
    tx_hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"

    # API response intended for user
    client_response = {
        "status": "ok",
        "account_id": account_id,
        "tx_hash": tx_hash,
        "balance": "100.50",
    }
    # In response path, the data is untouched
    assert client_response["account_id"] == account_id
    assert client_response["tx_hash"] == tx_hash

    # In diagnostic trace path, the data is sanitized
    trace_record = sanitize_diagnostic_trace(client_response)
    assert trace_record["status"] == "ok"
    assert trace_record["balance"] == "100.50"
    assert trace_record["account_id"] != account_id
    assert trace_record["account_id"].startswith("corr_")
    assert trace_record["tx_hash"] != tx_hash
    assert trace_record["tx_hash"].startswith("corr_")
