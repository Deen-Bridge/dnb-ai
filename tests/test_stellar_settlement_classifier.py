"""Tests for distinguishing payment submission from ledger confirmation (#559)."""

from __future__ import annotations

from stellar_settlement_classifier import classify_payment_settlement


def test_callback_or_hash_alone_never_proves_settlement() -> None:
    pending = classify_payment_settlement(
        network="public",
        has_frontend_callback=True,
        submitted_tx_hash="b" * 64,
        ledger_lookup_status="pending",
    )
    assert pending.settlement_state == "pending"
    assert pending.is_settled is False

    unknown = classify_payment_settlement(
        network="public",
        has_frontend_callback=True,
        submitted_tx_hash="b" * 64,
        ledger_lookup_status="not_found",
    )
    assert unknown.settlement_state == "unknown"
    assert unknown.is_settled is False


def test_confirmed_failed_and_stale_provider_states() -> None:
    confirmed = classify_payment_settlement(
        network="public",
        submitted_tx_hash="b" * 64,
        ledger_lookup_status="found",
        ledger_tx_successful=True,
        ledger_sequence=123456,
    )
    assert confirmed.settlement_state == "confirmed"
    assert confirmed.is_settled is True
    assert confirmed.ledger == 123456

    failed = classify_payment_settlement(
        network="public",
        submitted_tx_hash="b" * 64,
        ledger_lookup_status="found",
        ledger_tx_successful=False,
        ledger_sequence=123456,
    )
    assert failed.settlement_state == "failed"
    assert failed.is_settled is False

    stale = classify_payment_settlement(
        network="public",
        submitted_tx_hash="b" * 64,
        ledger_lookup_status="found",
        ledger_tx_successful=True,
        ledger_sequence=123456,
        provider_age_seconds=180,
        max_provider_age_seconds=60,
    )
    assert stale.settlement_state == "unknown"
    assert stale.is_settled is False
    assert stale.is_provider_stale is True
