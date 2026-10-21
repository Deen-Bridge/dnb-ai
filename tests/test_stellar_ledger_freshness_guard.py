"""Tests for preventing stale ledger data from being presented as current (#577)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from stellar_ledger_freshness_guard import evaluate_cached_stellar_result


def test_separate_windows_for_account_tx_and_static_facts() -> None:
    t0 = datetime(2026, 10, 21, 8, 0, 0, tzinfo=UTC)

    fresh_acct = evaluate_cached_stellar_result(
        category="account_state",
        network="public",
        observed_at=t0,
        now=t0 + timedelta(seconds=10),
        ledger_sequence=500,
        data={"balance": "100"},
    )
    assert fresh_acct.is_current is True
    assert fresh_acct.freshness_label == "current"

    stale_acct = evaluate_cached_stellar_result(
        category="account_state",
        network="public",
        observed_at=t0,
        now=t0 + timedelta(seconds=25),
        ledger_sequence=500,
        data={"balance": "100"},
    )
    assert stale_acct.is_current is False
    assert stale_acct.freshness_label == "stale_cached_snapshot"

    static_fact = evaluate_cached_stellar_result(
        category="static_protocol_fact",
        network="public",
        observed_at=t0,
        now=t0 + timedelta(hours=2),
        ledger_sequence=None,
        data={"stroops_per_xlm": 10000000},
    )
    assert static_fact.is_current is True
