"""Tests for Stellar source protocol version and freshness tracking (#552)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from stellar_source_freshness import StellarSourceFreshnessTracker


def test_unchanged_updated_stale_and_missing_freshness_states() -> None:
    tracker = StellarSourceFreshnessTracker(stale_threshold_days=14)
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)

    tracker.record_chunk(
        chunk_id="soroban-fees#1",
        source_url="https://developers.stellar.org/docs/soroban/fees",
        content="Protocol 21 fee calculation overview.",
        protocol_version="protocol-21",
        checked_at=t0,
    )

    unchanged = tracker.evaluate_freshness("soroban-fees#1", now=t0 + timedelta(days=5))
    assert unchanged.freshness_state == "unchanged"
    assert not unchanged.is_stale
    assert unchanged.protocol_version == "protocol-21"

    stale = tracker.evaluate_freshness("soroban-fees#1", now=t0 + timedelta(days=20))
    assert stale.freshness_state == "stale"
    assert stale.is_stale

    tracker.record_chunk(
        chunk_id="soroban-fees#1",
        source_url="https://developers.stellar.org/docs/soroban/fees",
        content="Protocol 22 updated fee calculation overview.",
        protocol_version="protocol-22",
        checked_at=t0 + timedelta(days=21),
    )
    updated = tracker.evaluate_freshness("soroban-fees#1", now=t0 + timedelta(days=22))
    assert updated.freshness_state == "updated"
    assert not updated.is_stale
    assert updated.protocol_version == "protocol-22"

    missing = tracker.evaluate_freshness("unknown-chunk", now=t0)
    assert missing.freshness_state == "missing"
    assert missing.is_stale
