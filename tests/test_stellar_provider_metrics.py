"""Tests for privacy-safe Stellar provider metrics (#578)."""

from __future__ import annotations

import pytest

from stellar_provider_metrics import StellarProviderMetricsCollector


def test_records_only_allowed_aggregate_dimensions_and_rejects_identifiers() -> None:
    collector = StellarProviderMetricsCollector()
    ev = collector.record_call(
        provider="horizon",
        operation_category="account_summary",
        duration_ms=42.5,
        outcome="success",
        retry_count=0,
    )
    assert set(ev.to_dict().keys()) == {
        "provider",
        "operation_category",
        "duration_ms",
        "outcome",
        "retry_count",
    }
    assert collector.summary()["total_calls"] == 1

    with pytest.raises(ValueError, match="Disallowed operation_category"):
        collector.record_call(
            provider="horizon",
            operation_category="G" + "A" * 55,
            duration_ms=10.0,
            outcome="success",
        )
