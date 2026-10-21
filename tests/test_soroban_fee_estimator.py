"""Tests for Soroban resource fee estimator explanation (#571)."""

from __future__ import annotations

from soroban_fee_estimator import explain_soroban_resource_fee


def test_timestamped_estimate_and_unavailable_fallback() -> None:
    est = explain_soroban_resource_fee(
        network="public",
        rpc_estimate={
            "base_inclusion_fee_stroops": 100,
            "resource_fee_stroops": 44900,
            "observed_at": "2026-10-21T08:10:00Z",
        },
    )
    assert est.status == "estimated"
    assert est.is_estimate is True
    assert est.total_estimated_fee_stroops == 45000
    assert est.observed_at == "2026-10-21T08:10:00Z"

    unavail = explain_soroban_resource_fee(network="testnet", rpc_estimate=None)
    assert unavail.status == "unavailable"
    assert unavail.total_estimated_fee_stroops is None
