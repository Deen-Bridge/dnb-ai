"""Tests for deterministic Stellar USDC amount and fee arithmetic (#524)."""

from __future__ import annotations

import pytest

from stellar_usdc_math import TRUSTED_USDC_ISSUERS, calculate_stellar_usdc_breakdown


def test_deterministic_decimal_conversion_and_network_issuer() -> None:
    res = calculate_stellar_usdc_breakdown("25.125", network="mainnet", operation_count=2)
    assert res.amount_decimal == "25.1250000"
    assert res.amount_stroops == 251_250_000
    assert res.fee_stroops == 200
    assert res.fee_xlm_decimal == "0.0000200"
    assert res.asset_issuer == TRUSTED_USDC_ISSUERS["mainnet"]
    assert res.live_exchange_rate_used is False


def test_rejects_float_zero_negative_unsupported_asset_and_huge_values() -> None:
    with pytest.raises(ValueError):
        calculate_stellar_usdc_breakdown(10.5)
    with pytest.raises(ValueError):
        calculate_stellar_usdc_breakdown("0")
    with pytest.raises(ValueError):
        calculate_stellar_usdc_breakdown("-5.00")
    with pytest.raises(ValueError):
        calculate_stellar_usdc_breakdown("10", asset_code="FAKE")
    with pytest.raises(ValueError):
        calculate_stellar_usdc_breakdown("999999999999")
