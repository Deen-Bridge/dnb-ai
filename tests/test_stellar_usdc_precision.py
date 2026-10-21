"""Tests for USDC precision and fee boundary behavior in AI tools (#582)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from stellar_usdc_precision import (
    MAX_STELLAR_AMOUNT,
    MIN_STROOP_AMOUNT,
    calculate_net_payout_after_fee,
    parse_stellar_decimal_amount,
    to_stroops,
)


def test_zero_smallest_and_maximum_stellar_bounds() -> None:
    assert parse_stellar_decimal_amount("0") == Decimal("0.0000000")
    assert parse_stellar_decimal_amount("0.0000001") == MIN_STROOP_AMOUNT
    assert to_stroops("0.0000001") == 1
    assert parse_stellar_decimal_amount(str(MAX_STELLAR_AMOUNT)) == MAX_STELLAR_AMOUNT

    net = calculate_net_payout_after_fee(gross_amount="10.0000000", fee_stroops=100)
    assert net == Decimal("9.9999900")


def test_rejects_excess_decimals_scientific_notation_and_overflow() -> None:
    with pytest.raises(ValueError, match="at most 7 decimal places"):
        parse_stellar_decimal_amount("1.00000001")

    with pytest.raises(ValueError, match="Scientific notation"):
        parse_stellar_decimal_amount("1e-5")

    with pytest.raises(ValueError, match="exceeds Stellar int64"):
        parse_stellar_decimal_amount("922337203685.4775808")
