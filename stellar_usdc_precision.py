"""Deterministic Stellar USDC and stroop precision boundary calculator (#582)."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

STROOPS_PER_UNIT = Decimal("10000000")  # 10^7 (7 decimal places on Stellar)
MIN_STROOP_AMOUNT = Decimal("0.0000001")
MAX_STELLAR_AMOUNT = Decimal("922337203685.4775807")


def parse_stellar_decimal_amount(raw: str) -> Decimal:
    """Parse and validate a Stellar amount string with strict 7-decimal stroop precision."""
    cleaned = raw.strip()
    if not cleaned or "e" in cleaned.lower():
        raise ValueError("Scientific notation and empty amounts are forbidden")
    try:
        val = Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal amount: {raw}") from exc

    if val < Decimal("0"):
        raise ValueError("Negative Stellar amounts are forbidden")
    if val > MAX_STELLAR_AMOUNT:
        raise ValueError("Amount exceeds Stellar int64 stroop maximum")

    exponent = val.as_tuple().exponent
    if not isinstance(exponent, int) or exponent < -7:
        raise ValueError("Stellar amounts support at most 7 decimal places (1 stroop)")
    return val.quantize(MIN_STROOP_AMOUNT)


def to_stroops(amount: str) -> int:
    """Convert a validated Stellar decimal string to exact integer stroops."""
    dec = parse_stellar_decimal_amount(amount)
    return int(dec * STROOPS_PER_UNIT)


def calculate_net_payout_after_fee(*, gross_amount: str, fee_stroops: int) -> Decimal:
    """Compute exact Decimal net amount after deducting integer stroop fee."""
    if fee_stroops < 0:
        raise ValueError("fee_stroops cannot be negative")
    gross_stroops = to_stroops(gross_amount)
    if fee_stroops > gross_stroops:
        raise ValueError("Fee exceeds gross amount")
    net_stroops = gross_stroops - fee_stroops
    return (Decimal(net_stroops) / STROOPS_PER_UNIT).quantize(MIN_STROOP_AMOUNT)
