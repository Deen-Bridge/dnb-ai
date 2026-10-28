"""Deterministic Stellar USDC amount and fee arithmetic without floating-point drift (#524)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation

STELLAR_SCALE = Decimal("0.0000001")
STROOPS_PER_UNIT = 10_000_000
DEFAULT_BASE_FEE_STROOPS = 100

TRUSTED_USDC_ISSUERS: dict[str, str] = {
    "mainnet": "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
    "testnet": "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5",
}


@dataclass(frozen=True)
class DeterministicPaymentBreakdown:
    network: str
    asset_code: str
    asset_issuer: str
    amount_decimal: str
    amount_stroops: int
    fee_stroops: int
    fee_xlm_decimal: str
    live_exchange_rate_used: bool = False


def calculate_stellar_usdc_breakdown(
    amount_input: str | int | Decimal | float,
    network: str = "mainnet",
    asset_code: str = "USDC",
    operation_count: int = 1,
    base_fee_stroops: int = DEFAULT_BASE_FEE_STROOPS,
) -> DeterministicPaymentBreakdown:
    """Compute exact minor units (stroops) and XLM network fee using Decimal math only."""
    if isinstance(amount_input, float):
        raise ValueError("Floating-point inputs are forbidden; pass a string or Decimal.")
    net = network.strip().lower()
    if net not in TRUSTED_USDC_ISSUERS:
        raise ValueError(f"Unsupported Stellar network: {network}")
    if asset_code != "USDC":
        raise ValueError(f"Unsupported asset: {asset_code}")
    if operation_count < 1 or base_fee_stroops < DEFAULT_BASE_FEE_STROOPS:
        raise ValueError("Invalid operation count or base fee.")

    try:
        dec = Decimal(str(amount_input))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal amount: {amount_input}") from exc

    if dec <= Decimal("0"):
        raise ValueError("Amount must be strictly positive.")
    if dec > Decimal("1000000000"):
        raise ValueError("Amount exceeds maximum per-transaction safety bound.")

    quantized = dec.quantize(STELLAR_SCALE, rounding=ROUND_HALF_EVEN)
    amount_stroops = int((quantized * STROOPS_PER_UNIT).to_integral_exact())
    total_fee_stroops = operation_count * base_fee_stroops
    fee_xlm = (Decimal(total_fee_stroops) / Decimal(STROOPS_PER_UNIT)).quantize(STELLAR_SCALE)

    return DeterministicPaymentBreakdown(
        network=net,
        asset_code="USDC",
        asset_issuer=TRUSTED_USDC_ISSUERS[net],
        amount_decimal=format(quantized, "f"),
        amount_stroops=amount_stroops,
        fee_stroops=total_fee_stroops,
        fee_xlm_decimal=format(fee_xlm, "f"),
        live_exchange_rate_used=False,
    )
