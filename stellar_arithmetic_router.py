"""Route Stellar USDC, fee, and stroop arithmetic to deterministic tools (#583)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

STROOPS_PER_XLM = Decimal("10000000")


@dataclass(frozen=True)
class DeterministicArithmeticResult:
    """Structured result emitted by the deterministic Stellar calculator."""

    operation: str  # "xlm_to_stroops", "stroops_to_xlm", "add_amounts", "fee_total"
    exact_numeric_output: str
    unit: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def execute_deterministic_stellar_math(
    *,
    operation: str,
    values: list[str],
) -> DeterministicArithmeticResult:
    """Execute Stellar arithmetic deterministically using Decimal and never LLM guessing."""
    op = operation.strip().lower()
    if not values:
        raise ValueError("values list cannot be empty")
    decimals = [Decimal(v.strip()) for v in values]
    if any(d < Decimal("0") for d in decimals):
        raise ValueError("Stellar arithmetic operands cannot be negative")

    if op == "xlm_to_stroops":
        stroops = int((decimals[0] * STROOPS_PER_XLM).to_integral_exact())
        out = str(stroops)
        return DeterministicArithmeticResult(
            operation=op,
            exact_numeric_output=out,
            unit="stroops",
            explanation=f"{decimals[0]} XLM equals exactly {out} stroops.",
        )

    if op == "stroops_to_xlm":
        xlm = (decimals[0] / STROOPS_PER_XLM).quantize(Decimal("0.0000001"))
        out = str(xlm)
        return DeterministicArithmeticResult(
            operation=op,
            exact_numeric_output=out,
            unit="XLM",
            explanation=f"{int(decimals[0])} stroops equals exactly {out} XLM.",
        )

    if op in {"add_amounts", "fee_total"}:
        total = sum(decimals, Decimal("0")).quantize(Decimal("0.0000001"))
        out = str(total)
        return DeterministicArithmeticResult(
            operation=op,
            exact_numeric_output=out,
            unit="USDC_OR_XLM",
            explanation=f"Deterministic sum is {out}.",
        )

    raise ValueError(f"Unsupported arithmetic operation: {operation}")
