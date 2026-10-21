"""Summarize Stellar transaction operations in a stable response schema (#560)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

KNOWN_OPERATION_TYPES: frozenset[str] = frozenset(
    {
        "create_account",
        "payment",
        "path_payment_strict_receive",
        "path_payment_strict_send",
        "change_trust",
    }
)


@dataclass(frozen=True)
class NormalizedOperationSummary:
    """Bounded, provider-agnostic summary of a single Stellar operation."""

    operation_type: str
    is_known_type: bool
    source_account: str | None
    destination_account: str | None
    asset_code: str | None
    asset_issuer: str | None
    amount: str | None
    trust_limit: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_transaction_operations(
    operations: list[dict[str, Any]], *, max_operations: int = 25
) -> list[NormalizedOperationSummary]:
    """Normalize payment, path-payment, account, and trustline operations safely."""
    if max_operations <= 0:
        raise ValueError("max_operations must be positive")
    summaries: list[NormalizedOperationSummary] = []
    for raw in operations[:max_operations]:
        op_type = str(raw.get("type", "unknown")).strip().lower()
        known = op_type in KNOWN_OPERATION_TYPES
        src = str(raw["source_account"]).strip() if raw.get("source_account") else None
        dest = str(raw.get("to") or raw.get("account") or "").strip() or None
        if raw.get("asset_type") == "native" or op_type == "create_account":
            code: str | None = "XLM"
            issuer: str | None = None
        else:
            code = str(raw["asset_code"]).strip() if raw.get("asset_code") else None
            issuer = str(raw["asset_issuer"]).strip() if raw.get("asset_issuer") else None
        amt_raw = raw.get("amount") or raw.get("starting_balance")
        amount = str(amt_raw).strip() if amt_raw is not None else None
        limit = str(raw["limit"]).strip() if raw.get("limit") is not None else None

        summaries.append(
            NormalizedOperationSummary(
                operation_type=op_type,
                is_known_type=known,
                source_account=src,
                destination_account=dest,
                asset_code=code,
                asset_issuer=issuer,
                amount=amount,
                trust_limit=limit,
            )
        )
    return summaries
