"""Explain Stellar reserve and sponsorship effects for account changes (#566)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class ReserveSponsorshipExplanation:
    """Grounded explanation of Stellar minimum balance reserve vs transaction fees."""

    network: str
    subentries_count: int
    sponsoring_count: int
    sponsored_count: int
    base_reserve_xlm: str | None
    minimum_balance_xlm: str | None
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_reserve_and_sponsorship(
    *,
    network: str,
    subentries_count: int,
    sponsoring_count: int = 0,
    sponsored_count: int = 0,
    configured_base_reserve_xlm: Decimal | None = None,
) -> ReserveSponsorshipExplanation:
    """Compute minimum balance reserve when network base_reserve is provided."""
    if subentries_count < 0 or sponsoring_count < 0 or sponsored_count < 0:
        raise ValueError("Counts cannot be negative")
    effective_entries = 2 + subentries_count + sponsoring_count - sponsored_count
    if effective_entries < 0:
        effective_entries = 0

    if configured_base_reserve_xlm is not None:
        min_bal = configured_base_reserve_xlm * Decimal(effective_entries)
        return ReserveSponsorshipExplanation(
            network=network.strip().lower(),
            subentries_count=subentries_count,
            sponsoring_count=sponsoring_count,
            sponsored_count=sponsored_count,
            base_reserve_xlm=str(configured_base_reserve_xlm),
            minimum_balance_xlm=str(min_bal),
            explanation=(
                f"Minimum balance reserve locks {min_bal} XLM across {effective_entries} "
                "effective reserve entries (2 base + subentries + sponsoring - sponsored). "
                "Reserve is locked balance, distinct from per-operation transaction fees."
            ),
        )

    return ReserveSponsorshipExplanation(
        network=network.strip().lower(),
        subentries_count=subentries_count,
        sponsoring_count=sponsoring_count,
        sponsored_count=sponsored_count,
        base_reserve_xlm=None,
        minimum_balance_xlm=None,
        explanation=(
            f"Account has {effective_entries} effective reserve entries (2 base + subentries "
            "+ sponsoring - sponsored). Minimum reserve is locked XLM and is separate from "
            "transaction fees."
        ),
    )
