"""Explain Stellar DEX offers and order-book results (#568)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OrderBookOfferSummary:
    """Bounded summary of a single public Stellar DEX offer."""

    selling_asset: str
    buying_asset: str
    amount: str
    price: str


@dataclass(frozen=True)
class StellarOrderBookExplanation:
    """Read-only interpretation of a Stellar order-book snapshot."""

    network: str
    base_asset: str
    counter_asset: str
    observed_at: str
    top_bids: tuple[OrderBookOfferSummary, ...]
    top_asks: tuple[OrderBookOfferSummary, ...]
    execution_disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_dex_orderbook_snapshot(
    *,
    network: str,
    base_asset: str,
    counter_asset: str,
    observed_at: str,
    bids: list[dict[str, Any]],
    asks: list[dict[str, Any]],
    max_depth: int = 5,
) -> StellarOrderBookExplanation:
    """Summarize public DEX offers while distinguishing open offers from settled trades."""
    if max_depth <= 0 or max_depth > 20:
        raise ValueError("max_depth must be between 1 and 20")

    norm_bids = tuple(
        OrderBookOfferSummary(
            selling_asset=counter_asset,
            buying_asset=base_asset,
            amount=str(b.get("amount", "0")),
            price=str(b.get("price", "0")),
        )
        for b in bids[:max_depth]
    )
    norm_asks = tuple(
        OrderBookOfferSummary(
            selling_asset=base_asset,
            buying_asset=counter_asset,
            amount=str(a.get("amount", "0")),
            price=str(a.get("price", "0")),
        )
        for a in asks[:max_depth]
    )

    return StellarOrderBookExplanation(
        network=network.strip().lower(),
        base_asset=base_asset.strip(),
        counter_asset=counter_asset.strip(),
        observed_at=observed_at.strip(),
        top_bids=norm_bids,
        top_asks=norm_asks,
        execution_disclaimer=(
            "Order-book offers represent open Resting limit orders at the observation time, "
            "not completed trades or a guaranteed execution price."
        ),
    )
