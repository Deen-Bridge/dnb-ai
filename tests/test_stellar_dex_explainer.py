"""Tests for Stellar DEX offers and order-book explainer (#568)."""

from __future__ import annotations

from stellar_dex_explainer import explain_dex_orderbook_snapshot


def test_bounds_depth_and_distinguishes_offers_from_completed_trades() -> None:
    snap = explain_dex_orderbook_snapshot(
        network="public",
        base_asset="XLM",
        counter_asset="USDC:G" + "A" * 55,
        observed_at="2026-10-21T08:00:00Z",
        bids=[{"amount": "100", "price": "0.12"} for _ in range(10)],
        asks=[{"amount": "150", "price": "0.13"} for _ in range(10)],
        max_depth=3,
    )
    assert len(snap.top_bids) == 3
    assert len(snap.top_asks) == 3
    assert "not completed trades or a guaranteed execution price" in snap.execution_disclaimer
