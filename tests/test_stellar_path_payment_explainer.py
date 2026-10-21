"""Tests for Stellar path-payment route explainer (#569)."""

from __future__ import annotations

from stellar_path_payment_explainer import explain_path_payment_quote


def test_preserves_route_order_and_marks_as_quote_only() -> None:
    res = explain_path_payment_quote(
        network="public",
        source_asset={"asset_code": "XLM", "asset_issuer": None},
        intermediate_hops=[{"asset_code": "AQUA", "asset_issuer": "G" + "A" * 55}],
        destination_asset={"asset_code": "USDC", "asset_issuer": "G" + "B" * 55},
        source_amount="100.0000000",
        destination_amount="11.8500000",
        quoted_at="2026-10-21T08:05:00Z",
    )
    assert res.is_quote_only is True
    assert [h.asset_code for h in res.route_hops] == ["XLM", "AQUA", "USDC"]
    assert res.route_hops[2].asset_issuer == "G" + "B" * 55
    assert "not a submitted or settled payment" in res.slippage_guidance
