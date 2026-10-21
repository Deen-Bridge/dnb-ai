"""Tests for separating anchor assets from native and issued Stellar assets (#573)."""

from __future__ import annotations

from stellar_asset_taxonomy import classify_asset_representation


def test_separates_native_issued_and_anchor_representations() -> None:
    native = classify_asset_representation(category="native")
    assert native.category == "native_xlm"
    assert native.is_settled_on_stellar_ledger is True

    issued = classify_asset_representation(
        category="issued",
        stellar_asset_code="USDC",
        stellar_asset_issuer="G" + "A" * 55,
    )
    assert issued.category == "issued_stellar_asset"
    assert issued.is_settled_on_stellar_ledger is True

    anchor = classify_asset_representation(
        category="anchor",
        anchor_name="MoneyGram Anchor",
        external_currency="USD",
        has_verified_onchain_trustline_balance=False,
    )
    assert anchor.category == "anchor_offchain_representation"
    assert anchor.is_settled_on_stellar_ledger is False
    assert "NOT an on-chain Stellar balance" in anchor.terminology_guidance
