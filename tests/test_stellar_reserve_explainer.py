"""Tests for Stellar reserve and sponsorship explainer (#566)."""

from __future__ import annotations

from decimal import Decimal

from stellar_reserve_explainer import explain_reserve_and_sponsorship


def test_numeric_and_qualitative_reserve_explanations() -> None:
    num = explain_reserve_and_sponsorship(
        network="public",
        subentries_count=3,
        sponsoring_count=1,
        sponsored_count=2,
        configured_base_reserve_xlm=Decimal("0.5"),
    )
    assert num.minimum_balance_xlm == "2.0"
    assert "distinct from per-operation transaction fees" in num.explanation

    qual = explain_reserve_and_sponsorship(
        network="testnet",
        subentries_count=1,
        configured_base_reserve_xlm=None,
    )
    assert qual.minimum_balance_xlm is None
    assert "separate from transaction fees" in qual.explanation
