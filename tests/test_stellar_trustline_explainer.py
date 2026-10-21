"""Tests for Stellar trustline requirements explainer (#554)."""

from __future__ import annotations

from stellar_trustline_explainer import explain_trustline_requirement

ISSUER = "G" + "C" * 55


def test_native_xlm_requires_no_trustline() -> None:
    res = explain_trustline_requirement(
        network="public", asset_code="XLM", asset_issuer=None
    )
    assert res.asset_type == "native"
    assert res.can_receive_payment is True
    assert res.subentry_reserve_increase == 0


def test_missing_existing_and_auth_required_trustline_cases() -> None:
    missing = explain_trustline_requirement(
        network="testnet",
        asset_code="USDC",
        asset_issuer=ISSUER,
        has_trustline=False,
    )
    assert missing.scenario == "missing_trustline"
    assert missing.can_receive_payment is False
    assert "op_no_trust" in missing.explanation

    unauth = explain_trustline_requirement(
        network="testnet",
        asset_code="USDC",
        asset_issuer=ISSUER,
        has_trustline=True,
        issuer_auth_required=True,
        is_trustline_authorized=False,
    )
    assert unauth.scenario == "auth_required_unapproved"
    assert unauth.can_receive_payment is False

    existing = explain_trustline_requirement(
        network="public",
        asset_code="USDC",
        asset_issuer=ISSUER,
        has_trustline=True,
        issuer_auth_required=True,
        is_trustline_authorized=True,
    )
    assert existing.scenario == "existing_trustline"
    assert existing.can_receive_payment is True
