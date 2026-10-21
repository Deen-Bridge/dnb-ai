"""Tests for lookalike USDC asset warning and issuer identity guard (#555)."""

from __future__ import annotations

from stellar_asset_issuer_guard import ReviewedAssetIssuerRegistry

CANONICAL_USDC_PUB = "G" + "D" * 55
LOOKALIKE_USDC_PUB = "G" + "E" * 55


def test_canonical_lookalike_and_unknown_asset_cases() -> None:
    registry = ReviewedAssetIssuerRegistry({"public": {"USDC": CANONICAL_USDC_PUB}})

    canonical = registry.assess_issued_asset(network="public", asset_code="USDC", asset_issuer=CANONICAL_USDC_PUB)
    assert canonical.is_canonical is True
    assert canonical.verification_status == "canonical"
    assert canonical.warning_message is None

    lookalike = registry.assess_issued_asset(network="public", asset_code="USDC", asset_issuer=LOOKALIKE_USDC_PUB)
    assert lookalike.is_canonical is False
    assert lookalike.verification_status == "lookalike_warning"
    assert lookalike.warning_message is not None
    assert "Lookalike asset warning" in lookalike.warning_message

    unknown = registry.assess_issued_asset(network="public", asset_code="EURC", asset_issuer=LOOKALIKE_USDC_PUB)
    assert unknown.is_canonical is False
    assert unknown.verification_status == "unknown_asset"
