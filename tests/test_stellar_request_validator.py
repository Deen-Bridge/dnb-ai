"""Tests for API boundary validation of Stellar parameters (#581)."""

from __future__ import annotations

from stellar_request_validator import validate_stellar_tool_parameters


def test_validates_parameters_and_returns_field_specific_422_errors() -> None:
    ok = validate_stellar_tool_parameters(
        network="public",
        asset_code="USDC",
        asset_issuer="G" + "A" * 55,
        tx_hash="a" * 64,
        amount="25.5000000",
    )
    assert ok.is_valid is True
    assert ok.status_code == 200

    bad = validate_stellar_tool_parameters(
        network="mainnet-invalid",
        asset_code="USDC",
        asset_issuer="bad-issuer",
        tx_hash="short",
        amount="-5",
    )
    assert bad.is_valid is False
    assert bad.status_code == 422
    fields = {e.field for e in bad.errors}
    assert fields == {"network", "asset_issuer", "tx_hash", "amount"}
