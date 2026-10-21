"""Tests for explicit Stellar network context boundary (#556)."""

from __future__ import annotations

from stellar_network_boundary import validate_stellar_network_context


def test_accepts_testnet_and_public_explicitly() -> None:
    t = validate_stellar_network_context(requested_network="testnet")
    assert t.ok is True
    assert t.network == "testnet"

    p = validate_stellar_network_context(
        requested_network="public", configured_network="public"
    )
    assert p.ok is True
    assert p.network == "public"


def test_rejects_missing_invalid_and_contradictory_network() -> None:
    missing = validate_stellar_network_context(requested_network=None)
    assert missing.ok is False
    assert missing.error_code == "MISSING_NETWORK_CONTEXT"

    invalid = validate_stellar_network_context(requested_network="futurenet-custom")
    assert invalid.ok is False
    assert invalid.error_code == "UNSUPPORTED_NETWORK_CONTEXT"
    assert invalid.network == "futurenet-custom"

    contradictory = validate_stellar_network_context(
        requested_network="public", configured_network="testnet"
    )
    assert contradictory.ok is False
    assert contradictory.error_code == "CONTRADICTORY_NETWORK_CONTEXT"
    assert contradictory.network == "public"
