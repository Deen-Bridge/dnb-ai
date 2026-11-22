"""Production contract checks for Stellar ledger and provider freshness."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest

import stellar
from errors import APIException
from stellar_contract_state import evaluate_provider_freshness, validate_contract_request

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize("network", ["testnet", "public"])
def test_request_contract_accepts_configured_network_and_canonical_usdc(network: str) -> None:
    assert (
        validate_contract_request(
            network=network,
            configured_network=network,
            asset_code="USDC",
            asset_issuer="G" + "A" * 55,
            expected_issuer="G" + "A" * 55,
            public_key_valid=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"network": "invalid"}, "UNSUPPORTED_NETWORK_CONTEXT"),
        ({"asset_issuer": "G" + "B" * 55}, "UNVERIFIED_ASSET_ISSUER"),
        ({"asset_code": "FAKE"}, "UNVERIFIED_ASSET_ISSUER"),
        ({"public_key_valid": False}, "INVALID_PUBLIC_KEY_EVIDENCE"),
    ],
)
def test_request_contract_rejects_untrusted_inputs_before_provider_access(overrides, expected) -> None:
    assert (
        validate_contract_request(
            network=overrides.get("network", "testnet"),
            configured_network="testnet",
            asset_code=overrides.get("asset_code", "USDC"),
            asset_issuer=overrides.get("asset_issuer", "G" + "A" * 55),
            expected_issuer="G" + "A" * 55,
            public_key_valid=overrides.get("public_key_valid", True),
        )
        == expected
    )


@pytest.mark.parametrize("network", ["testnet", "public"])
def test_fresh_provider_evidence_is_accepted_on_both_networks(network: str) -> None:
    result = evaluate_provider_freshness(
        {
            "network": network,
            "history_latest_ledger": 1200501,
            "history_latest_ledger_closed_at": (NOW - timedelta(seconds=10)).isoformat(),
        },
        now=NOW,
    )
    assert result.ok
    assert result.ledger_sequence == 1200501


@pytest.mark.parametrize("network", ["testnet", "public"])
def test_account_lookup_uses_configured_network_and_issuer(network: str) -> None:
    with patch.object(stellar, "STELLAR_NETWORK", network), patch.object(stellar, "Server") as provider:
        server = provider.return_value
        server.root.return_value.call.return_value = {
            "history_latest_ledger": 1200501,
            "history_latest_ledger_closed_at": datetime.now(UTC).isoformat(),
        }
        server.accounts.return_value.account_id.return_value.call.return_value = {
            "balances": [
                {
                    "asset_code": "USDC",
                    "asset_issuer": stellar.USDC_ISSUERS[network],
                    "balance": "12.5",
                }
            ]
        }
        assert stellar.fetch_usdc_balance("GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5") == Decimal("12.5")
        provider.assert_called_once_with(stellar.HORIZON_URLS[network])


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        ({}, "MISSING_LEDGER_EVIDENCE"),
        (
            {
                "network": "public",
                "history_latest_ledger": 1200501,
                "history_latest_ledger_closed_at": (NOW + timedelta(seconds=1)).isoformat(),
            },
            "FUTURE_LEDGER_EVIDENCE",
        ),
        (
            {
                "network": "testnet",
                "history_latest_ledger": "bad",
                "history_latest_ledger_closed_at": NOW.isoformat(),
            },
            "MISSING_LEDGER_EVIDENCE",
        ),
    ],
)
def test_provider_freshness_fails_closed(evidence, reason) -> None:
    assert not evaluate_provider_freshness(evidence, now=NOW).ok
    assert evaluate_provider_freshness(evidence, now=NOW).reason == reason


@pytest.mark.parametrize("network", ["testnet", "public"])
def test_stale_provider_freshness_fails_closed_on_both_networks(network: str) -> None:
    evidence = {
        "network": network,
        "history_latest_ledger": 1200501,
        "history_latest_ledger_closed_at": (NOW - timedelta(seconds=31)).isoformat(),
    }
    result = evaluate_provider_freshness(evidence, now=NOW)
    assert not result.ok
    assert result.reason == "STALE_LEDGER_EVIDENCE"


def test_invalid_asset_contract_stops_before_horizon_access(monkeypatch) -> None:
    monkeypatch.setattr(stellar, "STELLAR_NETWORK", "unsupported")
    with patch.object(stellar, "Server") as server, pytest.raises(APIException):
        stellar.fetch_usdc_balance("GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5")
    server.assert_not_called()


def test_stale_latest_ledger_prevents_account_lookup(monkeypatch) -> None:
    monkeypatch.setattr(stellar, "STELLAR_NETWORK", "testnet")
    with patch.object(stellar, "Server") as provider:
        server = provider.return_value
        server.root.return_value.call.return_value = {
            "history_latest_ledger": 1200501,
            "history_latest_ledger_closed_at": "2020-01-01T00:00:00+00:00",
        }
        with pytest.raises(APIException):
            stellar.fetch_usdc_balance("GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5")
        server.accounts.assert_not_called()


def test_provider_failure_prevents_account_lookup(monkeypatch) -> None:
    monkeypatch.setattr(stellar, "STELLAR_NETWORK", "public")
    with patch.object(stellar, "Server") as provider:
        server = provider.return_value
        server.root.return_value.call.side_effect = TimeoutError("synthetic timeout")
        with pytest.raises(APIException) as error:
            stellar.fetch_usdc_balance("GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5")
        server.accounts.assert_not_called()
    assert error.value.status_code == 503
    assert "synthetic timeout" not in str(error.value.detail)
