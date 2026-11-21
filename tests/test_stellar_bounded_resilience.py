"""Deterministic tests for bounded Horizon/Soroban provider access and error resilience.

Implements acceptance criteria for [PROD-AUDIT AI-088] / Issue #864:
- Pre-flight network, asset, and issuer checks before provider access.
- Strict refusal of secret keys before any Horizon network call.
- Bounded retries with exponential backoff on transient Horizon errors.
- Clean 503 error reporting on provider failure exhaustion.
- Bounded balance inspection and pagination protection.
- Mainnet vs testnet deterministic issuer matching.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from stellar_sdk.exceptions import NotFoundError

import stellar
from stellar import (
    HORIZON_MAX_BALANCES_INSPECTED,
    HORIZON_MAX_RETRIES,
    USDC_ISSUERS,
    fetch_usdc_balance,
    usdc_issuer,
)

VALID_TESTNET_KEY = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
VALID_OTHER_KEY = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
# Synthetic secret key fixture for testing pre-flight refusal
SYNTHETIC_SECRET_KEY = "S" + "B" * 55


class TestStellarBoundedResilience:
    def test_secret_key_refused_before_provider_access(self):
        """A secret key must be refused immediately without touching Horizon."""
        with patch.object(stellar, "Server") as mock_server:
            with pytest.raises(HTTPException) as exc:
                fetch_usdc_balance(SYNTHETIC_SECRET_KEY)
            assert exc.value.status_code == 400
            assert "Secret keys are never accepted" in exc.value.detail
            mock_server.assert_not_called()

    def test_unsupported_network_refused_before_provider_access(self):
        """Unsupported network configurations must fail fast before provider access."""
        with patch.object(stellar, "Server") as mock_server:
            with pytest.raises(HTTPException) as exc:
                fetch_usdc_balance(VALID_TESTNET_KEY, network="unsupported-cluster")
            assert exc.value.status_code == 400
            assert "Unsupported Stellar network" in exc.value.detail
            mock_server.assert_not_called()

    def test_transient_failures_retry_and_succeed(self):
        """Transient errors retry up to max_retries and succeed when node recovers."""
        mock_server = MagicMock()
        mock_accounts = mock_server.accounts.return_value.account_id.return_value

        # First 2 attempts fail with connection error, 3rd succeeds
        account_data = {
            "balances": [
                {
                    "asset_code": "USDC",
                    "asset_issuer": usdc_issuer("testnet"),
                    "balance": "250.5000000",
                }
            ]
        }
        mock_accounts.call.side_effect = [
            ConnectionError("Temporary DNS failure"),
            TimeoutError("Horizon read timeout"),
            account_data,
        ]

        with patch.object(stellar, "Server", return_value=mock_server):
            with patch("time.sleep"):  # No real sleeps during test
                balance = fetch_usdc_balance(VALID_TESTNET_KEY, network="testnet")
                assert balance == Decimal("250.5000000")
                assert mock_accounts.call.call_count == 3

    def test_retries_exhaustion_raises_sanitized_503(self):
        """Exhausting all retries raises a 503 Service Unavailable with sanitized details."""
        mock_server = MagicMock()
        mock_accounts = mock_server.accounts.return_value.account_id.return_value
        mock_accounts.call.side_effect = ConnectionError("Node unreachable")

        with patch.object(stellar, "Server", return_value=mock_server):
            with patch("time.sleep"):
                with pytest.raises(HTTPException) as exc:
                    fetch_usdc_balance(VALID_TESTNET_KEY, network="testnet")
                assert exc.value.status_code == 503
                assert "Stellar Horizon provider temporarily unavailable" in exc.value.detail
                assert mock_accounts.call.call_count == HORIZON_MAX_RETRIES + 1

    def test_404_not_found_is_not_retried(self):
        """404 indicates account absence on ledger and must not be retried."""
        mock_server = MagicMock()
        mock_accounts = mock_server.accounts.return_value.account_id.return_value
        mock_accounts.call.side_effect = NotFoundError(MagicMock())

        with patch.object(stellar, "Server", return_value=mock_server):
            with pytest.raises(HTTPException) as exc:
                fetch_usdc_balance(VALID_TESTNET_KEY, network="testnet")
            assert exc.value.status_code == 404
            assert mock_accounts.call.call_count == 1

    def test_balances_inspection_is_bounded(self):
        """Account with excess balances is bounded to HORIZON_MAX_BALANCES_INSPECTED."""
        mock_server = MagicMock()
        mock_accounts = mock_server.accounts.return_value.account_id.return_value

        # Create 200 dummy non-USDC trustlines followed by USDC at index 150
        large_balances = [
            {"asset_code": f"TKN{i}", "asset_issuer": VALID_OTHER_KEY, "balance": "10"} for i in range(150)
        ]
        # USDC placed beyond HORIZON_MAX_BALANCES_INSPECTED (100)
        large_balances.append(
            {
                "asset_code": "USDC",
                "asset_issuer": usdc_issuer("testnet"),
                "balance": "999.0",
            }
        )

        mock_accounts.call.return_value = {"balances": large_balances}

        with patch.object(stellar, "Server", return_value=mock_server):
            # Because inspection is bounded to HORIZON_MAX_BALANCES_INSPECTED, the entry at index 150 is ignored
            assert len(large_balances) > HORIZON_MAX_BALANCES_INSPECTED
            balance = fetch_usdc_balance(VALID_TESTNET_KEY, network="testnet")
            assert balance is None

    def test_mainnet_and_testnet_issuer_separation(self):
        """Mainnet and testnet strictly require their network's respective USDC issuer."""
        mainnet_issuer = USDC_ISSUERS["mainnet"]
        testnet_issuer = USDC_ISSUERS["testnet"]
        assert mainnet_issuer != testnet_issuer

        mock_server = MagicMock()
        mock_accounts = mock_server.accounts.return_value.account_id.return_value

        # Account has mainnet USDC issuer
        mock_accounts.call.return_value = {
            "balances": [
                {
                    "asset_code": "USDC",
                    "asset_issuer": mainnet_issuer,
                    "balance": "1000.0",
                }
            ]
        }

        with patch.object(stellar, "Server", return_value=mock_server):
            # Querying on mainnet succeeds
            bal_main = fetch_usdc_balance(VALID_TESTNET_KEY, network="mainnet")
            assert bal_main == Decimal("1000.0")

            # Querying on testnet ignores the balance because issuer doesn't match testnet issuer
            bal_test = fetch_usdc_balance(VALID_TESTNET_KEY, network="testnet")
            assert bal_test is None

    def test_malformed_balance_amount_fails_safe(self):
        """Malformed or negative balance strings are safely treated as invalid."""
        mock_server = MagicMock()
        mock_accounts = mock_server.accounts.return_value.account_id.return_value
        mock_accounts.call.return_value = {
            "balances": [
                {
                    "asset_code": "USDC",
                    "asset_issuer": usdc_issuer("testnet"),
                    "balance": "NOT_A_NUMBER",
                }
            ]
        }

        with patch.object(stellar, "Server", return_value=mock_server):
            assert fetch_usdc_balance(VALID_TESTNET_KEY, network="testnet") is None
