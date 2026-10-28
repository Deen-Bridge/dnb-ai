"""Tests for read-only Horizon account summary tool (#553)."""

from __future__ import annotations

from stellar_horizon_account_tool import summarize_horizon_account

VALID_G = "G" + "A" * 55
SECRET_S = "S" + "A" * 55


def test_successful_account_summary() -> None:
    def fake_fetcher(acct: str, net: str) -> dict[str, object]:
        return {
            "id": acct,
            "sequence": "123456",
            "last_modified_ledger": 54321,
            "last_modified_time": "2026-10-21T07:00:00Z",
            "thresholds": {"low_threshold": 1, "med_threshold": 2, "high_threshold": 3},
            "balances": [
                {"asset_type": "native", "balance": "150.5000000"},
                {
                    "asset_type": "credit_alphanum4",
                    "asset_code": "USDC",
                    "asset_issuer": "G" + "B" * 55,
                    "balance": "42.0000000",
                    "limit": "10000.0000000",
                    "is_authorized": True,
                },
            ],
        }

    res = summarize_horizon_account(account_id=VALID_G, network="testnet", provider_fetcher=fake_fetcher)
    assert res.status == "ok"
    assert res.native_balance == "150.5000000"
    assert len(res.trustlines) == 1
    assert res.trustlines[0].asset_code == "USDC"


def test_rejects_secret_seed_without_echoing_it() -> None:
    res = summarize_horizon_account(account_id=SECRET_S, network="testnet", provider_fetcher=lambda a, n: None)
    assert res.status == "invalid_input"
    assert res.error_code == "SECRET_KEY_FORBIDDEN"
    assert res.account_id is None


def test_handles_malformed_not_found_and_provider_error() -> None:
    malformed = summarize_horizon_account(account_id="not-a-key", network="public", provider_fetcher=lambda a, n: None)
    assert malformed.error_code == "MALFORMED_PUBLIC_ACCOUNT"

    empty = summarize_horizon_account(account_id=VALID_G, network="public", provider_fetcher=lambda a, n: None)
    assert empty.status == "not_found"

    def failing_fetcher(acct: str, net: str) -> dict[str, object] | None:
        raise RuntimeError("Horizon timeout")

    err = summarize_horizon_account(account_id=VALID_G, network="public", provider_fetcher=failing_fetcher)
    assert err.status == "provider_error"
    assert err.error_code == "HORIZON_PROVIDER_ERROR"
