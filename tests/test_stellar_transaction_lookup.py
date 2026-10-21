"""Tests for read-only Stellar transaction lookup by hash (#558)."""

from __future__ import annotations

from stellar_transaction_lookup import lookup_stellar_transaction

VALID_HASH = "a" * 64


def test_found_and_not_found_transaction_lookup() -> None:
    found = lookup_stellar_transaction(
        tx_hash=VALID_HASH,
        network="public",
        provider_fetcher=lambda h, n: {
            "hash": h,
            "successful": True,
            "ledger": 998877,
            "created_at": "2026-10-21T07:30:00Z",
            "fee_charged": "100",
            "memo_type": "text",
        },
    )
    assert found.lookup_outcome == "found"
    assert found.tx_status == "successful"
    assert found.ledger == 998877

    not_found = lookup_stellar_transaction(
        tx_hash=VALID_HASH,
        network="public",
        provider_fetcher=lambda h, n: None,
    )
    assert not_found.lookup_outcome == "unknown_not_found"
    assert not_found.tx_status == "unknown"


def test_malformed_timeout_and_provider_error_cases() -> None:
    bad = lookup_stellar_transaction(
        tx_hash="1234", network="testnet", provider_fetcher=lambda h, n: None
    )
    assert bad.lookup_outcome == "invalid_input"
    assert bad.error_code == "MALFORMED_TX_HASH"

    def raise_timeout(h: str, n: str) -> dict[str, object] | None:
        raise TimeoutError("Timed out")

    to_res = lookup_stellar_transaction(
        tx_hash=VALID_HASH, network="testnet", provider_fetcher=raise_timeout
    )
    assert to_res.lookup_outcome == "timeout"
    assert to_res.error_code == "PROVIDER_TIMEOUT"

    def raise_err(h: str, n: str) -> dict[str, object] | None:
        raise RuntimeError("502 Bad Gateway")

    err_res = lookup_stellar_transaction(
        tx_hash=VALID_HASH, network="testnet", provider_fetcher=raise_err
    )
    assert err_res.lookup_outcome == "provider_error"
    assert err_res.error_code == "PROVIDER_ERROR"
