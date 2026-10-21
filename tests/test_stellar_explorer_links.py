"""Tests for verified Stellar explorer link generator (#561)."""

from __future__ import annotations

from stellar_explorer_links import build_verified_explorer_tx_link

TX = "f" * 64


def test_generates_allowlisted_testnet_and_public_links() -> None:
    pub = build_verified_explorer_tx_link(network="public", tx_hash=TX)
    assert pub.ok is True
    assert pub.url == f"https://stellar.expert/explorer/public/tx/{TX}"

    test = build_verified_explorer_tx_link(network="testnet", tx_hash=TX, explorer="stellarchain")
    assert test.ok is True
    assert test.url == f"https://testnet.stellarchain.io/transactions/{TX}"


def test_rejects_arbitrary_host_and_malformed_hash() -> None:
    bad_host = build_verified_explorer_tx_link(network="public", tx_hash=TX, explorer="https://evil.example.com")
    assert bad_host.ok is False
    assert bad_host.error_code == "DISALLOWED_EXPLORER_HOST"

    bad_hash = build_verified_explorer_tx_link(network="public", tx_hash="not-a-hash/../../evil")
    assert bad_hash.ok is False
    assert bad_hash.error_code == "MALFORMED_TX_HASH"
