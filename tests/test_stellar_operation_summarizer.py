"""Tests for Stellar transaction operation summarizer (#560)."""

from __future__ import annotations

from stellar_operation_summarizer import summarize_transaction_operations


def test_normalizes_payment_trustline_and_unknown_operations() -> None:
    ops = [
        {
            "type": "payment",
            "source_account": "G" + "A" * 55,
            "to": "G" + "B" * 55,
            "asset_type": "credit_alphanum4",
            "asset_code": "USDC",
            "asset_issuer": "G" + "C" * 55,
            "amount": "25.0000000",
            "raw_internal_token": "should-not-leak",
        },
        {
            "type": "change_trust",
            "source_account": "G" + "B" * 55,
            "asset_code": "USDC",
            "asset_issuer": "G" + "C" * 55,
            "limit": "1000.0000000",
        },
        {
            "type": "bump_sequence",
            "source_account": "G" + "A" * 55,
        },
    ]
    res = summarize_transaction_operations(ops)
    assert len(res) == 3
    assert res[0].operation_type == "payment"
    assert res[0].is_known_type is True
    assert res[0].asset_code == "USDC"
    assert "raw_internal_token" not in res[0].to_dict()
    assert res[1].operation_type == "change_trust"
    assert res[1].trust_limit == "1000.0000000"
    assert res[2].operation_type == "bump_sequence"
    assert res[2].is_known_type is False
