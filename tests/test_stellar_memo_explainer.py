"""Tests for Stellar memo types and limitations explainer (#562)."""

from __future__ import annotations

import pytest

from stellar_memo_explainer import explain_stellar_memo


def test_explains_text_id_hash_and_return_memos_with_public_warning() -> None:
    for mtype, expected_label in [
        ("text", "MEMO_TEXT"),
        ("id", "MEMO_ID"),
        ("hash", "MEMO_HASH"),
        ("return", "MEMO_RETURN"),
    ]:
        res = explain_stellar_memo(mtype)
        assert res.protocol_label == expected_label
        assert res.is_public_on_chain is True
        assert "public ledger" in res.privacy_warning


def test_rejects_unknown_memo_type() -> None:
    with pytest.raises(ValueError, match="Unsupported Stellar memo_type"):
        explain_stellar_memo("encrypted_private")
