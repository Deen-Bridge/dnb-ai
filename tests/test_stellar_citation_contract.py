"""Tests for Stellar protocol passage citations (#551)."""

from __future__ import annotations

import pytest

from stellar_citation_contract import (
    StellarSourceCitation,
    build_grounded_stellar_answer,
)


def test_multiple_sources_and_serialization() -> None:
    c1 = StellarSourceCitation(
        passage_id="doc-accounts#p1",
        title="Stellar Accounts",
        canonical_url="https://developers.stellar.org/docs/learn/accounts",
    )
    c2 = StellarSourceCitation(
        passage_id="doc-assets#p3",
        title="Stellar Issued Assets",
        canonical_url="https://developers.stellar.org/docs/learn/assets",
    )
    res = build_grounded_stellar_answer(
        answer_text="Accounts require a trustline before receiving issued assets.",
        is_protocol_specific=True,
        citations=[c1, c2, c1],
    )
    payload = res.to_dict()
    assert payload["status"] == "grounded"
    assert len(payload["citations"]) == 2
    assert payload["citations"][0]["passage_id"] == "doc-accounts#p1"


def test_missing_evidence_on_protocol_claim_returns_insufficient_evidence() -> None:
    res = build_grounded_stellar_answer(
        answer_text="Minimum base reserve is 0.5 XLM per subentry.",
        is_protocol_specific=True,
        citations=[],
    )
    assert res.status == "insufficient_evidence"
    assert len(res.citations) == 0


def test_general_explanation_labeled_clearly_without_citation() -> None:
    res = build_grounded_stellar_answer(
        answer_text="Public-key cryptography uses a public address and a private key.",
        is_protocol_specific=False,
        citations=[],
    )
    assert res.status == "general_explanation"
    assert res.answer_text.startswith("[General Explanation]")


def test_invalid_citation_rejected() -> None:
    with pytest.raises(ValueError, match="Citation requires"):
        build_grounded_stellar_answer(
            answer_text="Claim",
            is_protocol_specific=True,
            citations=[
                StellarSourceCitation(
                    passage_id="",
                    title="Missing ID",
                    canonical_url="http://insecure.example.com",
                )
            ],
        )
