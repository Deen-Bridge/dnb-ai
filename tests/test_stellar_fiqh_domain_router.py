"""Tests for distinguishing Stellar protocol and Islamic jurisprudence queries (#584)."""

from __future__ import annotations

from stellar_fiqh_domain_router import classify_query_domain


def test_classifies_stellar_islamic_and_mixed_queries_separately() -> None:
    s = classify_query_domain("How do trustlines and stroops work on Stellar Horizon?")
    assert s.domain == "stellar_protocol"
    assert s.required_corpora == ("official_stellar_docs",)

    i = classify_query_domain("What is the nisab for zakat according to Quran and Hadith?")
    assert i.domain == "islamic_knowledge"
    assert i.required_corpora == ("authenticated_islamic_sources",)

    m = classify_query_domain("How can I pay sadaqah using USDC on a Stellar trustline?")
    assert m.domain == "mixed_domain"
    assert "official_stellar_docs" in m.required_corpora
    assert "authenticated_islamic_sources" in m.required_corpora
