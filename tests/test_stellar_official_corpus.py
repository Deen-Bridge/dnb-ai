"""Tests for versioned official Stellar documentation corpus (#550)."""

from __future__ import annotations

import pytest

from stellar_official_corpus import (
    OfficialStellarCorpus,
    compute_content_hash,
    is_allowed_stellar_source,
)


def test_allowlist_accepts_only_official_https_stellar_urls() -> None:
    assert is_allowed_stellar_source("https://developers.stellar.org/docs/learn/fundamentals")
    assert is_allowed_stellar_source("https://soroban.stellar.org/docs")
    assert not is_allowed_stellar_source("http://developers.stellar.org/docs")
    assert not is_allowed_stellar_source("https://community-blog.example.com/stellar")


def test_ingest_retains_metadata_and_retrieves_passages_deterministically() -> None:
    corpus = OfficialStellarCorpus(version="2026.10.1")
    content = (
        "Stellar accounts hold balances and sign transactions using public-key cryptography.\n\n"
        "Trustlines are explicit opt-in lines required before holding issued assets on Stellar."
    )
    doc = corpus.ingest_document(
        doc_id="stellar-accounts-v1",
        title="Stellar Accounts and Trustlines",
        canonical_url="https://developers.stellar.org/docs/learn/accounts",
        retrieval_date="2026-10-21",
        content=content,
    )
    assert doc.content_hash == compute_content_hash(content)
    p2 = corpus.get_passage("stellar-accounts-v1#p2")
    assert p2 is not None
    assert p2.canonical_url == "https://developers.stellar.org/docs/learn/accounts"
    assert p2.retrieval_date == "2026-10-21"
    assert p2.content_hash == doc.content_hash

    results = corpus.search_passages("trustlines issued assets")
    assert len(results) >= 1
    assert results[0].passage_id == "stellar-accounts-v1#p2"


def test_disallowed_source_rejected() -> None:
    corpus = OfficialStellarCorpus()
    with pytest.raises(ValueError, match="Disallowed documentation source"):
        corpus.ingest_document(
            doc_id="bad-1",
            title="Unofficial Forum Post",
            canonical_url="https://reddit.com/r/stellar/post",
            retrieval_date="2026-10-21",
            content="Some forum comment",
        )
