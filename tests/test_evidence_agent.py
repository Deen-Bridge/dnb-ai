"""Tests for the evidence verification agent wrapper (#433).

The agent delegates citation extraction and verification to ``verifier``. These
tests exercise the wrapper through its public ``run_evidence_agent`` interface
using the bundled corpus, so they stay offline and deterministic.
"""

import pytest

from evidence_agent import run_evidence_agent
from verifier import VerificationStatus


def _results(text: str) -> list[dict]:
    payload = run_evidence_agent(text)
    assert set(payload) == {"results"}
    return payload["results"]


def test_supported_claim_returns_verified_quran_citation() -> None:
    text = (
        'The Qur\u2019an opens with Quran 1:1 "In the name of Allah, the Entirely Merciful, the Especially Merciful."'
    )

    results = _results(text)

    assert len(results) == 1
    assert results[0]["source"] == "quran"
    assert results[0]["status"] == VerificationStatus.VERIFIED
    assert results[0]["confidence"] > 0
    assert results[0]["defer_recommended"] is False


def test_unsupported_claim_flags_mismatched_quote() -> None:
    text = 'Quran 1:1 "This is a fabricated translation string."'

    results = _results(text)

    assert len(results) == 1
    citation = results[0]
    assert citation["source"] == "quran"
    assert citation["status"] == VerificationStatus.MISMATCH
    assert "correct_text" in citation
    assert "does not match" in citation["reason"]
    assert citation["defer_recommended"] is True


def test_insufficient_evidence_is_reported_as_unverified() -> None:
    text = "The Prophet said this in Bukhari 42."

    results = _results(text)

    assert len(results) == 1
    citation = results[0]
    assert citation["source"] == "hadith"
    assert citation["status"] == VerificationStatus.UNVERIFIED
    assert "not available" in citation["reason"]
    assert citation["defer_recommended"] is True


def test_text_without_citations_returns_no_results() -> None:
    assert _results("There is no primary-source citation in this sentence.") == []


def test_malformed_reference_is_reported_as_mismatch() -> None:
    results = _results('Quran 1:999 "A quote that cannot exist."')

    assert len(results) == 1
    citation = results[0]
    assert citation["status"] == VerificationStatus.MISMATCH
    assert "does not exist" in citation["reason"]


def test_multiple_citations_are_verified_independently() -> None:
    text = (
        'Quran 1:1 "In the name of Allah, the Entirely Merciful, the Especially Merciful." '
        "And the narration is recorded in Bukhari 42."
    )

    results = _results(text)

    assert len(results) == 2
    by_source = {citation["source"]: citation for citation in results}
    assert by_source["quran"]["status"] == VerificationStatus.VERIFIED
    assert by_source["hadith"]["status"] == VerificationStatus.UNVERIFIED


def test_non_string_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        run_evidence_agent(None)  # type: ignore[arg-type]
