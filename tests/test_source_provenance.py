"""Tests for source provenance and uncertainty reporting (#526)."""

from __future__ import annotations

from source_provenance import SourceAttribution, build_provenance_envelope


def test_missing_or_unverified_sources_never_fabricate_citations() -> None:
    env = build_provenance_envelope([])
    assert env.has_verifiable_sources is False
    assert env.citations == []
    assert env.citation_validation_status == "no_verified_sources_found"


def test_conflicting_and_stale_sources_expose_clear_uncertainty() -> None:
    src = SourceAttribution(
        source_id="q-2-255",
        title="Surah Al-Baqarah",
        reference="Qur'an 2:255",
        scholar_or_collection="Tanzil Uthmani",
        retrieved_at_iso="2020-01-01T00:00:00Z",
        verified_in_corpus=True,
    )
    stale_env = build_provenance_envelope([src])
    assert stale_env.has_verifiable_sources is True
    assert stale_env.citation_validation_status == "verified_with_stale_timestamp"

    conflict_env = build_provenance_envelope([src], conflicting_sources=True)
    assert conflict_env.citation_validation_status == "conflicting_verified_sources"
