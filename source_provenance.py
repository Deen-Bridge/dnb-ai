"""Expose verifiable source provenance and uncertainty separately from model confidence (#526)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class SourceAttribution:
    source_id: str
    title: str
    reference: str
    scholar_or_collection: str
    retrieved_at_iso: str
    verified_in_corpus: bool


@dataclass(frozen=True)
class ProvenanceEnvelope:
    has_verifiable_sources: bool
    citations: list[SourceAttribution]
    citation_validation_status: str
    uncertainty_explanation: str


def build_provenance_envelope(
    sources: list[SourceAttribution],
    conflicting_sources: bool = False,
    stale_threshold_days: int = 365,
) -> ProvenanceEnvelope:
    """Validate retrieved references without fabricating citations when sources are missing or conflicting."""
    verified = [s for s in sources if s.verified_in_corpus and s.reference.strip()]
    if not verified:
        return ProvenanceEnvelope(
            has_verifiable_sources=False,
            citations=[],
            citation_validation_status="no_verified_sources_found",
            uncertainty_explanation=(
                "No verifiable primary Qur'an, hadith, or fiqh references were found in the retrieved corpus; "
                "no citation has been generated."
            ),
        )

    stale_found = False
    now = datetime.now(UTC)
    for src in verified:
        try:
            ts = datetime.fromisoformat(src.retrieved_at_iso.replace("Z", "+00:00"))
            if (now - ts).days > stale_threshold_days:
                stale_found = True
        except ValueError:
            stale_found = True

    if conflicting_sources:
        status = "conflicting_verified_sources"
        uncertainty = "Multiple verified sources present differing textual scopes or scholarly interpretations."
    elif stale_found:
        status = "verified_with_stale_timestamp"
        uncertainty = (
            "References are verified in the corpus, but one or more retrieval timestamps exceed the freshness window."
        )
    else:
        status = "verified"
        uncertainty = "All cited references were matched deterministically against the retrieved corpus."

    return ProvenanceEnvelope(
        has_verifiable_sources=True,
        citations=verified,
        citation_validation_status=status,
        uncertainty_explanation=uncertainty,
    )
