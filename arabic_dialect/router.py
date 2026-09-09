"""FastAPI router for Arabic Dialect Support endpoints (#136)."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from arabic_dialect.dialects import DIALECT_PATTERNS, dialect_classifier
from arabic_dialect.models import (
    ArabicDialect,
    DialectAnalysisResult,
    DialectAnalyzeRequest,
    DialectNormalizeRequest,
    DialectNormalizeResponse,
    DialectSegment,
    DialectTerm,
)
from arabic_dialect.terminology import extract_terms_from_text, terminology_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/arabic-dialect", tags=["arabic-dialect"])


@router.post("/analyze", response_model=DialectAnalysisResult)
async def analyze_arabic_dialect(body: DialectAnalyzeRequest) -> DialectAnalysisResult:
    """Analyze Arabic text for dialect features and map dialectal terms to MSA.

    Returns the detected dialect with confidence and markers, the list of
    dialectal terms found (with their MSA equivalents), a per-segment
    classification, and the text normalized to MSA.
    """
    start_time = time.perf_counter()
    text = body.text.strip()

    profile = dialect_classifier.classify_dialect(text)
    terms = extract_terms_from_text(text)
    normalized_msa, _ = dialect_classifier.normalize_to_msa(text)

    # Segment classification: split on sentence boundaries and classify each.
    segments: list[DialectSegment] = []
    for part in _split_segments(text):
        seg_profile = dialect_classifier.classify_dialect(part)
        segments.append(
            DialectSegment(
                text=part,
                dialect=seg_profile.primary_dialect,
                confidence=seg_profile.confidence,
            )
        )

    return DialectAnalysisResult(
        original_text=text,
        normalized_msa=normalized_msa,
        dialect=profile,
        detected_terms=terms,
        segments=segments,
        processing_time_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
    )


@router.post("/normalize", response_model=DialectNormalizeResponse)
async def normalize_arabic_dialect(body: DialectNormalizeRequest) -> DialectNormalizeResponse:
    """Normalize dialectal terms in Arabic text to Modern Standard Arabic."""
    normalized_msa, replaced = dialect_classifier.normalize_to_msa(body.text)
    return DialectNormalizeResponse(
        original_text=body.text,
        normalized_msa=normalized_msa,
        replaced_terms=replaced,
    )


@router.get("/dialects")
async def list_dialects() -> dict[str, Any]:
    """List supported Arabic dialects and their key markers."""
    profiles: dict[str, Any] = {}
    for dialect, markers in DIALECT_PATTERNS.items():
        profiles[dialect.value] = {
            "name": dialect.name,
            "sample_markers": list(markers),
        }
    return {
        "supported_dialects": [d.value for d in ArabicDialect],
        "dialect_profiles": profiles,
    }


@router.get("/terms", response_model=list[DialectTerm])
async def search_terms(
    query: str | None = Query(None, description="Search term, MSA equivalent, transliteration, or gloss"),
    dialect: ArabicDialect | None = Query(None, description="Filter by dialect"),
    category: str | None = Query(None, description="Thematic category filter"),
    limit: int = Query(50, ge=1, le=150),
) -> list[DialectTerm]:
    """Search the dialectal Islamic terminology lexicon."""
    return terminology_db.search_terms(query=query, dialect=dialect, category=category, limit=limit)


@router.get("/terms/{term_id}", response_model=DialectTerm)
async def get_term_by_id(term_id: str) -> DialectTerm:
    """Retrieve a single dialectal term by its ID."""
    term = terminology_db.get_term_by_id(term_id)
    if not term:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dialectal term with ID '{term_id}' not found.",
        )
    return term


def _split_segments(text: str) -> list[str]:
    """Split text into sentence-ish segments for per-segment dialect analysis."""
    import re as _re

    parts = [p.strip() for p in _re.split(r"[.!؟?،,؛;]+", text) if p and p.strip()]
    return parts or ([text] if text else [])
