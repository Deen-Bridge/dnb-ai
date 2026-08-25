"""FastAPI router for Swahili Islamic Language Processing endpoints."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from swahili.code_switching import code_switch_processor
from swahili.cultural_context import cultural_context_engine
from swahili.dialects import DIALECT_MARKERS, dialect_classifier
from swahili.generator import swahili_response_enhancer
from swahili.loanwords import loanword_analyzer
from swahili.models import (
    CodeSwitchResult,
    IslamicDomain,
    IslamicTerm,
    SwahiliAnalysisResult,
    SwahiliAnalyzeRequest,
    SwahiliCodeSwitchRequest,
    SwahiliDialect,
    SwahiliNormalizeRequest,
    SwahiliNormalizeResponse,
    SwahiliPromptEnhancement,
)
from swahili.terminology import terminology_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/swahili", tags=["swahili"])


@router.post("/analyze", response_model=SwahiliAnalysisResult)
async def analyze_swahili_text(body: SwahiliAnalyzeRequest) -> SwahiliAnalysisResult:
    """Perform full linguistic, loanword, dialect, code-switching, and cultural analysis."""
    start_time = time.perf_counter()

    tokens = loanword_analyzer.tokenize_swahili(body.text)
    detected_terms = terminology_db.extract_terms_from_text(body.text)
    loanwords = loanword_analyzer.extract_loanwords(body.text)
    dialect = dialect_classifier.classify_dialect(body.text)
    code_switch = code_switch_processor.analyze_code_switching(body.text)
    cultural = cultural_context_engine.extract_context(body.text)
    normalized_text, _ = dialect_classifier.normalize_to_standard(body.text)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return SwahiliAnalysisResult(
        original_text=body.text,
        normalized_text=normalized_text,
        tokens=tokens,
        detected_terms=detected_terms,
        loanwords=loanwords,
        dialect=dialect,
        code_switch=code_switch,
        cultural_context=cultural,
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.post("/normalize", response_model=SwahiliNormalizeResponse)
async def normalize_swahili_text(body: SwahiliNormalizeRequest) -> SwahiliNormalizeResponse:
    """Normalize regional dialect terms and spelling variants into Standard Swahili (Kiswahili Sanifu)."""
    normalized_text, replaced = dialect_classifier.normalize_to_standard(body.text)
    return SwahiliNormalizeResponse(
        original_text=body.text,
        normalized_text=normalized_text,
        replaced_terms=replaced,
    )


@router.get("/terms", response_model=list[IslamicTerm])
async def search_islamic_terms(
    query: str | None = Query(None, description="Search term in Swahili, Arabic, or English"),
    category: IslamicDomain | None = Query(None, description="Islamic thematic domain"),
    limit: int = Query(50, ge=1, le=150),
) -> list[IslamicTerm]:
    """Search the Swahili Islamic terminology database."""
    return terminology_db.search_terms(query=query, category=category, limit=limit)


@router.get("/terms/{term_id}", response_model=IslamicTerm)
async def get_term_by_id(term_id: str) -> IslamicTerm:
    """Retrieve details for a specific Islamic term by its ID."""
    term = terminology_db.get_term_by_id(term_id)
    if not term:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Islamic term with ID '{term_id}' not found.",
        )
    return term


@router.get("/dialects")
async def list_dialects() -> dict[str, Any]:
    """List supported East African Swahili dialects and key regional markers."""
    dialects_data = {}
    for dialect, profile in DIALECT_MARKERS.items():
        dialects_data[dialect.value] = {
            "name": dialect.name,
            "is_coastal": profile["is_coastal"],
            "sample_markers": list(profile["words"].keys()),
        }
    return {
        "supported_dialects": [d.value for d in SwahiliDialect],
        "dialect_profiles": dialects_data,
    }


@router.post("/code-switch", response_model=CodeSwitchResult)
async def analyze_code_switching_endpoint(body: SwahiliCodeSwitchRequest) -> CodeSwitchResult:
    """Analyze multi-lingual code-switching segments and Islamic formulas in text."""
    return code_switch_processor.analyze_code_switching(body.text)


@router.post("/enhance-prompt", response_model=SwahiliPromptEnhancement)
async def enhance_swahili_prompt(body: SwahiliAnalyzeRequest) -> SwahiliPromptEnhancement:
    """Generate prompt enhancement with Islamic glossary and East African cultural notes."""
    return swahili_response_enhancer.build_prompt_enhancement(body.text)
