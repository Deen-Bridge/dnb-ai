"""HTTP surface for the Malay/Indonesian language module (#879).

The endpoints mirror :mod:`docs.languages.md` §2.6 and reuse the existing
``malay`` engines rather than reimplementing normalization or terminology.
Everything here is offline and deterministic.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from malay.dialects import DialectHandler
from malay.models import MalayQueryRequest, MalayQueryResponse
from malay.optimizer import QueryOptimizer
from malay.processor import MalayProcessor
from malay.terminology import TerminologyLexicon

router = APIRouter(prefix="/malay", tags=["malay"])

_processor = MalayProcessor()
_lexicon = TerminologyLexicon()

DIALECTS: tuple[dict[str, str], ...] = (
    {"id": "malaysia", "label": "Malaysian Malay", "script": "latin"},
    {"id": "indonesia", "label": "Indonesian", "script": "latin"},
)


class MalayNormalizeRequest(BaseModel):
    text: str = Field(..., description="Input text in Malay or Indonesian")
    dialect: str | None = Field(None, description="Target dialect: 'malaysia' or 'indonesia'")


class MalayNormalizeResponse(BaseModel):
    original_text: str
    normalized_text: str
    detected_dialect: str
    processing_time_ms: float = 0.0


class MalayTerm(BaseModel):
    id: str
    malay_term: str | None = None
    indonesian_term: str | None = None
    jawi_term: str | None = None
    english_equivalent: str | None = None
    category: str | None = None
    variants: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class MalayDialect(BaseModel):
    id: str
    label: str
    script: str


class MalayRouter:
    """Backwards-compatible topic router retained for existing callers."""

    @staticmethod
    def route_request(text: str) -> str:
        if any(term in text.lower() for term in ["shalat", "zakat", "haji", "puasa"]):
            return "fiqh_ibadat"
        return "general_islamic_qa"


def _as_term(term: dict[str, Any]) -> MalayTerm:
    return MalayTerm(
        id=str(term.get("id", "")),
        malay_term=term.get("malay_term"),
        indonesian_term=term.get("indonesian_term"),
        jawi_term=term.get("jawi_term"),
        english_equivalent=term.get("english_equivalent"),
        category=term.get("category"),
        variants=list(term.get("variants", [])),
        raw=term,
    )


def _matches(term: dict[str, Any], needle: str) -> bool:
    haystack = [
        str(term.get("id", "")),
        str(term.get("malay_term", "")),
        str(term.get("indonesian_term", "")),
        str(term.get("jawi_term", "")),
        str(term.get("english_equivalent", "")),
        *[str(v) for v in term.get("variants", [])],
    ]
    return any(needle in value.casefold() for value in haystack)


@router.post("/analyze", response_model=MalayQueryResponse)
def analyze(request: MalayQueryRequest) -> MalayQueryResponse:
    """Run the full Malay/Indonesian facade pipeline."""
    return _processor.process(request)


@router.post("/normalize", response_model=MalayNormalizeResponse)
def normalize(request: MalayNormalizeRequest) -> MalayNormalizeResponse:
    """Normalize dialect spelling without answering the question."""
    started = time.perf_counter()
    dialect = request.dialect or DialectHandler.detect_dialect(request.text)
    normalized = QueryOptimizer.optimize_query(request.text)
    normalized = DialectHandler.normalize_dialect(normalized, dialect)
    return MalayNormalizeResponse(
        original_text=request.text,
        normalized_text=normalized,
        detected_dialect=dialect,
        processing_time_ms=(time.perf_counter() - started) * 1000.0,
    )


@router.get("/dialects", response_model=list[MalayDialect])
def dialects() -> list[MalayDialect]:
    """List the dialects the normalization layer understands."""
    return [MalayDialect(**dialect) for dialect in DIALECTS]


@router.get("/terms", response_model=list[MalayTerm])
def terms(
    query: str | None = None,
    limit: int = Query(50, ge=1, le=150),
) -> list[MalayTerm]:
    """Search the offline Malay/Indonesian terminology catalog."""
    needle = (query or "").strip().casefold()
    found = [term for term in _lexicon.terms if not needle or _matches(term, needle)]
    return [_as_term(term) for term in found[:limit]]


@router.get("/terms/{term_id}", response_model=MalayTerm)
def term(term_id: str) -> MalayTerm:
    """Return one catalog term by id, or 404."""
    for entry in _lexicon.terms:
        if entry.get("id") == term_id:
            return _as_term(entry)
    raise HTTPException(status_code=404, detail=f"Unknown Malay term id: {term_id}")
