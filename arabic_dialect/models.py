"""Data models and schemas for the Arabic Dialect Support subsystem (#136).

Covers the three major regional varieties the issue calls for — Egyptian,
Gulf (Khaleeji), and Levantine (Shami) — plus Modern Standard Arabic (MSA)
as the reference variety everything is normalized toward.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ArabicDialect(str, Enum):
    """Major Arabic dialectal varieties supported by the subsystem."""

    MSA = "msa"  # Modern Standard Arabic (فصحى) — the reference variety
    EGYPTIAN = "egyptian"  # اللهجة المصرية
    GULF = "gulf"  # اللهجة الخليجية (Khaleeji)
    LEVANTINE = "levantine"  # اللهجة الشامية (Shami)
    UNKNOWN = "unknown"


class DialectRegion(str, Enum):
    """Geographic region associated with a dialect."""

    MASHREQ = "mashreq"
    EGYPT = "egypt"
    GULF = "gulf"
    LEVANT = "levant"
    NONE = "none"


class DialectProfile(BaseModel):
    """Classification of the dialect of an Arabic text segment."""

    primary_dialect: ArabicDialect
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_markers: list[str] = Field(default_factory=list)
    normalized_equivalents: dict[str, str] = Field(
        default_factory=dict,
        description="dialectal term → MSA equivalent",
    )
    is_msa: bool = False


class DialectTerm(BaseModel):
    """A dialectal Islamic term and its mapping to MSA and references."""

    id: str
    term: str = Field(..., description="The dialectal term as written")
    dialect: ArabicDialect
    msa_equivalent: str = Field(..., description="The MSA term the dialectal form maps to")
    transliteration: str = Field("", description="Latin-script transliteration of the term")
    english_equivalent: str = Field("", description="English gloss")
    category: str = Field("general", description="Thematic category, e.g. worship, jurisprudence")
    variants: list[str] = Field(default_factory=list, description="Spelling/regional variants")
    notes: str | None = Field(None, description="Usage notes, e.g. register or region")


class DialectAnalysisResult(BaseModel):
    """Aggregated dialect analysis of an Arabic text."""

    original_text: str
    normalized_msa: str = Field(..., description="Text with dialectal terms mapped to MSA")
    dialect: DialectProfile
    detected_terms: list[DialectTerm] = Field(default_factory=list)
    segments: list[DialectSegment] = Field(default_factory=list)
    processing_time_ms: float = 0.0


class DialectSegment(BaseModel):
    """A contiguous span of text classified to one dialect."""

    text: str
    dialect: ArabicDialect
    confidence: float = Field(..., ge=0.0, le=1.0)


class DialectNormalizeRequest(BaseModel):
    """Request schema for /arabic-dialect/normalize."""

    text: str = Field(..., min_length=1)
    preserve_dialect: bool = False


class DialectNormalizeResponse(BaseModel):
    """Response schema for /arabic-dialect/normalize."""

    original_text: str
    normalized_msa: str
    replaced_terms: dict[str, str] = Field(default_factory=dict)


class DialectAnalyzeRequest(BaseModel):
    """Request schema for /arabic-dialect/analyze."""

    text: str = Field(..., min_length=1)
    target_dialect: ArabicDialect | None = Field(None, description="Optional desired response dialect")


DialectAnalysisResult.model_rebuild()
