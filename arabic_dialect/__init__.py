"""Arabic Dialect Support Package (#136).

Subsystem for recognizing and processing Egyptian, Gulf (Khaleeji), and
Levantine (Shami) Arabic dialect features in Islamic-context queries, and for
normalizing dialectal terminology to Modern Standard Arabic.
"""

from arabic_dialect.dialects import dialect_classifier
from arabic_dialect.models import (
    ArabicDialect,
    DialectAnalysisResult,
    DialectProfile,
    DialectSegment,
    DialectTerm,
)
from arabic_dialect.router import router
from arabic_dialect.terminology import extract_terms_from_text, terminology_db


def analyze_arabic_dialect(text: str) -> DialectAnalysisResult:
    """Convenience helper: full dialect analysis of an Arabic text string."""
    profile = dialect_classifier.classify_dialect(text)
    terms = extract_terms_from_text(text)
    normalized_msa, _ = dialect_classifier.normalize_to_msa(text)
    return DialectAnalysisResult(
        original_text=text,
        normalized_msa=normalized_msa,
        dialect=profile,
        detected_terms=terms,
        segments=[],
        processing_time_ms=0.0,
    )


__all__ = [
    "ArabicDialect",
    "DialectAnalysisResult",
    "DialectProfile",
    "DialectSegment",
    "DialectTerm",
    "analyze_arabic_dialect",
    "dialect_classifier",
    "extract_terms_from_text",
    "router",
    "terminology_db",
]
