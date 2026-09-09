"""Facade module for Swahili language processing subsystem.

Re-exports core engines, models, and singleton processors for root-level access.
"""

from __future__ import annotations

from swahili.code_switching import code_switch_processor
from swahili.cultural_context import cultural_context_engine
from swahili.dialects import dialect_classifier
from swahili.generator import swahili_response_enhancer
from swahili.loanwords import loanword_analyzer
from swahili.models import (
    CodeSwitchResult,
    CodeSwitchType,
    CulturalContext,
    DialectResult,
    IslamicDomain,
    IslamicTerm,
    LoanwordMatch,
    SwahiliAnalysisResult,
    SwahiliDialect,
    SwahiliPromptEnhancement,
    SwahiliToken,
)
from swahili.optimizer import swahili_query_optimizer
from swahili.router import router
from swahili.terminology import terminology_db


def analyze_swahili(text: str) -> SwahiliAnalysisResult:
    """Convenience helper to analyze a Swahili text string."""
    tokens = loanword_analyzer.tokenize_swahili(text)
    detected_terms = terminology_db.extract_terms_from_text(text)
    loanwords = loanword_analyzer.extract_loanwords(text)
    dialect = dialect_classifier.classify_dialect(text)
    code_switch = code_switch_processor.analyze_code_switching(text)
    cultural = cultural_context_engine.extract_context(text)
    normalized_text, _ = dialect_classifier.normalize_to_standard(text)

    return SwahiliAnalysisResult(
        original_text=text,
        normalized_text=normalized_text,
        tokens=tokens,
        detected_terms=detected_terms,
        loanwords=loanwords,
        dialect=dialect,
        code_switch=code_switch,
        cultural_context=cultural,
        processing_time_ms=0.0,
    )


__all__ = [
    "CodeSwitchResult",
    "CodeSwitchType",
    "CulturalContext",
    "DialectResult",
    "IslamicDomain",
    "IslamicTerm",
    "LoanwordMatch",
    "SwahiliAnalysisResult",
    "SwahiliDialect",
    "SwahiliPromptEnhancement",
    "SwahiliToken",
    "analyze_swahili",
    "code_switch_processor",
    "cultural_context_engine",
    "dialect_classifier",
    "loanword_analyzer",
    "router",
    "swahili_query_optimizer",
    "swahili_response_enhancer",
    "terminology_db",
]
