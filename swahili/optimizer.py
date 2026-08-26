"""Swahili Query Optimizer for Cross-Lingual & Islamic Corpus Retrieval."""

from __future__ import annotations

import logging
import re
from typing import Any

from swahili.dialects import dialect_classifier
from swahili.loanwords import loanword_analyzer
from swahili.terminology import terminology_db

logger = logging.getLogger(__name__)


class SwahiliQueryOptimizer:
    """Enhances Swahili search queries by generating multi-lingual Islamic search keys."""

    def __init__(self) -> None:
        self._term_db = terminology_db
        self._loanwords = loanword_analyzer
        self._dialects = dialect_classifier

    def optimize_query_for_retrieval(self, query: str) -> dict[str, Any]:
        """Generate expanded search terms across Swahili, Arabic, and English."""
        normalized_swahili, replaced_dialect = self._dialects.normalize_to_standard(query)
        detected_terms = self._term_db.extract_terms_from_text(normalized_swahili)
        loanword_matches = self._loanwords.extract_loanwords(normalized_swahili)

        arabic_terms: list[str] = []
        arabic_transliterations: list[str] = []
        english_equivalents: list[str] = []
        categories: set[str] = set()

        for term in detected_terms:
            arabic_terms.append(term.arabic_original)
            arabic_transliterations.append(term.arabic_transliteration)
            english_equivalents.append(term.english_equivalent)
            categories.add(term.category.value)

        for loan in loanword_matches:
            if loan.arabic_original not in arabic_terms:
                arabic_terms.append(loan.arabic_original)
            if loan.arabic_transliteration not in arabic_transliterations:
                arabic_transliterations.append(loan.arabic_transliteration)
            categories.add(loan.category)

        # Keyword extraction
        clean_words = [
            w.lower()
            for w in re.findall(r"[\w']+", normalized_swahili)
            if len(w) > 2
            and w.lower() not in {"hiki", "huyu", "yule", "gani", "nini", "wapi", "lini", "kwa", "katika", "yake"}
        ]

        # Combine into unified search string for vector and keyword search
        expanded_search_tokens = clean_words + [t.lower() for t in arabic_transliterations] + english_equivalents

        return {
            "original_query": query,
            "normalized_swahili": normalized_swahili,
            "dialect_replacements": replaced_dialect,
            "detected_categories": list(categories),
            "arabic_keywords": arabic_terms,
            "arabic_transliterations": arabic_transliterations,
            "english_keywords": english_equivalents,
            "expanded_search_string": " ".join(dict.fromkeys(expanded_search_tokens)),
        }


# Global singleton instance
swahili_query_optimizer = SwahiliQueryOptimizer()
