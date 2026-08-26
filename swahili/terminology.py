"""Swahili Islamic Terminology Engine — indexes, searches, and maps Islamic vocabulary."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from swahili.models import IslamicDomain, IslamicTerm

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "swahili_islamic_terms.json"


class SwahiliIslamicTerminology:
    """Manages the Swahili Islamic terminology database with multi-index lookups."""

    def __init__(self, data_file: Path = DATA_PATH) -> None:
        self.data_file = data_file
        self.terms: list[IslamicTerm] = []
        self._by_id: dict[str, IslamicTerm] = {}
        self._by_swahili_term: dict[str, IslamicTerm] = {}
        self._by_arabic: dict[str, IslamicTerm] = {}
        self._by_transliteration: dict[str, IslamicTerm] = {}
        self._by_variant: dict[str, IslamicTerm] = {}
        self._by_category: dict[IslamicDomain, list[IslamicTerm]] = {cat: [] for cat in IslamicDomain}
        self._normalized_lookup: dict[str, IslamicTerm] = {}
        self._load_database()

    def _normalize_key(self, text: str) -> str:
        """Normalize text for invariant case/punctuation dictionary keys."""
        cleaned = text.strip().lower()
        cleaned = re.sub(r"[\'`\"\-_]", "", cleaned)
        return cleaned

    def _load_database(self) -> None:
        """Load Islamic terms JSON into memory and build secondary indices."""
        if not self.data_file.exists():
            logger.warning("Swahili terminology file %s does not exist.", self.data_file)
            return

        try:
            with open(self.data_file, encoding="utf-8") as f:
                data = json.load(f)

            raw_terms = data.get("terms", [])
            for item in raw_terms:
                try:
                    category = IslamicDomain(item.get("category", "ibada"))
                except ValueError:
                    category = IslamicDomain.IBADA

                term = IslamicTerm(
                    id=item["id"],
                    swahili_term=item["swahili_term"],
                    arabic_original=item["arabic_original"],
                    arabic_transliteration=item["arabic_transliteration"],
                    english_equivalent=item["english_equivalent"],
                    category=category,
                    definition_sw=item["definition_sw"],
                    definition_en=item["definition_en"],
                    variants_sw=item.get("variants_sw", []),
                    dialect_notes=item.get("dialect_notes"),
                    common_misspellings=item.get("common_misspellings", []),
                    related_terms=item.get("related_terms", []),
                )

                self.terms.append(term)
                self._by_id[term.id] = term
                self._by_category[term.category].append(term)

                # Primary swahili term index
                sw_key = self._normalize_key(term.swahili_term)
                self._by_swahili_term[sw_key] = term
                self._normalized_lookup[sw_key] = term

                # Arabic script index
                ar_key = self._normalize_key(term.arabic_original)
                self._by_arabic[ar_key] = term
                self._normalized_lookup[ar_key] = term

                # Arabic transliteration index
                trans_key = self._normalize_key(term.arabic_transliteration)
                self._by_transliteration[trans_key] = term
                self._normalized_lookup[trans_key] = term

                # Variants index
                for variant in term.variants_sw:
                    v_key = self._normalize_key(variant)
                    self._by_variant[v_key] = term
                    self._normalized_lookup[v_key] = term

                # Common misspellings index
                for misspelling in term.common_misspellings:
                    m_key = self._normalize_key(misspelling)
                    self._normalized_lookup[m_key] = term

            logger.info("Loaded %d Swahili Islamic terms from %s", len(self.terms), self.data_file)
        except Exception:
            logger.exception("Failed to load Swahili terminology")

    def get_term_by_id(self, term_id: str) -> IslamicTerm | None:
        """Retrieve term by its unique ID."""
        return self._by_id.get(term_id)

    def lookup_term(self, text: str) -> IslamicTerm | None:
        """Direct lookup of a term using normalized string matching."""
        key = self._normalize_key(text)
        return self._normalized_lookup.get(key)

    def search_terms(
        self,
        query: str | None = None,
        category: IslamicDomain | None = None,
        limit: int = 50,
    ) -> list[IslamicTerm]:
        """Search Islamic terms with optional text query and category filtering."""
        results = self.terms

        if category:
            results = self._by_category.get(category, [])

        if not query or not query.strip():
            return results[:limit]

        normalized_query = self._normalize_key(query)
        matched: list[tuple[int, IslamicTerm]] = []

        for term in results:
            score = 0
            sw_norm = self._normalize_key(term.swahili_term)
            ar_norm = self._normalize_key(term.arabic_original)
            trans_norm = self._normalize_key(term.arabic_transliteration)
            en_norm = self._normalize_key(term.english_equivalent)

            # Exact matches get highest priority
            if normalized_query in (sw_norm, ar_norm, trans_norm):
                score += 100
            elif normalized_query in en_norm:
                score += 80
            elif normalized_query in [self._normalize_key(v) for v in term.variants_sw]:
                score += 90
            elif (
                normalized_query in sw_norm
                or normalized_query in trans_norm
                or normalized_query in self._normalize_key(term.definition_sw)
                or normalized_query in self._normalize_key(term.definition_en)
            ):
                score += 50

            if score > 0:
                matched.append((score, term))

        matched.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in matched[:limit]]

    def extract_terms_from_text(self, text: str) -> list[IslamicTerm]:
        """Scan input text and extract all recognized Islamic terms."""
        found_terms: dict[str, IslamicTerm] = {}
        normalized_words = re.findall(r"[\w']+", text.lower())

        # Check multi-word terms first (e.g. "zaka ya fitri", "uchaji mungu", "qur'ani tukufu", "kadhi mkuu")
        lower_text = text.lower()
        for term in self.terms:
            sw_lower = term.swahili_term.lower()
            if " " in sw_lower and sw_lower in lower_text:
                found_terms[term.id] = term
            for variant in term.variants_sw:
                v_lower = variant.lower()
                if " " in v_lower and v_lower in lower_text:
                    found_terms[term.id] = term

        # Single word term matching
        for word in normalized_words:
            matched = self.lookup_term(word)
            if matched:
                found_terms[matched.id] = matched

        return list(found_terms.values())


# Global singleton instance
terminology_db = SwahiliIslamicTerminology()
