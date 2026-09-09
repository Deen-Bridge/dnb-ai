"""Dialectal Islamic Terminology Lexicon (#136).

A curated lexicon mapping dialectal Islamic terms (Egyptian, Gulf, Levantine)
to their Modern Standard Arabic (MSA) equivalents, with transliteration,
English gloss, thematic category, and usage notes. The lexicon is embedded so
the subsystem runs fully offline, mirroring how the concordance and hadith
datasets are bundled. The categories mirror the domains the rest of the
service reasons about (worship, jurisprudence, creed, transactions, ethics).
"""

from __future__ import annotations

import re

from arabic_dialect.models import ArabicDialect, DialectTerm

# id, term, dialect, msa, transliteration, english, category, variants, notes
_RAW_TERMS: list[tuple[str, str, ArabicDialect, str, str, str, str, list[str], str | None]] = [
    # ---- Worship / ritual ----------------------------------------------------
    (
        "eg-jam3",
        "جامع",
        ArabicDialect.EGYPTIAN,
        "مسجد",
        "gāmiʿ",
        "mosque",
        "worship",
        ["الجامع"],
        "Egyptian term for mosque (used for the Friday congregational mosque).",
    ),
    (
        "eg-salat-alfajr",
        "صلاة الفجر",
        ArabicDialect.EGYPTIAN,
        "صلاة الفجر",
        "ṣalāt al-fajr",
        "dawn prayer",
        "worship",
        [],
        "Same term as MSA; kept to anchor the dialect lexicon in worship context.",
    ),
    (
        "eg-zakat",
        "الزكاة",
        ArabicDialect.EGYPTIAN,
        "الزكاة",
        "al-zakāh",
        "obligatory charity",
        "worship",
        [],
        "Pronunciation variant; standard meaning.",
    ),
    (
        "eg-dabh",
        "الذبح",
        ArabicDialect.EGYPTIAN,
        "الأضحية",
        "al-ḍabḥ",
        "sacrificial slaughter",
        "worship",
        ["ذبح العيد"],
        "Egyptian usage for the Eid sacrifice; MSA prefers al-udhiyah.",
    ),
    # ---- Jurisprudence / rulings ---------------------------------------------
    (
        "gulf-mo",
        "مو",
        ArabicDialect.GULF,
        "ليس",
        "mū",
        "is not",
        "jurisprudence",
        ["مب"],
        "Negative copula; 'مو حلال' → 'ليس حلالاً'.",
    ),
    (
        "lev-badi",
        "بدي",
        ArabicDialect.LEVANTINE,
        "أريد",
        "biddī",
        "I want",
        "general",
        ["بدو", "بدها"],
        "Levantine desire/volition verb; 'بدي أعرف الحكم' → 'أريد أن أعرف الحكم'.",
    ),
    # ---- Creed / theology ----------------------------------------------------
    (
        "eg-rabbena",
        "ربنا",
        ArabicDialect.EGYPTIAN,
        "ربنا",
        "rabbinā",
        "our Lord",
        "creed",
        [],
        "Common Egyptian invocation of Allah; same meaning as MSA.",
    ),
    (
        "gulf-alhamdulillah",
        "الحمدلله",
        ArabicDialect.GULF,
        "الحمد لله",
        "al-ḥamdu lillāh",
        "praise be to God",
        "creed",
        ["الحمد لله"],
        "Gulf orthographic variant of the standard phrase.",
    ),
    (
        "lev-yarza",
        "الله يرضى عنك",
        ArabicDialect.LEVANTINE,
        "رضي الله عنك",
        "allāh yarḍā ʿannak",
        "may God be pleased with you",
        "creed",
        [],
        "Levantine blessing formula; mapped to the MSA counterpart.",
    ),
    # ---- Transactions / dealings ---------------------------------------------
    (
        "gulf-wain",
        "وين",
        ArabicDialect.GULF,
        "أين",
        "wain",
        "where",
        "general",
        ["وين"],
        "Gulf interrogative; 'وين الزكاة؟' → 'أين الزكاة؟'.",
    ),
    (
        "lev-mashan",
        "مشان",
        ArabicDialect.LEVANTINE,
        "من أجل",
        "mashān",
        "for the sake of",
        "general",
        ["عشان"],
        "Levantine purposive preposition; 'مشان الله' → 'من أجل الله'.",
    ),
    (
        "eg-3ayez",
        "عايز",
        ArabicDialect.EGYPTIAN,
        "أريد",
        "ʿāyiz",
        "I want",
        "general",
        ["عايزة"],
        "Egyptian desire verb; 'عايز أسأل عن الحكم' → 'أريد أن أسأل عن الحكم'.",
    ),
]


class DialectalTerminology:
    """Lexicon of dialectal Islamic terms with search and lookup."""

    def __init__(self) -> None:
        self.terms: list[DialectTerm] = []
        self._by_id: dict[str, DialectTerm] = {}
        self._by_term: dict[str, DialectTerm] = {}
        self._by_dialect: dict[ArabicDialect, list[DialectTerm]] = {dialect: [] for dialect in ArabicDialect}
        self._load()

    def _load(self) -> None:
        for (
            term_id,
            term,
            dialect,
            msa,
            transliteration,
            english,
            category,
            variants,
            notes,
        ) in _RAW_TERMS:
            entry = DialectTerm(
                id=term_id,
                term=term,
                dialect=dialect,
                msa_equivalent=msa,
                transliteration=transliteration,
                english_equivalent=english,
                category=category,
                variants=variants,
                notes=notes,
            )
            self.terms.append(entry)
            self._by_id[term_id] = entry
            self._by_term[term] = entry
            for variant in variants:
                self._by_term.setdefault(variant, entry)
            self._by_dialect[dialect].append(entry)

    def get_term_by_id(self, term_id: str) -> DialectTerm | None:
        return self._by_id.get(term_id)

    def search_terms(
        self,
        query: str | None = None,
        dialect: ArabicDialect | None = None,
        category: str | None = None,
        limit: int = 50,
    ) -> list[DialectTerm]:
        """Search the lexicon by term, MSA equivalent, transliteration, or gloss."""
        results = list(self.terms)
        if dialect is not None and dialect is not ArabicDialect.UNKNOWN:
            results = [t for t in results if t.dialect is dialect]
        if category:
            results = [t for t in results if t.category == category]
        if query:
            lowered = query.strip().lower()
            results = [
                t
                for t in results
                if lowered in t.term.lower()
                or lowered in t.msa_equivalent.lower()
                or lowered in t.transliteration.lower()
                or lowered in t.english_equivalent.lower()
            ]
        return results[:limit]

    def lookup(self, term: str) -> DialectTerm | None:
        """Exact lookup of a term or variant."""
        return self._by_term.get(term.strip())


terminology_db = DialectalTerminology()


def extract_terms_from_text(text: str) -> list[DialectTerm]:
    """Return the dialectal lexicon terms present in the text."""
    found: list[DialectTerm] = []
    seen: set[str] = set()
    for term in terminology_db.terms:
        for candidate in [term.term, *term.variants]:
            if candidate and re.search(r"\b" + re.escape(candidate) + r"\b", text):
                if term.id not in seen:
                    found.append(term)
                    seen.add(term.id)
                break
    return found
