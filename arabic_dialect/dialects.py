"""Arabic Dialect Identification & Normalization Engine (#136).

Identifies Egyptian, Gulf (Khaleeji), and Levantine (Shami) dialect features
in Arabic text and normalizes dialectal Islamic terminology to Modern
Standard Arabic (MSA) equivalents, so a user writing in a regional variety is
understood against the canonical references the rest of the service uses.

Approach
--------
Deterministic marker matching: each dialect carries a lexicon of distinctive
words/phrases and lightweight patterns. Text is scored per dialect; the
highest-scoring dialect wins with a confidence derived from the score, and a
text with no markers is classified as MSA with high confidence. Normalization
replaces known dialectal forms with their MSA equivalents word by word, so
the rest of the pipeline (tafsir, concordance, chat) receives standard text.

The marker lexicons deliberately focus on Islamic-context vocabulary — the
terms a user reaches for when discussing religious topics in their dialect —
because that is the domain this service operates in; a general-purpose
dialect classifier is out of scope here.
"""

from __future__ import annotations

import re

from arabic_dialect.models import ArabicDialect, DialectProfile

# ---------------------------------------------------------------------------
# Dialect marker lexicons: dialectal term → MSA equivalent
# ---------------------------------------------------------------------------

# Egyptian (اللهجة المصرية)
EGYPTIAN_MARKERS: dict[str, str] = {
    # Religious / Islamic-context vocabulary. Pan-Arabic formulas such as
    # "الحمد لله" are deliberately NOT markers: they are shared by every
    # dialect and MSA, so they carry no dialectal signal.
    "ربنا": "ربنا",  # Our Lord (used broadly)
    "جامع": "مسجد",
    "سيدنا": "سيدنا",
    "الست": "السيدة",
    "النهاردة": "اليوم",
    "امبارح": "أمس",
    "دلوقتي": "الآن",
    "عايز": "أريد",
    "عايزة": "أريد",
    "إزيك": "كيف حالك",
    "عامل إيه": "كيف حالك",
    "كده": "هكذا",
    "ليه": "لماذا",
    "إمتى": "متى",
    "فين": "أين",
    "إزاي": "كيف",
    "آه": "نعم",
    "لأ": "لا",
    "مين": "من",
    "إيه": "ماذا",
    "خالص": "إطلاقا",
    "أوي": "جدًا",
    "بس": "فقط",
    "حاجة": "شيء",
    "واحد": "أحد",
    # Islamic terms in Egyptian pronunciation
    "الزكاة": "الزكاة",
    "الذبح": "الأضحية",
}

# Gulf / Khaleeji (اللهجة الخليجية)
GULF_MARKERS: dict[str, str] = {
    "شنو": "ماذا",
    "وش": "ماذا",
    "شلون": "كيف",
    "وين": "أين",
    "إمتى": "متى",
    "ليش": "لماذا",
    "الحين": "الآن",
    "بس": "فقط",
    "عقب": "بعد",
    "هلا": "أهلا",
    "مب": "ليس",
    "مو": "ليس",
    "أبوي": "أبي",
    "أمي": "أمي",
    "أخوي": "أخي",
    "أختي": "أختي",
    "تونا": "الآن",
    "دي": "هذه",
    "ذا": "هذا",
}

# Levantine / Shami (اللهجة الشامية)
LEVANTINE_MARKERS: dict[str, str] = {
    "شو": "ماذا",
    "كيفك": "كيف حالك",
    "وين": "أين",
    "ليش": "لماذا",
    "إمتى": "متى",
    "هلق": "الآن",
    "بس": "فقط",
    "مشان": "من أجل",
    "عشان": "من أجل",
    "بدي": "أريد",
    "بدك": "تريد",
    "هاد": "هذا",
    "هيدا": "هذا",
    "هيك": "هكذا",
    "مبارح": "أمس",
    "بكرة": "غدا",
    "الله يرضى عنك": "رضي الله عنك",
    "يا شيخ": "يا شيخ",
}

# Patterns that strongly indicate a dialect (matched case-insensitively on the
# normalized Arabic text).
DIALECT_PATTERNS: dict[ArabicDialect, tuple[str, ...]] = {
    ArabicDialect.EGYPTIAN: (
        r"\bعايز\b",
        r"\bإزيك\b",
        r"\bكده\b",
        r"\bدلوقتي\b",
        r"\bامبارح\b",
    ),
    ArabicDialect.GULF: (
        r"\bشنو\b",
        r"\bوش\b",
        r"\bشلون\b",
        r"\bوين\b",
        r"\bليش\b",
        r"\bالحين\b",
    ),
    ArabicDialect.LEVANTINE: (
        r"\bشو\b",
        r"\bكيفك\b",
        r"\bهلق\b",
        r"\bبدي\b",
        r"\bمشان\b",
    ),
}

_MARKERS_BY_DIALECT: dict[ArabicDialect, dict[str, str]] = {
    ArabicDialect.EGYPTIAN: EGYPTIAN_MARKERS,
    ArabicDialect.GULF: GULF_MARKERS,
    ArabicDialect.LEVANTINE: LEVANTINE_MARKERS,
}


class ArabicDialectClassifier:
    """Classifies Arabic text into a regional dialect and normalizes to MSA."""

    def _score_profiles(self, lower_text: str) -> tuple[dict[ArabicDialect, float], list[str], dict[str, str]]:
        scores: dict[ArabicDialect, float] = {
            ArabicDialect.EGYPTIAN: 0.0,
            ArabicDialect.GULF: 0.0,
            ArabicDialect.LEVANTINE: 0.0,
        }
        detected_markers: list[str] = []
        normalized_map: dict[str, str] = {}

        for dialect, markers in _MARKERS_BY_DIALECT.items():
            for dialectal_word, msa_word in markers.items():
                if re.search(r"\b" + re.escape(dialectal_word) + r"\b", lower_text):
                    scores[dialect] += 1.5
                    detected_markers.append(dialectal_word)
                    normalized_map[dialectal_word] = msa_word

        for dialect, patterns in DIALECT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, lower_text):
                    scores[dialect] += 2.0
                    detected_markers.append(f"pattern:{pattern}")

        return scores, detected_markers, normalized_map

    def classify_dialect(self, text: str) -> DialectProfile:
        """Analyze Arabic text to detect regional dialect features."""
        lower_text = text.strip().lower()
        scores, detected_markers, normalized_map = self._score_profiles(lower_text)

        best_dialect = ArabicDialect.MSA
        highest_score = 0.0
        for dialect, score in scores.items():
            if score > highest_score:
                highest_score = score
                best_dialect = dialect

        if highest_score <= 0.0:
            return DialectProfile(
                primary_dialect=ArabicDialect.MSA,
                confidence=0.95,
                detected_markers=[],
                normalized_equivalents={},
                is_msa=True,
            )

        # Confidence scales with marker density, capped for non-MSA dialects.
        confidence = min(0.98, 0.60 + (highest_score * 0.1))
        return DialectProfile(
            primary_dialect=best_dialect,
            confidence=round(confidence, 2),
            detected_markers=detected_markers,
            normalized_equivalents=normalized_map,
            is_msa=False,
        )

    def normalize_to_msa(self, text: str) -> tuple[str, dict[str, str]]:
        """Map dialectal terms in the text to their MSA equivalents.

        Replaces the longest dialectal marker first so multi-word phrases
        (e.g. ``الحمد لله``) are matched before their single-word parts.
        """
        normalized_text = text
        replaced: dict[str, str] = {}
        all_markers: list[tuple[str, str]] = []
        for markers in _MARKERS_BY_DIALECT.values():
            all_markers.extend(markers.items())
        all_markers.sort(key=lambda item: -len(item[0]))

        for dialectal_word, msa_word in all_markers:
            pattern = r"\b" + re.escape(dialectal_word) + r"\b"
            if re.search(pattern, normalized_text, re.IGNORECASE):
                normalized_text = re.sub(pattern, msa_word, normalized_text, flags=re.IGNORECASE)
                replaced[dialectal_word] = msa_word

        return normalized_text, replaced

    def detect_dialect_terms(self, text: str) -> list[str]:
        """Return the dialectal marker words present in the text."""
        lower_text = text.strip().lower()
        found: list[str] = []
        for markers in _MARKERS_BY_DIALECT.values():
            for dialectal_word in markers:
                if re.search(r"\b" + re.escape(dialectal_word) + r"\b", lower_text):
                    if dialectal_word not in found:
                        found.append(dialectal_word)
        return found


# Global singleton instance
dialect_classifier = ArabicDialectClassifier()
