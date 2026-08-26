"""Regional Swahili Dialect & Variation Engine for East Africa."""

from __future__ import annotations

import logging
import re
from typing import Any

from swahili.models import DialectResult, SwahiliDialect

logger = logging.getLogger(__name__)

# Markers characteristic of specific regional varieties
DIALECT_MARKERS: dict[SwahiliDialect, dict[str, Any]] = {
    SwahiliDialect.PWANI_MVITA: {
        "words": {
            "kuvua": "kutawadha",
            "mvyee": "mzee",
            "choo cha kule": "choo",
            "chuo": "madrasa",
            "mwanzi": "msikiti",
            "nduu": "ndugu",
            "nyuni": "ndege",
            "mahare": "mahari",
        },
        "patterns": [
            r"\bkuvua\b",
            r"\bchoo cha kule\b",
            r"\bmvyee\b",
            r"\bnduu yangu\b",
        ],
        "is_coastal": True,
    },
    SwahiliDialect.PWANI_AMU: {
        "words": {
            "yuzi": "kutawadha",
            "kuteua": "kuchagua",
            "muweneji": "mwenyeji",
            "ziumbe": "viumbe",
            "moliwa": "mola wetu",
            "kwelea": "kuelewa",
            "mtumwa": "mtume",
        },
        "patterns": [
            r"\byuzi\b",
            r"\bmoliwa\b",
            r"\bziumbe\b",
            r"\bmaulidi ya amu\b",
        ],
        "is_coastal": True,
    },
    SwahiliDialect.PWANI_UNGUJA: {
        "words": {
            "kwa nini basi": "kwa nini",
            "mkebe": "kopo",
            "haja ya": "lazima",
            "shehe": "sheikh",
            "kufuturu": "kula futari",
        },
        "patterns": [
            r"\bkisiwani\b",
            r"\bunguja\b",
            r"\bzenj\b",
            r"\bofisi ya mufti zanzibar\b",
        ],
        "is_coastal": True,
    },
    SwahiliDialect.BARA_INLAND: {
        "words": {
            "kupiga sala": "kuswali",
            "kufunga ramadhani": "kufunga saumu ya ramadhani",
            "kutoa zaka": "kutoa zaka",
            "kufanya maombi": "kuomba dua",
            "kusoma kurani": "kusoma qur'ani",
        },
        "patterns": [
            r"\bkupiga sala\b",
            r"\bmaombi ya kiislamu\b",
            r"\bbakwata\b",
            r"\btanzania bara\b",
            r"\bupcountry\b",
        ],
        "is_coastal": False,
    },
    SwahiliDialect.SHENG_URBAN: {
        "words": {
            "kushika wudhu": "kutawadha",
            "kupiga swala": "kuswali",
            "ku-fast": "kufunga saumu",
            "m-god": "mwenyezi mungu",
            "madhe": "mama",
            "ubao": "njaa ya saumu",
            "shehe": "sheikh",
        },
        "patterns": [
            r"\bkushika wudhu\b",
            r"\bkupiga swala\b",
            r"\bku-fast\b",
            r"\bm-god\b",
            r"\bsheng\b",
        ],
        "is_coastal": False,
    },
}


class SwahiliDialectClassifier:
    """Classifies regional Swahili varieties and normalizes dialect-specific terms."""

    def classify_dialect(self, text: str) -> DialectResult:
        """Analyze text to detect regional dialect features and confidence."""
        lower_text = text.lower()
        scores: dict[SwahiliDialect, float] = {d: 0.0 for d in SwahiliDialect}
        detected_markers: list[str] = []
        normalized_map: dict[str, str] = {}

        for dialect, profile in DIALECT_MARKERS.items():
            # Check pattern matches
            for pat in profile["patterns"]:
                if re.search(pat, lower_text):
                    scores[dialect] += 2.0
                    detected_markers.append(f"pattern:{pat}")

            # Check individual word markers
            for marker_word, standard_form in profile["words"].items():
                if re.search(r"\b" + re.escape(marker_word) + r"\b", lower_text):
                    scores[dialect] += 1.5
                    detected_markers.append(marker_word)
                    normalized_map[marker_word] = standard_form

        # Coastal lexical density indicators
        coastal_keywords = ["sheikh", "ustaadh", "maulidi", "kibarazani", "futari", "daku", "swalat", "kadhi"]
        coastal_hits = sum(1 for w in coastal_keywords if w in lower_text)
        if coastal_hits >= 2:
            scores[SwahiliDialect.PWANI_UNGUJA] += coastal_hits * 0.5

        # Determine highest scoring dialect
        best_dialect = SwahiliDialect.SANIFU
        highest_score = 0.0

        for dialect, score in scores.items():
            if score > highest_score:
                highest_score = score
                best_dialect = dialect

        if highest_score == 0.0:
            best_dialect = SwahiliDialect.SANIFU
            confidence = 0.90
            is_coastal = False
        else:
            confidence = min(0.98, 0.60 + (highest_score * 0.1))
            is_coastal = DIALECT_MARKERS.get(best_dialect, {}).get("is_coastal", False)

        return DialectResult(
            primary_dialect=best_dialect,
            confidence=round(confidence, 2),
            is_coastal=is_coastal,
            detected_markers=detected_markers,
            normalized_equivalents=normalized_map,
        )

    def normalize_to_standard(self, text: str) -> tuple[str, dict[str, str]]:
        """Convert dialect-specific phrases into Standard Swahili (Kiswahili Sanifu)."""
        normalized_text = text
        replaced: dict[str, str] = {}

        for _dialect, profile in DIALECT_MARKERS.items():
            for dialect_word, standard_word in profile["words"].items():
                pattern = r"\b" + re.escape(dialect_word) + r"\b"
                if re.search(pattern, normalized_text, re.IGNORECASE):
                    normalized_text = re.sub(pattern, standard_word, normalized_text, flags=re.IGNORECASE)
                    replaced[dialect_word] = standard_word

        return normalized_text, replaced


# Global singleton instance
dialect_classifier = SwahiliDialectClassifier()
