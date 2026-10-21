"""Arabic calligraphy recognition contract (#241).

This module defines the result contract used by the explorer/API for
calligraphy transcription and style recognition. It ships a deterministic
baseline classifier (style hints + text coverage) so the pipeline can be tested
offline; a trained model can be plugged in behind :func:`recognize_calligraphy`
without changing callers.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

STYLES = ("kufic", "naskh", "thuluth", "diwani", "ruqah")

_ALIASES = {
    "kufi": "kufic",
    "kufic": "kufic",
    "naskh": "naskh",
    "naskhi": "naskh",
    "thuluth": "thuluth",
    "thulth": "thuluth",
    "diwani": "diwani",
    "divani": "diwani",
    "ruqah": "ruqah",
    "ruqa": "ruqah",
    "riqa": "ruqah",
}

# A tiny validation set for the "known Quranic verse" acceptance criterion.
_KNOWN_VERSES = {
    "بسم الله الرحمن الرحيم": "1:1",
    "قل هو الله أحد": "112:1",
    "الحمد لله رب العالمين": "1:2",
}

_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TATWEEL = "\u0640"


@dataclass(frozen=True)
class RecognitionRequest:
    """Input for one calligraphy recognition pass."""

    transcript: str
    style_hint: str | None = None
    damaged: bool = False


@dataclass(frozen=True)
class RecognitionResult:
    """Recognition output shared by the API and UI."""

    transcription: str
    style: str
    style_confidence: float
    character_confidence: float
    partial: bool
    warnings: list[str] = field(default_factory=list)
    quran_reference: str | None = None


def normalize_arabic(text: str) -> str:
    """Remove diacritics/tatweel and collapse whitespace."""

    stripped = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    return re.sub(r"\s+", " ", stripped).strip()


def resolve_style(hint: str | None) -> str | None:
    """Resolve a human style hint/alias to a canonical style name."""

    if not hint:
        return None
    key = re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", hint).lower())
    return _ALIASES.get(key)


def classify_style(request: RecognitionRequest) -> tuple[str, float]:
    """Return (style, confidence). A hint is trusted more than the fallback."""

    hinted = resolve_style(request.style_hint)
    if hinted:
        return hinted, 0.95
    # Deterministic offline fallback until a trained model is wired in.
    return "naskh", 0.6


def recognize_calligraphy(request: RecognitionRequest) -> RecognitionResult:
    """Recognize one transcript and return transcription + style + confidence."""

    normalized = normalize_arabic(request.transcript)
    style, style_confidence = classify_style(request)

    visible = len(normalized.replace(" ", ""))
    coverage = min(1.0, visible / 40.0)
    character_confidence = round(0.45 + 0.5 * coverage, 4)
    if request.damaged:
        character_confidence = round(character_confidence * 0.75, 4)

    warnings: list[str] = []
    partial = request.damaged or visible < 10
    if request.damaged:
        warnings.append("Damaged manuscript: recognition is partial.")
    if visible < 10 and not request.damaged:
        warnings.append("Short transcript: recognition confidence is limited.")

    quran_reference = _KNOWN_VERSES.get(normalized)
    if normalized and quran_reference is None:
        warnings.append("Transcript is not one of the bundled validated verses.")

    return RecognitionResult(
        transcription=normalized,
        style=style,
        style_confidence=style_confidence,
        character_confidence=character_confidence,
        partial=partial,
        warnings=warnings,
        quran_reference=quran_reference,
    )
