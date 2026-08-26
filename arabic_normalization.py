"""Multi-level Arabic text normalization (#191).

Why this exists
---------------
Arabic normalization used to live in three unrelated ad-hoc copies:
``verifier.normalize_arabic`` (tashkeel strip + alef folding),
``arabic_ocr.normalize_arabic`` (alef/hamza/ta-marbuta folding for OCR
output), and ``audio_hadith.normalize_text`` (tashkeel strip for transcript
matching). Each implemented its own subset of rules, none could preserve
theologically significant diacritics, and none reported what it had done.

This module is the single canonical pipeline. It is offline and deterministic,
built on precompiled Unicode-range regexes so a normal call is microseconds:

- **Multi-level**: ``NONE`` (identity), ``LIGHT`` (NFC, tatweel removal,
  whitespace collapse — safe to display), ``FULL`` (search-grade folding:
  tashkeel, alef variants, hamza carriers, alef maksura, ta marbuta).
- **Context-aware / Quranic preservation**: ``preserve="auto"`` detects
  heavily-diacritized or marked-up text (ayah ornaments, basmala, honorifics)
  and keeps its diacritics — stripping harakat from Quranic text can change
  meaning (e.g. يُضِلّ vs يُضلّ readings) — while still applying letter folding.
- **Configurable**: every rule group can be toggled through
  ``NormalizationConfig`` without subclassing.
- **Quality metrics**: ``normalize_with_metrics`` reports diacritic density
  before/after and the share of letters changed, so callers can monitor
  normalization impact instead of guessing.

Search recall improves because ``مُحَمَّد`` and ``محمد`` (and ``إسلام`` /
``اسلام`` / ``الإسلام`` stems) collapse to one key; precision is protected by
the Quranic-preservation path.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Unicode character classes (precompiled once)
# ---------------------------------------------------------------------------

# Harakat and Quranic annotation marks: fathatan..sukun, superscript alef,
# small high marks. Mirrors verifier.TASHKEEL_REGEX but complete.
TASHKEEL_PATTERN = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]")

TATWEEL_PATTERN = re.compile(r"\u0640")

# Alef variants: hamza above/below, madda, wasla -> plain alef.
ALEF_VARIANTS_PATTERN = re.compile(r"[\u0623\u0625\u0622\u0671]")

# Standalone hamza carries no consonantal information in search keys.
STANDALONE_HAMZA_PATTERN = re.compile(r"\u0621")

# Hamza on carrier folds back onto the base letter for loose matching.
HAMZA_ON_WAW_PATTERN = re.compile(r"\u0624")
HAMZA_ON_YA_PATTERN = re.compile(r"\u0626")

# Alef maksura -> ya; final form is orthographic variation only.
ALEF_MAKSURA_PATTERN = re.compile(r"\u0649")

# Ta marbuta -> ha; agreement suffix variation only in search contexts.
TA_MARBUTA_PATTERN = re.compile(r"\u0629")

WHITESPACE_PATTERN = re.compile(r"\s+")

# Arabic punctuation that carries no search value: comma, semicolon,
# question mark, Arabic date separator, ornate parentheses content kept.
ARABIC_PUNCTUATION_PATTERN = re.compile(r"[،؛؟«»\u066D\u060C]")

# ---------------------------------------------------------------------------
# Quranic-text heuristics
# ---------------------------------------------------------------------------

_QURANIC_MARKS = ("\ufdfb", "\ufdfd", "\ufdf2")  # bismillah, sajdah, salawat ligatures
_QURANIC_PATTERNS = (
    re.compile(r"[\u06D6-\u06ED]"),  # Quranic annotation marks (small waqf signs)
    re.compile(r"﴿|﴾"),  # ornate ayah brackets
    re.compile(r"ﷺ|عليه\s*السلام|رضي\s*الله\s*عنه"),  # sacred-name honorifics
)


def diacritic_density(text: str) -> float:
    """Share of Arabic letter positions carrying a tashkeel mark."""
    letters = sum(1 for ch in text if "\u0621" <= ch <= "\u064a")
    if letters == 0:
        return 0.0
    marks = len(TASHKEEL_PATTERN.findall(text))
    return round(marks / letters, 4)


def looks_like_quranic(text: str) -> bool:
    """Heuristic: heavy diacritization or explicit Quranic markup.

    A diacritic density above 0.25 is far beyond ordinary prose (even fully
    vocalized news text rarely exceeds ~0.15) and strongly suggests Quranic
    or liturgical material whose marks must not be stripped silently.
    """
    if any(mark in text for mark in _QURANIC_MARKS):
        return True
    if any(pattern.search(text) for pattern in _QURANIC_PATTERNS):
        return True
    return diacritic_density(text) > 0.25


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class NormalizationLevel(str, Enum):
    """How aggressive the pipeline is.

    - ``NONE``: return input unchanged.
    - ``LIGHT``: display-safe — NFC, tatweel, whitespace, stray punctuation.
      Diacritics and letter forms untouched.
    - ``FULL``: search-grade — everything in LIGHT plus tashkeel removal and
      letter-form folding (alef variants, hamza carriers, alef maksura,
      ta marbuta, standalone hamza).
    """

    NONE = "none"
    LIGHT = "light"
    FULL = "full"


@dataclass(frozen=True)
class NormalizationConfig:
    """Toggle individual rule groups without new pipeline code."""

    level: NormalizationLevel = NormalizationLevel.FULL
    preserve: str = "never"  # "never" | "auto" | "always"
    fold_alef_variants: bool = True
    fold_hamza_carriers: bool = True
    fold_alef_maksura: bool = True
    fold_ta_marbuta: bool = True
    remove_standalone_hamza: bool = True


DEFAULT_CONFIG = NormalizationConfig()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _apply_light(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = TATWEEL_PATTERN.sub("", text)
    text = ARABIC_PUNCTUATION_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _apply_full(text: str, config: NormalizationConfig, preserve_diacritics: bool) -> str:
    text = _apply_light(text)
    if not preserve_diacritics:
        text = TASHKEEL_PATTERN.sub("", text)
    if config.fold_alef_variants:
        text = ALEF_VARIANTS_PATTERN.sub("\u0627", text)
    if config.remove_standalone_hamza:
        text = STANDALONE_HAMZA_PATTERN.sub("", text)
    if config.fold_hamza_carriers:
        text = HAMZA_ON_WAW_PATTERN.sub("\u0648", text)
        text = HAMZA_ON_YA_PATTERN.sub("\u064a", text)
    if config.fold_alef_maksura:
        text = ALEF_MAKSURA_PATTERN.sub("\u064a", text)
    if config.fold_ta_marbuta:
        text = TA_MARBUTA_PATTERN.sub("\u0647", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_arabic_text(
    text: str | None,
    config: NormalizationLevel | NormalizationConfig | None = None,
) -> str:
    """Normalize *text* according to ``config``. Never raises on odd input.

    ``config`` accepts a bare level for convenience or a full
    ``NormalizationConfig``. With ``preserve="auto"``, Quranic-looking input
    keeps its diacritics even at FULL level (letter folding still applies);
    with ``preserve="always"`` nothing vocalized is ever stripped.
    """
    if not text:
        return ""

    if config is None:
        resolved = DEFAULT_CONFIG
    elif isinstance(config, NormalizationLevel):
        resolved = NormalizationConfig(level=config)
    else:
        resolved = config

    if resolved.level is NormalizationLevel.NONE:
        return text

    if resolved.level is NormalizationLevel.LIGHT:
        return _apply_light(text)

    preserve_diacritics = resolved.preserve == "always" or (resolved.preserve == "auto" and looks_like_quranic(text))
    return _apply_full(text, resolved, preserve_diacritics)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


@dataclass
class NormalizationMetrics:
    """What normalization did to a piece of text."""

    original_chars: int = 0
    normalized_chars: int = 0
    diacritics_removed: int = 0
    letters_changed: int = 0
    diacritic_density_before: float = 0.0
    diacritic_density_after: float = 0.0
    preserved_quranic: bool = False
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def change_ratio(self) -> float:
        """Share of original characters affected by normalization."""
        if self.original_chars == 0:
            return 0.0
        return round(self.letters_changed / self.original_chars, 4)


def normalize_with_metrics(
    text: str | None,
    config: NormalizationLevel | NormalizationConfig | None = None,
) -> tuple[str, NormalizationMetrics]:
    """Normalize and report exactly what happened, for monitoring pipelines."""
    source = text or ""
    normalized = normalize_arabic_text(source, config)

    resolved: NormalizationConfig
    if isinstance(config, NormalizationConfig):
        resolved = config
    else:
        resolved = NormalizationConfig(level=config or NormalizationLevel.FULL)

    preserved = resolved.level is NormalizationLevel.FULL and (
        resolved.preserve == "always" or (resolved.preserve == "auto" and looks_like_quranic(source))
    )

    before_marks = len(TASHKEEL_PATTERN.findall(source))
    after_marks = len(TASHKEEL_PATTERN.findall(normalized))
    after_letters = sum(1 for ch in normalized if "\u0621" <= ch <= "\u064a")

    metrics = NormalizationMetrics(
        original_chars=len(source),
        normalized_chars=len(normalized),
        diacritics_removed=max(0, before_marks - after_marks),
        diacritic_density_before=diacritic_density(source),
        diacritic_density_after=round(after_marks / after_letters, 4) if after_letters else 0.0,
        preserved_quranic=preserved,
    )
    metrics.letters_changed = max(0, len(source) - len(normalized)) + abs(before_marks - after_marks)
    return normalized, metrics
