"""Direction metadata for chat answers (#878).

Clients should not have to guess whether to render a response with
``dir="rtl"`` or ``dir="ltr"``. This module is the single typed helper used by
the API contract and by tests. It uses the effective language when it is known
and otherwise falls back to the dominant script of the answer text.

Arabic, Urdu and Persian are always right-to-left. Malay and Indonesian are
left-to-right in Latin script but right-to-left when written in Jawi
(Arabic script). Everything else is left-to-right.
"""

from __future__ import annotations

from typing import Literal

Direction = Literal["ltr", "rtl"]
Script = Literal["arabic", "latin", "none"]

RTL_LANGUAGES: frozenset[str] = frozenset({"ar", "ur", "fa"})
JAWI_CAPABLE_LANGUAGES: frozenset[str] = frozenset({"ms", "id"})
# The advertised set. Kept here so unknown codes can be distinguished from
# known left-to-right languages; tests assert it equals SUPPORTED_LANGUAGES.
KNOWN_LANGUAGES: frozenset[str] = frozenset({"ar", "en", "ur", "ms", "fr", "tr", "id", "bn", "fa", "ha", "sw", "tl"})

# Arabic, Arabic Supplement, Arabic Extended-A/B, and the Arabic presentation
# forms used by Jawi and other Arabic-script orthographies.
_ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def _base_language(language: str | None) -> str:
    if not language:
        return ""
    return language.strip().lower().split("-", 1)[0]


def _is_arabic_script(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in _ARABIC_RANGES)


def dominant_script(text: str | None) -> Script:
    """Return the dominant script of ``text`` (Arabic, Latin, or none)."""
    if not text:
        return "none"
    arabic = sum(1 for char in text if _is_arabic_script(char))
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    if arabic == 0 and latin == 0:
        return "none"
    return "arabic" if arabic >= latin else "latin"


def direction_for(language: str | None, text: str | None = None) -> Direction:
    """Resolve the text direction for an effective language and optional text.

    ``language`` is the effective BCP-47 code (``None`` means auto-detect).
    When the language is unknown, the response text's dominant script decides.
    Empty or script-neutral text defaults to ``"ltr"``.
    """
    base = _base_language(language)
    if base in RTL_LANGUAGES:
        return "rtl"
    if base in JAWI_CAPABLE_LANGUAGES:
        if not text:
            return "ltr"
        arabic = sum(1 for char in text if _is_arabic_script(char))
        latin = sum(1 for char in text if char.isascii() and char.isalpha())
        # A Latin-script Malay answer may quote Arabic. Treat it as Jawi only
        # when Arabic script is a substantial share of the text.
        return "rtl" if arabic > 0 and arabic >= latin / 3 else "ltr"
    if base in KNOWN_LANGUAGES:
        return "ltr"
    return "rtl" if dominant_script(text) == "arabic" else "ltr"
