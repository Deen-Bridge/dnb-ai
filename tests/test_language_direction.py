"""Unit tests for the language direction contract (#878, #879).

Offline: no browser, no network, no provider call.
"""

from __future__ import annotations

import pytest

from language_direction import direction_for, dominant_script

ARABIC = "الصبر مفتاح الفرج"
URDU = "صبر کامیابی کی کنجی ہے"
PERSIAN = "صبر کلید پیروزی است"
JAWI = "صلاة dan puasa"
ENGLISH = "Patience is the key to relief"
SWAHILI_ENGLISH = "Subira ni ufunguo wa faraja, and patience is a virtue"


@pytest.mark.parametrize(
    ("language", "text", "expected"),
    [
        ("ar", ARABIC, "rtl"),
        ("ur", URDU, "rtl"),
        ("fa", PERSIAN, "rtl"),
        ("ar-SA", ARABIC, "rtl"),
        ("ur-PK", URDU, "rtl"),
        ("fa-IR", PERSIAN, "rtl"),
        ("ms", JAWI, "rtl"),
        ("id", JAWI, "rtl"),
        ("ms", "Solat dan puasa", "ltr"),
        ("id", "Shalat dan puasa", "ltr"),
        ("en", ENGLISH, "ltr"),
        ("sw", "Subira ni ufunguo wa faraja", "ltr"),
        ("sw", SWAHILI_ENGLISH, "ltr"),
        ("fr", "La patience est une vertu", "ltr"),
        ("tl", "Ang pasensya ay susi", "ltr"),
    ],
)
def test_direction_for_language(language: str, text: str, expected: str) -> None:
    assert direction_for(language, text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (ARABIC, "rtl"),
        (URDU, "rtl"),
        (PERSIAN, "rtl"),
        (JAWI, "ltr"),  # auto/unknown follows the dominant Latin script
        (ENGLISH, "ltr"),
        (SWAHILI_ENGLISH, "ltr"),
        ("", "ltr"),
        (None, "ltr"),
        ("1234 !!!", "ltr"),
    ],
)
def test_auto_and_unknown_language_use_dominant_script(text: str | None, expected: str) -> None:
    assert direction_for(None, text) == expected
    assert direction_for("zz", text) == expected


def test_dominant_script_detects_mixed_text() -> None:
    assert dominant_script("الصبر sabr") == "arabic"
    assert dominant_script("sabr is patience الصبر") == "latin"
    assert dominant_script("") == "none"
    assert dominant_script(None) == "none"


def test_case_and_whitespace_insensitive_language_code() -> None:
    assert direction_for(" AR ") == "rtl"
    assert direction_for("UR-pk") == "rtl"
    assert direction_for("MS", JAWI) == "rtl"
    assert direction_for("EN-gb") == "ltr"
