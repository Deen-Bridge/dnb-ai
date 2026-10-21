"""Tests for the unified ``languages/`` package grouping (#417)."""

from __future__ import annotations

import languages
from languages import (
    arabic,
    arabic_dialect,
    arabic_ocr,
    calligraphy,
    calligraphy_ocr,
    swahili,
    swahili_processor,
)


def test_languages_package_exposes_all_language_submodules() -> None:
    for mod in (
        arabic,
        arabic_ocr,
        swahili_processor,
        calligraphy,
        calligraphy_ocr,
        arabic_dialect,
        swahili,
    ):
        assert mod is not None
    assert hasattr(arabic_ocr, "router")
    assert hasattr(swahili_processor, "router")
    assert hasattr(calligraphy, "router")
    assert hasattr(calligraphy_ocr, "router")
    assert "arabic_ocr" in languages.__all__
