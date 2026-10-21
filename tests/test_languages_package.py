"""Tests for the unified ``languages/`` package grouping (#417)."""

from __future__ import annotations

import languages
from languages import (
    arabic_dialect,
    arabic_ocr,
    calligraphy,
    swahili_processor,
    urdu_processor,
)


def test_languages_package_exposes_all_language_submodules() -> None:
    for mod in (
        arabic_ocr,
        swahili_processor,
        urdu_processor,
        calligraphy,
        arabic_dialect,
    ):
        assert mod is not None
    assert hasattr(arabic_ocr, "router")
    assert hasattr(swahili_processor, "router")
    assert hasattr(urdu_processor, "router")
    assert hasattr(calligraphy, "router")
    assert "arabic_ocr" in languages.__all__
