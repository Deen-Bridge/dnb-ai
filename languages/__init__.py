"""Unified ``languages`` package grouping Arabic, Swahili, and Calligraphy modules (#417).

Provides structured package-level imports (`languages.arabic`,
`languages.arabic_ocr`, `languages.swahili_processor`,
`languages.calligraphy`, `languages.calligraphy_ocr`,
`languages.arabic_dialect`, and `languages.swahili`) while maintaining full
backwards compatibility with root-level module imports.
"""

from __future__ import annotations

import arabic as arabic
import arabic_dialect as arabic_dialect
import arabic_ocr as arabic_ocr
import calligraphy as calligraphy
import calligraphy_ocr as calligraphy_ocr
import swahili as swahili
import swahili_processor as swahili_processor

__all__ = [
    "arabic",
    "arabic_dialect",
    "arabic_ocr",
    "calligraphy",
    "calligraphy_ocr",
    "swahili",
    "swahili_processor",
]
