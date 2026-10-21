"""Unified ``languages`` package grouping Arabic, Swahili, Urdu, Malay, and Calligraphy modules (#417).

Provides structured package-level imports (`languages.arabic_ocr`,
`languages.swahili_processor`, `languages.urdu_processor`,
`languages.calligraphy`, `languages.arabic_dialect`, `languages.malay`,
and `languages.swahili`) while maintaining full backwards compatibility with
root-level module imports.
"""

from __future__ import annotations

import malay as malay
import urdu_processor as urdu_processor

import arabic_dialect as arabic_dialect
import arabic_ocr as arabic_ocr
import calligraphy as calligraphy
import swahili as swahili
import swahili_processor as swahili_processor

__all__ = [
    "arabic_dialect",
    "arabic_ocr",
    "calligraphy",
    "malay",
    "swahili",
    "swahili_processor",
    "urdu_processor",
]
