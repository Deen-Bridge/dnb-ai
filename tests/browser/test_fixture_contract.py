"""Validate the committed browser fixtures against the ChatResponse contract (#878).

This file needs no browser, so it runs even when Playwright is unavailable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from main import SUPPORTED_LANGUAGES

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda path: path.stem)
def test_browser_fixture_matches_chat_response_contract(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["direction"] in {"ltr", "rtl"}
    assert payload["language"] in SUPPORTED_LANGUAGES
    assert payload["requested_language"] == payload["language"]
    assert isinstance(payload["response"], str) and payload["response"]
    for citation in payload["citations"]:
        assert re.fullmatch(r"\d{1,3}:\d{1,3}(-\d{1,3})?", citation["reference"])
        assert citation["quote"]
        assert citation["quote_direction"] in {"rtl", "ltr"}
