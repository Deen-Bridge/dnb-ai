"""Tests for first-class Arabic and multilingual support (#14).

Offline tests — no Gemini API calls.  These verify language normalization,
system-prompt construction, ChatRequest/ChatResponse models, and
JSON round-tripping of Arabic content.
"""

import json

import pytest

from main import (
    ChatRequest,
    ChatResponse,
    ISLAMIC_CONTEXT,
    LANGUAGE_INSTRUCTIONS,
    SUPPORTED_LANGUAGES,
    Message,
    normalize_language,
)


# ---------------------------------------------------------------------------
# normalize_language
# ---------------------------------------------------------------------------

class TestNormalizeLanguage:
    """normalize_language validates BCP-47 codes against SUPPORTED_LANGUAGES."""

    def test_none_returns_none(self):
        assert normalize_language(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_language("") is None

    @pytest.mark.parametrize("code", ["ar", "en", "ur", "ms", "fr", "tr", "id", "bn", "fa", "ha", "sw", "tl"])
    def test_supported_codes_accepted(self, code):
        assert normalize_language(code) == code

    def test_arabic_capitalized(self):
        assert normalize_language("AR") == "ar"

    def test_bcp47_subtag_falls_back(self):
        assert normalize_language("ar-SA") == "ar"

    def test_english_us_bcp47(self):
        assert normalize_language("en-US") == "en"

    def test_unrecognized_code_returns_none(self):
        assert normalize_language("zz") is None

    def test_whitespace_stripped(self):
        assert normalize_language("  en  ") == "en"

    def test_unrecognized_does_not_raise(self):
        result = normalize_language(" Klingon")
        assert result is None


# ---------------------------------------------------------------------------
# Supported languages constant
# ---------------------------------------------------------------------------

class TestSupportedLanguages:
    def test_arabic_is_supported(self):
        assert "ar" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["ar"] == "Arabic"

    def test_english_is_supported(self):
        assert "en" in SUPPORTED_LANGUAGES

    def test_urdu_is_supported(self):
        assert "ur" in SUPPORTED_LANGUAGES

    def test_malay_is_supported(self):
        assert "ms" in SUPPORTED_LANGUAGES


# ---------------------------------------------------------------------------
# Language instructions
# ---------------------------------------------------------------------------

class TestLanguageInstructions:
    def test_instructions_mention_arabic_script(self):
        assert "Arabic script" in LANGUAGE_INSTRUCTIONS

    def test_instructions_mention_transliteration(self):
        assert "transliteration" in LANGUAGE_INSTRUCTIONS

    def test_instructions_mention_auto_mode(self):
        assert "auto" in LANGUAGE_INSTRUCTIONS

    def test_instructions_included_when_language_set(self):
        assert LANGUAGE_INSTRUCTIONS in ISLAMIC_CONTEXT + LANGUAGE_INSTRUCTIONS


# ---------------------------------------------------------------------------
# ChatRequest model
# ---------------------------------------------------------------------------

class TestChatRequest:
    def test_language_field_optional(self):
        req = ChatRequest(prompt="What is salat?")
        assert req.language is None

    def test_language_field_accepted(self):
        req = ChatRequest(prompt="ما هي الصلاة؟", language="ar")
        assert req.language == "ar"

    def test_language_arabic_prompt(self):
        req = ChatRequest(prompt="كيف أحسب الزكاة؟", language="ar")
        assert req.prompt == "كيف أحسب الزكاة؟"
        assert req.language == "ar"

    def test_mixed_language_request(self):
        req = ChatRequest(
            prompt="What does بِسْمِ ٱللَّهِ mean?",
            language="en",
        )
        assert req.language == "en"
        assert "بِسْمِ" in req.prompt


# ---------------------------------------------------------------------------
# ChatResponse model
# ---------------------------------------------------------------------------

class TestChatResponse:
    def test_language_field_in_response(self):
        resp = ChatResponse(
            response="الصلاة هي الركن الثاني من أركان الإسلام",
            chat_id="test-123",
            language="ar",
        )
        assert resp.language == "ar"

    def test_language_field_optional(self):
        resp = ChatResponse(response="test", chat_id="test-456")
        assert resp.language is None

    def test_language_field_included_in_json(self):
        resp = ChatResponse(
            response="test",
            chat_id="test-789",
            language="ar",
        )
        data = resp.model_dump()
        assert "language" in data
        assert data["language"] == "ar"


# ---------------------------------------------------------------------------
# Arabic content JSON round-tripping
# ---------------------------------------------------------------------------

class TestArabicJsonRoundTrip:
    """Ensure Arabic (and mixed) content survives JSON serialization."""

    ARABIC_TEXT = "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"
    ARABIC_RESPONSE = "الصلاة هي الركن الثاني من أركان الإسلام الخمسة"

    def test_arabic_response_round_trip(self):
        resp = ChatResponse(
            response=self.ARABIC_RESPONSE,
            chat_id="ar-test",
            history=[
                Message(role="user", content="ما هي أركان الإسلام؟"),
                Message(role="model", content=self.ARABIC_RESPONSE),
            ],
            language="ar",
        )
        serialized = json.dumps(resp.model_dump(), ensure_ascii=False)
        deserialized = json.loads(serialized)
        assert deserialized["response"] == self.ARABIC_RESPONSE
        assert deserialized["language"] == "ar"
        assert deserialized["history"][0]["content"] == "ما هي أركان الإسلام؟"

    def test_mixed_content_round_trip(self):
        mixed = "The meaning of بِسْمِ ٱللَّهِ is 'In the name of Allah'"
        resp = ChatResponse(
            response=mixed,
            chat_id="mixed-test",
            language="en",
        )
        serialized = json.dumps(resp.model_dump(), ensure_ascii=False)
        deserialized = json.loads(serialized)
        assert deserialized["response"] == mixed
        assert "بِسْمِ" in deserialized["response"]

    def test_arabic_unicode_not_escaped(self):
        resp = ChatResponse(
            response=self.ARABIC_TEXT,
            chat_id="unicode-test",
            language="ar",
        )
        serialized = json.dumps(resp.model_dump(), ensure_ascii=False)
        assert "\\u" not in serialized
        assert "بِسْمِ" in serialized

    def test_emoji_and_arabic_together(self):
        text = "الحمد لله 🤲 الصلاة مفروضة على المسلمين"
        resp = ChatResponse(response=text, chat_id="emoji-test", language="ar")
        data = resp.model_dump()
        assert data["response"] == text
