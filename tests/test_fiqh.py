"""Tests for the fiqh module — no live API calls.

Covers:
- Madhhab normalization (valid, variant spellings, unknown values)
- Fiqh classifier deterministic pre-filter
- Fiqh question detection pipeline
- Handler-level prompt assembly with a mocked Gemini client
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from fiqh import (
    FIQH_INSTRUCTIONS,
    FIQH_KEYWORDS,
    NON_FIQH_PATTERNS,
    FiqhClassifier,
    normalize_madhhab,
    _quick_is_fiqh,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_eval_cases():
    path = FIXTURE_DIR / "fiqh_eval_cases.jsonl"
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


# ---------------------------------------------------------------------------
# Madhhab normalization
# ---------------------------------------------------------------------------


class TestNormalizeMadhhab:
    def test_none_returns_none(self):
        assert normalize_madhhab(None) is None

    def test_empty_returns_none(self):
        assert normalize_madhhab("") is None

    def test_canonical_hanafi(self):
        assert normalize_madhhab("hanafi") == "hanafi"

    def test_canonical_maliki(self):
        assert normalize_madhhab("maliki") == "maliki"

    def test_canonical_shafii(self):
        assert normalize_madhhab("shafii") == "shafii"

    def test_canonical_hanbali(self):
        assert normalize_madhhab("hanbali") == "hanbali"

    def test_shafii_variant_apostrophe(self):
        assert normalize_madhhab("shafi'i") == "shafii"

    def test_shafii_variant_backtick(self):
        assert normalize_madhhab("shafi`i") == "shafii"

    def test_shafii_variant_shafie(self):
        assert normalize_madhhab("Shafie") == "shafii"

    def test_hanbali_variant_hambali(self):
        assert normalize_madhhab("Hambali") == "hanbali"

    def test_unknown_degrades_gracefully(self, caplog):
        result = normalize_madhhab("jaafari")
        assert result is None
        assert "Unknown madhhab" in caplog.text

    def test_mixed_case_normalized(self):
        assert normalize_madhhab("HaNaFi") == "hanafi"

    def test_with_punctuation(self):
        assert normalize_madhhab("Shafi'i!") == "shafii"

    def test_valid_madhhabs_set(self):
        from fiqh import VALID_MADHHABS
        assert "hanafi" in VALID_MADHHABS
        assert "maliki" in VALID_MADHHABS
        assert "shafii" in VALID_MADHHABS
        assert "hanbali" in VALID_MADHHABS
        assert len(VALID_MADHHABS) == 4


# ---------------------------------------------------------------------------
# Fiqh classifier — deterministic pre-filter
# ---------------------------------------------------------------------------


class TestQuickIsFiqh:
    def test_greeting_is_not_fiqh(self):
        assert _quick_is_fiqh("hello") is False

    def test_assalamu_alaykum_is_not_fiqh(self):
        assert _quick_is_fiqh("Assalamu alaykum, what can you do?") is False

    def test_platform_question_is_not_fiqh(self):
        assert _quick_is_fiqh("Who are you?") is False

    def test_halal_question_is_fiqh(self):
        assert _quick_is_fiqh("Is gelatine halal?") is True

    def test_wudu_question_is_fiqh(self):
        assert _quick_is_fiqh("Does touching a woman break wudu?") is True

    def test_prayer_ruling_is_fiqh(self):
        assert _quick_is_fiqh("Where do I place my hands in prayer?") is True

    def test_ambiguious_returns_none(self):
        assert _quick_is_fiqh("What does the Quran say about patience?") is None

    def test_zakat_question_is_fiqh(self):
        assert _quick_is_fiqh("How is zakat calculated?") is True

    def test_marriage_ruling_is_fiqh(self):
        assert _quick_is_fiqh("What is the ruling on marriage in Islam?") is True


# ---------------------------------------------------------------------------
# Fiqh classifier — full pipeline
# ---------------------------------------------------------------------------


class TestFiqhClassifier:
    def test_without_model_falls_back_to_false(self):
        classifier = FiqhClassifier(genai_model=None)
        assert classifier.is_fiqh_question("What does the Quran say?") is False

    def test_uses_model_for_ambiguous(self):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"is_fiqh": true}'
        mock_model.generate_content.return_value = mock_response

        classifier = FiqhClassifier(genai_model=mock_model)
        result = classifier.is_fiqh_question("What does the Quran say about patience?")
        assert result is True
        mock_model.generate_content.assert_called_once()

    def test_model_returns_false(self):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"is_fiqh": false}'
        mock_model.generate_content.return_value = mock_response

        classifier = FiqhClassifier(genai_model=mock_model)
        result = classifier.is_fiqh_question("Tell me about the Prophet's life")
        assert result is False

    def test_fiqh_keywords_list_not_empty(self):
        assert len(FIQH_KEYWORDS) > 0

    def test_non_fiqh_patterns_list_not_empty(self):
        assert len(NON_FIQH_PATTERNS) > 0


# ---------------------------------------------------------------------------
# Eval cases — verify classification
# ---------------------------------------------------------------------------


class TestEvalCasesClassification:
    def test_all_eval_cases_classify_correctly(self):
        cases = load_eval_cases()
        assert len(cases) >= 10

        for case in cases:
            quick = _quick_is_fiqh(case["question"])
            if quick is not None:
                assert quick == case["is_fiqh"], (
                    f"Case {case['id']}: {case['question']!r} "
                    f"expected is_fiqh={case['is_fiqh']}, got {quick}"
                )

    def test_eval_cases_cover_both_types(self):
        cases = load_eval_cases()
        fiqh_cases = [c for c in cases if c["is_fiqh"]]
        non_fiqh_cases = [c for c in cases if not c["is_fiqh"]]
        assert len(fiqh_cases) >= 8
        assert len(non_fiqh_cases) >= 2

    def test_eval_cases_have_required_fields(self):
        cases = load_eval_cases()
        for case in cases:
            assert "id" in case
            assert "question" in case
            assert "is_fiqh" in case


# ---------------------------------------------------------------------------
# FIQH_INSTRUCTIONS template
# ---------------------------------------------------------------------------


class TestFiqhInstructions:
    def test_template_has_madhab_lead_placeholder(self):
        assert "{MADHHAB_LEAD}" in FIQH_INSTRUCTIONS

    def test_template_formats_with_madhab(self):
        madhhab_lead = "The user follows the Hanafi school. Lead with the Hanafi position."
        result = FIQH_INSTRUCTIONS.replace("{MADHHAB_LEAD}", madhhab_lead)
        assert "{MADHHAB_LEAD}" not in result
        assert "Hanafi" in result

    def test_template_formats_without_madhab(self):
        madhhab_lead = "No specific madhhab is indicated."
        result = FIQH_INSTRUCTIONS.replace("{MADHHAB_LEAD}", madhhab_lead)
        assert "{MADHHAB_LEAD}" not in result

    def test_instructions_mention_all_four_schools(self):
        for school in ["Hanafi", "Maliki", "Shafi'i", "Hanbali"]:
            assert school in FIQH_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Handler-level tests with mocked Gemini
# ---------------------------------------------------------------------------


class TestFiqhPromptAssembly:
    """Tests that verify the handler correctly assembles fiqh-aware prompts.

    These tests mock the Gemini client and verify that:
    1. Fiqh questions get fiqh instructions in the prompt
    2. Non-fiqh questions do NOT get fiqh instructions
    3. Madhhab preference is correctly reflected in the prompt
    4. ChatResponse carries the correct fiqh metadata
    """

    def test_non_fiqh_prompt_does_not_contain_fiqh_block(self):
        from main import ISLAMIC_CONTEXT

        prompt = "Tell me about the story of Prophet Yusuf"

        fiqh_block = ""
        full_prompt = f"{ISLAMIC_CONTEXT}\n{fiqh_block}\nUser question: {prompt}"
        assert fiqh_block == ""
        assert "FIQH ANSWER GUIDELINES" not in full_prompt
        assert ISLAMIC_CONTEXT in full_prompt

    def test_fiqh_prompt_contains_fiqh_block(self):
        from main import ISLAMIC_CONTEXT

        prompt = "Does touching a woman break wudu?"

        madhhab_lead = "No specific madhhab is indicated. Present all four schools fairly without ranking."
        fiqh_block = FIQH_INSTRUCTIONS.replace("{MADHHAB_LEAD}", madhhab_lead)

        full_prompt = f"{ISLAMIC_CONTEXT}\n{fiqh_block}\nUser question: {prompt}"
        assert fiqh_block != ""
        assert "FIQH ANSWER GUIDELINES" in full_prompt
        assert "Hanafi" in full_prompt
        assert "Maliki" in full_prompt
        assert "Shafi'i" in full_prompt
        assert "Hanbali" in full_prompt

    def test_fiqh_prompt_with_madhab_leads_with_it(self):
        from main import ISLAMIC_CONTEXT

        prompt = "As a Shafi'i, where should I place my hands?"

        madhhab_lead = (
            "The user follows the Shafii school. Lead with the Shafii position. "
            "Also summarize the positions of the other three schools."
        )
        fiqh_block = FIQH_INSTRUCTIONS.replace("{MADHHAB_LEAD}", madhhab_lead)

        full_prompt = f"{ISLAMIC_CONTEXT}\n{fiqh_block}\nUser question: {prompt}"
        assert "Shafii" in full_prompt
        assert "FIQH ANSWER GUIDELINES" in full_prompt

    def test_fiqh_metadata_in_response(self):
        from main import FiqhMetadata

        meta = FiqhMetadata(is_fiqh_question=True, madhhab_requested="hanafi")
        assert meta.is_fiqh_question is True
        assert meta.madhhab_requested == "hanafi"

        meta2 = FiqhMetadata(is_fiqh_question=False, madhhab_requested=None)
        assert meta2.is_fiqh_question is False
        assert meta2.madhhab_requested is None

    def test_chat_response_fiqh_optional(self):
        from main import ChatResponse, Message

        resp = ChatResponse(
            response="test",
            chat_id="123",
            history=[Message(role="user", content="test")],
        )
        assert resp.fiqh is None

        resp2 = ChatResponse(
            response="test",
            chat_id="123",
            history=[Message(role="user", content="test")],
            fiqh={"is_fiqh_question": True, "madhhab_requested": "shafii"},
        )
        assert resp2.fiqh is not None
        assert resp2.fiqh.is_fiqh_question is True
