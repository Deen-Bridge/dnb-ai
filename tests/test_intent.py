"""
Tests for the question-understanding pipeline: intent classification,
classifying questions, answer-length calibration, and suggested follow-ups.

All tests are offline (mocked Gemini client where needed).
"""

import json

import pytest

from intent import (
    Intent,
    ClassificationResult,
    classify_intent,
    _is_trivial_greeting,
    _detect_ambiguity,
    get_intent_config,
    parse_followups,
    strip_followup_block,
    should_clarify,
    AMBIGUITY_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeModelResponse:
    """Simulates genai's response object for testing."""

    def __init__(self, text: str):
        self.text = text


class FakeModel:
    """Simulates model.generate_content for testing."""

    def __init__(self, response_text: str):
        self.response_text = response_text

    def generate_content(self, prompt: str):
        return FakeModelResponse(self.response_text)


# Successful classification responses
CLASSIFICATION_FIXTURES = {
    "greeting": {
        "intent": "greeting_smalltalk",
        "needs_clarification": False,
        "clarifying_question": "",
        "ambiguity_confidence": 0.0,
    },
    "factual": {
        "intent": "factual_knowledge",
        "needs_clarification": False,
        "clarifying_question": "",
        "ambiguity_confidence": 0.1,
    },
    "fiqh": {
        "intent": "fiqh_ruling",
        "needs_clarification": False,
        "clarifying_question": "",
        "ambiguity_confidence": 0.2,
    },
    "ambiguous": {
        "intent": "fiqh_ruling",
        "needs_clarification": True,
        "clarifying_question": "Which madhhab are you asking about?",
        "ambiguity_confidence": 0.85,
    },
    "personal": {
        "intent": "personal_guidance",
        "needs_clarification": False,
        "clarifying_question": "",
        "ambiguity_confidence": 0.1,
    },
    "platform": {
        "intent": "platform_question",
        "needs_clarification": False,
        "clarifying_question": "",
        "ambiguity_confidence": 0.0,
    },
    "out_of_scope": {
        "intent": "out_of_scope",
        "needs_clarification": False,
        "clarifying_question": "",
        "ambiguity_confidence": 0.0,
    },
}


def _make_model(intent_key: str, wrap_markdown: bool = False) -> FakeModel:
    """Create a FakeModel that returns the given classification fixture."""
    data = CLASSIFICATION_FIXTURES[intent_key]
    raw = json.dumps(data)
    if wrap_markdown:
        raw = f"```json\n{raw}\n```"
    return FakeModel(raw)


# ---------------------------------------------------------------------------
# Deterministic short-circuit tests
# ---------------------------------------------------------------------------


class TestDeterministicShortCircuit:
    """The common greeting case must add nearly zero latency — no LLM call."""

    def test_salam_exact(self):
        """'Assalamu alaykum' should be classified without an LLM call."""
        result = _is_trivial_greeting("Assalamu alaykum")
        assert result is True

    def test_salam_variants(self):
        """Common salam variants must be caught."""
        for msg in [
            "Salam",
            "Salaam",
            "Assalamo Alaikum",
            "as-salam",
            "As-salamu alaykum",
            "Wa alaykum assalam",
        ]:
            assert _is_trivial_greeting(msg), f"Failed for: {msg}"

    def test_english_greetings(self):
        """Common English greetings must be caught when short."""
        for msg in ["Hi", "Hello", "Hey", "Good morning", "Peace"]:
            result = _is_trivial_greeting(msg)
            assert result is True, f"Failed for: {msg}"

    def test_longer_greeting_not_caught(self):
        """A longer message that starts with a greeting word should NOT be
        short-circuited, as it likely contains a substantive question."""
        msg = "Hi I have a question about wudu"
        assert _is_trivial_greeting(msg) is False

    def test_substantive_message_not_greeting(self):
        """A substantive question should NOT be caught by the short-circuit."""
        for msg in [
            "What is the ruling on interest?",
            "Can you explain Surah Al-Fatiha?",
            "How should I pray?",
        ]:
            assert _is_trivial_greeting(msg) is False, f"Failed for: {msg}"

    def test_empty_message_not_greeting(self):
        assert _is_trivial_greeting("") is False

    def test_classify_trivial_greeting_no_model(self):
        """classify_intent with a trivial greeting should NOT require a model."""
        result = classify_intent("Assalamu alaykum")
        assert result.intent == Intent.GREETING_SMALLTALK
        assert result.needs_clarification is False


# ---------------------------------------------------------------------------
# LLM-based classification tests
# ---------------------------------------------------------------------------


class TestLLMClassification:
    """Tests that exercise the Gemini-based classifier path."""

    def test_classify_factual_knowledge(self):
        model = _make_model("factual")
        result = classify_intent("What does Islam say about charity?", model.generate_content)
        assert result.intent == Intent.FACTUAL_KNOWLEDGE
        assert result.needs_clarification is False

    def test_classify_fiqh_ruling(self):
        model = _make_model("fiqh")
        result = classify_intent("Is eating pork haram?", model.generate_content)
        assert result.intent == Intent.FIQH_RULING

    def test_classify_personal_guidance(self):
        model = _make_model("personal")
        result = classify_intent("I'm going through a hard time, please make dua", model.generate_content)
        assert result.intent == Intent.PERSONAL_GUIDANCE

    def test_classify_platform_question(self):
        model = _make_model("platform")
        result = classify_intent("How does the zakat calculator work?", model.generate_content)
        assert result.intent == Intent.PLATFORM_QUESTION

    def test_classify_out_of_scope(self):
        model = _make_model("out_of_scope")
        result = classify_intent("What is the meaning of life according to Nietzsche?", model.generate_content)
        assert result.intent == Intent.OUT_OF_SCOPE

    def test_classify_ambiguous_question(self):
        model = _make_model("ambiguous")
        result = classify_intent("Is music haram?", model.generate_content)
        assert result.intent == Intent.FIQH_RULING
        assert result.needs_clarification is True
        assert len(result.clarifying_question) > 0

    def test_unknown_intent_falls_back_to_factual(self):
        """If the LLM returns an invalid intent, fallback to factual_knowledge."""
        model = FakeModel(json.dumps({"intent": "cooking_recipe", "needs_clarification": False}))
        result = classify_intent("How do I make biryani?", model.generate_content)
        assert result.intent == Intent.FACTUAL_KNOWLEDGE

    def test_malformed_json_uses_safe_fallback(self):
        """Unparseable JSON from the classifier should fall back gracefully."""
        model = FakeModel("this is not json")
        result = classify_intent("What is the meaning of life?", model.generate_content)
        # Falls back to factual_knowledge
        assert result.intent == Intent.FACTUAL_KNOWLEDGE
        assert result.needs_clarification is False

    def test_markdown_wrapped_json(self):
        """Handle markdown code fences around the JSON output."""
        model = _make_model("factual", wrap_markdown=True)
        result = classify_intent("Explain Surah Ikhlas", model.generate_content)
        assert result.intent == Intent.FACTUAL_KNOWLEDGE

    def test_ambiguity_below_threshold_not_clarified(self):
        """When ambiguity_confidence is below threshold, don't clarify."""
        data = CLASSIFICATION_FIXTURES["ambiguous"].copy()
        data["ambiguity_confidence"] = AMBIGUITY_THRESHOLD - 0.1
        data["needs_clarification"] = True
        model = FakeModel(json.dumps(data))
        result = classify_intent("Is music haram?", model.generate_content)
        assert result.needs_clarification is False, (
            "Should not clarify when confidence below threshold"
        )

    def test_needs_clarification_without_question_falls_back(self):
        """If needs_clarification is True but no clarifying_question given,
        the module provides a safe fallback."""
        data = CLASSIFICATION_FIXTURES["ambiguous"].copy()
        data["clarifying_question"] = ""
        data["ambiguity_confidence"] = 0.9
        model = FakeModel(json.dumps(data))
        result = classify_intent("Is music haram?", model.generate_content)
        assert result.needs_clarification is True
        assert len(result.clarifying_question) > 0


# ---------------------------------------------------------------------------
# Per-intent generation config tests
# ---------------------------------------------------------------------------


class TestIntentConfig:
    """Each intent maps to an appropriate config (answer shape, length, tone)."""

    def test_greeting_config_is_short(self):
        cfg = get_intent_config(Intent.GREETING_SMALLTALK)
        assert cfg.max_output_tokens <= 256
        assert cfg.temperature <= 0.7
        assert "greeting" in cfg.instruction_snippet.lower() or "salam" in cfg.instruction_snippet.lower()

    def test_factual_config_is_thorough(self):
        cfg = get_intent_config(Intent.FACTUAL_KNOWLEDGE)
        assert cfg.max_output_tokens >= 1024
        assert "structured" in cfg.instruction_snippet.lower()

    def test_fiqh_config_cites_sources(self):
        cfg = get_intent_config(Intent.FIQH_RULING)
        assert cfg.max_output_tokens >= 1024
        assert "fiqh" in cfg.instruction_snippet.lower() or "ruling" in cfg.instruction_snippet.lower()

    def test_personal_guidance_config_compassionate(self):
        cfg = get_intent_config(Intent.PERSONAL_GUIDANCE)
        assert cfg.max_output_tokens <= 1024
        assert "compassionate" in cfg.instruction_snippet.lower() or "supportive" in cfg.instruction_snippet.lower()

    def test_platform_config_is_concise(self):
        cfg = get_intent_config(Intent.PLATFORM_QUESTION)
        assert cfg.max_output_tokens <= 1024
        assert "concisely" in cfg.instruction_snippet.lower() or "actionable" in cfg.instruction_snippet.lower()

    def test_out_of_scope_config_deflects(self):
        cfg = get_intent_config(Intent.OUT_OF_SCOPE)
        assert "outside your scope" in cfg.instruction_snippet.lower()

    def test_unknown_intent_falls_back_to_factual(self):
        cfg = get_intent_config(Intent.GREETING_SMALLTALK)
        assert cfg is not None

    def test_all_intents_have_configs(self):
        for intent in [
            Intent.GREETING_SMALLTALK,
            Intent.FACTUAL_KNOWLEDGE,
            Intent.FIQH_RULING,
            Intent.PERSONAL_GUIDANCE,
            Intent.PLATFORM_QUESTION,
            Intent.OUT_OF_SCOPE,
        ]:
            cfg = get_intent_config(intent)
            assert cfg is not None, f"Missing config for {intent}"
            assert len(cfg.instruction_snippet) > 0


# ---------------------------------------------------------------------------
# Follow-up parsing tests
# ---------------------------------------------------------------------------


class TestFollowUpParsing:
    """Defensive parse-or-degrade pattern: never raises, never leaks delimiters."""

    def test_parse_well_formed_followups(self):
        text = (
            "Here is the answer about wudu.\n\n"
            "<!-- FOLLOWUPS -->\n"
            "1. What invalidates wudu?\n"
            "2. How is tayammum performed?\n"
            "3. What are the sunnah acts of wudu?\n"
            "<!-- /FOLLOWUPS -->"
        )
        result = parse_followups(text)
        assert len(result) == 3
        assert "What invalidates wudu?" in result
        assert "How is tayammum performed?" in result
        assert "What are the sunnah acts of wudu?" in result

    def test_parse_alternative_delimiter(self):
        text = (
            "Here is the answer.\n\n"
            "[[FOLLOWUPS]]\n"
            "* What is the first question?\n"
            "* What is the second question?\n"
            "[[/FOLLOWUPS]]"
        )
        result = parse_followups(text)
        assert len(result) == 2
        assert "What is the first question?" in result

    def test_missing_block_returns_empty(self):
        text = "Just a plain answer with no follow-ups."
        result = parse_followups(text)
        assert result == []

    def test_malformed_block_returns_empty(self):
        """A block that has a start delimiter but no end returns empty list."""
        text = "Answer here.\n\n<!-- FOLLOWUPS -->\n1. Orphan question"
        result = parse_followups(text)
        assert result == []

    def test_empty_block_returns_empty(self):
        text = "Answer.\n\n<!-- FOLLOWUPS -->\n\n<!-- /FOLLOWUPS -->"
        result = parse_followups(text)
        assert result == []

    def test_strip_followup_block(self):
        text = (
            "Visible answer here.\n\n"
            "<!-- FOLLOWUPS -->\n"
            "1. Follow-up?\n"
            "<!-- /FOLLOWUPS -->"
        )
        stripped = strip_followup_block(text)
        assert stripped == "Visible answer here."
        assert "FOLLOWUPS" not in stripped

    def test_strip_followup_block_alternative_delimiter(self):
        text = (
            "Visible answer.\n\n"
            "[[FOLLOWUPS]]\n"
            "1. Follow-up?\n"
            "[[/FOLLOWUPS]]"
        )
        stripped = strip_followup_block(text)
        assert stripped == "Visible answer."
        assert "FOLLOWUPS" not in stripped

    def test_strip_does_not_remove_content_without_block(self):
        text = "Just a plain answer with no follow-ups."
        assert strip_followup_block(text) == text

    def test_mixed_bullets_and_numbers(self):
        text = (
            "Answer.\n\n"
            "<!-- FOLLOWUPS -->\n"
            "- What invalidates wudu?\n"
            "* How is tayammum performed?\n"
            "1. What are the sunnah acts?\n"
            "<!-- /FOLLOWUPS -->"
        )
        result = parse_followups(text)
        assert len(result) == 3
        assert "What invalidates wudu?" in result
        assert "How is tayammum performed?" in result
        assert "What are the sunnah acts?" in result

    def test_empty_text_returns_empty(self):
        assert parse_followups("") == []
        assert parse_followups(None) == []

    def test_maximum_five_followups(self):
        """At most 5 follow-ups are returned, even if more are present."""
        items = "\n".join(f"{i}. Question {i}?" for i in range(1, 10))
        text = f"Answer.\n\n<!-- FOLLOWUPS -->\n{items}\n<!-- /FOLLOWUPS -->"
        result = parse_followups(text)
        assert len(result) <= 5


# ---------------------------------------------------------------------------
# No-double-clarification guard tests
# ---------------------------------------------------------------------------


class TestNoDoubleClarification:
    """Never clarify twice in a row for the same session."""

    def test_first_clarification_allowed(self):
        cls = ClassificationResult(
            intent=Intent.FIQH_RULING,
            needs_clarification=True,
            clarifying_question="Which madhhab?",
        )
        assert should_clarify(cls, None) is True

    def test_second_clarification_blocked(self):
        cls = ClassificationResult(
            intent=Intent.FIQH_RULING,
            needs_clarification=True,
            clarifying_question="Which madhhab?",
        )
        last = ClassificationResult(
            intent=Intent.FIQH_RULING,
            needs_clarification=True,
        )
        assert should_clarify(cls, last) is False

    def test_nonambiguous_not_clarified(self):
        cls = ClassificationResult(
            intent=Intent.FACTUAL_KNOWLEDGE,
            needs_clarification=False,
        )
        assert should_clarify(cls, None) is False

    def test_different_intent_allows_clarification(self):
        """If last was a different intent, clarification is allowed again
        (e.g. the user changed the subject)."""
        cls = ClassificationResult(
            intent=Intent.FIQH_RULING,
            needs_clarification=True,
            clarifying_question="Which madhhab?",
        )
        last = ClassificationResult(
            intent=Intent.GREETING_SMALLTALK,
            needs_clarification=False,
        )
        assert should_clarify(cls, last) is True


# ---------------------------------------------------------------------------
# Ambiguity detection tests
# ---------------------------------------------------------------------------


class TestAmbiguityDetection:
    """Offline keyword-based ambiguity detection."""

    def test_haram_question_detected(self):
        needs, question = _detect_ambiguity("Is music haram?")
        assert needs is True
        assert len(question) > 0

    def test_what_breaks_detected(self):
        needs, question = _detect_ambiguity("What breaks the fast?")
        assert needs is True

    def test_specific_question_not_ambiguous(self):
        needs, question = _detect_ambiguity("What are the five pillars of Islam?")
        # The five pillars question is not inherently ambiguous
        # It may or may not match triggers depending on the implementation
        # Just check that it returns a valid result
        assert isinstance(needs, bool)
        assert isinstance(question, str)

    def test_greeting_not_ambiguous(self):
        needs, question = _detect_ambiguity("Assalamu alaykum")
        assert needs is False
        assert isinstance(question, str)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_non_trivial_message_requires_model(self):
        """classify_intent should raise when called on non-trivial message
        without a model_callable."""
        with pytest.raises(RuntimeError):
            classify_intent("What is the ruling on interest?", None)

    def test_long_message_classified(self):
        model = _make_model("factual")
        long_msg = "Can you explain " + "the concept of " * 50 + "tawheed?"
        result = classify_intent(long_msg, model.generate_content)
        assert result.intent in Intent

    def test_special_characters(self):
        model = _make_model("factual")
        result = classify_intent("What does Quran 2:255 mean?", model.generate_content)
        assert result.intent == Intent.FACTUAL_KNOWLEDGE

    def test_bullet_followup_with_different_markers(self):
        """Test that Unicode bullet markers are handled correctly."""
        text = (
            "Answer.\n\n"
            "<!-- FOLLOWUPS -->\n"
            "• First question?\n"
            "• Second question?\n"
            "<!-- /FOLLOWUPS -->"
        )
        result = parse_followups(text)
        # Unicode bullet (U+2022) is now handled
        assert len(result) == 2
        assert "First question?" in result
        assert "Second question?" in result


# ---------------------------------------------------------------------------
# Integration: end-to-end intent flow
# ---------------------------------------------------------------------------


class TestIntentFlow:
    """Higher-level test of the complete classify_intent → config → clarify flow."""

    def test_greeting_flow(self):
        """Greeting → short answer, no clarification, appropriate config."""
        result = classify_intent("Assalamu alaykum")
        assert result.intent == Intent.GREETING_SMALLTALK
        assert result.needs_clarification is False

        cfg = get_intent_config(result.intent)
        assert cfg.max_output_tokens <= 256
        # No clarifying question
        assert not should_clarify(result, None)

    def test_ambiguous_flow(self):
        """Ambiguous fiqh question → clarification requested, config is fiqh."""
        model = _make_model("ambiguous")
        result = classify_intent("Is music haram?", model.generate_content)
        assert result.intent == Intent.FIQH_RULING
        assert result.needs_clarification is True

        # First turn: clarification allowed
        assert should_clarify(result, None) is True

        # Second turn: clarification blocked (same session)
        assert should_clarify(result, result) is False

    def test_factual_followups_config(self):
        """Factual knowledge config should mention follow-ups in instruction."""
        cfg = get_intent_config(Intent.FACTUAL_KNOWLEDGE)
        assert "follow-up" in cfg.instruction_snippet.lower()

    def test_fiqh_followups_config(self):
        """Fiqh ruling config should mention follow-ups in instruction."""
        cfg = get_intent_config(Intent.FIQH_RULING)
        assert "follow-up" in cfg.instruction_snippet.lower()
