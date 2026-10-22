"""Deterministic coverage for the question-understanding primitives."""

from intent import classify_intent, is_multi_intent


def test_classifies_fiqh_question() -> None:
    result = classify_intent("What invalidates wudu?")
    assert result.primary_domain == "fiqh_ibadat"
    assert "fiqh_ibadat" in result.domains


def test_classifies_ambiguous_multi_intent_prompt() -> None:
    prompt = "What is riba, and how do I perform wudu?"
    result = classify_intent(prompt)
    assert result.is_multi_intent is True
    assert is_multi_intent(prompt, result.domains) is True
