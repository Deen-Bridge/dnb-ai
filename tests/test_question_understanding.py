"""Deterministic coverage for the question-understanding primitives."""

from intent import classify_intent, is_multi_intent


def test_classifies_fiqh_question() -> None:
    result = classify_intent("What invalidates wudu?")
    assert result.primary_intent == "fiqh"
    assert result.confidence > 0


def test_classifies_ambiguous_multi_intent_prompt() -> None:
    prompt = "What is riba, and how should I pray when travelling?"
    result = classify_intent(prompt)
    assert result.is_multi_intent is True
    assert is_multi_intent(prompt, result.domains) is True
