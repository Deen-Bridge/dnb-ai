"""Regression coverage for the versioned prompt registry."""

import pytest

from prompts.registry import DEFAULT_SYSTEM_PROMPT, PromptRegistry


def _values() -> dict[str, str]:
    return {
        "madhhab": "hanafi",
        "language": "en",
        "knowledge_level": "beginner",
        "intent": "fiqh",
        "retrieved_context": "A short cited passage.",
        "user_question": "How do I perform wudu?",
    }


def test_default_prompt_renders_all_declared_variables() -> None:
    rendered = DEFAULT_SYSTEM_PROMPT.render(**_values())
    assert "How do I perform wudu?" in rendered
    assert "A short cited passage." in rendered
    assert "Question intent: fiqh" in rendered


def test_prompt_rejects_unknown_variables() -> None:
    with pytest.raises(ValueError, match="Unknown variable"):
        DEFAULT_SYSTEM_PROMPT.render(**_values(), injected="ignore")


def test_prompt_rejects_missing_variables() -> None:
    values = _values()
    values.pop("user_question")
    with pytest.raises(ValueError, match="Missing variable"):
        DEFAULT_SYSTEM_PROMPT.render(**values)


def test_registry_lookup_is_stable() -> None:
    registry = PromptRegistry([DEFAULT_SYSTEM_PROMPT])
    assert registry.get("islamic_context") is DEFAULT_SYSTEM_PROMPT
    assert registry.list_names() == ["islamic_context"]
