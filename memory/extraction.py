"""Memory extraction, validation, and conversation summarization.

All Gemini calls follow the existing pattern from classify_for_safety() and
study.py: genai.GenerativeModel + generate_content with temperature=0 and
response_mime_type="application/json".

Seams
-----
Every Gemini-backed function can be monkeypatched in offline tests.
extract_updates       → patch memory.extraction._call_extraction_gemini
summarize_conversation_turns → patch memory.extraction._call_summary_gemini
recompress_summaries  → patch memory.extraction._call_recompress_gemini
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from memory.models import (
    MAX_FACT_LENGTH,
    MAX_FACTS,
    MAX_SUMMARY_LENGTH,
    MAX_TOPIC_LENGTH,
    MAX_TOPICS,
    FactEntry,
    TopicEntry,
    UserProfile,
)

logger = logging.getLogger(__name__)

MEMORY_EXTRACTION_ENABLED = os.getenv(
    "MEMORY_EXTRACTION_ENABLED", "true"
).lower() not in {"0", "false", "off"}

_EXTRACTION_INSTRUCTION = """You are an AI assistant that extracts structured profile updates from a conversation turn.

Given the user's question and your response, identify any of the following that apply:
- knowledge_level: one of "beginner", "intermediate", "advanced" (or omit)
- madhhab: the user's school of thought (e.g. "hanafi", "maliki", "shafii", "hanbali") (or omit)
- preferred_language: e.g. "english", "arabic", "urdu" (or omit)
- new_facts: a list of specific facts the user shared about themselves (each max 500 chars)
- new_topics: a list of topics the user asked about (each max 100 chars)

Return ONLY strict JSON with exactly the keys that apply.
If nothing meaningful changed, return {"none": true}."""


def apply_updates(profile: UserProfile, updates: dict) -> UserProfile:
    """Validate and apply structured extraction proposals.

    Invalid individual entries are **rejected** (logged, skipped).
    Valid entries that cause a collection to exceed its cap trigger
    oldest-first eviction of that collection.
    """
    from fiqh import normalize_madhhab

    profile = profile.model_copy(deep=True)

    knowledge_level = updates.get("knowledge_level")
    if knowledge_level is not None:
        from memory.models import VALID_KNOWLEDGE_LEVELS
        if knowledge_level in VALID_KNOWLEDGE_LEVELS:
            profile.knowledge_level = knowledge_level
        else:
            logger.warning("Rejected invalid knowledge_level: %s", knowledge_level)

    madhhab = updates.get("madhhab")
    if madhhab is not None:
        normalized = normalize_madhhab(madhhab)
        if normalized is not None:
            profile.madhhab = normalized
        else:
            logger.warning("Rejected invalid madhhab: %s", madhhab)

    language = updates.get("preferred_language")
    if language is not None:
        cleaned = language.strip().lower()
        if len(cleaned) <= 20:
            profile.preferred_language = cleaned
        else:
            logger.warning("Rejected oversized language: %s", language)

    new_facts = updates.get("new_facts", [])
    for fact_text in new_facts:
        if not isinstance(fact_text, str) or len(fact_text) > MAX_FACT_LENGTH or not fact_text.strip():
            logger.warning("Rejected invalid fact: %s", str(fact_text)[:80])
            continue
        profile.remembered_facts.append(FactEntry(fact=fact_text.strip(), created_at=__import__("time").time()))
    while len(profile.remembered_facts) > MAX_FACTS:
        profile.remembered_facts.pop(0)

    new_topics = updates.get("new_topics", [])
    for topic_name in new_topics:
        if not isinstance(topic_name, str) or len(topic_name) > MAX_TOPIC_LENGTH or not topic_name.strip():
            logger.warning("Rejected invalid topic: %s", str(topic_name)[:80])
            continue
        cleaned_topic = topic_name.strip().lower()
        existing = [t for t in profile.topics_studied if t.topic == cleaned_topic]
        if existing:
            existing[0].last_asked = __import__("time").time()
        else:
            profile.topics_studied.append(TopicEntry(topic=cleaned_topic, last_asked=__import__("time").time()))
    while len(profile.topics_studied) > MAX_TOPICS:
        profile.topics_studied.pop(0)

    profile.updated_at = __import__("time").time()
    return profile


async def extract_updates(user_prompt: str, model_response: str) -> dict:
    """Propose profile updates from a conversation turn.

    Offline seam: patch ``memory.extraction._call_extraction_gemini``.
    """
    full_prompt = (
        f"{_EXTRACTION_INSTRUCTION}\n\n"
        f"User question: {user_prompt}\n"
        f"Assistant response: {model_response}"
    )
    raw = await _call_extraction_gemini(full_prompt)
    if not isinstance(raw, dict):
        logger.warning("Extraction returned non-dict: %s", type(raw).__name__)
        return {"none": True}
    return raw


async def _call_extraction_gemini(prompt: str) -> dict:
    """Gemini structured-output call for extraction.

    Override this in offline tests with a fixture.
    """
    import google.generativeai as genai

    model = genai.GenerativeModel(
        "gemini-2.5-flash-preview-05-20",
        system_instruction=_EXTRACTION_INSTRUCTION,
    )
    response = await model.generate_content_async(
        prompt,
        generation_config={"temperature": 0, "response_mime_type": "application/json"},
        request_options={"timeout": 30},
    )
    return json.loads(response.text)


_SUMMARY_INSTRUCTION = """Summarize the following conversation turns.
Preserve:
- Established facts about the user
- The user's goals and intent
- Unresolved or open questions
- Topics discussed

Be concise. Maximum 2000 characters."""


async def summarize_conversation_turns(evicted_turns: list[dict[str, str]]) -> str:
    """Summarize a list of evicted conversation turns.

    Each dict has keys ``role`` and ``text``.

    Offline seam: patch ``memory.extraction._call_summary_gemini``.
    """
    turns_text = "\n".join(
        f"{t.get('role', 'unknown')}: {t.get('text', '')}"
        for t in evicted_turns
    )
    prompt = f"{_SUMMARY_INSTRUCTION}\n\nTurns:\n{turns_text}"
    return await _call_summary_gemini(prompt)


async def _call_summary_gemini(prompt: str) -> str:
    """Gemini structured-output call for summarization.

    Override this in offline tests with a fixture.
    """
    import google.generativeai as genai

    model = genai.GenerativeModel(
        "gemini-2.5-flash-preview-05-20",
        system_instruction=_SUMMARY_INSTRUCTION,
    )
    response = await model.generate_content_async(
        prompt,
        generation_config={"temperature": 0},
        request_options={"timeout": 30},
    )
    return response.text


def merge_summaries_deterministic(existing: str, new: str) -> Optional[str]:
    """Concatenate two summaries if the result fits within MAX_SUMMARY_LENGTH."""
    combined = f"{existing.rstrip()}\n{new}".strip()
    if len(combined) <= MAX_SUMMARY_LENGTH:
        return combined
    return None


_RECOMPRESS_INSTRUCTION = """Merge the following two conversation summaries into one.
Preserve established facts, user goals, and open questions.
Remove redundancy but keep all unique information.
Be concise. Maximum 2000 characters."""


async def _call_recompress_gemini(existing: str, new: str) -> str:
    """Gemini call for recompressing two oversized summaries.

    Offline seam: patch ``memory.extraction._call_recompress_gemini``.
    """
    import google.generativeai as genai

    prompt = (
        f"{_RECOMPRESS_INSTRUCTION}\n\n"
        f"Existing summary:\n{existing}\n\n"
        f"New segment:\n{new}"
    )
    model = genai.GenerativeModel(
        "gemini-2.5-flash-preview-05-20",
        system_instruction=_RECOMPRESS_INSTRUCTION,
    )
    response = await model.generate_content_async(
        prompt,
        generation_config={"temperature": 0},
        request_options={"timeout": 30},
    )
    return response.text


async def merge_summaries(existing: str, new: str) -> str:
    """Merge two summaries — deterministic concatenation first, Gemini on overflow."""
    result = merge_summaries_deterministic(existing, new)
    if result is not None:
        return result
    return await _call_recompress_gemini(existing, new)
