"""Tests for memory extraction, validation, and summarization — no live Gemini.

All Gemini-backed functions are monkeypatched with fixtures.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from memory.extraction import (
    MAX_FACT_LENGTH,
    MAX_FACTS,
    MAX_SUMMARY_LENGTH,
    MAX_TOPICS,
    apply_updates,
    merge_summaries,
    merge_summaries_deterministic,
    summarize_conversation_turns,
)
from memory.models import UserProfile


# ---------------------------------------------------------------------------
# apply_updates
# ---------------------------------------------------------------------------


def _empty_profile() -> UserProfile:
    return UserProfile(user_id="test-user")


class TestApplyUpdates:
    def test_empty_updates_is_noop(self):
        profile = _empty_profile()
        result = apply_updates(profile, {})
        assert result.knowledge_level is None
        assert result.madhhab is None

    def test_knowledge_level_accepted(self):
        profile = apply_updates(_empty_profile(), {"knowledge_level": "beginner"})
        assert profile.knowledge_level == "beginner"

    def test_invalid_knowledge_level_rejected(self):
        profile = apply_updates(_empty_profile(), {"knowledge_level": "expert"})
        assert profile.knowledge_level is None

    def test_madhhab_accepted(self):
        profile = apply_updates(_empty_profile(), {"madhhab": "shafii"})
        assert profile.madhhab == "shafii"

    def test_invalid_madhhab_rejected(self):
        profile = apply_updates(_empty_profile(), {"madhhab": "zahiri"})
        assert profile.madhhab is None

    def test_new_facts_appended(self):
        profile = apply_updates(_empty_profile(), {"new_facts": ["is a convert"]})
        assert len(profile.remembered_facts) == 1
        assert profile.remembered_facts[0].fact == "is a convert"

    def test_oversized_fact_rejected(self):
        profile = apply_updates(
            _empty_profile(),
            {"new_facts": ["x" * (MAX_FACT_LENGTH + 1)]},
        )
        assert len(profile.remembered_facts) == 0

    def test_empty_fact_rejected(self):
        profile = apply_updates(_empty_profile(), {"new_facts": ["   "]})
        assert len(profile.remembered_facts) == 0

    def test_fact_eviction_oldest_removed(self):
        updates = {"new_facts": [f"fact-{i}" for i in range(MAX_FACTS + 1)]}
        profile = apply_updates(_empty_profile(), updates)
        assert len(profile.remembered_facts) == MAX_FACTS
        assert profile.remembered_facts[0].fact == "fact-1"
        assert profile.remembered_facts[-1].fact == f"fact-{MAX_FACTS}"

    def test_new_topics_appended(self):
        profile = apply_updates(_empty_profile(), {"new_topics": ["zakat"]})
        assert len(profile.topics_studied) == 1
        assert profile.topics_studied[0].topic == "zakat"

    def test_existing_topic_updated(self):
        profile = apply_updates(_empty_profile(), {"new_topics": ["zakat"]})
        ts1 = profile.topics_studied[0].last_asked
        profile = apply_updates(profile, {"new_topics": ["zakat"]})
        assert len(profile.topics_studied) == 1
        assert profile.topics_studied[0].last_asked >= ts1

    def test_topic_eviction_oldest_removed(self):
        updates = {"new_topics": [f"topic-{i}" for i in range(MAX_TOPICS + 1)]}
        profile = apply_updates(_empty_profile(), updates)
        assert len(profile.topics_studied) == MAX_TOPICS
        assert profile.topics_studied[0].topic == "topic-1"

    def test_updated_at_timestamp_set(self):
        profile = apply_updates(_empty_profile(), {"knowledge_level": "advanced"})
        profile2 = apply_updates(profile, {"madhhab": "maliki"})
        assert profile2.updated_at >= profile.updated_at


# ---------------------------------------------------------------------------
# summarize_conversation_turns (seam)
# ---------------------------------------------------------------------------


class TestSummarizeConversationTurns:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    @patch("memory.extraction._call_summary_gemini", new_callable=AsyncMock)
    async def test_returns_summary(self, mock_call):
        mock_call.return_value = "user asked about salah"
        turns = [{"role": "user", "text": "tell me about salah"}]
        result = await summarize_conversation_turns(turns)
        assert result == "user asked about salah"
        mock_call.assert_called_once()


# ---------------------------------------------------------------------------
# merge_summaries_deterministic
# ---------------------------------------------------------------------------


class TestMergeSummariesDeterministic:
    def test_short_summaries_concatenated(self):
        result = merge_summaries_deterministic("fact a", "fact b")
        assert result == "fact a\nfact b"

    def test_existing_empty(self):
        result = merge_summaries_deterministic("", "fact b")
        assert result == "fact b"

    def test_oversized_returns_none(self):
        big = "x" * MAX_SUMMARY_LENGTH
        result = merge_summaries_deterministic(big, "more")
        assert result is None


# ---------------------------------------------------------------------------
# merge_summaries (async, with seam)
# ---------------------------------------------------------------------------


class TestMergeSummaries:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    @patch("memory.extraction._call_recompress_gemini", new_callable=AsyncMock)
    async def test_deterministic_path_used_when_fits(self, mock_recompress):
        result = await merge_summaries("short", "also short")
        assert result == "short\nalso short"
        mock_recompress.assert_not_called()

    @patch("memory.extraction._call_recompress_gemini", new_callable=AsyncMock)
    async def test_gemini_path_used_on_overflow(self, mock_recompress):
        mock_recompress.return_value = "recompressed summary"
        big = "x" * MAX_SUMMARY_LENGTH
        result = await merge_summaries(big, "more")
        assert result == "recompressed summary"
        mock_recompress.assert_called_once()
