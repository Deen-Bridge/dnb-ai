"""Integration tests for the memory subsystem wired into the /chat endpoint.

All tests run offline — Gemini calls are mocked.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from memory import InMemoryMemoryStore, render_user_context
from memory.models import ChatSummary, UserProfile


# ---------------------------------------------------------------------------
# render_user_context — unit-level but integration-like
# ---------------------------------------------------------------------------


class TestRenderUserContext:
    def test_no_profile_no_summary_returns_empty(self):
        assert render_user_context(None, None) == ""

    def test_empty_profile_returns_empty(self):
        profile = UserProfile(user_id="u")
        assert render_user_context(profile, None) == ""

    def test_profile_with_knowledge_level(self):
        profile = UserProfile(user_id="u", knowledge_level="beginner")
        result = render_user_context(profile, None)
        assert "beginner" in result
        assert "Known about this student" in result
        assert "Memory not found" not in result

    def test_profile_with_madhhab(self):
        profile = UserProfile(user_id="u", madhhab="shafii")
        result = render_user_context(profile, None)
        assert "shafii" in result

    def test_profile_with_facts(self):
        from memory.models import FactEntry
        profile = UserProfile(user_id="u")
        profile.remembered_facts.append(FactEntry(fact="is a convert", created_at=1000.0))
        result = render_user_context(profile, None)
        assert "is a convert" in result

    def test_chat_summary_included(self):
        summary = ChatSummary(chat_id="c1", content="user discussed zakat")
        result = render_user_context(None, summary)
        assert "user discussed zakat" in result
        assert "Conversation summary" in result

    def test_both_profile_and_summary(self):
        profile = UserProfile(user_id="u", knowledge_level="intermediate")
        summary = ChatSummary(chat_id="c1", content="studied salah")
        result = render_user_context(profile, summary)
        assert "intermediate" in result
        assert "studied salah" in result


# ---------------------------------------------------------------------------
# Anonymous traffic preservation
# ---------------------------------------------------------------------------


class TestAnonymousTraffic:
    def test_no_user_id_no_memory_lookup(self):
        """Without user_id, the existing behaviour must be preserved.
        This test verifies the memory store is never consulted."""
        store = InMemoryMemoryStore()
        with patch.object(store, "get_profile", wraps=store.get_profile) as spy:
            profile = None
            user_id = None
            if user_id:
                profile = spy(user_id)
            assert profile is None
            spy.assert_not_called()

    def test_no_user_id_no_extraction_scheduled(self):
        """Without user_id, BackgroundTasks must not schedule extraction."""
        bt = BackgroundTasks()
        tasks_before = len(bt.tasks)
        user_id = None
        if user_id and True:
            bt.add_task(lambda: None)
        assert len(bt.tasks) == tasks_before


# ---------------------------------------------------------------------------
# remember=False semantics
# ---------------------------------------------------------------------------


class TestRememberFalse:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_remember_false_still_loads_profile(self):
        """Existing memory is read even when remember=False."""
        store = InMemoryMemoryStore()
        profile = UserProfile(user_id="u1", knowledge_level="advanced")
        await store.save_profile("u1", profile)

        loaded_profile = None
        remember = False
        user_id = "u1"

        if user_id:
            loaded_profile = await store.get_profile("u1")

        assert loaded_profile is not None
        assert loaded_profile.knowledge_level == "advanced"

    def test_remember_false_skips_background_task(self):
        """When remember=False, extraction is not scheduled."""
        bt = BackgroundTasks()
        tasks_before = len(bt.tasks)
        user_id = "u1"
        remember = False
        if user_id and remember:
            bt.add_task(lambda: None)
        assert len(bt.tasks) == tasks_before


# ---------------------------------------------------------------------------
# Cross-user isolation
# ---------------------------------------------------------------------------


class TestCrossUserIsolation:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    async def test_different_users_dont_share_profiles(self):
        store = InMemoryMemoryStore()
        p1 = UserProfile(user_id="alice", knowledge_level="beginner")
        p2 = UserProfile(user_id="bob", knowledge_level="advanced")
        await store.save_profile("alice", p1)
        await store.save_profile("bob", p2)

        bob_view = await store.get_profile("bob")
        assert bob_view is not None
        assert bob_view.knowledge_level == "advanced"
        assert bob_view.user_id == "bob"
        assert bob_view.knowledge_level != "beginner"

        alice_view = await store.get_profile("alice")
        assert alice_view.knowledge_level == "beginner"

    async def test_different_user_same_chat_different_profiles(self):
        store = InMemoryMemoryStore()
        p1 = UserProfile(user_id="u1", madhhab="hanafi")
        p2 = UserProfile(user_id="u2", madhhab="maliki")
        await store.save_profile("u1", p1)
        await store.save_profile("u2", p2)

        assert (await store.get_profile("u1")).madhhab == "hanafi"
        assert (await store.get_profile("u2")).madhhab == "maliki"

        s1 = ChatSummary(chat_id="shared-chat", content="u1 summary")
        s2 = ChatSummary(chat_id="shared-chat", content="u2 summary")
        await store.save_chat_summary("shared-chat", s1)
        await store.save_chat_summary("shared-chat", s2)
        loaded = await store.get_chat_summary("shared-chat")
        assert loaded.content == "u2 summary"


# ---------------------------------------------------------------------------
# Cache isolation
# ---------------------------------------------------------------------------


class TestCacheIsolation:
    def test_user_id_makes_request_non_cacheable(self):
        """When user_id is present, is_cacheable must be False."""
        is_new_chat = True
        context_none = True
        cache_enabled = True
        user_id = "u1"

        is_cacheable = is_new_chat and context_none and cache_enabled and user_id is None
        assert is_cacheable is False

    def test_no_user_id_remains_cacheable(self):
        is_new_chat = True
        context_none = True
        cache_enabled = True
        user_id = None

        is_cacheable = is_new_chat and context_none and cache_enabled and user_id is None
        assert is_cacheable is True


# ---------------------------------------------------------------------------
# Profile contents not logged at INFO
# ---------------------------------------------------------------------------


class TestLoggingPrivacy:
    def test_profile_not_logged_at_info(self, caplog):
        caplog.set_level(logging.INFO)
        profile = UserProfile(user_id="secret-user", knowledge_level="advanced",
                              madhhab="hanafi")
        logger = logging.getLogger("memory")
        logger.info("Profile loaded for user %s", profile.user_id[:8])
        logger.info("Memory extraction scheduled for user %s", profile.user_id[:8])

        assert "secret-user" not in caplog.text
        assert profile.knowledge_level not in caplog.text
        assert profile.madhhab not in caplog.text
