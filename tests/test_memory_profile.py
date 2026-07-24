"""Tests for memory models and store — no live Redis needed.

All tests use InMemoryMemoryStore directly or RedisMemoryStore with
a mocked redis client.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock

import pytest

from pydantic import ValidationError

from memory.models import (
    MAX_FACT_LENGTH,
    MAX_SUMMARY_LENGTH,
    MAX_TOPIC_LENGTH,
    ChatSummary,
    FactEntry,
    TopicEntry,
    UserProfile,
)
from memory.store import (
    InMemoryMemoryStore,
    RedisMemoryStore,
)

# ---------------------------------------------------------------------------
# UserProfile model
# ---------------------------------------------------------------------------


class TestUserProfile:
    def test_minimal_profile(self):
        p = UserProfile(user_id="user-1")
        assert p.user_id == "user-1"
        assert p.knowledge_level is None
        assert p.madhhab is None
        assert p.remembered_facts == []
        assert p.topics_studied == []

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            UserProfile(user_id="u", injected="harmful")

    def test_fact_max_length_enforced(self):
        with pytest.raises(ValidationError):
            FactEntry(fact="x" * (MAX_FACT_LENGTH + 1), created_at=time.time())

    def test_topic_max_length_enforced(self):
        with pytest.raises(ValidationError):
            TopicEntry(topic="x" * (MAX_TOPIC_LENGTH + 1), last_asked=time.time())

    def test_summary_max_length_enforced(self):
        with pytest.raises(ValidationError):
            ChatSummary(chat_id="c", content="x" * (MAX_SUMMARY_LENGTH + 1))

    def test_timestamps_set_on_creation(self):
        p = UserProfile(user_id="u")
        assert p.created_at > 0
        assert p.updated_at > 0


# ---------------------------------------------------------------------------
# InMemoryMemoryStore
# ---------------------------------------------------------------------------


class TestInMemoryMemoryStore:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    @pytest.fixture(autouse=True)
    def store(self):
        self.store = InMemoryMemoryStore()
        yield

    async def test_load_nonexistent_profile(self):
        profile = await self.store.get_profile("no-such-user")
        assert profile is None

    async def test_save_and_load_profile_roundtrip(self):
        profile = UserProfile(user_id="u1", knowledge_level="beginner")
        await self.store.save_profile("u1", profile)
        loaded = await self.store.get_profile("u1")
        assert loaded is not None
        assert loaded.user_id == "u1"
        assert loaded.knowledge_level == "beginner"

    async def test_delete_profile(self):
        profile = UserProfile(user_id="u1")
        await self.store.save_profile("u1", profile)
        deleted = await self.store.delete_profile("u1")
        assert deleted is True
        loaded = await self.store.get_profile("u1")
        assert loaded is None

    async def test_delete_nonexistent_returns_false(self):
        deleted = await self.store.delete_profile("no-such-user")
        assert deleted is False

    async def test_multiple_profiles_independent(self):
        p1 = UserProfile(user_id="u1", knowledge_level="beginner")
        p2 = UserProfile(user_id="u2", knowledge_level="advanced")
        await self.store.save_profile("u1", p1)
        await self.store.save_profile("u2", p2)
        assert (await self.store.get_profile("u1")).knowledge_level == "beginner"
        assert (await self.store.get_profile("u2")).knowledge_level == "advanced"

    async def test_chat_summary_save_and_load(self):
        summary = ChatSummary(chat_id="c1", content="test content", turn_count=5)
        await self.store.save_chat_summary("c1", summary)
        loaded = await self.store.get_chat_summary("c1")
        assert loaded is not None
        assert loaded.content == "test content"
        assert loaded.turn_count == 5

    async def test_chat_summary_delete(self):
        summary = ChatSummary(chat_id="c1", content="test")
        await self.store.save_chat_summary("c1", summary)
        deleted = await self.store.delete_chat_summary("c1")
        assert deleted is True
        assert await self.store.get_chat_summary("c1") is None

    async def test_chat_summaries_scoped_by_chat_id(self):
        s1 = ChatSummary(chat_id="c1", content="from c1")
        s2 = ChatSummary(chat_id="c2", content="from c2")
        await self.store.save_chat_summary("c1", s1)
        await self.store.save_chat_summary("c2", s2)
        assert (await self.store.get_chat_summary("c1")).content == "from c1"
        assert (await self.store.get_chat_summary("c2")).content == "from c2"


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


class TestStoreTTL:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    @pytest.fixture(autouse=True)
    def store(self, monkeypatch):
        monkeypatch.setattr("memory.store.MEMORY_TTL_SECONDS", 0)
        self.store = InMemoryMemoryStore()
        yield

    async def test_expired_profile_returns_none(self):
        profile = UserProfile(user_id="u1")
        await self.store.save_profile("u1", profile)
        loaded = await self.store.get_profile("u1")
        assert loaded is None

    async def test_expired_summary_returns_none(self):
        summary = ChatSummary(chat_id="c1", content="test")
        await self.store.save_chat_summary("c1", summary)
        loaded = await self.store.get_chat_summary("c1")
        assert loaded is None


# ---------------------------------------------------------------------------
# RedisMemoryStore (mocked Redis client)
# ---------------------------------------------------------------------------


class TestRedisMemoryStore:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    @pytest.fixture(autouse=True)
    def store(self, monkeypatch):
        fake_redis = AsyncMock()
        fake_redis.get = AsyncMock(return_value=None)
        fake_redis.setex = AsyncMock()
        fake_redis.delete = AsyncMock(return_value=1)
        monkeypatch.setattr("memory.store._redis_available", True)

        store = RedisMemoryStore("redis://fake:6379")
        store._redis = fake_redis
        self.store = store
        self.fake_redis = fake_redis
        yield

    async def test_get_profile_nonexistent(self):
        self.fake_redis.get.return_value = None
        profile = await self.store.get_profile("u1")
        assert profile is None

    async def test_save_and_load_profile(self):
        profile = UserProfile(user_id="u1", knowledge_level="intermediate")
        await self.store.save_profile("u1", profile)
        raw = profile.model_dump_json()
        stored = json.loads(raw)
        self.fake_redis.get.return_value = json.dumps(stored)
        loaded = await self.store.get_profile("u1")
        assert loaded is not None
        assert loaded.knowledge_level == "intermediate"

    async def test_delete_profile(self):
        result = await self.store.delete_profile("u1")
        assert result is True
        self.fake_redis.delete.assert_called_with("memory:profile:u1")

    async def test_corrupt_profile_returns_none(self):
        self.fake_redis.get.return_value = "not-json-at-all"
        profile = await self.store.get_profile("u1")
        assert profile is None

    async def test_chat_summary_save_and_load(self):
        summary = ChatSummary(chat_id="c1", content="hello")
        await self.store.save_chat_summary("c1", summary)
        raw = summary.model_dump_json()
        stored = json.loads(raw)
        self.fake_redis.get.return_value = json.dumps(stored)
        loaded = await self.store.get_chat_summary("c1")
        assert loaded is not None
        assert loaded.content == "hello"
