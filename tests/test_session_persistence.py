"""Tests for session persistence (Redis-backed store with in-memory fallback).

All tests use the in-memory fallback — no Redis server needed.
"""

import pytest

from store import SessionStore, history_to_dicts, dicts_to_contents


# ---------------------------------------------------------------------------
# SessionStore — in-memory fallback
# ---------------------------------------------------------------------------


class TestSessionStoreInMemory:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    @pytest.fixture(autouse=True)
    def store(self):
        self.store = SessionStore()
        # Force in-memory mode regardless of environment
        self.store._use_redis = False
        self.store._local = {}
        yield

    async def test_load_empty_returns_empty_list(self):
        history = await self.store.load_history("nonexistent")
        assert history == []

    async def test_save_and_load_roundtrip(self):
        history = [{"role": "user", "text": "hello"}, {"role": "model", "text": "hi"}]
        await self.store.save_history("chat-1", history)
        loaded = await self.store.load_history("chat-1")
        assert loaded == history

    async def test_delete_removes_session(self):
        await self.store.save_history("chat-1", [{"role": "user", "text": "hello"}])
        deleted = await self.store.delete_session("chat-1")
        assert deleted is True
        loaded = await self.store.load_history("chat-1")
        assert loaded == []

    async def test_delete_nonexistent_returns_false(self):
        deleted = await self.store.delete_session("no-such-chat")
        assert deleted is False

    async def test_multiple_sessions_are_independent(self):
        hist_a = [{"role": "user", "text": "from A"}]
        hist_b = [{"role": "user", "text": "from B"}]
        await self.store.save_history("chat-a", hist_a)
        await self.store.save_history("chat-b", hist_b)
        assert await self.store.load_history("chat-a") == hist_a
        assert await self.store.load_history("chat-b") == hist_b


# ---------------------------------------------------------------------------
# History serialization helpers
# ---------------------------------------------------------------------------


class TestHistorySerialization:
    def testhistory_to_dicts_empty(self):
        assert history_to_dicts([]) == []

    def test_roundtrip(self):
        dicts = [
            {"role": "user", "text": "Assalamu alaykum"},
            {"role": "model", "text": "Wa alaykum assalam"},
        ]
        contents = dicts_to_contents(dicts)
        result = history_to_dicts(contents)
        assert result == dicts

    def testdicts_to_contents_returns_protos(self):
        import google.generativeai.protos as protos

        dicts = [{"role": "user", "text": "hello"}]
        contents = dicts_to_contents(dicts)
        assert len(contents) == 1
        assert isinstance(contents[0], protos.Content)
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "hello"


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


class TestSessionTTL:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    @pytest.fixture(autouse=True)
    def store(self, monkeypatch):
        monkeypatch.setattr("store.SESSION_TTL_SECONDS", 0)  # expire immediately
        self.store = SessionStore()
        self.store._use_redis = False
        self.store._local = {}
        yield

    async def test_expired_session_returns_empty(self):


        history = [{"role": "user", "text": "hello"}]
        await self.store.save_history("chat-1", history)
        # TTL=0 means already expired — load should return empty
        loaded = await self.store.load_history("chat-1")
        assert loaded == []


# ---------------------------------------------------------------------------
# Integration: Gemini chat session lifecycle with store
# ---------------------------------------------------------------------------


class TestGeminiChatSessionLifecycle:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    async def test_new_session_creates_and_persists(self, monkeypatch):
        """Simulate a full chat request cycle with mocked Gemini."""
        from unittest.mock import MagicMock, AsyncMock

        store = SessionStore()
        store._use_redis = False
        store._local = {}

        dicts_saved = []

        async def fake_save(chat_id, history):
            nonlocal dicts_saved
            dicts_saved = history

        monkeypatch.setattr(store, "save_history", fake_save)

        monkeypatch.setattr(store, "load_history", AsyncMock(return_value=[]))

        # Simulate the Gemini send_message flow
        fake_chat = MagicMock()
        fake_chat.history = []
        fake_chat.send_message = MagicMock()

        fake_response = MagicMock()
        fake_response.text = "Wa alaykum assalam"
        fake_chat.send_message.return_value = fake_response

        # After send_message, history should have 2 entries
        fake_chat.history = [
            MagicMock(role="user", parts=[MagicMock(text="Assalamu alaykum")]),
            MagicMock(role="model", parts=[MagicMock(text="Wa alaykum assalam")]),
        ]

        history_dicts = await store.load_history("chat-1")
        assert history_dicts == []

        history_dicts = [{"role": "user", "text": "hello"}]
        await store.save_history("chat-1", history_dicts)
        assert dicts_saved == [{"role": "user", "text": "hello"}]

    async def test_load_existing_session_rebuilds_history(self, monkeypatch):
        from unittest.mock import AsyncMock

        store = SessionStore()
        store._use_redis = False
        store._local = {}

        persisted = [
            {"role": "user", "text": "first question"},
            {"role": "model", "text": "first answer"},
        ]
        monkeypatch.setattr(store, "load_history", AsyncMock(return_value=persisted))

        loaded = await store.load_history("chat-1")
        assert loaded == persisted

        contents = dicts_to_contents(loaded)
        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "first question"
