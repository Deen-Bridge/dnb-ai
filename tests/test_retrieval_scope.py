"""Adversarial tests for deny-by-default retrieval authorization scoping (#91).

Test matrix
-----------
1.  RetrievalScope construction — factories, frozen, bad input.
2.  derive_scope — principal id / anonymous paths.
3.  assert_scope_match — body≠principal → PermissionError; happy paths.
4.  visible_to — post-check: owner match, cross-user, public, anonymous.
5.  cache_scope_key — partition key correctness.
6.  InMemoryMemoryStore scope enforcement — cross-user, anonymous, own data.
7.  SemanticCache scope partitioning — cross-principal entries never served.
8.  Denied-retrieval logging — WARNING messages emitted with no PII leak.
9.  Integration via /chat, /memory/{user_id}, /user/{user_id}/chats — 403
    on cross-principal access; own data is accessible.
10. Bypass-attempt tests — direct scope manipulation rejected.

All tests run offline; Gemini calls are mocked.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Set GEMINI_API_KEY before importing main so pydantic-settings doesn't raise.
# This must happen before any `import main` statement in this module.
# ---------------------------------------------------------------------------
os.environ.setdefault("GEMINI_API_KEY", "test-key-for-scope-tests")

from httpx import ASGITransport, AsyncClient

from memory import InMemoryMemoryStore
from memory.models import ChatSummary, UserProfile
from retrieval.scope import (
    PublishVisibility,
    RetrievalScope,
    UserRole,
    assert_scope_match,
    cache_scope_key,
    derive_scope,
    visible_to,
)
from semantic_cache import CacheEntry, SemanticCache, get_cache, set_fake_embedding

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

V_EXACT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
V_PARA = np.array([0.99, np.sqrt(1 - 0.99**2), 0.0, 0.0], dtype=np.float32)
V_OTHER = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)


@dataclass
class _OwnedRecord:
    """Minimal record with a user_id ownership claim."""
    user_id: str
    content: str = "secret"


@dataclass
class _PublicRecord:
    """Record with no user_id — public content."""
    content: str = "public"


# ---------------------------------------------------------------------------
# 1. RetrievalScope construction
# ---------------------------------------------------------------------------


class TestRetrievalScopeConstruction:
    def test_anonymous_factory(self):
        scope = RetrievalScope.anonymous()
        assert scope.principal_user_id is None
        assert scope.role == UserRole.ANONYMOUS
        assert scope.publish_visibility == PublishVisibility.PUBLIC

    def test_for_user_factory(self):
        scope = RetrievalScope.for_user("alice")
        assert scope.principal_user_id == "alice"
        assert scope.role == UserRole.USER
        assert scope.publish_visibility == PublishVisibility.PRIVATE

    def test_for_user_with_role(self):
        scope = RetrievalScope.for_user("bob", UserRole.REVIEWER)
        assert scope.role == UserRole.REVIEWER

    def test_frozen_immutable(self):
        scope = RetrievalScope.for_user("alice")
        with pytest.raises((AttributeError, TypeError)):
            scope.principal_user_id = "bob"  # type: ignore[misc]

    def test_for_user_empty_string_raises(self):
        with pytest.raises(ValueError):
            RetrievalScope.for_user("")

    def test_for_user_whitespace_raises(self):
        with pytest.raises(ValueError):
            RetrievalScope.for_user("   ")


# ---------------------------------------------------------------------------
# 2. derive_scope
# ---------------------------------------------------------------------------


class TestDeriveScope:
    def test_derive_with_principal(self):
        scope = derive_scope(principal_user_id="alice")
        assert scope.principal_user_id == "alice"
        assert scope.role == UserRole.ANONYMOUS  # default

    def test_derive_with_role(self):
        scope = derive_scope(principal_user_id="alice", role=UserRole.ADMIN)
        assert scope.role == UserRole.ADMIN

    def test_derive_anonymous_when_none(self):
        scope = derive_scope(principal_user_id=None)
        assert scope.principal_user_id is None
        assert scope.role == UserRole.ANONYMOUS

    def test_derive_anonymous_when_empty_string(self):
        scope = derive_scope(principal_user_id="")
        assert scope.principal_user_id is None


# ---------------------------------------------------------------------------
# 3. assert_scope_match
# ---------------------------------------------------------------------------


class TestAssertScopeMatch:
    def test_body_none_always_passes(self):
        scope = RetrievalScope.for_user("alice")
        assert_scope_match(scope, None)  # must not raise

    def test_body_matches_principal_passes(self):
        scope = RetrievalScope.for_user("alice")
        assert_scope_match(scope, "alice")  # must not raise

    def test_body_differs_from_principal_raises(self):
        scope = RetrievalScope.for_user("alice")
        with pytest.raises(PermissionError):
            assert_scope_match(scope, "bob")

    def test_unauthenticated_scope_with_body_user_id_raises(self):
        scope = RetrievalScope.anonymous()
        with pytest.raises(PermissionError):
            assert_scope_match(scope, "alice")

    def test_unauthenticated_scope_with_none_passes(self):
        scope = RetrievalScope.anonymous()
        assert_scope_match(scope, None)  # no conflict


# ---------------------------------------------------------------------------
# 4. visible_to — post-check
# ---------------------------------------------------------------------------


class TestVisibleTo:
    def test_public_record_accessible_to_anonymous(self):
        record = _PublicRecord()
        scope = RetrievalScope.anonymous()
        assert visible_to(record, scope) is True

    def test_public_record_accessible_to_user(self):
        record = _PublicRecord()
        scope = RetrievalScope.for_user("alice")
        assert visible_to(record, scope) is True

    def test_owned_record_accessible_to_owner(self):
        record = _OwnedRecord(user_id="alice")
        scope = RetrievalScope.for_user("alice")
        assert visible_to(record, scope) is True

    def test_owned_record_denied_to_anonymous(self):
        record = _OwnedRecord(user_id="alice")
        scope = RetrievalScope.anonymous()
        assert visible_to(record, scope) is False

    def test_owned_record_denied_to_different_user(self):
        """Core cross-user isolation: user B cannot see user A's record."""
        record = _OwnedRecord(user_id="alice")
        scope = RetrievalScope.for_user("bob")
        assert visible_to(record, scope) is False

    def test_owned_record_denied_under_public_scope(self):
        """PUBLIC-visibility scope must not serve private owned records."""
        record = _OwnedRecord(user_id="alice")
        scope = RetrievalScope(
            principal_user_id="alice",
            role=UserRole.USER,
            publish_visibility=PublishVisibility.PUBLIC,
        )
        assert visible_to(record, scope) is False

    def test_record_without_owner_attr_is_public(self):
        """A record with no user_id attribute is treated as public."""
        scope = RetrievalScope.anonymous()
        assert visible_to(object(), scope) is True

    def test_record_with_none_owner_attr_is_public(self):
        """A record where user_id is None is treated as public."""
        record = MagicMock()
        record.user_id = None
        scope = RetrievalScope.anonymous()
        assert visible_to(record, scope) is True


# ---------------------------------------------------------------------------
# 5. cache_scope_key
# ---------------------------------------------------------------------------


class TestCacheScopeKey:
    def test_anonymous_scope_returns_bare_key(self):
        scope = RetrievalScope.anonymous()
        assert cache_scope_key(scope, "what-are-the-five-pillars") == "what-are-the-five-pillars"

    def test_user_scope_returns_namespaced_key(self):
        scope = RetrievalScope.for_user("alice")
        key = cache_scope_key(scope, "five-pillars")
        assert key == "user:alice:five-pillars"

    def test_different_users_get_different_keys(self):
        ka = cache_scope_key(RetrievalScope.for_user("alice"), "q")
        kb = cache_scope_key(RetrievalScope.for_user("bob"), "q")
        assert ka != kb

    def test_user_key_never_matches_public_key(self):
        public_key = cache_scope_key(RetrievalScope.anonymous(), "q")
        user_key = cache_scope_key(RetrievalScope.for_user("alice"), "q")
        assert public_key != user_key


# ---------------------------------------------------------------------------
# 6. InMemoryMemoryStore — scope enforcement
# ---------------------------------------------------------------------------


class TestMemoryStoreScopeEnforcement:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    async def test_own_profile_accessible_with_correct_scope(self):
        store = InMemoryMemoryStore()
        await store.save_profile("alice", UserProfile(user_id="alice", knowledge_level="beginner"))
        scope = RetrievalScope.for_user("alice")
        profile = await store.get_profile("alice", scope=scope)
        assert profile is not None
        assert profile.knowledge_level == "beginner"

    async def test_cross_user_profile_denied(self):
        """User B with a scope for B cannot read user A's profile."""
        store = InMemoryMemoryStore()
        await store.save_profile("alice", UserProfile(user_id="alice", knowledge_level="advanced"))
        scope_bob = RetrievalScope.for_user("bob")
        result = await store.get_profile("alice", scope=scope_bob)
        assert result is None

    async def test_anonymous_scope_cannot_read_profile(self):
        store = InMemoryMemoryStore()
        await store.save_profile("alice", UserProfile(user_id="alice"))
        scope = RetrievalScope.anonymous()
        result = await store.get_profile("alice", scope=scope)
        assert result is None

    async def test_no_scope_preserves_legacy_behaviour(self):
        """scope=None (default) must not break existing unchecked callers."""
        store = InMemoryMemoryStore()
        await store.save_profile("alice", UserProfile(user_id="alice", knowledge_level="intermediate"))
        profile = await store.get_profile("alice")  # no scope kwarg
        assert profile is not None
        assert profile.knowledge_level == "intermediate"

    async def test_own_summary_accessible_with_correct_scope(self):
        store = InMemoryMemoryStore()
        summary = ChatSummary(chat_id="alice:chat1", content="alice's summary")
        await store.save_chat_summary("alice:chat1", summary)
        scope = RetrievalScope.for_user("alice")
        result = await store.get_chat_summary("alice:chat1", scope=scope)
        assert result is not None
        assert result.content == "alice's summary"

    async def test_cross_user_summary_denied(self):
        """Bob cannot read a summary stored under alice's chat key."""
        store = InMemoryMemoryStore()
        summary = ChatSummary(chat_id="alice:chat1", content="secret summary")
        await store.save_chat_summary("alice:chat1", summary)
        scope_bob = RetrievalScope.for_user("bob")
        result = await store.get_chat_summary("alice:chat1", scope=scope_bob)
        assert result is None

    async def test_anonymous_scope_cannot_read_prefixed_summary(self):
        store = InMemoryMemoryStore()
        summary = ChatSummary(chat_id="alice:chat1", content="private")
        await store.save_chat_summary("alice:chat1", summary)
        scope = RetrievalScope.anonymous()
        result = await store.get_chat_summary("alice:chat1", scope=scope)
        assert result is None

    async def test_unprefixed_summary_accessible_anonymously(self):
        """A summary stored without a user_id prefix is public (legacy behaviour)."""
        store = InMemoryMemoryStore()
        summary = ChatSummary(chat_id="plain-chat-id", content="public summary")
        await store.save_chat_summary("plain-chat-id", summary)
        scope = RetrievalScope.anonymous()
        result = await store.get_chat_summary("plain-chat-id", scope=scope)
        assert result is not None

    async def test_user_a_and_b_profiles_isolated(self):
        """Two users' profiles are fully isolated in the same store."""
        store = InMemoryMemoryStore()
        await store.save_profile("alice", UserProfile(user_id="alice", madhhab="hanafi"))
        await store.save_profile("bob", UserProfile(user_id="bob", madhhab="maliki"))

        alice_scope = RetrievalScope.for_user("alice")
        bob_scope = RetrievalScope.for_user("bob")

        alice_profile = await store.get_profile("alice", scope=alice_scope)
        bob_profile = await store.get_profile("bob", scope=bob_scope)

        # Each user gets their own data
        assert alice_profile.madhhab == "hanafi"
        assert bob_profile.madhhab == "maliki"

        # Neither can read the other's profile
        assert await store.get_profile("bob", scope=alice_scope) is None
        assert await store.get_profile("alice", scope=bob_scope) is None


# ---------------------------------------------------------------------------
# 7. SemanticCache scope partitioning
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_cache_fixture():
    cache = get_cache()
    cache.clear()
    cache.hits = 0
    cache.misses = 0
    cache.bypasses = 0
    cache.evictions = 0
    set_fake_embedding(None)
    with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True):
        yield
    cache.clear()


class TestSemanticCacheScopePartition:
    def test_public_entry_served_to_public_lookup(self):
        cache = SemanticCache()
        with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True):
            cache.put(V_EXACT, "public answer", "cid-pub", [], scope_key=None)
            entry = cache.get(V_EXACT, scope_key=None)
        assert entry is not None
        assert entry.response == "public answer"

    def test_user_entry_served_to_same_user(self):
        cache = SemanticCache()
        user_key = "user:alice:q"
        with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True):
            cache.put(V_EXACT, "alice's answer", "cid-alice", [], scope_key=user_key)
            entry = cache.get(V_EXACT, scope_key=user_key)
        assert entry is not None
        assert entry.response == "alice's answer"

    def test_user_entry_not_served_to_different_user(self):
        """User B must never receive user A's cached answer even at 1.0 similarity."""
        cache = SemanticCache()
        alice_key = "user:alice:q"
        bob_key = "user:bob:q"
        with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True):
            cache.put(V_EXACT, "alice's private answer", "cid-alice", [], scope_key=alice_key)
            entry = cache.get(V_EXACT, scope_key=bob_key)
        assert entry is None

    def test_user_entry_not_served_to_public_lookup(self):
        """A per-user entry must not be served to an anonymous/public lookup."""
        cache = SemanticCache()
        user_key = "user:alice:q"
        with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True):
            cache.put(V_EXACT, "private answer", "cid-alice", [], scope_key=user_key)
            entry = cache.get(V_EXACT, scope_key=None)
        assert entry is None

    def test_public_entry_not_served_to_user_lookup(self):
        """A public entry must not be served under a per-user scope_key."""
        cache = SemanticCache()
        with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True):
            cache.put(V_EXACT, "public answer", "cid-pub", [], scope_key=None)
            entry = cache.get(V_EXACT, scope_key="user:alice:q")
        assert entry is None

    def test_two_users_see_own_cached_answers_independently(self):
        """Alice and Bob can each have their own cached entry for the same vector."""
        cache = SemanticCache()
        alice_key = "user:alice:q"
        bob_key = "user:bob:q"
        with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True):
            cache.put(V_EXACT, "alice answer", "cid-alice", [], scope_key=alice_key)
            cache.put(V_PARA, "bob answer", "cid-bob", [], scope_key=bob_key)

            alice_entry = cache.get(V_EXACT, scope_key=alice_key)
            bob_entry = cache.get(V_PARA, scope_key=bob_key)

        assert alice_entry is not None and alice_entry.response == "alice answer"
        assert bob_entry is not None and bob_entry.response == "bob answer"

    def test_scope_key_stored_on_entry(self):
        cache = SemanticCache()
        with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True):
            cache.put(V_EXACT, "resp", "cid", [], scope_key="user:alice:q")
        assert cache._entries[0].scope_key == "user:alice:q"

    def test_public_entry_scope_key_is_none(self):
        cache = SemanticCache()
        with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True):
            cache.put(V_EXACT, "resp", "cid", [], scope_key=None)
        assert cache._entries[0].scope_key is None


# ---------------------------------------------------------------------------
# 8. Denied-retrieval logging
# ---------------------------------------------------------------------------


class TestDeniedRetrievalLogging:
    @pytest.mark.asyncio
    async def test_cross_user_profile_denial_logged_as_warning(self, caplog):
        """A cross-user profile access attempt must emit at least one WARNING."""
        store = InMemoryMemoryStore()
        await store.save_profile("alice", UserProfile(user_id="alice"))
        scope = RetrievalScope.for_user("bob")

        with caplog.at_level(logging.WARNING):
            result = await store.get_profile("alice", scope=scope)

        assert result is None
        warning_texts = " ".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
        assert warning_texts, "No WARNING emitted for cross-user profile access"

    @pytest.mark.asyncio
    async def test_anonymous_scope_denial_logged(self, caplog):
        """An anonymous scope attempting to read a profile must emit a WARNING."""
        store = InMemoryMemoryStore()
        await store.save_profile("alice", UserProfile(user_id="alice"))
        scope = RetrievalScope.anonymous()

        with caplog.at_level(logging.WARNING):
            result = await store.get_profile("alice", scope=scope)

        assert result is None
        warning_texts = " ".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
        assert warning_texts, "No WARNING emitted for anonymous profile access"

    def test_scope_mismatch_logs_without_full_user_id(self, caplog):
        """Log messages must not emit the full user_id (truncate to 8 chars)."""
        long_user_id = "very-long-user-id-that-should-be-truncated"
        scope = RetrievalScope.for_user("alice12345678")
        with caplog.at_level(logging.WARNING):
            with pytest.raises(PermissionError):
                assert_scope_match(scope, long_user_id)
        for record in caplog.records:
            assert long_user_id not in record.message

    def test_visible_to_denial_logged(self, caplog):
        with caplog.at_level(logging.WARNING):
            record = _OwnedRecord(user_id="alice")
            scope = RetrievalScope.for_user("bob")
            result = visible_to(record, scope)
        assert result is False
        assert any("retrieval_denied" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 9. Integration tests via HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_env(monkeypatch):
    import main
    monkeypatch.setattr(main, "GEMINI_API_KEY", "test-key")
    main.genai.configure(api_key="test-key")
    monkeypatch.setattr(main, "sessions", {})
    monkeypatch.setattr(main, "active_chats", {})
    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "false")
    monkeypatch.setattr(main, "SEMANTIC_CACHE_ENABLED", False)
    monkeypatch.setattr(main, "zakat_retriever", AsyncMock(return_value=None))
    monkeypatch.setattr(main, "purchase_retriever", AsyncMock(return_value=None))
    monkeypatch.setattr(main, "tafsir_retriever", AsyncMock(return_value=None))
    monkeypatch.setattr(main, "enqueue_for_review", AsyncMock())


@pytest.mark.asyncio
async def test_chat_cross_user_body_id_returns_403(mock_env):
    """Sending X-User-Id: alice but body user_id: bob must return 403."""
    import main
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/chat",
            json={"prompt": "Hello", "user_id": "bob"},
            headers={"X-User-Id": "alice"},
        )
    assert res.status_code == 403
    assert "Forbidden" in res.json()["detail"]


@pytest.mark.asyncio
async def test_chat_matching_user_id_succeeds(mock_env, monkeypatch):
    """Sending X-User-Id: alice and body user_id: alice must succeed (200)."""
    import main

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Test response"
    mock_resp.candidates = [MagicMock(finish_reason="STOP")]
    mock_resp.prompt_feedback = None
    mock_session.send_message_async = AsyncMock(return_value=mock_resp)
    mock_session.history = []
    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/chat",
            json={"prompt": "Hello", "user_id": "alice"},
            headers={"X-User-Id": "alice"},
        )
    assert res.status_code == 200


@pytest.mark.asyncio
async def test_chat_no_user_id_no_principal_succeeds(mock_env, monkeypatch):
    """Anonymous request with no X-User-Id and no body user_id must succeed (200)."""
    import main

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Answer"
    mock_resp.candidates = [MagicMock(finish_reason="STOP")]
    mock_resp.prompt_feedback = None
    mock_session.send_message_async = AsyncMock(return_value=mock_resp)
    mock_session.history = []
    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/chat", json={"prompt": "Hello"})
    assert res.status_code == 200


@pytest.mark.asyncio
async def test_memory_get_cross_user_returns_403(mock_env):
    """GET /memory/bob with X-User-Id: alice must return 403."""
    import main
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/memory/bob", headers={"X-User-Id": "alice"})
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_memory_get_own_id_returns_404_or_200(mock_env):
    """GET /memory/alice with X-User-Id: alice must return 200 or 404 (not 403)."""
    import main
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/memory/alice", headers={"X-User-Id": "alice"})
    assert res.status_code in (200, 404)  # 404 = no profile, but not 403


@pytest.mark.asyncio
async def test_memory_delete_cross_user_returns_403(mock_env):
    """DELETE /memory/bob with X-User-Id: alice must return 403."""
    import main
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.delete("/memory/bob", headers={"X-User-Id": "alice"})
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_user_chats_cross_user_returns_403(mock_env):
    """GET /user/bob/chats with X-User-Id: alice must return 403."""
    import main
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/user/bob/chats", headers={"X-User-Id": "alice"})
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_user_chats_own_id_succeeds(mock_env):
    """GET /user/alice/chats with X-User-Id: alice must return 200."""
    import main
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/user/alice/chats", headers={"X-User-Id": "alice"})
    assert res.status_code == 200


@pytest.mark.asyncio
async def test_memory_no_principal_with_user_path_returns_403(mock_env):
    """GET /memory/alice with no X-User-Id at all must return 403 (anonymous cannot read user memory)."""
    import main
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/memory/alice")  # no X-User-Id
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_chat_stream_cross_user_body_id_returns_403(mock_env):
    """POST /chat/stream with X-User-Id: alice and body user_id: bob must return 403."""
    import main
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/chat/stream",
            json={"prompt": "Hello", "user_id": "bob"},
            headers={"X-User-Id": "alice"},
        )
    assert res.status_code == 403


# ---------------------------------------------------------------------------
# 10. Bypass-attempt tests
# ---------------------------------------------------------------------------


class TestBypassAttempts:

    async def test_spoofed_user_id_in_body_denied(self):
        """A caller who sends the *correct* API key but the *wrong* body user_id
        must be rejected with PermissionError before any retrieval happens."""
        scope = RetrievalScope.for_user("alice")
        with pytest.raises(PermissionError):
            assert_scope_match(scope, "eve")  # eve trying to act as alice

    async def test_scope_with_no_principal_cannot_read_any_profile(self):
        """An anonymous scope is structurally denied all per-user data."""
        store = InMemoryMemoryStore()
        for uid in ["alice", "bob", "charlie"]:
            await store.save_profile(uid, UserProfile(user_id=uid, knowledge_level="advanced"))

        anon_scope = RetrievalScope.anonymous()
        for uid in ["alice", "bob", "charlie"]:
            result = await store.get_profile(uid, scope=anon_scope)
            assert result is None, f"Anonymous scope leaked profile for {uid}"

    async def test_cache_cross_user_bypass_fails_even_at_perfect_similarity(self):
        """Even at cosine similarity = 1.0, user B must not get user A's cached entry."""
        cache = SemanticCache()
        alice_key = "user:alice:q"
        bob_key = "user:bob:q"
        with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True):
            # Store Alice's entry under the exact same vector Bob will query
            cache.put(V_EXACT, "alice private answer", "cid-alice", [], scope_key=alice_key)
            # Bob queries with the SAME vector — must get nothing
            entry = cache.get(V_EXACT, scope_key=bob_key)
        assert entry is None, "Cache isolation breach: Alice's entry was served to Bob"

    async def test_visible_to_rejects_record_even_with_matching_principal_under_public_scope(self):
        """A PUBLIC-visibility scope cannot access owned records,
        even when the principal id matches the owner."""
        record = _OwnedRecord(user_id="alice")
        public_scope = RetrievalScope(
            principal_user_id="alice",
            role=UserRole.USER,
            publish_visibility=PublishVisibility.PUBLIC,
        )
        assert visible_to(record, public_scope) is False

    def test_scope_key_cannot_be_forged_by_choosing_similar_string(self):
        """Cache scope keys are exact-string compared, so a near-match is not a match."""
        scope_alice = "user:alice:q"
        scope_alice_variant = "user:alice_:q"  # typo / injection attempt
        cache = SemanticCache()
        with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True):
            cache.put(V_EXACT, "alice answer", "cid", [], scope_key=scope_alice)
            entry = cache.get(V_EXACT, scope_key=scope_alice_variant)
        assert entry is None
