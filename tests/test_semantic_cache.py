"""Tests for the semantic response cache — no live API calls.

Fake embedding function returns hand-built vectors so all tests run offline.
"""

import time
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from semantic_cache import (
    SEMANTIC_CACHE_MAX_ENTRIES,
    CacheEntry,
    cosine_similarity,
    get_cache,
    get_chat_exact_cache,
    get_token_quota_tracker,
    invalidate_public_cache,
    invalidate_user_cache,
    normalize_text,
    set_fake_embedding,
)


@dataclass
class FakeMessage:
    """Mirrors main.Message without importing from main.py (avoids genai dep)."""

    role: str
    content: str


# ---------------------------------------------------------------------------
# Hand-built embedding vectors for known prompts
#
# We use small 4-D unit-ish vectors so cosine similarities are easy to
# reason about:
#   v_exact  = [1, 0, 0, 0]   — exact match
#   v_para   = [0.99, 0.14, 0, 0]   — ≈0.99 cosine to v_exact (paraphrase)
#   v_near   = [0.87, 0.5, 0, 0]    — ≈0.87 cosine to v_exact (below 0.95)
#   v_other  = [0, 0, 1, 0]   — unrelated
# ---------------------------------------------------------------------------

V_EXACT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
V_PARA = np.array([0.99, np.sqrt(1 - 0.99**2), 0.0, 0.0], dtype=np.float32)
V_NEAR = np.array([0.87, np.sqrt(1 - 0.87**2), 0.0, 0.0], dtype=np.float32)
V_OTHER = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

# Expected cosine similarities:
#   cos(V_EXACT, V_EXACT) ≈ 1.0
#   cos(V_EXACT, V_PARA)  ≈ 0.99
#   cos(V_EXACT, V_NEAR)  ≈ 0.87
#   cos(V_EXACT, V_OTHER) ≈ 0.0


def test_cosine_similarity_values():
    assert cosine_similarity(V_EXACT, V_EXACT) == pytest.approx(1.0, abs=1e-6)
    assert cosine_similarity(V_EXACT, V_PARA) == pytest.approx(0.99, abs=1e-2)
    assert cosine_similarity(V_EXACT, V_NEAR) == pytest.approx(0.87, abs=1e-2)
    assert cosine_similarity(V_EXACT, V_OTHER) == pytest.approx(0.0, abs=1e-6)


def test_normalize_text():
    assert normalize_text("  What   ARE the  Five Pillars? ") == "what are the five pillars?"


# ---------------------------------------------------------------------------
# Semantic cache tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_cache():
    cache = get_cache()
    cache.clear()
    cache.hits = 0
    cache.misses = 0
    cache.bypasses = 0
    cache.evictions = 0
    cache.tokens_saved = 0
    set_fake_embedding(None)

    # Reset exact cache
    exact_cache = get_chat_exact_cache()
    exact_cache.clear()

    # Reset token quota tracker
    quota_tracker = get_token_quota_tracker()
    quota_tracker._window_seconds = 3600.0
    quota_tracker.reset()

    patcher = patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True)
    patcher.start()
    yield
    patcher.stop()


def make_history(prompt: str, response: str):
    return [
        FakeMessage(role="user", content=prompt),
        FakeMessage(role="model", content=response),
    ]


class TestExactHit:
    def test_exact_match_is_found(self):
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("what are the five pillars", "answer")
        cache.put(V_EXACT, "answer", "cid-1", history)
        set_fake_embedding(V_EXACT)
        entry = cache.get(V_EXACT)
        assert entry is not None
        assert entry.response == "answer"

    def test_stats_hit_incremented(self):
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("q", "a")
        cache.put(V_EXACT, "answer", "cid-1", history)
        cache.get(V_EXACT)
        assert cache.hits == 1

    def test_paraphrase_above_threshold(self):
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("what are the five pillars", "answer")
        cache.put(V_EXACT, "answer", "cid-1", history)
        set_fake_embedding(V_PARA)
        entry = cache.get(V_PARA)
        assert entry is not None
        assert entry.response == "answer"

    def test_near_miss_below_threshold(self):
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("can I combine prayers while traveling", "answer_a")
        cache.put(V_EXACT, "answer_a", "cid-1", history)
        set_fake_embedding(V_NEAR)
        entry = cache.get(V_NEAR)
        assert entry is None
        assert cache.misses == 1


class TestTTL:
    def test_expired_entry_not_served(self):
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("q", "a")
        cache.put(V_EXACT, "answer", "cid-1", history)
        # Manually expire the entry
        cache._entries[0].expires_at = time.time() - 1
        entry = cache.get(V_EXACT)
        assert entry is None
        assert cache.misses == 1

    def test_lazy_expiry_evicts(self):
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("q", "a")
        cache.put(V_EXACT, "answer", "cid-1", history)
        cache._entries[0].expires_at = time.time() - 1
        cache.get(V_EXACT)
        assert len(cache._entries) == 0


class TestBypass:
    def test_bypass_increments_counter(self):
        cache = get_cache()
        cache.bypasses += 1
        stats = cache.get_stats()
        assert stats["bypasses"] == 1


class TestHistoryContextExclusion:
    def test_new_chat_without_context_is_cacheable(self):
        """This test verifies the logic that determines cacheability.
        The actual decision lives in main.py's chat() — here we just
        verify the cache itself doesn't impose extra restrictions."""
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("q", "a")
        cache.put(V_EXACT, "answer", "cid-1", history)
        entry = cache.get(V_EXACT)
        assert entry is not None


class TestStatsCounters:
    def test_stats_return_all_keys(self):
        cache = get_cache()
        stats = cache.get_stats()
        expected_keys = {
            "hits",
            "misses",
            "bypasses",
            "evictions",
            "hit_rate",
            "size",
            "max_entries",
            "threshold",
            "ttl_seconds",
            "enabled",
            "tokens_saved",
        }
        assert set(stats.keys()) == expected_keys

    def test_hit_rate_zero_when_empty(self):
        cache = get_cache()
        assert cache.get_stats()["hit_rate"] == 0.0

    def test_hit_rate_after_hits_and_misses(self):
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("q", "a")
        cache.put(V_EXACT, "answer", "cid-1", history)
        cache.get(V_EXACT)
        cache.get(V_OTHER)
        assert cache.hits == 1
        assert cache.misses == 1
        assert cache.get_stats()["hit_rate"] == 0.5


class TestEviction:
    def test_lru_eviction_when_full(self):
        cache = get_cache()
        max_entries = SEMANTIC_CACHE_MAX_ENTRIES

        # Fill to capacity
        set_fake_embedding(V_EXACT)
        for i in range(max_entries + 1):
            vec = np.array([float(i), 0.0, 0.0, 0.0], dtype=np.float32)
            cache.put(vec, f"answer-{i}", f"cid-{i}", make_history(f"q{i}", f"a{i}"))

        assert len(cache._entries) == max_entries
        assert cache.evictions >= 1


class TestCacheDisabled:
    def test_get_returns_none_when_disabled(self):
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("q", "a")
        cache.put(V_EXACT, "answer", "cid-1", history)
        with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", False):
            entry = cache.get(V_EXACT)
        assert entry is None

    def test_put_does_not_store_when_disabled(self):
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("q", "a")
        with patch("semantic_cache.SEMANTIC_CACHE_ENABLED", False):
            cache.put(V_EXACT, "answer", "cid-1", history)
        assert len(cache._entries) == 0


class TestCosineSimilarityEdgeCases:
    def test_zero_vector(self):
        zero = np.zeros(4, dtype=np.float32)
        assert cosine_similarity(zero, V_EXACT) == 0.0
        assert cosine_similarity(V_EXACT, zero) == 0.0
        assert cosine_similarity(zero, zero) == 0.0


class TestEntryExpiredProperty:
    def test_expired_property(self):
        entry = CacheEntry(
            embedding=V_EXACT,
            response="test",
            chat_id="cid",
            history=[],
            expires_at=time.time() - 1,
        )
        assert entry.expired is True

    def test_not_expired(self):
        entry = CacheEntry(
            embedding=V_EXACT,
            response="test",
            chat_id="cid",
            history=[],
            expires_at=time.time() + 3600,
        )
        assert entry.expired is False


class TestScopeIsolation:
    def test_public_scope_does_not_match_user_scope(self):
        """Public entries should never be returned to user-scoped lookups."""
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("what is zakat", "zakat is...")

        # Store in public scope
        cache.put(V_EXACT, "public answer", "cid-1", history, scope="public", token_count=100)

        # Try to retrieve with user scope - should miss
        set_fake_embedding(V_EXACT)
        entry = cache.get(V_EXACT, scope="user:user123")
        assert entry is None
        assert cache.misses == 1

    def test_user_scope_does_not_match_another_user_scope(self):
        """User A's entries should never be returned to user B."""
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("what is zakat", "zakat is...")

        # Store for user A
        cache.put(V_EXACT, "user A answer", "cid-1", history, scope="user:userA", token_count=100)

        # Try to retrieve for user B - should miss
        set_fake_embedding(V_EXACT)
        entry = cache.get(V_EXACT, scope="user:userB")
        assert entry is None
        assert cache.misses == 1

    def test_same_scope_matches(self):
        """Entries should match when scope is identical."""
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("what is zakat", "zakat is...")

        # Store for user A
        cache.put(V_EXACT, "user A answer", "cid-1", history, scope="user:userA", token_count=100)

        # Retrieve for user A - should hit
        set_fake_embedding(V_EXACT)
        entry = cache.get(V_EXACT, scope="user:userA")
        assert entry is not None
        assert entry.response == "user A answer"
        assert cache.hits == 1

    def test_invalidate_by_scope(self):
        """Invalidation by scope should only remove entries in that scope."""
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("q", "a")

        # Store entries in different scopes
        cache.put(V_EXACT, "public answer", "cid-1", history, scope="public", token_count=100)
        cache.put(V_EXACT, "user A answer", "cid-2", history, scope="user:userA", token_count=100)

        # Invalidate user A's cache
        invalidated = cache.invalidate_by_scope("user:userA")
        assert invalidated == 1
        assert len(cache._entries) == 1

        # Public entry should still be accessible
        set_fake_embedding(V_EXACT)
        entry = cache.get(V_EXACT, scope="public")
        assert entry is not None
        assert entry.response == "public answer"

    def test_invalidate_user_cache_function(self):
        """The invalidate_user_cache function should clear both cache tiers."""
        cache = get_cache()
        exact_cache = get_chat_exact_cache()

        # Add entries to both caches for user A
        history = make_history("q", "a")
        cache.put(V_EXACT, "semantic answer", "cid-1", history, scope="user:userA", token_count=100)
        exact_cache.put("user:userA:q", {"response": "exact answer", "history": history}, token_count=50)

        # Invalidate user A's cache
        invalidated = invalidate_user_cache("userA")
        assert invalidated == 2  # One from semantic, one from exact

        # Both should be empty for user A
        set_fake_embedding(V_EXACT)
        assert cache.get(V_EXACT, scope="user:userA") is None
        assert exact_cache.get("user:userA:q") is None

    def test_invalidate_public_cache_function(self):
        """The invalidate_public_cache function should clear public entries."""
        cache = get_cache()
        exact_cache = get_chat_exact_cache()

        # Add public entries
        history = make_history("q", "a")
        cache.put(V_EXACT, "semantic answer", "cid-1", history, scope="public", token_count=100)
        exact_cache.put("public:q", {"response": "exact answer", "history": history}, token_count=50)

        # Invalidate public cache
        invalidated = invalidate_public_cache()
        assert invalidated == 2

        # Both should be empty for public scope
        set_fake_embedding(V_EXACT)
        assert cache.get(V_EXACT, scope="public") is None
        assert exact_cache.get("public:q") is None


class TestExactCache:
    def test_exact_match_hit(self):
        """Exact cache should return identical prompts."""
        exact_cache = get_chat_exact_cache()
        history = make_history("q", "a")

        exact_cache.put("public:what is zakat", {"response": "zakat is...", "history": history}, token_count=50)
        result = exact_cache.get("public:what is zakat")

        assert result is not None
        assert result["response"] == "zakat is..."
        assert exact_cache.hits == 1

    def test_exact_miss_different_prompt(self):
        """Exact cache should miss on different prompts."""
        exact_cache = get_chat_exact_cache()
        history = make_history("q", "a")

        exact_cache.put("public:what is zakat", {"response": "zakat is...", "history": history}, token_count=50)
        result = exact_cache.get("public:what is prayer")

        assert result is None
        assert exact_cache.misses == 1

    def test_exact_cache_scope_isolation(self):
        """Exact cache should respect scope prefixes."""
        exact_cache = get_chat_exact_cache()
        history = make_history("q", "a")

        exact_cache.put("public:q", {"response": "public answer", "history": history}, token_count=50)
        exact_cache.put("user:userA:q", {"response": "user A answer", "history": history}, token_count=50)

        # Each scope should only see its own entries
        assert exact_cache.get("public:q")["response"] == "public answer"
        assert exact_cache.get("user:userA:q")["response"] == "user A answer"
        assert exact_cache.get("user:userB:q") is None

    def test_exact_cache_invalidate_by_prefix(self):
        """Exact cache should support prefix-based invalidation."""
        exact_cache = get_chat_exact_cache()
        history = make_history("q", "a")

        exact_cache.put("public:q1", {"response": "a1", "history": history}, token_count=50)
        exact_cache.put("public:q2", {"response": "a2", "history": history}, token_count=50)
        exact_cache.put("user:userA:q", {"response": "user a", "history": history}, token_count=50)

        # Invalidate all public entries
        invalidated = exact_cache.invalidate_by_prefix("public:")
        assert invalidated == 2
        assert exact_cache.get("public:q1") is None
        assert exact_cache.get("public:q2") is None
        assert exact_cache.get("user:userA:q") is not None


class TestTokenSavings:
    def test_semantic_cache_tracks_tokens_saved(self):
        """Semantic cache should track tokens saved on hits."""
        cache = get_cache()
        set_fake_embedding(V_EXACT)
        history = make_history("q", "a")

        cache.put(V_EXACT, "answer", "cid-1", history, scope="public", token_count=150)
        set_fake_embedding(V_EXACT)
        cache.get(V_EXACT, scope="public")

        assert cache.tokens_saved == 150
        stats = cache.get_stats()
        assert stats["tokens_saved"] == 150

    def test_exact_cache_tracks_tokens_saved(self):
        """Exact cache should track tokens saved on hits."""
        exact_cache = get_chat_exact_cache()
        history = make_history("q", "a")

        exact_cache.put("public:q", {"response": "answer", "history": history}, token_count=75)
        exact_cache.get("public:q")

        assert exact_cache.tokens_saved == 75
        stats = exact_cache.get_stats()
        assert stats["tokens_saved"] == 75


class TestTokenQuotaTracker:
    def test_quota_allows_under_limit(self):
        """Quota tracker should allow requests under the limit."""
        tracker = get_token_quota_tracker()

        allowed, retry_after = tracker.is_allowed("user1", 1000)
        assert allowed is True
        assert retry_after is None

    def test_quota_blocks_over_limit(self):
        """Quota tracker should block requests over the limit."""
        tracker = get_token_quota_tracker()
        quota = tracker._quota

        # Use up the quota
        allowed, _ = tracker.is_allowed("user1", quota - 1)
        assert allowed is True

        # Next request should be blocked
        allowed, retry_after = tracker.is_allowed("user1", 10)
        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0

    def test_quota_is_per_key(self):
        """Quota should be tracked separately per key."""
        tracker = get_token_quota_tracker()
        quota = tracker._quota

        # User 1 uses quota
        tracker.is_allowed("user1", quota)

        # User 2 should still be allowed
        allowed, _ = tracker.is_allowed("user2", quota)
        assert allowed is True

    def test_quota_sweeps_expired_entries(self):
        """Quota tracker should sweep expired entries."""
        tracker = get_token_quota_tracker()
        tracker._window_seconds = 0.1  # Very short window for testing

        # Add usage
        tracker.is_allowed("user1", 1000)

        # Wait for window to expire
        import time as time_module

        time_module.sleep(0.15)

        # Should be allowed again
        allowed, _ = tracker.is_allowed("user1", 1000)
        assert allowed is True

    def test_quota_get_usage(self):
        """Quota tracker should report current usage."""
        tracker = get_token_quota_tracker()

        tracker.is_allowed("user1", 500)
        tracker.is_allowed("user1", 300)

        usage = tracker.get_usage("user1")
        assert usage == 800
