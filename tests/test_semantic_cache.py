"""Tests for the semantic response cache — no live API calls.

Fake embedding function returns hand-built vectors so all tests run offline.
"""

from dataclasses import dataclass
import time
from unittest.mock import patch

import numpy as np
import pytest

from semantic_cache import (
    SEMANTIC_CACHE_MAX_ENTRIES,
    CacheEntry,
    cosine_similarity,
    get_cache,
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
    set_fake_embedding(None)
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
            "hits", "misses", "bypasses", "evictions", "hit_rate",
            "size", "max_entries", "threshold", "ttl_seconds", "enabled",
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
