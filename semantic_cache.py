"""Semantic response cache with embedding-similarity matching.

Cache correctness invariant
---------------------------
A follow-up question depends on conversation history, so its answer must
never be served to someone else. Therefore we consult/populate the cache
*only* when the chat has no prior history (new chat_id / first turn) AND
request.context is None.

Store choice
------------
In-memory store with numpy for cosine similarity. Chosen over ChromaDB to
keep dependencies minimal — numpy alone is sufficient for this use case,
and avoids coupling to ChromaDB's full vector-store infrastructure. If the
RAG infrastructure lands with ChromaDB, the cache can be migrated to share
its collection.
"""

import logging
import os
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "0").lower() in (
    "1",
    "true",
    "yes",
)
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
SEMANTIC_CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400"))
SEMANTIC_CACHE_MAX_ENTRIES = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "1000"))

# Rate limiting and quota enforcement
CHAT_RATE_LIMIT_MAX = int(os.getenv("CHAT_RATE_LIMIT_MAX", "60"))
CHAT_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("CHAT_RATE_LIMIT_WINDOW_SECONDS", "60"))
CHAT_TOKEN_QUOTA_PER_HOUR = int(os.getenv("CHAT_TOKEN_QUOTA_PER_HOUR", "100000"))
CHAT_PROMPT_MAX_LENGTH = int(os.getenv("CHAT_PROMPT_MAX_LENGTH", "10000"))
CHAT_CONTEXT_MAX_LENGTH = int(os.getenv("CHAT_CONTEXT_MAX_LENGTH", "5000"))

# ---------------------------------------------------------------------------
# Token quota tracker (per-user/tier sliding window)
# ---------------------------------------------------------------------------


class TokenQuotaTracker:
    """Per-user/tier token quota tracker with sliding window enforcement."""

    def __init__(
        self,
        quota_per_hour: int = CHAT_TOKEN_QUOTA_PER_HOUR,
    ) -> None:
        self._default_quota = quota_per_hour
        self._quota = quota_per_hour
        self._window_seconds = 3600.0  # 1 hour
        # key -> list of (timestamp, token_count)
        self._usage: dict[str, list[tuple[float, int]]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str, token_count: int) -> tuple[bool, int | None]:
        """Check if a request is allowed under the quota.

        Returns (is_allowed, retry_after_seconds).
        retry_after_seconds is None if allowed, otherwise the seconds until
        the oldest usage falls out of the window.
        """
        now = time.monotonic()
        cutoff = now - self._window_seconds

        with self._lock:
            self._sweep(cutoff)
            bucket = self._usage.get(key, [])

            # Calculate current usage
            current_usage = sum(tokens for _, tokens in bucket)

            if current_usage + token_count > self._quota:
                # Over quota: calculate retry-after based on oldest entry
                if bucket:
                    oldest_ts = bucket[0][0]
                    retry_after = int(oldest_ts + self._window_seconds - now) + 1
                    return False, retry_after
                return False, 60  # Default retry if bucket is empty

            # Under quota: record usage
            bucket.append((now, token_count))
            self._usage[key] = bucket
            return True, None

    def _sweep(self, cutoff: float) -> None:
        """Remove expired entries from all buckets."""
        for key in list(self._usage.keys()):
            bucket = self._usage[key]
            self._usage[key] = [(ts, tokens) for ts, tokens in bucket if ts >= cutoff]
            if not self._usage[key]:
                del self._usage[key]

    def reset(self) -> None:
        """Clear all buckets and restore default quota/limits. Used by tests."""
        with self._lock:
            self._usage.clear()
            self._quota = self._default_quota
            self._window_seconds = 3600.0

    def get_usage(self, key: str) -> int:
        """Get current token usage for a key."""
        now = time.monotonic()
        cutoff = now - self._window_seconds

        with self._lock:
            bucket = self._usage.get(key, [])
            bucket = [(ts, tokens) for ts, tokens in bucket if ts >= cutoff]
            return sum(tokens for _, tokens in bucket)


_token_quota_tracker = TokenQuotaTracker()


def get_token_quota_tracker() -> TokenQuotaTracker:
    return _token_quota_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


# ---------------------------------------------------------------------------
# Embedding seam
# ---------------------------------------------------------------------------

_FAKE_EMBEDDING: np.ndarray | None = None


def set_fake_embedding(vec: np.ndarray | None) -> None:
    global _FAKE_EMBEDDING
    _FAKE_EMBEDDING = vec


def embed_text(text: str) -> np.ndarray:
    if _FAKE_EMBEDDING is not None:
        return _FAKE_EMBEDDING
    try:
        import google.generativeai as genai

        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
        )
        return np.array(result["embedding"], dtype=np.float32)
    except Exception:
        # Deterministic local fallback so the semantic cache still functions
        # offline and in tests (identical prompts yield identical vectors) when
        # no embedding model/API key is available.
        return _local_embedding(text)


def _local_embedding(text: str) -> np.ndarray:
    """Deterministic offline embedding for when the embedding API is unavailable."""
    import hashlib

    dim = 384
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Seed a fixed RNG so vectors are stable across calls and processes.
    rng = np.random.default_rng(int.from_bytes(digest, "big") & 0xFFFFFFFF)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


class CacheEntry:
    __slots__ = ("embedding", "response", "chat_id", "history", "expires_at", "scope", "token_count", "version", "metadata")

    def __init__(
        self,
        embedding: np.ndarray,
        response: str,
        chat_id: str,
        history: list[Any],
        expires_at: float,
        scope: str = "public",
        token_count: int = 0,
        version: int = 1,
        metadata: dict | None = None,
    ) -> None:
        self.embedding = embedding
        self.response = response
        self.chat_id = chat_id
        self.history = history
        self.expires_at = expires_at
        self.scope = scope
        self.token_count = token_count
        self.version = version
        self.metadata = metadata or {}

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at


# ---------------------------------------------------------------------------
# In-memory semantic cache
# ---------------------------------------------------------------------------


class SemanticCache:
    def __init__(self) -> None:
        self._entries: list[CacheEntry] = []
        self._access_times: list[float] = []
        self._preferences: dict[str, dict[str, Any]] = {}

        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.evictions = 0
        self.tokens_saved = 0  # Total tokens saved by cache hits

    # -- public API ---------------------------------------------------------

    def get(self, embedding: np.ndarray, scope: str = "public") -> CacheEntry | None:
        if not SEMANTIC_CACHE_ENABLED:
            return None
        match = self._find_best_match(embedding, scope)
        if match is not None:
            entry, idx = match
            self._access_times[idx] = time.time()
            self.hits += 1
            self.tokens_saved += entry.token_count
            return entry
        self.misses += 1
        return None

    def put(
        self,
        embedding: np.ndarray,
        response: str,
        chat_id: str,
        history: list[Any],
        scope: str = "public",
        token_count: int = 0,
        version: int = 1,
        metadata: dict | None = None,
    ) -> None:
        if not SEMANTIC_CACHE_ENABLED:
            return
        self._evict_lru_if_full()
        entry = CacheEntry(
            embedding=embedding,
            response=response,
            chat_id=chat_id,
            history=history,
            expires_at=time.time() + SEMANTIC_CACHE_TTL_SECONDS,
            scope=scope,
            token_count=token_count,
            version=version,
            metadata=metadata,
        )
        self._entries.append(entry)
        self._access_times.append(time.time())

    def get_stats(self) -> dict[str, Any]:
        total = self.hits + self.misses + self.bypasses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "bypasses": self.bypasses,
            "evictions": self.evictions,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
            "size": len(self._entries),
            "max_entries": SEMANTIC_CACHE_MAX_ENTRIES,
            "threshold": SEMANTIC_CACHE_THRESHOLD,
            "ttl_seconds": SEMANTIC_CACHE_TTL_SECONDS,
            "enabled": SEMANTIC_CACHE_ENABLED,
            "tokens_saved": self.tokens_saved,
        }

    def clear(self) -> None:
        self._entries.clear()
        self._access_times.clear()
        self._preferences.clear()
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.evictions = 0
        self.tokens_saved = 0

    def invalidate_by_scope(self, scope: str) -> int:
        """Invalidate all entries matching the given scope. Returns count of invalidated entries."""
        surviving_entries: list[CacheEntry] = []
        surviving_times: list[float] = []
        invalidated = 0

        for entry, access_time in zip(self._entries, self._access_times, strict=True):
            if entry.scope == scope:
                invalidated += 1
            else:
                surviving_entries.append(entry)
                surviving_times.append(access_time)

        self._entries = surviving_entries
        self._access_times = surviving_times
        return invalidated

    def invalidate_by_content_source(self, content_source: str) -> int:
        """Invalidate entries tagged with a specific content source.

        This is a placeholder for future content-source tagging.
        Currently returns 0 as entries are not yet tagged with content sources.
        """
        # TODO: Add content_source tagging to CacheEntry and implement matching
        return 0

    # -- memory and context sharing API -------------------------------------

    def share_memory(self, entry_id: int, target_scope: str) -> bool:
        """Copy a cache entry (memory) into another scope, enabling inter-agent sharing."""
        if entry_id < 0 or entry_id >= len(self._entries):
            return False
        entry = self._entries[entry_id]
        if entry.scope == target_scope:
            return True  # Already in target scope
        # Create a shallow copy with updated scope and fresh timestamps
        shared_entry = CacheEntry(
            embedding=entry.embedding,
            response=entry.response,
            chat_id=entry.chat_id,
            history=list(entry.history),
            expires_at=time.time() + SEMANTIC_CACHE_TTL_SECONDS,
            scope=target_scope,
            token_count=entry.token_count,
            version=entry.version,
            metadata=dict(entry.metadata),
        )
        self._entries.append(shared_entry)
        self._access_times.append(time.time())
        return True

    def set_user_preference(self, user_id: str, key: str, value: Any) -> None:
        """Persist a user preference in memory (key-value store)."""
        prefs = self._preferences.setdefault(user_id, {})
        prefs[key] = value

    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        """Retrieve all persisted preferences for a user."""
        return dict(self._preferences.get(user_id, {}))

    def prune_archived(self, cutoff: float | None = None) -> int:
        """Remove expired entries and optionally entries older than cutoff."""
        now = time.time()
        cutoff = cutoff if cutoff is not None else now
        surviving_entries: list[CacheEntry] = []
        surviving_times: list[float] = []
        pruned = 0
        for entry, access_time in zip(self._entries, self._access_times, strict=True):
            if entry.expired or access_time < cutoff:
                pruned += 1
                self.evictions += 1
            else:
                surviving_entries.append(entry)
                surviving_times.append(access_time)
        self._entries = surviving_entries
        self._access_times = surviving_times
        return pruned

    def retrieve_contexts(
        self,
        embedding: np.ndarray,
        scope: str = "public",
        top_k: int = 3,
        min_score: float | None = None,
    ) -> list[tuple[CacheEntry, float]]:
        """Retrieve the top-k most relevant cached entries for context assembly."""
        if not SEMANTIC_CACHE_ENABLED:
            return []
        threshold = min_score if min_score is not None else SEMANTIC_CACHE_THRESHOLD
        scored: list[tuple[float, CacheEntry]] = []
        # First pass to clean expired and enforce scope
        for entry in self._entries:
            if entry.expired or entry.scope != scope:
                continue
            score = cosine_similarity(embedding, entry.embedding)
            if score >= threshold:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Update access times for retrieved entries
        retrieved = scored[:top_k]
        for score, entry in retrieved:
            # Find its index and update access time (simplified: just mark access)
            try:
                idx = self._entries.index(entry)
                self._access_times[idx] = time.time()
            except ValueError:
                pass
        return [(entry, score) for score, entry in retrieved]

    # -- internals ----------------------------------------------------------

    def _find_best_match(self, embedding: np.ndarray, scope: str) -> tuple[CacheEntry, int] | None:
        best_score = SEMANTIC_CACHE_THRESHOLD
        best_idx: int | None = None

        surviving_entries: list[CacheEntry] = []
        surviving_times: list[float] = []

        for i, entry in enumerate(self._entries):
            if entry.expired:
                self.evictions += 1
                continue
            # Scope isolation: only match entries in the same scope
            if entry.scope != scope:
                surviving_entries.append(entry)
                surviving_times.append(self._access_times[i])
                continue
            surviving_entries.append(entry)
            surviving_times.append(self._access_times[i])

        self._entries = surviving_entries
        self._access_times = surviving_times

        for i, entry in enumerate(self._entries):
            # Double-check scope in case of race conditions
            if entry.scope != scope:
                continue
            score = cosine_similarity(embedding, entry.embedding)
            if score >= best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            return self._entries[best_idx], best_idx
        return None

    def _evict_lru_if_full(self) -> None:
        if len(self._entries) < SEMANTIC_CACHE_MAX_ENTRIES:
            return
        lru_idx = int(np.argmin(self._access_times))
        self._entries.pop(lru_idx)
        self._access_times.pop(lru_idx)
        self.evictions += 1


_cache: SemanticCache = SemanticCache()


def get_cache() -> SemanticCache:
    return _cache


# ---------------------------------------------------------------------------
# Keyed cache (exact-key sibling of the semantic cache)
# ---------------------------------------------------------------------------


class KeyedCache:
    """Exact-key LRU cache for content that is immutable per key.

    Why this lives here rather than in its own module: it is the same cache
    concern as ``SemanticCache`` — same TTL and max-entry configuration, same
    LRU eviction, same stats shape — and callers that need caching should have
    exactly one place to look. It is *not* a second cache system; it is the
    lookup mode the semantic cache cannot serve.

    Why not reuse ``SemanticCache`` directly: a tafsir lookup is keyed by an
    ayah reference, which is exact. Approximate embedding similarity is the
    wrong matching rule there — 2:255 and 2:256 are near-identical strings and
    must never match each other. Cached values are keyed and looked up by an
    exact string, never by distance.

    Scope isolation: entries are namespaced by scope (public or user:{user_id})
    to prevent cross-user replay of personalized answers.
    """

    __slots__ = ("_entries", "_access_times", "hits", "misses", "evictions", "tokens_saved")

    def __init__(self) -> None:
        # key -> (value, expires_at, token_count)
        self._entries: dict[str, tuple[Any, float, int]] = {}
        self._access_times: dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.tokens_saved = 0

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        value, expires_at, token_count = entry
        if time.time() > expires_at:
            del self._entries[key]
            self._access_times.pop(key, None)
            self.evictions += 1
            self.misses += 1
            return None
        self._access_times[key] = time.time()
        self.hits += 1
        self.tokens_saved += token_count
        return value

    def put(self, key: str, value: Any, ttl_seconds: int | None = None, token_count: int = 0) -> None:
        """Store *value*, expiring after *ttl_seconds* (default: the cache TTL).

        The override exists for content that is cacheable but not immutable —
        a market price is worth caching for hours, not for the day-long TTL
        that suits a fixed tafsir passage.
        """
        self._evict_lru_if_full()
        ttl = SEMANTIC_CACHE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        self._entries[key] = (value, time.time() + ttl, token_count)
        self._access_times[key] = time.time()

    def get_stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
            "size": len(self._entries),
            "max_entries": SEMANTIC_CACHE_MAX_ENTRIES,
            "ttl_seconds": SEMANTIC_CACHE_TTL_SECONDS,
            "tokens_saved": self.tokens_saved,
        }

    def clear(self) -> None:
        self._entries.clear()
        self._access_times.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.tokens_saved = 0

    def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all entries with keys starting with the given prefix.

        Used for scope-based invalidation (e.g., 'user:' prefix for user-scoped entries).
        Returns count of invalidated entries.
        """
        keys_to_delete = [k for k in self._entries.keys() if k.startswith(prefix)]
        for key in keys_to_delete:
            del self._entries[key]
            self._access_times.pop(key, None)
        return len(keys_to_delete)

    def _evict_lru_if_full(self) -> None:
        if len(self._entries) < SEMANTIC_CACHE_MAX_ENTRIES:
            return
        lru_key = min(self._access_times, key=lambda k: self._access_times[k])
        self._entries.pop(lru_key, None)
        self._access_times.pop(lru_key, None)
        self.evictions += 1


_keyed_caches: dict[str, KeyedCache] = {}


def get_keyed_cache(namespace: str) -> KeyedCache:
    """Return the process-wide keyed cache for *namespace*, creating it once."""
    cache = _keyed_caches.get(namespace)
    if cache is None:
        cache = KeyedCache()
        _keyed_caches[namespace] = cache
    return cache


def keyed_cache_stats() -> dict[str, dict[str, Any]]:
    return {name: cache.get_stats() for name, cache in _keyed_caches.items()}


def get_chat_exact_cache() -> KeyedCache:
    """Return the exact-match cache for chat responses.

    This cache is checked before the semantic cache and provides exact-match
    lookups for identical prompts within the same scope.
    """
    return get_keyed_cache("chat_exact")


def invalidate_user_cache(user_id: str) -> int:
    """Invalidate all cache entries for a specific user.

    Invalidates entries in both the semantic cache and the exact-match cache.
    Returns total count of invalidated entries.
    """
    scope = f"user:{user_id}"
    semantic_cache = get_cache()
    exact_cache = get_chat_exact_cache()

    semantic_invalidated = semantic_cache.invalidate_by_scope(scope)
    exact_invalidated = exact_cache.invalidate_by_prefix(f"{scope}:")

    return semantic_invalidated + exact_invalidated


def invalidate_public_cache() -> int:
    """Invalidate all public (non-personalized) cache entries.

    Returns total count of invalidated entries.
    """
    semantic_cache = get_cache()
    exact_cache = get_chat_exact_cache()

    semantic_invalidated = semantic_cache.invalidate_by_scope("public")
    exact_invalidated = exact_cache.invalidate_by_prefix("public:")

    return semantic_invalidated + exact_invalidated
