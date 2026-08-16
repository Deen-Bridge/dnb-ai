"""Memory store abstraction — Redis-backed with in-memory fallback.

REDIS_URL absent  → InMemoryMemoryStore (process-local, lost on restart).
REDIS_URL present → RedisMemoryStore. Connection failures surface
                    (log + raise) — never silently switch to in-memory,
                    because two workers would diverge on user memory.

Authorization
-------------
``get_profile`` and ``get_chat_summary`` accept an optional ``scope``
(``RetrievalScope``).  When a scope is supplied the store enforces:

  * **Anonymous scope** — per-user data is structurally inaccessible; the
    methods return ``None`` and log a denial.
  * **Principal scope** — the resolved profile / summary is post-checked with
    ``visible_to``; any record whose ``user_id`` does not match the principal
    is dropped and logged, never returned.

Passing ``scope=None`` (default) preserves the previous unchecked behaviour so
existing callers (e.g. the memory-extraction background task that already runs
under the right user id) continue to work unchanged.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import time

from memory.models import ChatSummary, UserProfile

logger = logging.getLogger(__name__)

MEMORY_TTL_SECONDS = int(os.getenv("MEMORY_TTL_DAYS", "90")) * 86400
REDIS_URL = os.getenv("REDIS_URL", "")

try:
    import redis.asyncio as aioredis

    _redis_available = True
except ImportError:
    _redis_available = False


def _profile_key(user_id: str) -> str:
    return f"memory:profile:{user_id}"


def _summary_key(chat_id: str) -> str:
    return f"memory:summary:{chat_id}"


# ---------------------------------------------------------------------------
# Scope-enforcement helpers (imported lazily to avoid circular imports between
# memory.store and retrieval.scope at module load time).
# ---------------------------------------------------------------------------


def _check_profile_scope(scope: object, user_id: str, profile: "UserProfile | None") -> "UserProfile | None":
    """Apply scope visibility rules to a loaded profile.

    Returns the profile if access is allowed, ``None`` if denied.
    """
    if scope is None:
        # Legacy / unchecked path — preserve existing behaviour.
        return profile

    # Import here (not at module top) to avoid circular dependencies during
    # early module initialisation.
    from retrieval.scope import RetrievalScope, UserRole, visible_to  # noqa: PLC0415

    assert isinstance(scope, RetrievalScope)

    if scope.principal_user_id is None:
        logger.warning(
            "retrieval_denied principal=anonymous requested_user_id=%s reason=anonymous scope cannot access profile",
            user_id[:8] if len(user_id) >= 8 else user_id,
        )
        return None

    if scope.principal_user_id != user_id:
        logger.warning(
            "retrieval_denied principal=%s requested_user_id=%s reason=cross-user profile access denied",
            scope.principal_user_id[:8],
            user_id[:8] if len(user_id) >= 8 else user_id,
        )
        return None

    if profile is not None and not visible_to(profile, scope):
        return None

    return profile


def _check_summary_scope(scope: object, chat_id: str, summary: "ChatSummary | None") -> "ChatSummary | None":
    """Apply scope visibility rules to a loaded chat summary.

    Chat summaries are keyed as ``{user_id}:{chat_id}``.  The user_id prefix
    is extracted and compared against the scope.
    """
    if scope is None:
        return summary

    from retrieval.scope import RetrievalScope, visible_to  # noqa: PLC0415

    assert isinstance(scope, RetrievalScope)

    # chat_id arriving here is expected to be "{user_id}:{actual_chat_id}".
    # Extract the owner from the prefix.
    if ":" in chat_id:
        owner_prefix = chat_id.split(":", 1)[0]
    else:
        # No owner prefix — treat as ownerless (backwards-compat with plain chat IDs).
        owner_prefix = None

    if scope.principal_user_id is None:
        if owner_prefix is not None:
            logger.warning(
                "retrieval_denied principal=anonymous chat_id=%s reason=anonymous scope cannot access summary",
                chat_id[:16] if len(chat_id) >= 16 else chat_id,
            )
            return None
        return summary

    if owner_prefix is not None and scope.principal_user_id != owner_prefix:
        logger.warning(
            "retrieval_denied principal=%s chat_owner=%s reason=cross-user summary access denied",
            scope.principal_user_id[:8],
            owner_prefix[:8] if len(owner_prefix) >= 8 else owner_prefix,
        )
        return None

    return summary


class MemoryStore(abc.ABC):
    @abc.abstractmethod
    async def get_profile(self, user_id: str, *, scope: object = None) -> UserProfile | None: ...

    @abc.abstractmethod
    async def save_profile(self, user_id: str, profile: UserProfile) -> None: ...

    @abc.abstractmethod
    async def delete_profile(self, user_id: str) -> bool: ...

    @abc.abstractmethod
    async def get_chat_summary(self, chat_id: str, *, scope: object = None) -> ChatSummary | None: ...

    @abc.abstractmethod
    async def save_chat_summary(self, chat_id: str, summary: ChatSummary) -> None: ...

    @abc.abstractmethod
    async def delete_chat_summary(self, chat_id: str) -> bool: ...


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._profiles: dict[str, tuple[float, UserProfile]] = {}
        self._summaries: dict[str, tuple[float, ChatSummary]] = {}

    async def get_profile(self, user_id: str, *, scope: object = None) -> UserProfile | None:
        entry = self._profiles.get(user_id)
        if entry is None:
            return None
        expires_at, profile = entry
        if time.monotonic() > expires_at:
            del self._profiles[user_id]
            return None
        return _check_profile_scope(scope, user_id, profile)

    async def save_profile(self, user_id: str, profile: UserProfile) -> None:
        self._profiles[user_id] = (time.monotonic() + MEMORY_TTL_SECONDS, profile)

    async def delete_profile(self, user_id: str) -> bool:
        return self._profiles.pop(user_id, None) is not None

    async def get_chat_summary(self, chat_id: str, *, scope: object = None) -> ChatSummary | None:
        entry = self._summaries.get(chat_id)
        if entry is None:
            return None
        expires_at, summary = entry
        if time.monotonic() > expires_at:
            del self._summaries[chat_id]
            return None
        return _check_summary_scope(scope, chat_id, summary)

    async def save_chat_summary(self, chat_id: str, summary: ChatSummary) -> None:
        self._summaries[chat_id] = (time.monotonic() + MEMORY_TTL_SECONDS, summary)

    async def delete_chat_summary(self, chat_id: str) -> bool:
        return self._summaries.pop(chat_id, None) is not None


class RedisMemoryStore(MemoryStore):
    def __init__(self, redis_url: str) -> None:
        if not _redis_available:
            raise RuntimeError("redis package not installed")
        self._redis = aioredis.from_url(redis_url, decode_responses=True)

    async def get_profile(self, user_id: str, *, scope: object = None) -> UserProfile | None:
        raw = await self._redis.get(_profile_key(user_id))
        if raw is None:
            return None
        try:
            profile = UserProfile.model_validate(json.loads(raw))
        except (json.JSONDecodeError, Exception):
            logger.warning("Corrupt profile for user %s", user_id)
            return None
        return _check_profile_scope(scope, user_id, profile)

    async def save_profile(self, user_id: str, profile: UserProfile) -> None:
        await self._redis.setex(
            _profile_key(user_id),
            MEMORY_TTL_SECONDS,
            profile.model_dump_json(),
        )

    async def delete_profile(self, user_id: str) -> bool:
        deleted = await self._redis.delete(_profile_key(user_id))
        return deleted > 0

    async def get_chat_summary(self, chat_id: str, *, scope: object = None) -> ChatSummary | None:
        raw = await self._redis.get(_summary_key(chat_id))
        if raw is None:
            return None
        try:
            summary = ChatSummary.model_validate(json.loads(raw))
        except (json.JSONDecodeError, Exception):
            logger.warning("Corrupt chat summary for chat %s", chat_id)
            return None
        return _check_summary_scope(scope, chat_id, summary)

    async def save_chat_summary(self, chat_id: str, summary: ChatSummary) -> None:
        await self._redis.setex(
            _summary_key(chat_id),
            MEMORY_TTL_SECONDS,
            summary.model_dump_json(),
        )

    async def delete_chat_summary(self, chat_id: str) -> bool:
        deleted = await self._redis.delete(_summary_key(chat_id))
        return deleted > 0
