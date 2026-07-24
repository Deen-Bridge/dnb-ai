"""Memory store abstraction — Redis-backed with in-memory fallback.

REDIS_URL absent  → InMemoryMemoryStore (process-local, lost on restart).
REDIS_URL present → RedisMemoryStore. Connection failures surface
                    (log + raise) — never silently switch to in-memory,
                    because two workers would diverge on user memory.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import time
from typing import Optional

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


class MemoryStore(abc.ABC):
    @abc.abstractmethod
    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        ...

    @abc.abstractmethod
    async def save_profile(self, user_id: str, profile: UserProfile) -> None:
        ...

    @abc.abstractmethod
    async def delete_profile(self, user_id: str) -> bool:
        ...

    @abc.abstractmethod
    async def get_chat_summary(self, chat_id: str) -> Optional[ChatSummary]:
        ...

    @abc.abstractmethod
    async def save_chat_summary(self, chat_id: str, summary: ChatSummary) -> None:
        ...

    @abc.abstractmethod
    async def delete_chat_summary(self, chat_id: str) -> bool:
        ...


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._profiles: dict[str, tuple[float, UserProfile]] = {}
        self._summaries: dict[str, tuple[float, ChatSummary]] = {}

    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        entry = self._profiles.get(user_id)
        if entry is None:
            return None
        expires_at, profile = entry
        if time.monotonic() > expires_at:
            del self._profiles[user_id]
            return None
        return profile

    async def save_profile(self, user_id: str, profile: UserProfile) -> None:
        self._profiles[user_id] = (time.monotonic() + MEMORY_TTL_SECONDS, profile)

    async def delete_profile(self, user_id: str) -> bool:
        return self._profiles.pop(user_id, None) is not None

    async def get_chat_summary(self, chat_id: str) -> Optional[ChatSummary]:
        entry = self._summaries.get(chat_id)
        if entry is None:
            return None
        expires_at, summary = entry
        if time.monotonic() > expires_at:
            del self._summaries[chat_id]
            return None
        return summary

    async def save_chat_summary(self, chat_id: str, summary: ChatSummary) -> None:
        self._summaries[chat_id] = (time.monotonic() + MEMORY_TTL_SECONDS, summary)

    async def delete_chat_summary(self, chat_id: str) -> bool:
        return self._summaries.pop(chat_id, None) is not None


class RedisMemoryStore(MemoryStore):
    def __init__(self, redis_url: str) -> None:
        if not _redis_available:
            raise RuntimeError("redis package not installed")
        self._redis = aioredis.from_url(redis_url, decode_responses=True)

    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        raw = await self._redis.get(_profile_key(user_id))
        if raw is None:
            return None
        try:
            return UserProfile.model_validate(json.loads(raw))
        except (json.JSONDecodeError, Exception):
            logger.warning("Corrupt profile for user %s", user_id)
            return None

    async def save_profile(self, user_id: str, profile: UserProfile) -> None:
        await self._redis.setex(
            _profile_key(user_id),
            MEMORY_TTL_SECONDS,
            profile.model_dump_json(),
        )

    async def delete_profile(self, user_id: str) -> bool:
        deleted = await self._redis.delete(_profile_key(user_id))
        return deleted > 0

    async def get_chat_summary(self, chat_id: str) -> Optional[ChatSummary]:
        raw = await self._redis.get(_summary_key(chat_id))
        if raw is None:
            return None
        try:
            return ChatSummary.model_validate(json.loads(raw))
        except (json.JSONDecodeError, Exception):
            logger.warning("Corrupt chat summary for chat %s", chat_id)
            return None

    async def save_chat_summary(self, chat_id: str, summary: ChatSummary) -> None:
        await self._redis.setex(
            _summary_key(chat_id),
            MEMORY_TTL_SECONDS,
            summary.model_dump_json(),
        )

    async def delete_chat_summary(self, chat_id: str) -> bool:
        deleted = await self._redis.delete(_summary_key(chat_id))
        return deleted > 0
