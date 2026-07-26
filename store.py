"""Session store abstraction — Redis-backed with in-memory fallback.

On startup, checks for a REDIS_URL environment variable. If set, connects to
Redis and uses it for session persistence (survives restarts, shared across
instances). Falls back to an in-process dict for local development.
"""

import json
import logging
import os
import time
from typing import Optional

import google.generativeai.protos as protos

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24h


try:
    import redis.asyncio as aioredis

    _redis_available = True
except ImportError:
    _redis_available = False


def history_to_dicts(contents: list) -> list[dict[str, str]]:
    """Serialize Gemini Content objects to simple dicts for storage."""
    result = []
    for c in contents:
        text = ""
        if hasattr(c, "parts") and c.parts:
            part = c.parts[0]
            text = part.text if hasattr(part, "text") else str(part)
        result.append({"role": c.role, "text": text})
    return result


def dicts_to_contents(dicts: list[dict[str, str]]) -> list:
    """Reconstruct Gemini Content objects from stored dicts."""
    return [
        protos.Content(role=d["role"], parts=[protos.Part(text=d["text"])])
        for d in dicts
    ]


class SessionStore:
    """Abstracts persistence of chat-session history.

    Each session's history is stored as a JSON-serialized list of
    ``{"role": str, "text": str}`` dicts, keyed by ``chat:{chat_id}``.
    """

    def __init__(self) -> None:
        self._redis: Optional[aioredis.Redis] = None
        self._local: dict[str, tuple[float, list[dict[str, str]]]] = {}
        self._use_redis = False

        if REDIS_URL and _redis_available:
            try:
                self._redis = aioredis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                )
                self._use_redis = True
                logger.info("SessionStore using Redis at %s", REDIS_URL)
            except Exception as exc:
                logger.warning(
                    "Failed to connect to Redis at %s (%s); falling back to in-memory",
                    REDIS_URL,
                    exc,
                )

        if not self._use_redis:
            logger.info("SessionStore using in-memory dict (local development)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load_history(self, chat_id: str) -> list[dict[str, str]]:
        """Return the persisted history for *chat_id*, or an empty list."""
        if self._use_redis:
            return await self._redis_load(chat_id)
        return self._local_load(chat_id)

    async def save_history(
        self, chat_id: str, history: list[dict[str, str]]
    ) -> None:
        """Persist *history* for *chat_id* with a TTL."""
        if self._use_redis:
            await self._redis_save(chat_id, history)
        else:
            self._local_save(chat_id, history)

    async def delete_session(self, chat_id: str) -> bool:
        """Remove the session from the store. Returns True if it existed."""
        if self._use_redis:
            return await self._redis_delete(chat_id)
        return self._local_delete(chat_id)

    # ------------------------------------------------------------------
    # Redis backend
    # ------------------------------------------------------------------

    async def _redis_load(self, chat_id: str) -> list[dict[str, str]]:
        raw = await self._redis.get(f"chat:{chat_id}")
        if raw is None:
            return []
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt history for chat %s; starting fresh", chat_id)
            return []

    async def _redis_save(
        self, chat_id: str, history: list[dict[str, str]]
    ) -> None:
        await self._redis.setex(
            f"chat:{chat_id}",
            SESSION_TTL_SECONDS,
            json.dumps(history),
        )

    async def _redis_delete(self, chat_id: str) -> bool:
        deleted = await self._redis.delete(f"chat:{chat_id}")
        return deleted > 0

    # ------------------------------------------------------------------
    # In-memory fallback backend (local development)
    # ------------------------------------------------------------------

    def _local_load(self, chat_id: str) -> list[dict[str, str]]:
        entry = self._local.get(chat_id)
        if entry is None:
            return []
        expires_at, history = entry
        if time.monotonic() > expires_at:
            del self._local[chat_id]
            return []
        return history

    def _local_save(
        self, chat_id: str, history: list[dict[str, str]]
    ) -> None:
        self._local[chat_id] = (
            time.monotonic() + SESSION_TTL_SECONDS,
            history,
        )

    def _local_delete(self, chat_id: str) -> bool:
        return self._local.pop(chat_id, None) is not None
