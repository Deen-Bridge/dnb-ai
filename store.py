"""Session store abstraction — Redis-backed with in-memory fallback.

On startup, checks for a REDIS_URL environment variable. If set, connects to
Redis and uses it for session persistence (survives restarts, shared across
instances). Falls back to an in-process dict with optional file persistence
for local development (STORE_FILE env var, default ./data/store.json).
"""

import json
import logging
import os
import time
from typing import Any

import google.generativeai.protos as protos

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24h
STORE_FILE = os.getenv("STORE_FILE", "./data/store.json")


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
    return [protos.Content(role=d["role"], parts=[protos.Part(text=d["text"])]) for d in dicts]


class SessionStore:
    """Abstracts persistence of chat-session history.

    Each session's history is stored as a JSON-serialized list of
    ``{"role": str, "text": str}`` dicts, keyed by ``chat:{chat_id}``.
    """

    def __init__(self) -> None:
        # Typed Any: only ever touched behind the _use_redis flag below.
        self._redis: Any = None
        self._local: dict[str, tuple[float, list[dict[str, str]]]] = {}
        self._local_meta: dict[str, tuple[float, float]] = {}
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
            self._load_from_disk()
            logger.info("SessionStore using in-memory dict (local development)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load_history(self, chat_id: str) -> list[dict[str, str]]:
        """Return the persisted history for *chat_id*, or an empty list."""
        if self._use_redis:
            return await self._redis_load(chat_id)
        return self._local_load(chat_id)

    async def save_history(self, chat_id: str, history: list[dict[str, str]]) -> None:
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

    async def add_user_chat(self, user_id: str, chat_id: str) -> None:
        """Associate *chat_id* with *user_id* and record creation time."""
        now = time.time()
        meta_key = f"chat_meta:{chat_id}"
        key = f"user_chats:{user_id}"
        if self._use_redis:
            await self._redis.sadd(key, chat_id)
            await self._redis.expire(key, SESSION_TTL_SECONDS)
            await self._redis.hsetnx(meta_key, "created_at", str(now))
            await self._redis.expire(meta_key, SESSION_TTL_SECONDS)
        else:
            # user_chats set
            entry = self._local.get(key, [time.monotonic() + SESSION_TTL_SECONDS, set()])
            if time.monotonic() > entry[0]:
                entry = [time.monotonic() + SESSION_TTL_SECONDS, set()]
            entry[1].add(chat_id)
            self._local[key] = entry
            # chat meta (only set once)
            if meta_key not in self._local_meta:
                self._local_meta[meta_key] = (time.monotonic() + SESSION_TTL_SECONDS, now)
            self._save_to_disk()

    async def get_user_chats(self, user_id: str) -> list[str]:
        """Return all chat IDs associated with *user_id*."""
        key = f"user_chats:{user_id}"
        if self._use_redis:
            return list(await self._redis.smembers(key))
        entry = self._local.get(key)
        if entry is None:
            return []
        expires_at, chats = entry
        if time.monotonic() > expires_at:
            del self._local[key]
            return []
        return list(chats)

    async def get_chat_created_at(self, chat_id: str) -> float | None:
        """Return the creation timestamp for *chat_id*, or None."""
        meta_key = f"chat_meta:{chat_id}"
        if self._use_redis:
            raw = await self._redis.hget(meta_key, "created_at")
            return float(raw) if raw else None
        entry = self._local_meta.get(meta_key)
        if entry is None:
            return None
        expires_at, ts = entry
        if time.monotonic() > expires_at:
            del self._local_meta[meta_key]
            return None
        return ts

    async def remove_user_chat(self, user_id: str, chat_id: str) -> None:
        """Remove *chat_id* from *user_id*'s chat list."""
        key = f"user_chats:{user_id}"
        if self._use_redis:
            await self._redis.srem(key, chat_id)
        else:
            entry = self._local.get(key)
            if entry and time.monotonic() <= entry[0]:
                entry[1].discard(chat_id)
                self._save_to_disk()

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

    async def _redis_save(self, chat_id: str, history: list[dict[str, str]]) -> None:
        await self._redis.setex(
            f"chat:{chat_id}",
            SESSION_TTL_SECONDS,
            json.dumps(history),
        )

    async def _redis_delete(self, chat_id: str) -> bool:
        deleted = await self._redis.delete(f"chat:{chat_id}")
        return deleted > 0

    # ------------------------------------------------------------------
    # File persistence (in-memory fallback only)
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Restore in-memory state from the JSON file on startup."""
        if not STORE_FILE:
            return
        try:
            with open(STORE_FILE) as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except Exception:
            logger.warning("Failed to load store from %s", STORE_FILE, exc_info=True)
            return

        now = time.monotonic()
        loaded = 0
        for key, entry in data.get("local", {}).items():
            expires_at = entry[0]
            if expires_at <= now:
                continue
            value = set(entry[1]) if key.startswith("user_chats:") else entry[1]
            self._local[key] = (expires_at, value)
            loaded += 1
        for key, entry in data.get("local_meta", {}).items():
            expires_at = entry[0]
            if expires_at <= now:
                continue
            self._local_meta[key] = (expires_at, entry[1])
            loaded += 1
        if loaded:
            logger.info("Restored %d entries from %s", loaded, STORE_FILE)

    def _save_to_disk(self) -> None:
        """Persist the current in-memory state to a JSON file."""
        if not STORE_FILE:
            return
        try:
            os.makedirs(os.path.dirname(STORE_FILE) or ".", exist_ok=True)
        except OSError:
            return

        now = time.monotonic()
        data: dict[str, dict] = {"local": {}, "local_meta": {}}
        for key, (expires_at, value) in list(self._local.items()):
            if expires_at <= now:
                continue
            data["local"][key] = [expires_at, list(value) if isinstance(value, set) else value]
        for key, (expires_at, ts) in list(self._local_meta.items()):
            if expires_at <= now:
                continue
            data["local_meta"][key] = [expires_at, ts]

        try:
            with open(STORE_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            logger.warning("Failed to save store to %s", STORE_FILE, exc_info=True)

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

    def _local_save(self, chat_id: str, history: list[dict[str, str]]) -> None:
        self._local[chat_id] = (
            time.monotonic() + SESSION_TTL_SECONDS,
            history,
        )
        self._save_to_disk()

    def _local_delete(self, chat_id: str) -> bool:
        existed = self._local.pop(chat_id, None) is not None
        self._save_to_disk()
        return existed
