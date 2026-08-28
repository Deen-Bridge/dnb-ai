"""Session store abstraction — Firestore-backed with Redis and in-memory fallbacks.

Persistence preference at runtime:

1. Firestore (firebase-admin) — when FIREBASE_SERVICE_ACCOUNT,
   FIREBASE_SERVICE_ACCOUNT_JSON, or GOOGLE_APPLICATION_CREDENTIALS is set and
   firebase-admin is installed. Durable, no TTL: a returning user's chats are
   never lost, which is the production configuration.
2. Redis — when REDIS_URL is set. Survives restarts but expires history after
   SESSION_TTL_SECONDS (default 24h).
3. In-memory dict with optional JSON-file persistence (STORE_FILE env var,
   default ./data/store.json) for local development.

Use :func:`create_session_store` to build the store; main.py calls it once at
startup.
"""

import asyncio
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

FIREBASE_SERVICE_ACCOUNT = os.getenv("FIREBASE_SERVICE_ACCOUNT", "")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Collection that holds every AI conversation. Layout:
#   ai_chats/{chat_id}                     -> {user_id, created_at, updated_at, message_count}
#   ai_chats/{chat_id}/messages/{seq:08d}  -> {seq, role, text}
CHAT_COLLECTION = os.getenv("CHAT_COLLECTION", "ai_chats")
_FIREBASE_APP_NAME = "dnb-ai-session-store"


try:
    import redis.asyncio as aioredis

    _redis_available = True
except ImportError:
    _redis_available = False

try:
    import firebase_admin
    from firebase_admin import credentials as fb_credentials, firestore as fb_firestore
    from google.cloud.firestore import FieldFilter

    _firebase_available = True
except ImportError:
    _firebase_available = False


def firebase_credentials_configured() -> bool:
    """True when the process has any Firebase service-account configuration."""
    return bool(FIREBASE_SERVICE_ACCOUNT or FIREBASE_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS)


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
        # Value shape varies by key: session history keys store
        # (expires_at, list[dict]); user_chats: keys store (expires_at, set[str]).
        self._local: dict[str, tuple[float, Any]] = {}
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
            # user_chats set — value is (expires_at, set[str])
            entry = self._local.get(key)
            if entry is None or time.monotonic() > entry[0]:
                expires_at = time.monotonic() + SESSION_TTL_SECONDS
                chats: set[str] = set()
            else:
                expires_at = entry[0]
                chats = entry[1]
            chats.add(chat_id)
            self._local[key] = (expires_at, chats)
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
        if time.monotonic() >= expires_at:
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


class FirestoreSessionStore:
    """Durable chat-history storage backed by Firestore (firebase-admin).

    Unlike :class:`SessionStore` there is no TTL: conversations persist until
    explicitly deleted, so a returning user sees their full history.

    Firestore's admin client is synchronous, so every public method runs its
    blocking work in a thread to keep the event loop responsive.
    """

    def __init__(self, client: Any = None, collection: str = CHAT_COLLECTION) -> None:
        if client is not None:
            # Test seam: callers inject a fake client to avoid live credentials.
            self._db = client
        else:
            if not _firebase_available:
                raise RuntimeError("firebase-admin is not installed")
            existing_apps = getattr(firebase_admin, "_apps", {})
            app = existing_apps.get(_FIREBASE_APP_NAME) or firebase_admin.initialize_app(
                self._resolve_credentials(),
                name=_FIREBASE_APP_NAME,
            )
            self._db = fb_firestore.client(app)
        self._collection = collection

    # ------------------------------------------------------------------
    # Public API (async; same signatures as SessionStore)
    # ------------------------------------------------------------------

    async def load_history(self, chat_id: str) -> list[dict[str, str]]:
        return await asyncio.to_thread(self._load_history_sync, chat_id)

    async def save_history(self, chat_id: str, history: list[dict[str, str]]) -> None:
        await asyncio.to_thread(self._save_history_sync, chat_id, history)

    async def delete_session(self, chat_id: str) -> bool:
        return await asyncio.to_thread(self._delete_session_sync, chat_id)

    async def add_user_chat(self, user_id: str, chat_id: str) -> None:
        await asyncio.to_thread(self._add_user_chat_sync, user_id, chat_id)

    async def get_user_chats(self, user_id: str) -> list[str]:
        return await asyncio.to_thread(self._get_user_chats_sync, user_id)

    async def get_chat_created_at(self, chat_id: str) -> float | None:
        return await asyncio.to_thread(self._get_chat_created_at_sync, chat_id)

    async def remove_user_chat(self, user_id: str, chat_id: str) -> None:
        await asyncio.to_thread(self._remove_user_chat_sync, user_id, chat_id)

    # ------------------------------------------------------------------
    # Blocking internals
    # ------------------------------------------------------------------

    def _chat_ref(self, chat_id: str) -> Any:
        return self._db.collection(self._collection).document(chat_id)

    def _messages_ref(self, chat_id: str) -> Any:
        return self._chat_ref(chat_id).collection("messages")

    def _load_history_sync(self, chat_id: str) -> list[dict[str, str]]:
        docs = self._messages_ref(chat_id).order_by("seq").stream()
        return [{"role": d.to_dict().get("role", ""), "text": d.to_dict().get("text", "")} for d in docs]

    def _save_history_sync(self, chat_id: str, history: list[dict[str, str]]) -> None:
        chat_ref = self._chat_ref(chat_id)
        chat_doc = chat_ref.get()
        existing_count = (chat_doc.to_dict() or {}).get("message_count", 0) if chat_doc.exists else 0
        now = time.time()

        new_messages = history[existing_count:]
        if new_messages or not chat_doc.exists:
            batch = self._db.batch()
            for i, msg in enumerate(new_messages, start=existing_count):
                batch.set(
                    self._messages_ref(chat_id).document(_seq_id(i)),
                    {
                        "seq": i,
                        "role": msg.get("role", ""),
                        "text": msg.get("text", ""),
                    },
                )
            if chat_doc.exists:
                batch.update(
                    chat_ref,
                    {"message_count": len(history), "updated_at": now},
                )
            else:
                # The chat doc is created here (batch.update would fail on a
                # missing document); add_user_chat fills in user_id/created_at.
                batch.set(
                    chat_ref,
                    {"message_count": len(history), "updated_at": now},
                    merge=True,
                )
            batch.commit()
        else:
            chat_ref.update({"updated_at": now})

    def _delete_session_sync(self, chat_id: str) -> bool:
        chat_ref = self._chat_ref(chat_id)
        if not chat_ref.get().exists:
            return False
        # Firestore requires subcollection docs to be deleted individually.
        for doc in self._messages_ref(chat_id).stream():
            doc.reference.delete()
        chat_ref.delete()
        return True

    def _add_user_chat_sync(self, user_id: str, chat_id: str) -> None:
        chat_ref = self._chat_ref(chat_id)
        chat_doc = chat_ref.get()
        now = time.time()
        if not chat_doc.exists:
            chat_ref.set(
                {
                    "user_id": user_id,
                    "created_at": now,
                    "updated_at": now,
                    "message_count": 0,
                }
            )
        else:
            update: dict[str, object] = {"user_id": user_id}
            if chat_doc.to_dict().get("created_at") is None:
                # save_history may have created the doc first; backfill the
                # creation time so chat lists can sort by it.
                update["created_at"] = now
            chat_ref.update(update)

    def _get_user_chats_sync(self, user_id: str) -> list[str]:
        chats = []
        docs = self._db.collection(self._collection).where(filter=FieldFilter("user_id", "==", user_id)).stream()
        for doc in docs:
            chats.append(doc.id)
        return chats

    def _get_chat_created_at_sync(self, chat_id: str) -> float | None:
        chat_doc = self._chat_ref(chat_id).get()
        if not chat_doc.exists:
            return None
        created_at = (chat_doc.to_dict() or {}).get("created_at")
        return float(created_at) if created_at is not None else None

    def _remove_user_chat_sync(self, user_id: str, chat_id: str) -> None:
        chat_ref = self._chat_ref(chat_id)
        chat_doc = chat_ref.get()
        if chat_doc.exists and (chat_doc.to_dict() or {}).get("user_id") == user_id:
            chat_ref.update({"user_id": None})

    @staticmethod
    def _resolve_credentials() -> Any:
        """Build the admin credential from env config, or ADC when none is set."""
        raw_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
        if raw_json:
            try:
                return fb_credentials.Certificate(json.loads(raw_json))
            except Exception as exc:  # noqa: BLE001 - try the file-based vars next
                logger.warning("Could not parse FIREBASE_SERVICE_ACCOUNT_JSON (%s)", exc)
        for var in ("FIREBASE_SERVICE_ACCOUNT", "GOOGLE_APPLICATION_CREDENTIALS"):
            path = os.getenv(var)
            if path:
                try:
                    return fb_credentials.Certificate(path)
                except Exception as exc:  # noqa: BLE001 - invalid file, try ADC
                    logger.warning("Could not load Firebase credentials from %s (%s)", var, exc)
        return fb_credentials.ApplicationDefault()


def _seq_id(seq: int) -> str:
    """Zero-padded doc id so ordering by id matches ordering by seq."""
    return f"{seq:08d}"


def create_session_store() -> SessionStore | FirestoreSessionStore:
    """Build the configured session store: Firestore > Redis/in-memory.

    Falls back to the existing Redis/in-memory :class:`SessionStore` whenever
    Firestore cannot be initialised, so a missing or broken credential never
    takes the chat service down.
    """
    if firebase_credentials_configured():
        if not _firebase_available:
            logger.warning(
                "Firebase credentials are set but firebase-admin is not installed; "
                "falling back to Redis/in-memory chat storage"
            )
            return SessionStore()
        try:
            store = FirestoreSessionStore()
            logger.info("SessionStore using Firestore (collection=%s)", CHAT_COLLECTION)
            return store
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "Could not initialise Firestore chat storage (%s); falling back to Redis/in-memory",
                exc,
            )
    return SessionStore()
