"""Feedback storage for the Deen Bridge AI service.

Stores per-message ratings and failure categories so the team can measure
answer quality and grow the evaluation dataset from real user pain rather
than guesses.

Storage backends (selected at import time):
  - Redis   — if REDIS_URL is set (aligns with issue #3's store direction)
  - SQLite  — fallback for local dev and free-tier Render

Abuse resistance:
  - One record per (chat_id, message_id): resubmission overwrites (idempotent)
  - comment capped at COMMENT_MAX_CHARS characters (validated server-side)
  - categories validated against FEEDBACK_TAXONOMY
  - per-IP rate limiting via simple in-process sliding-window counter
    (stopgap until issue #9 provides real auth/rate-limiting infrastructure)
  - SQLite bounded by SQLITE_MAX_RECORDS; Redis keys carry TTL

Admin endpoints are protected by ADMIN_TOKEN (stopgap — replaced by #9).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEEDBACK_TAXONOMY = {
    "incorrect_information",
    "wrong_or_missing_citation",
    "one_sided_fiqh_answer",
    "too_vague",
    "too_long",
    "wrong_language",
    "poor_adab",
    "refused_unnecessarily",
    "other",
}

COMMENT_MAX_CHARS = 1000

# Redis TTL for feedback records (30 days)
REDIS_TTL_SECONDS = 60 * 60 * 24 * 30

# SQLite cap — oldest records are pruned when this is exceeded
SQLITE_MAX_RECORDS = 50_000

# Rate-limiting: max submissions per IP per window
RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW_SECONDS = 60


# ---------------------------------------------------------------------------
# Rate limiter (in-process sliding window — stopgap for #9)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple per-IP sliding-window rate limiter (in-process, non-persistent)."""

    def __init__(
        self,
        max_calls: int = RATE_LIMIT_MAX,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self._max = max_calls
        self._window = window_seconds
        self._buckets: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            bucket = self._buckets[ip]
            # Evict timestamps outside the window
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self._max:
                return False
            bucket.append(now)
            return True


rate_limiter = RateLimiter()


# ---------------------------------------------------------------------------
# Feedback record dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRecord:
    feedback_id: str
    chat_id: str
    message_id: str
    rating: str                       # "up" | "down"
    categories: List[str] = field(default_factory=list)
    comment: Optional[str] = None
    prompt: Optional[str] = None
    answer: Optional[str] = None
    model_name: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None
    created_at: str = ""              # ISO-8601 UTC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "chat_id": self.chat_id,
            "message_id": self.message_id,
            "rating": self.rating,
            "categories": self.categories,
            "comment": self.comment,
            "prompt": self.prompt,
            "answer": self.answer,
            "model_name": self.model_name,
            "generation_config": self.generation_config,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FeedbackRecord":
        gen_cfg = d.get("generation_config")
        if isinstance(gen_cfg, str):
            try:
                gen_cfg = json.loads(gen_cfg)
            except Exception:
                gen_cfg = None
        cats = d.get("categories", [])
        if isinstance(cats, str):
            try:
                cats = json.loads(cats)
            except Exception:
                cats = []
        return FeedbackRecord(
            feedback_id=d["feedback_id"],
            chat_id=d["chat_id"],
            message_id=d["message_id"],
            rating=d["rating"],
            categories=cats,
            comment=d.get("comment"),
            prompt=d.get("prompt"),
            answer=d.get("answer"),
            model_name=d.get("model_name"),
            generation_config=gen_cfg,
            created_at=d.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Storage back-ends
# ---------------------------------------------------------------------------

class FeedbackStore:
    """Abstract interface — concrete implementations below."""

    def upsert(self, record: FeedbackRecord) -> None:
        raise NotImplementedError

    def get(self, chat_id: str, message_id: str) -> Optional[FeedbackRecord]:
        raise NotImplementedError

    def list_records(
        self,
        rating: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackRecord]:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError


# ── SQLite store ────────────────────────────────────────────────────────────

_SQLITE_PATH = os.getenv("FEEDBACK_DB_PATH", "feedback.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id       TEXT NOT NULL,
    chat_id           TEXT NOT NULL,
    message_id        TEXT NOT NULL,
    rating            TEXT NOT NULL,
    categories        TEXT NOT NULL DEFAULT '[]',
    comment           TEXT,
    prompt            TEXT,
    answer            TEXT,
    model_name        TEXT,
    generation_config TEXT,
    created_at        TEXT NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_rating    ON feedback(rating);
CREATE INDEX IF NOT EXISTS idx_feedback_created   ON feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_model     ON feedback(model_name);
"""


class SQLiteFeedbackStore(FeedbackStore):
    """Thread-safe SQLite store.  Prunes oldest rows when SQLITE_MAX_RECORDS is reached."""

    def __init__(self, db_path: str = _SQLITE_PATH) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_CREATE_TABLE)
        conn.commit()
        conn.close()

    def _prune(self, conn: sqlite3.Connection) -> None:
        count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        if count > SQLITE_MAX_RECORDS:
            excess = count - SQLITE_MAX_RECORDS
            conn.execute(
                "DELETE FROM feedback WHERE rowid IN "
                "(SELECT rowid FROM feedback ORDER BY created_at ASC LIMIT ?)",
                (excess,),
            )

    def upsert(self, record: FeedbackRecord) -> None:
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO feedback
                (feedback_id, chat_id, message_id, rating, categories,
                 comment, prompt, answer, model_name, generation_config, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(chat_id, message_id) DO UPDATE SET
                feedback_id       = excluded.feedback_id,
                rating            = excluded.rating,
                categories        = excluded.categories,
                comment           = excluded.comment,
                prompt            = excluded.prompt,
                answer            = excluded.answer,
                model_name        = excluded.model_name,
                generation_config = excluded.generation_config,
                created_at        = excluded.created_at
            """,
            (
                record.feedback_id,
                record.chat_id,
                record.message_id,
                record.rating,
                json.dumps(record.categories),
                record.comment,
                record.prompt,
                record.answer,
                record.model_name,
                json.dumps(record.generation_config) if record.generation_config else None,
                record.created_at,
            ),
        )
        self._prune(conn)
        conn.commit()

    def get(self, chat_id: str, message_id: str) -> Optional[FeedbackRecord]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM feedback WHERE chat_id=? AND message_id=?",
            (chat_id, message_id),
        ).fetchone()
        return FeedbackRecord.from_dict(dict(row)) if row else None

    def list_records(
        self,
        rating: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackRecord]:
        conn = self._conn()
        sql = "SELECT * FROM feedback WHERE 1=1"
        params: list = []
        if rating:
            sql += " AND rating=?"
            params.append(rating)
        if category:
            # categories stored as JSON array string — use LIKE for simplicity
            sql += " AND categories LIKE ?"
            params.append(f'%"{category}"%')
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [FeedbackRecord.from_dict(dict(r)) for r in rows]

    def stats(self) -> Dict[str, Any]:
        conn = self._conn()

        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='up'").fetchone()[0]
        down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='down'").fetchone()[0]

        # Per-category counts
        cat_counts: Dict[str, Dict[str, int]] = {}
        for row in conn.execute("SELECT categories, rating FROM feedback").fetchall():
            try:
                cats = json.loads(row["categories"]) if row["categories"] else []
            except Exception:
                cats = []
            for cat in cats:
                if cat not in cat_counts:
                    cat_counts[cat] = {"up": 0, "down": 0}
                cat_counts[cat][row["rating"]] = cat_counts[cat].get(row["rating"], 0) + 1

        # Per-model counts
        model_rows = conn.execute(
            "SELECT model_name, rating, COUNT(*) as cnt "
            "FROM feedback GROUP BY model_name, rating"
        ).fetchall()
        model_counts: Dict[str, Dict[str, int]] = {}
        for r in model_rows:
            name = r["model_name"] or "unknown"
            if name not in model_counts:
                model_counts[name] = {"up": 0, "down": 0}
            model_counts[name][r["rating"]] = r["cnt"]

        # Time-bucket counts (day / week)
        day_rows = conn.execute(
            "SELECT substr(created_at,1,10) as day, rating, COUNT(*) as cnt "
            "FROM feedback GROUP BY day, rating ORDER BY day DESC LIMIT 14"
        ).fetchall()
        by_day: Dict[str, Dict[str, int]] = {}
        for r in day_rows:
            d = r["day"]
            if d not in by_day:
                by_day[d] = {"up": 0, "down": 0}
            by_day[d][r["rating"]] = r["cnt"]

        return {
            "total": total,
            "up": up,
            "down": down,
            "up_ratio": round(up / total, 4) if total else None,
            "by_category": cat_counts,
            "by_model": model_counts,
            "by_day": by_day,
        }


# ── Redis store ─────────────────────────────────────────────────────────────

def _build_redis_store() -> Optional["RedisFeedbackStore"]:
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis as _redis  # type: ignore
        client = _redis.from_url(redis_url, decode_responses=True)
        client.ping()
        logger.info("Feedback store: Redis (%s)", redis_url.split("@")[-1])
        return RedisFeedbackStore(client)
    except Exception as exc:
        logger.warning("Redis unavailable (%s); falling back to SQLite.", exc)
        return None


class RedisFeedbackStore(FeedbackStore):
    """Redis-backed store.

    Key layout:
      feedback:<chat_id>:<message_id>  → JSON hash (TTL REDIS_TTL_SECONDS)
      feedback:index:rating:<rating>   → sorted set  score=unix_timestamp
      feedback:index:cat:<cat>         → sorted set  score=unix_timestamp
      feedback:index:model:<name>      → sorted set  score=unix_timestamp
    """

    _PREFIX = "feedback"

    def __init__(self, client: Any) -> None:
        self._r = client

    def _record_key(self, chat_id: str, message_id: str) -> str:
        return f"{self._PREFIX}:{chat_id}:{message_id}"

    def upsert(self, record: FeedbackRecord) -> None:
        key = self._record_key(record.chat_id, record.message_id)
        ts = time.time()
        data = record.to_dict()
        data["categories"] = json.dumps(data["categories"])
        data["generation_config"] = (
            json.dumps(data["generation_config"]) if data["generation_config"] else ""
        )
        pipe = self._r.pipeline()
        pipe.hset(key, mapping={k: (v or "") for k, v in data.items()})
        pipe.expire(key, REDIS_TTL_SECONDS)
        pipe.zadd(f"{self._PREFIX}:index:rating:{record.rating}", {key: ts})
        for cat in record.categories:
            pipe.zadd(f"{self._PREFIX}:index:cat:{cat}", {key: ts})
        model_key = record.model_name or "unknown"
        pipe.zadd(f"{self._PREFIX}:index:model:{model_key}", {key: ts})
        pipe.execute()

    def get(self, chat_id: str, message_id: str) -> Optional[FeedbackRecord]:
        key = self._record_key(chat_id, message_id)
        data = self._r.hgetall(key)
        if not data:
            return None
        return FeedbackRecord.from_dict(data)

    def _fetch_keys(self, index_key: str, limit: int) -> List[str]:
        return self._r.zrevrange(index_key, 0, limit - 1)

    def _fetch_records(self, keys: List[str]) -> List[FeedbackRecord]:
        pipe = self._r.pipeline()
        for k in keys:
            pipe.hgetall(k)
        results = pipe.execute()
        records = []
        for data in results:
            if data:
                try:
                    records.append(FeedbackRecord.from_dict(data))
                except Exception:
                    pass
        return records

    def list_records(
        self,
        rating: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackRecord]:
        if rating:
            keys = self._fetch_keys(f"{self._PREFIX}:index:rating:{rating}", limit)
        elif category:
            keys = self._fetch_keys(f"{self._PREFIX}:index:cat:{category}", limit)
        else:
            # Merge up + down sorted sets
            up_keys = self._fetch_keys(f"{self._PREFIX}:index:rating:up", limit)
            down_keys = self._fetch_keys(f"{self._PREFIX}:index:rating:down", limit)
            seen: set = set()
            merged = []
            for k in up_keys + down_keys:
                if k not in seen:
                    seen.add(k)
                    merged.append(k)
            keys = merged[:limit]
        if category and rating:
            # Intersect manually
            cat_keys = set(self._fetch_keys(f"{self._PREFIX}:index:cat:{category}", limit * 2))
            keys = [k for k in keys if k in cat_keys][:limit]
        return self._fetch_records(keys)

    def stats(self) -> Dict[str, Any]:
        up = self._r.zcard(f"{self._PREFIX}:index:rating:up")
        down = self._r.zcard(f"{self._PREFIX}:index:rating:down")
        total = up + down

        cat_counts: Dict[str, Dict[str, int]] = {}
        for cat in FEEDBACK_TAXONOMY:
            n = self._r.zcard(f"{self._PREFIX}:index:cat:{cat}")
            if n:
                cat_counts[cat] = {"total": n}

        return {
            "total": total,
            "up": up,
            "down": down,
            "up_ratio": round(up / total, 4) if total else None,
            "by_category": cat_counts,
            "by_model": {},   # full aggregation omitted for Redis brevity
            "by_day": {},
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def _build_store() -> FeedbackStore:
    redis_store = _build_redis_store()
    if redis_store is not None:
        return redis_store
    logger.info("Feedback store: SQLite (%s)", _SQLITE_PATH)
    return SQLiteFeedbackStore()


store: FeedbackStore = _build_store()
