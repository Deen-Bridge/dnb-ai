"""Answer-feedback capture for the Deen Bridge AI service.

Stores per-message ratings and failure categories so the team can measure
answer quality and grow the evaluation dataset from real user pain rather than
guesses. This is the capture-and-storage half of issue #43; the scholar-review
queue (#56) owns human vetting of low-confidence answers, and they share
storage direction (Redis when configured) rather than inventing parallel ones.

Storage backends (selected at import time):
  - Redis   — when REDIS_URL is set (aligns with the session/queue store direction)
  - SQLite  — fallback for local dev and free-tier Render

Abuse resistance:
  - One record per (chat_id, message_id): resubmission overwrites (idempotent)
  - comment capped at COMMENT_MAX_CHARS characters (validated server-side)
  - categories validated against FEEDBACK_TAXONOMY
  - per-IP rate limiting via an in-process sliding-window counter
    (stopgap until real auth/rate-limiting infrastructure lands)
  - SQLite bounded by SQLITE_MAX_RECORDS; Redis keys carry a TTL

Admin endpoints are protected by ADMIN_TOKEN (stopgap).
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
from typing import Any

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


def env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive int from the environment, falling back on nonsense.

    A malformed tuning value must not crash boot: main.py imports this module,
    so an unguarded ``int(os.getenv(...))`` here would take the whole app down
    with a traceback instead of degrading to a sane default.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%s is below the minimum %s; using %s", name, value, minimum, default)
        return default
    return value


# Redis TTL for feedback records (30 days).
REDIS_TTL_SECONDS = 60 * 60 * 24 * 30

# SQLite cap — oldest records are pruned when this is exceeded.
SQLITE_MAX_RECORDS = 50_000

# Rate limiting: max submissions per IP per window.
RATE_LIMIT_MAX = env_int("FEEDBACK_RATE_LIMIT_MAX", 20)
RATE_LIMIT_WINDOW_SECONDS = env_int("FEEDBACK_RATE_LIMIT_WINDOW", 60)


# ---------------------------------------------------------------------------
# Rate limiter (in-process sliding window — stopgap)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Per-IP sliding-window rate limiter (in-process, non-persistent)."""

    def __init__(
        self,
        max_calls: int = RATE_LIMIT_MAX,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self._max = max_calls
        self._window = window_seconds
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            self._sweep(cutoff)
            bucket = self._buckets[ip]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self._max:
                return False
            bucket.append(now)
            return True

    def _sweep(self, cutoff: float) -> None:
        """Drop buckets whose newest timestamp is outside the window.

        Without this, one entry accumulates per distinct IP and is never
        reclaimed. The key comes from a client-controlled X-Forwarded-For, so
        an attacker could otherwise grow this dict without bound. A bucket
        whose most-recent hit is older than the window can hold nothing live,
        so it is safe to drop entirely.
        """
        stale = [ip for ip, bucket in self._buckets.items() if not bucket or bucket[-1] < cutoff]
        for ip in stale:
            del self._buckets[ip]

    def reset(self) -> None:
        """Clear all buckets. Used by tests so limiter state never leaks between them."""
        with self._lock:
            self._buckets.clear()


rate_limiter = RateLimiter()


# ---------------------------------------------------------------------------
# Feedback record
# ---------------------------------------------------------------------------


@dataclass
class FeedbackRecord:
    feedback_id: str
    chat_id: str
    message_id: str
    rating: str  # "up" | "down"
    categories: list[str] = field(default_factory=list)
    comment: str | None = None
    prompt: str | None = None
    answer: str | None = None
    model_name: str | None = None
    generation_config: dict[str, Any] | None = None
    created_at: str = ""  # ISO-8601 UTC

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(d: dict[str, Any]) -> FeedbackRecord:
        gen_cfg = d.get("generation_config")
        if isinstance(gen_cfg, str):
            try:
                gen_cfg = json.loads(gen_cfg) if gen_cfg else None
            except (json.JSONDecodeError, TypeError):
                gen_cfg = None
        cats = d.get("categories", [])
        if isinstance(cats, str):
            try:
                cats = json.loads(cats) if cats else []
            except (json.JSONDecodeError, TypeError):
                cats = []
        return FeedbackRecord(
            feedback_id=d["feedback_id"],
            chat_id=d["chat_id"],
            message_id=d["message_id"],
            rating=d["rating"],
            categories=cats,
            comment=d.get("comment") or None,
            prompt=d.get("prompt") or None,
            answer=d.get("answer") or None,
            model_name=d.get("model_name") or None,
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

    def get(self, chat_id: str, message_id: str) -> FeedbackRecord | None:
        raise NotImplementedError

    def list_records(
        self,
        rating: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[FeedbackRecord]:
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:
        raise NotImplementedError


# -- Quality Judge Agent -----------------------------------------------------


QUALITY_DIMENSIONS = (
    'accuracy',
    'completeness',
    'clarity',
    'scholarly_rigor',
    'appropriateness',
    'balance',
    'citation_quality',
    'reasoning',
)

DEFAULT_QUALITY_THRESHOLD = 0.7


class QualityJudgeAgent:
    '''Heuristic multi-dimensional answer quality judge.

    Scores an answer across QUALITY_DIMENSIONS using lightweight text
    heuristics, produces per-dimension feedback, gap lists, improvement
    recommendations, and a final re-generation decision based on a
    configurable threshold.
    '''

    def __init__(self, threshold: float = DEFAULT_QUALITY_THRESHOLD) -> None:
        self.threshold = threshold

    def evaluate(self, answer: str, prompt: str | None = None) -> dict[str, Any]:
        '''Return a full quality report for *answer*.'''
        scores = {
            dimension: self._score_dimension(dimension, answer, prompt)
            for dimension in QUALITY_DIMENSIONS
        }
        overall = sum(scores.values()) / len(scores)
        gaps = self._find_gaps(scores)
        recommendations = self._recommend_improvements(scores, gaps)
        needs_regeneration = self.should_regenerate(overall, gaps)
        return {
            'overall_score': round(overall, 3),
            'dimension_scores': {k: round(v, 3) for k, v in scores.items()},
            'gaps': gaps,
            'recommendations': recommendations,
            'needs_regeneration': needs_regeneration,
            'threshold': self.threshold,
        }

    def should_regenerate(self, overall: float, gaps: list[str]) -> bool:
        '''Regenerate when the overall score falls below threshold or critical gaps exist.'''
        if overall < self.threshold:
            return True
        critical = {'accuracy', 'completeness'}
        return any(g in critical for g in gaps)

    def _score_dimension(self, dimension: str, answer: str, prompt: str | None) -> float:
        '''Heuristic per-dimension scoring. Each returns a float in [0, 1].'''
        if dimension == 'accuracy':
            return self._score_accuracy(answer)
        if dimension == 'completeness':
            return self._score_completeness(answer, prompt)
        if dimension == 'clarity':
            return self._score_clarity(answer)
        if dimension == 'scholarly_rigor':
            return self._score_scholarly_rigor(answer)
        if dimension == 'appropriateness':
            return self._score_appropriateness(answer, prompt)
        if dimension == 'balance':
            return self._score_balance(answer)
        if dimension == 'citation_quality':
            return self._score_citation_quality(answer)
        if dimension == 'reasoning':
            return self._score_reasoning(answer)
        return 0.0

    def _score_accuracy(self, answer: str) -> float:
        # Very light heuristic: penalize hedging, reward definite statements.
        hedge_words = {'maybe', 'might', 'perhaps', 'could be', 'possibly'}
        lowered = answer.lower()
        if not answer.strip():
            return 0.0
        hits = sum(w in lowered for w in hedge_words)
        return max(0.0, 1.0 - (hits * 0.15))

    def _score_completeness(self, answer: str, prompt: str | None) -> float:
        # Reward length relative to a crude expectation, penalize very short answers.
        if not answer.strip():
            return 0.0
        length = len(answer.split())
        if length < 10:
            return 0.2
        if length < 30:
            return 0.6
        if prompt and len(prompt.split()) > 20 and length < 60:
            return 0.5
        return min(1.0, length / 200.0)

    def _score_clarity(self, answer: str) -> float:
        # Reward short sentences, penalize extremely long ones.
        sentences = [s for s in answer.replace(chr(10), ' ').split('. ') if s]
        if not sentences:
            return 0.0
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_len <= 20:
            return 1.0
        if avg_len <= 35:
            return 0.7
        return max(0.0, 1.0 - (avg_len - 35) / 50.0)

    def _score_scholarly_rigor(self, answer: str) -> float:
        # Reward presence of technical terms, citations, structured markers.
        lowered = answer.lower()
        indicators = ['according to', 'evidence', 'study', 'research', 'hadith', 'quran', 'surah', 'ayah']
        hits = sum(ind in lowered for ind in indicators)
        return min(1.0, 0.3 + hits * 0.15)

    def _score_appropriateness(self, answer: str, prompt: str | None) -> float:
        # If no prompt, assume appropriate. Otherwise check language/style fit.
        if not prompt:
            return 1.0
        # Very rough: penalize if answer is extremely long for a short prompt.
        p_len = len(prompt.split())
        a_len = len(answer.split())
        if p_len < 5 and a_len > 100:
            return 0.4
        return 1.0 if a_len < 300 else 0.8

    def _score_balance(self, answer: str) -> float:
        # Look for multiple viewpoints or contrasting phrases.
        lowered = answer.lower()
        balance_markers = ['some scholars', 'others', 'contrast', 'however', 'on the other hand', 'alternative view']
        hits = sum(m in lowered for m in balance_markers)
        return min(1.0, hits * 0.25)

    def _score_citation_quality(self, answer: str) -> float:
        # Reward explicit references to sources.
        lowered = answer.lower()
        indicators = ['hadith', 'quran', 'surah', 'source', 'citation']
        hits = sum(ind in lowered for ind in indicators)
        return min(1.0, 0.2 + hits * 0.2)

    def _score_reasoning(self, answer: str) -> float:
        # Reward logical connectives.
        lowered = answer.lower()
        reasons = ['therefore', 'because', 'thus', 'hence', 'consequently', 'as a result']
        hits = sum(r in lowered for r in reasons)
        return min(1.0, hits * 0.2)

    def _find_gaps(self, scores: dict[str, float]) -> list[str]:
        return [dim for dim, score in scores.items() if score < 0.5]

    def _recommend_improvements(self, scores: dict[str, float], gaps: list[str]) -> list[str]:
        recommendations = []
        for dim in gaps:
            if dim == 'accuracy':
                recommendations.append('Verify factual claims and add reliable sources.')
            elif dim == 'completeness':
                recommendations.append('Expand coverage to address all parts of the question.')
            elif dim == 'clarity':
                recommendations.append('Break long sentences into shorter, clearer ones.')
            elif dim == 'scholarly_rigor':
                recommendations.append('Add scholarly references and technical terminology.')
            elif dim == 'appropriateness':
                recommendations.append('Adjust depth and detail to match the scope of the question.')
            elif dim == 'balance':
                recommendations.append('Present multiple scholarly views fairly.')
            elif dim == 'citation_quality':
                recommendations.append('Include precise citations with source details.')
            elif dim == 'reasoning':
                recommendations.append('Make the logical step-by-step reasoning explicit.')
        return recommendations


# -- SQLite store -----------------------------------------------------------

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
    """Thread-safe SQLite store; prunes oldest rows past SQLITE_MAX_RECORDS."""

    def __init__(self, db_path: str = _SQLITE_PATH) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if getattr(self._local, "conn", None) is None:
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
                "DELETE FROM feedback WHERE rowid IN (SELECT rowid FROM feedback ORDER BY created_at ASC LIMIT ?)",
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

    def get(self, chat_id: str, message_id: str) -> FeedbackRecord | None:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM feedback WHERE chat_id=? AND message_id=?",
            (chat_id, message_id),
        ).fetchone()
        return FeedbackRecord.from_dict(dict(row)) if row else None

    def list_records(
        self,
        rating: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[FeedbackRecord]:
        conn = self._conn()
        sql = "SELECT * FROM feedback WHERE 1=1"
        params: list = []
        if rating:
            sql += " AND rating=?"
            params.append(rating)
        if category:
            # categories stored as a JSON array string — LIKE on the quoted token
            sql += " AND categories LIKE ?"
            params.append(f'%"{category}"%')
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [FeedbackRecord.from_dict(dict(r)) for r in rows]

    def stats(self) -> dict[str, Any]:
        conn = self._conn()

        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='up'").fetchone()[0]
        down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='down'").fetchone()[0]

        cat_counts: dict[str, dict[str, int]] = {}
        for row in conn.execute("SELECT categories, rating FROM feedback").fetchall():
            try:
                cats = json.loads(row["categories"]) if row["categories"] else []
            except (json.JSONDecodeError, TypeError):
                cats = []
            for cat in cats:
                bucket = cat_counts.setdefault(cat, {"up": 0, "down": 0})
                bucket[row["rating"]] = bucket.get(row["rating"], 0) + 1

        model_rows = conn.execute(
            "SELECT model_name, rating, COUNT(*) as cnt FROM feedback GROUP BY model_name, rating"
        ).fetchall()
        model_counts: dict[str, dict[str, int]] = {}
        for r in model_rows:
            name = r["model_name"] or "unknown"
            bucket = model_counts.setdefault(name, {"up": 0, "down": 0})
            bucket[r["rating"]] = r["cnt"]

        # Limit distinct *days*, not grouped rows: GROUP BY day, rating yields
        # up to two rows per day, so a plain LIMIT 14 would return as few as
        # seven days when both ratings occur.
        day_rows = conn.execute(
            "SELECT substr(created_at,1,10) as day, rating, COUNT(*) as cnt "
            "FROM feedback "
            "WHERE substr(created_at,1,10) IN ("
            "  SELECT DISTINCT substr(created_at,1,10) FROM feedback "
            "  ORDER BY 1 DESC LIMIT 14"
            ") "
            "GROUP BY day, rating ORDER BY day DESC"
        ).fetchall()
        by_day: dict[str, dict[str, int]] = {}
        for r in day_rows:
            bucket = by_day.setdefault(r["day"], {"up": 0, "down": 0})
            bucket[r["rating"]] = r["cnt"]

        return {
            "total": total,
            "up": up,
            "down": down,
            "up_ratio": round(up / total, 4) if total else None,
            "by_category": cat_counts,
            "by_model": model_counts,
            "by_day": by_day,
        }


# -- Redis store ------------------------------------------------------------


class RedisFeedbackStore(FeedbackStore):
    """Redis-backed store.

    Key layout:
      feedback:<chat_id>:<message_id>  -> JSON hash (TTL REDIS_TTL_SECONDS)
      feedback:index:rating:<rating>   -> sorted set, score = unix timestamp
      feedback:index:cat:<cat>         -> sorted set, score = unix timestamp
      feedback:index:model:<name>      -> sorted set, score = unix timestamp
    """

    _PREFIX = "feedback"

    def __init__(self, client: Any) -> None:
        self._r = client

    def _record_key(self, chat_id: str, message_id: str) -> str:
        return f"{self._PREFIX}:{chat_id}:{message_id}"

    def upsert(self, record: FeedbackRecord) -> None:
        key = self._record_key(record.chat_id, record.message_id)
        ts = time.time()

        # Idempotent overwrite must also fix the indexes: a re-rating (down->up)
        # or a changed category set would otherwise leave the key in the old
        # rating/category sorted sets forever, so list_records and stats would
        # double-count it. Remove the previous memberships before re-adding.
        previous = self.get(record.chat_id, record.message_id)

        data = record.to_dict()
        data["categories"] = json.dumps(data["categories"])
        data["generation_config"] = json.dumps(data["generation_config"]) if data["generation_config"] else ""
        pipe = self._r.pipeline()
        if previous is not None:
            pipe.zrem(f"{self._PREFIX}:index:rating:{previous.rating}", key)
            for cat in previous.categories:
                pipe.zrem(f"{self._PREFIX}:index:cat:{cat}", key)
            pipe.zrem(f"{self._PREFIX}:index:model:{previous.model_name or 'unknown'}", key)

        pipe.hset(key, mapping={k: (v if v is not None else "") for k, v in data.items()})
        pipe.expire(key, REDIS_TTL_SECONDS)
        pipe.zadd(f"{self._PREFIX}:index:rating:{record.rating}", {key: ts})
        for cat in record.categories:
            pipe.zadd(f"{self._PREFIX}:index:cat:{cat}", {key: ts})
        pipe.zadd(f"{self._PREFIX}:index:model:{record.model_name or 'unknown'}", {key: ts})
        pipe.execute()

    def get(self, chat_id: str, message_id: str) -> FeedbackRecord | None:
        data = self._r.hgetall(self._record_key(chat_id, message_id))
        return FeedbackRecord.from_dict(data) if data else None

    def _fetch_keys(self, index_key: str, limit: int) -> list[str]:
        return self._r.zrevrange(index_key, 0, limit - 1)

    def _fetch_records(self, keys: list[str]) -> list[FeedbackRecord]:
        if not keys:
            return []
        pipe = self._r.pipeline()
        for k in keys:
            pipe.hgetall(k)
        records = []
        for data in pipe.execute():
            if data:
                try:
                    records.append(FeedbackRecord.from_dict(data))
                except (KeyError, TypeError):
                    continue
        return records

    def list_records(
        self,
        rating: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[FeedbackRecord]:
        if rating:
            keys = self._fetch_keys(f"{self._PREFIX}:index:rating:{rating}", limit)
        elif category:
            keys = self._fetch_keys(f"{self._PREFIX}:index:cat:{category}", limit)
        else:
            up_keys = self._fetch_keys(f"{self._PREFIX}:index:rating:up", limit)
            down_keys = self._fetch_keys(f"{self._PREFIX}:index:rating:down", limit)
            seen: set = set()
            keys = []
            for k in up_keys + down_keys:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
            keys = keys[:limit]
        if category and rating:
            cat_keys = set(self._fetch_keys(f"{self._PREFIX}:index:cat:{category}", limit * 2))
            keys = [k for k in keys if k in cat_keys][:limit]
        return self._fetch_records(keys)

    def stats(self) -> dict[str, Any]:
        up = self._r.zcard(f"{self._PREFIX}:index:rating:up")
        down = self._r.zcard(f"{self._PREFIX}:index:rating:down")
        total = up + down

        cat_counts: dict[str, dict[str, int]] = {}
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
            "by_model": {},  # full per-model aggregation omitted for Redis brevity
            "by_day": {},
        }


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _build_redis_store() -> RedisFeedbackStore | None:
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis as _redis  # type: ignore

        client = _redis.from_url(redis_url, decode_responses=True)
        client.ping()
        logger.info("Feedback store: Redis (%s)", redis_url.split("@")[-1])
        return RedisFeedbackStore(client)
    except Exception as exc:  # noqa: BLE001 - any Redis failure degrades to SQLite
        logger.warning("Redis unavailable (%s); falling back to SQLite.", exc)
        return None


def build_store() -> FeedbackStore:
    """Return the configured feedback store: Redis when reachable, else SQLite.

    The single place backend selection happens, so the service, the export
    script, and tests all agree on which store is live rather than each
    hardcoding SQLite.
    """
    redis_store = _build_redis_store()
    if redis_store is not None:
        return redis_store
    logger.info("Feedback store: SQLite (%s)", _SQLITE_PATH)
    return SQLiteFeedbackStore()


store: FeedbackStore = build_store()
