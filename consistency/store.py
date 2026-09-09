"""Factual claim storage abstraction — Redis-backed with in-memory fallback.

Provides fast multi-index retrieval of historical factual claims by topic,
entity, user ID, session ID, and semantic similarity.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import re

from consistency.models import FactualClaim

logger = logging.getLogger(__name__)

CLAIM_TTL_SECONDS = int(os.getenv("CLAIM_TTL_DAYS", "180")) * 86400
MAX_IN_MEMORY_CLAIMS = int(os.getenv("MAX_IN_MEMORY_CLAIMS", "50000"))
REDIS_URL = os.getenv("REDIS_URL", "")

try:
    import redis.asyncio as aioredis

    _redis_available = True
except ImportError:
    _redis_available = False


def _normalize_key(text: str) -> str:
    """Normalize string for indexing."""
    cleaned = text.lower().strip()
    return re.sub(r"[^\w\s]", "", cleaned)


class ClaimStore(abc.ABC):
    """Abstract interface for long-term factual claim persistence and querying."""

    @abc.abstractmethod
    async def save_claim(self, claim: FactualClaim) -> None: ...

    @abc.abstractmethod
    async def save_claims(self, claims: list[FactualClaim]) -> None: ...

    @abc.abstractmethod
    async def get_claims_by_topic(self, topic: str, limit: int = 50) -> list[FactualClaim]: ...

    @abc.abstractmethod
    async def get_claims_by_entity(self, entity: str, limit: int = 50) -> list[FactualClaim]: ...

    @abc.abstractmethod
    async def get_claims_by_user(self, user_id: str, limit: int = 100) -> list[FactualClaim]: ...

    @abc.abstractmethod
    async def get_claims_by_session(self, session_id: str) -> list[FactualClaim]: ...

    @abc.abstractmethod
    async def find_relevant_claims(
        self,
        query: str,
        topic: str | None = None,
        entity: str | None = None,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[FactualClaim]: ...

    @abc.abstractmethod
    async def get_all_claims(self, limit: int = 1000) -> list[FactualClaim]: ...

    @abc.abstractmethod
    async def count_claims(self) -> int: ...

    @abc.abstractmethod
    async def delete_user_claims(self, user_id: str) -> int: ...


class InMemoryClaimStore(ClaimStore):
    """Process-local multi-indexed claim store for fast testing and local development."""

    def __init__(self, max_capacity: int = MAX_IN_MEMORY_CLAIMS) -> None:
        self.max_capacity = max_capacity
        self._claims: dict[str, FactualClaim] = {}
        self._by_topic: dict[str, list[str]] = {}
        self._by_entity: dict[str, list[str]] = {}
        self._by_user: dict[str, list[str]] = {}
        self._by_session: dict[str, list[str]] = {}
        self._chronological: list[str] = []

    async def save_claim(self, claim: FactualClaim) -> None:
        if not claim.normalized_text:
            claim.normalized_text = _normalize_key(claim.text)

        # Evict oldest if exceeding capacity
        if len(self._claims) >= self.max_capacity and claim.claim_id not in self._claims:
            oldest_id = self._chronological.pop(0)
            self._evict_claim_id(oldest_id)

        self._claims[claim.claim_id] = claim
        if claim.claim_id not in self._chronological:
            self._chronological.append(claim.claim_id)

        # Update topic index
        norm_topic = _normalize_key(claim.topic)
        if norm_topic:
            self._by_topic.setdefault(norm_topic, [])
            if claim.claim_id not in self._by_topic[norm_topic]:
                self._by_topic[norm_topic].append(claim.claim_id)

        # Update entity index
        if claim.entity:
            norm_entity = _normalize_key(claim.entity)
            self._by_entity.setdefault(norm_entity, [])
            if claim.claim_id not in self._by_entity[norm_entity]:
                self._by_entity[norm_entity].append(claim.claim_id)

        # Update user index
        if claim.user_id:
            self._by_user.setdefault(claim.user_id, [])
            if claim.claim_id not in self._by_user[claim.user_id]:
                self._by_user[claim.user_id].append(claim.claim_id)

        # Update session index
        session_ref = claim.session_id or claim.chat_id
        if session_ref:
            self._by_session.setdefault(session_ref, [])
            if claim.claim_id not in self._by_session[session_ref]:
                self._by_session[session_ref].append(claim.claim_id)

    async def save_claims(self, claims: list[FactualClaim]) -> None:
        for c in claims:
            await self.save_claim(c)

    def _evict_claim_id(self, claim_id: str) -> None:
        claim = self._claims.pop(claim_id, None)
        if not claim:
            return

        norm_topic = _normalize_key(claim.topic)
        if norm_topic in self._by_topic and claim_id in self._by_topic[norm_topic]:
            self._by_topic[norm_topic].remove(claim_id)

        if claim.entity:
            norm_entity = _normalize_key(claim.entity)
            if norm_entity in self._by_entity and claim_id in self._by_entity[norm_entity]:
                self._by_entity[norm_entity].remove(claim_id)

        if claim.user_id and claim.user_id in self._by_user and claim_id in self._by_user[claim.user_id]:
            self._by_user[claim.user_id].remove(claim_id)

        session_ref = claim.session_id or claim.chat_id
        if session_ref and session_ref in self._by_session and claim_id in self._by_session[session_ref]:
            self._by_session[session_ref].remove(claim_id)

    async def get_claims_by_topic(self, topic: str, limit: int = 50) -> list[FactualClaim]:
        norm_topic = _normalize_key(topic)
        ids = self._by_topic.get(norm_topic, [])
        # Return most recent first
        return [self._claims[cid] for cid in reversed(ids) if cid in self._claims][:limit]

    async def get_claims_by_entity(self, entity: str, limit: int = 50) -> list[FactualClaim]:
        norm_entity = _normalize_key(entity)
        ids = self._by_entity.get(norm_entity, [])
        return [self._claims[cid] for cid in reversed(ids) if cid in self._claims][:limit]

    async def get_claims_by_user(self, user_id: str, limit: int = 100) -> list[FactualClaim]:
        ids = self._by_user.get(user_id, [])
        return [self._claims[cid] for cid in reversed(ids) if cid in self._claims][:limit]

    async def get_claims_by_session(self, session_id: str) -> list[FactualClaim]:
        ids = self._by_session.get(session_id, [])
        return [self._claims[cid] for cid in ids if cid in self._claims]

    async def find_relevant_claims(
        self,
        query: str,
        topic: str | None = None,
        entity: str | None = None,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[FactualClaim]:
        candidates: set[str] = set()

        if topic:
            norm_topic = _normalize_key(topic)
            candidates.update(self._by_topic.get(norm_topic, []))
            # Also check partial topic matching
            for t_key, ids in self._by_topic.items():
                if norm_topic in t_key or t_key in norm_topic:
                    candidates.update(ids)

        if entity:
            norm_entity = _normalize_key(entity)
            candidates.update(self._by_entity.get(norm_entity, []))
            for e_key, ids in self._by_entity.items():
                if norm_entity in e_key or e_key in norm_entity:
                    candidates.update(ids)

        if user_id and user_id in self._by_user:
            candidates.update(self._by_user[user_id])

        # If no specific key filters matched, search all claims
        target_ids = candidates if candidates else set(self._claims.keys())
        if not target_ids:
            return []

        query_tokens = set(_normalize_key(query).split())
        scored: list[tuple[float, FactualClaim]] = []

        for cid in target_ids:
            claim = self._claims.get(cid)
            if not claim:
                continue

            claim_tokens = set(claim.normalized_text.split())
            if not claim_tokens:
                claim_tokens = set(_normalize_key(claim.text).split())

            overlap = len(query_tokens & claim_tokens)
            score = overlap / max(1, len(query_tokens | claim_tokens))

            # Boost score if topic or entity matches
            if topic and _normalize_key(claim.topic) == _normalize_key(topic):
                score += 0.3
            if entity and claim.entity and _normalize_key(claim.entity) == _normalize_key(entity):
                score += 0.4

            if score > 0.05 or not query_tokens:
                scored.append((score, claim))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:limit]]

    async def get_all_claims(self, limit: int = 1000) -> list[FactualClaim]:
        return [self._claims[cid] for cid in reversed(self._chronological)][:limit]

    async def count_claims(self) -> int:
        return len(self._claims)

    async def delete_user_claims(self, user_id: str) -> int:
        ids = list(self._by_user.get(user_id, []))
        deleted_count = 0
        for cid in ids:
            if cid in self._claims:
                self._evict_claim_id(cid)
                if cid in self._chronological:
                    self._chronological.remove(cid)
                deleted_count += 1
        self._by_user.pop(user_id, None)
        return deleted_count


class RedisClaimStore(ClaimStore):
    """Distributed Redis-backed factual claim store."""

    def __init__(self, redis_url: str) -> None:
        if not _redis_available:
            raise RuntimeError("redis package not installed")
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._fallback_in_memory = InMemoryClaimStore()

    def _claim_key(self, claim_id: str) -> str:
        return f"consistency:claim:{claim_id}"

    def _topic_set_key(self, topic: str) -> str:
        return f"consistency:topic:{_normalize_key(topic)}"

    def _entity_set_key(self, entity: str) -> str:
        return f"consistency:entity:{_normalize_key(entity)}"

    def _user_set_key(self, user_id: str) -> str:
        return f"consistency:user:{user_id}"

    def _session_set_key(self, session_id: str) -> str:
        return f"consistency:session:{session_id}"

    async def save_claim(self, claim: FactualClaim) -> None:
        if not claim.normalized_text:
            claim.normalized_text = _normalize_key(claim.text)

        payload = claim.model_dump_json()
        pipe = self._redis.pipeline()
        pipe.setex(self._claim_key(claim.claim_id), CLAIM_TTL_SECONDS, payload)

        norm_topic = _normalize_key(claim.topic)
        if norm_topic:
            pipe.sadd(self._topic_set_key(norm_topic), claim.claim_id)
            pipe.expire(self._topic_set_key(norm_topic), CLAIM_TTL_SECONDS)

        if claim.entity:
            pipe.sadd(self._entity_set_key(claim.entity), claim.claim_id)
            pipe.expire(self._entity_set_key(claim.entity), CLAIM_TTL_SECONDS)

        if claim.user_id:
            pipe.sadd(self._user_set_key(claim.user_id), claim.claim_id)
            pipe.expire(self._user_set_key(claim.user_id), CLAIM_TTL_SECONDS)

        session_ref = claim.session_id or claim.chat_id
        if session_ref:
            pipe.sadd(self._session_set_key(session_ref), claim.claim_id)
            pipe.expire(self._session_set_key(session_ref), CLAIM_TTL_SECONDS)

        await pipe.execute()
        await self._fallback_in_memory.save_claim(claim)

    async def save_claims(self, claims: list[FactualClaim]) -> None:
        for c in claims:
            await self.save_claim(c)

    async def get_claims_by_topic(self, topic: str, limit: int = 50) -> list[FactualClaim]:
        norm_topic = _normalize_key(topic)
        ids = await self._redis.smembers(self._topic_set_key(norm_topic))
        if not ids:
            return []
        return await self._fetch_claims(list(ids)[:limit])

    async def get_claims_by_entity(self, entity: str, limit: int = 50) -> list[FactualClaim]:
        ids = await self._redis.smembers(self._entity_set_key(entity))
        if not ids:
            return []
        return await self._fetch_claims(list(ids)[:limit])

    async def get_claims_by_user(self, user_id: str, limit: int = 100) -> list[FactualClaim]:
        ids = await self._redis.smembers(self._user_set_key(user_id))
        if not ids:
            return []
        return await self._fetch_claims(list(ids)[:limit])

    async def get_claims_by_session(self, session_id: str) -> list[FactualClaim]:
        ids = await self._redis.smembers(self._session_set_key(session_id))
        if not ids:
            return []
        return await self._fetch_claims(list(ids))

    async def _fetch_claims(self, claim_ids: list[str]) -> list[FactualClaim]:
        if not claim_ids:
            return []
        keys = [self._claim_key(cid) for cid in claim_ids]
        raw_items = await self._redis.mget(keys)
        claims: list[FactualClaim] = []
        for raw in raw_items:
            if raw:
                try:
                    claims.append(FactualClaim.model_validate(json.loads(raw)))
                except Exception as exc:
                    logger.warning("Corrupt claim in Redis: %s", exc)
        return claims

    async def find_relevant_claims(
        self,
        query: str,
        topic: str | None = None,
        entity: str | None = None,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[FactualClaim]:
        # Fast local candidate search + fetch
        return await self._fallback_in_memory.find_relevant_claims(
            query=query, topic=topic, entity=entity, user_id=user_id, limit=limit
        )

    async def get_all_claims(self, limit: int = 1000) -> list[FactualClaim]:
        return await self._fallback_in_memory.get_all_claims(limit=limit)

    async def count_claims(self) -> int:
        return await self._fallback_in_memory.count_claims()

    async def delete_user_claims(self, user_id: str) -> int:
        ids = await self._redis.smembers(self._user_set_key(user_id))
        if not ids:
            return 0
        pipe = self._redis.pipeline()
        for cid in ids:
            pipe.delete(self._claim_key(cid))
        pipe.delete(self._user_set_key(user_id))
        await pipe.execute()
        await self._fallback_in_memory.delete_user_claims(user_id)
        return len(ids)


def create_claim_store() -> ClaimStore:
    """Factory: REDIS_URL set -> RedisClaimStore, else InMemoryClaimStore."""
    url = os.getenv("REDIS_URL", "")
    if url:
        logger.info("ClaimStore using Redis at %s", url)
        return RedisClaimStore(url)
    logger.info("ClaimStore using in-memory store")
    return InMemoryClaimStore()
