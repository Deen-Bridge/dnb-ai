"""Tests for the per-user retrieval layer (per-user RAG).

Everything runs offline: the dnb-backend HTTP calls are replaced by fetcher
seams, the embedding call is replaced by a deterministic bag-of-words embedder,
and Gemini is mocked. No API keys or network are used.

The load-bearing guarantees proven here:
- Deny-by-default: an unauthenticated turn retrieves nothing and never fetches.
- Isolation: a user only ever sees their own records — enforced pre-fetch by
  the store key and re-checked post-retrieval — proven by an adversarial unit
  test and an integration test through the real /chat handler.
- Retrieve, don't stuff: only records above the relevance floor, capped at
  top-k, reach the prompt block; unrelated same-user records are excluded.
- Graceful degrade: one signal's fetcher raising does not fail the turn.
- Source attribution is present on the attached context.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
import uuid
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from main import app
from memory import personal_context
from memory.models import PersonalContextBundle, PersonalRecord
from memory.personal_context import (
    build_personal_context,
    ingest_personal_records,
)
from memory.store import InMemoryMemoryStore

# ---------------------------------------------------------------------------
# Deterministic offline embedder (bag-of-words over a hashed vocabulary)
# ---------------------------------------------------------------------------

# A wide space keeps distinct words on distinct axes, so cosine similarity
# reflects real shared-word overlap rather than hash collisions.
_EMBED_DIM = 4096


def fake_embed(text: str) -> np.ndarray:
    """A stable, offline embedding: cosine reflects shared-word overlap."""
    vec = np.zeros(_EMBED_DIM, dtype=float)
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % _EMBED_DIM
        vec[idx] += 1.0
    return vec


@pytest.fixture
def offline_embeddings(monkeypatch):
    monkeypatch.setattr(personal_context, "embed_text", fake_embed)


# ---------------------------------------------------------------------------
# Record + store helpers
# ---------------------------------------------------------------------------


def record(
    user_id: str, text: str, *, record_type: str = "course_progress", rid: str = "r1", source: str = "course-progress"
) -> PersonalRecord:
    return PersonalRecord(
        user_id=user_id,
        record_type=record_type,
        record_id=rid,
        text=text,
        source=source,
    )


def seed(store: InMemoryMemoryStore, user_id: str, records: list[PersonalRecord]) -> None:
    bundle = PersonalContextBundle(user_id=user_id, records=records, fetched_at=time.time())
    asyncio.run(store.save_personal_context(user_id, bundle))


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Store: dual-backend user-scoped roundtrip + isolation
# ---------------------------------------------------------------------------


class TestPersonalStore:
    def test_roundtrip_and_ttl_key(self):
        store = InMemoryMemoryStore()
        rec = record("alice", "Your progress is 50 percent")
        run(store.save_personal_context("alice", PersonalContextBundle(user_id="alice", records=[rec])))
        loaded = run(store.get_personal_context("alice"))
        assert loaded is not None
        assert loaded.records[0].text == "Your progress is 50 percent"

    def test_missing_user_returns_none(self):
        store = InMemoryMemoryStore()
        assert run(store.get_personal_context("nobody")) is None

    def test_users_do_not_share_records(self):
        store = InMemoryMemoryStore()
        run(
            store.save_personal_context(
                "alice", PersonalContextBundle(user_id="alice", records=[record("alice", "alice data")])
            )
        )
        run(
            store.save_personal_context(
                "bob", PersonalContextBundle(user_id="bob", records=[record("bob", "bob data")])
            )
        )
        assert run(store.get_personal_context("bob")).records[0].text == "bob data"
        assert run(store.get_personal_context("alice")).records[0].text == "alice data"
        assert run(store.delete_personal_context("alice")) is True
        assert run(store.get_personal_context("alice")) is None
        # Deleting alice must not touch bob.
        assert run(store.get_personal_context("bob")) is not None


# ---------------------------------------------------------------------------
# Deny-by-default
# ---------------------------------------------------------------------------


class TestDenyByDefault:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    async def test_no_user_id_returns_none_and_never_fetches(self, monkeypatch):
        called = False

        async def boom(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("must not ingest for an unauthenticated turn")

        monkeypatch.setattr(personal_context, "ingest_personal_records", boom)
        store = InMemoryMemoryStore()
        result = await build_personal_context("my progress", user_id=None, auth_token="jwt", store=store)
        assert result is None
        assert called is False

    async def test_no_auth_token_returns_none(self, monkeypatch):
        async def boom(*args, **kwargs):
            raise AssertionError("must not ingest without a token")

        monkeypatch.setattr(personal_context, "ingest_personal_records", boom)
        store = InMemoryMemoryStore()
        result = await build_personal_context("my progress", user_id="alice", auth_token=None, store=store)
        assert result is None


# ---------------------------------------------------------------------------
# Seeded retrieval: answers "my progress" / "my next lesson" / "my pledges"
# ---------------------------------------------------------------------------


class TestSeededRetrieval:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    def _seed_alice(self, store):
        recs = [
            record("alice", "Your progress in the Alpha course is fifty percent complete", rid="p"),
            record("alice", "Your next lesson in the Alpha course is Purification", rid="l"),
            record(
                "alice",
                "Your pledges include one hundred USDC to the Masjid campaign",
                record_type="pledge",
                rid="pl",
                source="pledges",
            ),
        ]
        bundle = PersonalContextBundle(user_id="alice", records=recs, fetched_at=time.time())
        return bundle

    async def test_answers_my_progress(self, monkeypatch, offline_embeddings):
        monkeypatch.setattr(personal_context, "PERSONAL_CONTEXT_RELEVANCE_FLOOR", 0.05)
        store = InMemoryMemoryStore()
        await store.save_personal_context("alice", self._seed_alice(store))
        ctx = await build_personal_context("what is my progress", user_id="alice", auth_token="jwt", store=store)
        assert ctx is not None
        assert "fifty percent complete" in ctx.prompt_block

    async def test_answers_my_next_lesson(self, monkeypatch, offline_embeddings):
        monkeypatch.setattr(personal_context, "PERSONAL_CONTEXT_RELEVANCE_FLOOR", 0.05)
        store = InMemoryMemoryStore()
        await store.save_personal_context("alice", self._seed_alice(store))
        ctx = await build_personal_context("what is my next lesson", user_id="alice", auth_token="jwt", store=store)
        assert ctx is not None
        assert "Purification" in ctx.prompt_block

    async def test_answers_my_pledges(self, monkeypatch, offline_embeddings):
        monkeypatch.setattr(personal_context, "PERSONAL_CONTEXT_RELEVANCE_FLOOR", 0.05)
        store = InMemoryMemoryStore()
        await store.save_personal_context("alice", self._seed_alice(store))
        ctx = await build_personal_context("what are my pledges", user_id="alice", auth_token="jwt", store=store)
        assert ctx is not None
        assert "Masjid campaign" in ctx.prompt_block

    async def test_source_attribution_present(self, monkeypatch, offline_embeddings):
        monkeypatch.setattr(personal_context, "PERSONAL_CONTEXT_RELEVANCE_FLOOR", 0.05)
        store = InMemoryMemoryStore()
        await store.save_personal_context("alice", self._seed_alice(store))
        ctx = await build_personal_context("what are my pledges", user_id="alice", auth_token="jwt", store=store)
        assert ctx is not None
        assert "Source: pledges" in ctx.prompt_block
        assert "[pledge]" in ctx.prompt_block


# ---------------------------------------------------------------------------
# Top-k relevance: unrelated same-user records are excluded (not dump-all)
# ---------------------------------------------------------------------------


class TestTopKRelevance:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    async def test_unrelated_record_excluded(self, monkeypatch, offline_embeddings):
        monkeypatch.setattr(personal_context, "PERSONAL_CONTEXT_RELEVANCE_FLOOR", 0.05)
        store = InMemoryMemoryStore()
        recs = [
            record("alice", "alpha beta gamma progress", rid="rel"),
            record(
                "alice",
                "brewing coffee beans on a stovetop",
                rid="unrel",
                record_type="saved_item",
                source="saved-items",
            ),
        ]
        await store.save_personal_context(
            "alice", PersonalContextBundle(user_id="alice", records=recs, fetched_at=time.time())
        )
        ctx = await build_personal_context("alpha beta gamma", user_id="alice", auth_token="jwt", store=store)
        assert ctx is not None
        assert "alpha beta gamma progress" in ctx.prompt_block
        assert "coffee" not in ctx.prompt_block
        assert ctx.info.selected_records == 1
        assert ctx.info.total_records == 2

    async def test_top_k_caps_selection(self, monkeypatch, offline_embeddings):
        monkeypatch.setattr(personal_context, "PERSONAL_CONTEXT_RELEVANCE_FLOOR", 0.05)
        monkeypatch.setattr(personal_context, "PERSONAL_CONTEXT_TOP_K", 2)
        store = InMemoryMemoryStore()
        recs = [
            record("alice", "alpha beta gamma markerA", rid="a"),
            record("alice", "alpha beta markerB filler", rid="b"),
            record("alice", "alpha markerC filler filler", rid="c"),
            record("alice", "markerD zzz yyy www", rid="d"),
        ]
        await store.save_personal_context(
            "alice", PersonalContextBundle(user_id="alice", records=recs, fetched_at=time.time())
        )
        ctx = await build_personal_context("alpha beta gamma", user_id="alice", auth_token="jwt", store=store)
        assert ctx is not None
        assert ctx.info.selected_records == 2
        assert "markerA" in ctx.prompt_block
        assert "markerB" in ctx.prompt_block
        assert "markerC" not in ctx.prompt_block  # below the top-k cut
        assert "markerD" not in ctx.prompt_block  # zero overlap, below floor


# ---------------------------------------------------------------------------
# Isolation: adversarial unit — a foreign record in the bundle is dropped
# ---------------------------------------------------------------------------


class TestIsolationUnit:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    async def test_foreign_record_is_dropped_post_retrieval(self, monkeypatch, offline_embeddings):
        monkeypatch.setattr(personal_context, "PERSONAL_CONTEXT_RELEVANCE_FLOOR", 0.05)
        store = InMemoryMemoryStore()
        # A tampered bundle stored under bob's key that smuggles one of alice's
        # records. The post-retrieval ownership re-check must drop it.
        recs = [
            record("bob", "alpha beta gamma bob record", rid="bobrec"),
            record("alice", "alpha beta gamma alice secret", rid="alicerec"),
        ]
        await store.save_personal_context(
            "bob", PersonalContextBundle(user_id="bob", records=recs, fetched_at=time.time())
        )
        ctx = await build_personal_context("alpha beta gamma", user_id="bob", auth_token="jwt", store=store)
        assert ctx is not None
        assert "bob record" in ctx.prompt_block
        assert "alice secret" not in ctx.prompt_block
        assert ctx.info.total_records == 1  # foreign record excluded before ranking


# ---------------------------------------------------------------------------
# Graceful degrade: one signal raising leaves the others intact
# ---------------------------------------------------------------------------


class TestGracefulDegrade:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    async def test_one_fetcher_raising_does_not_fail_ingest(self, monkeypatch):
        async def fake_fetch_json(path, auth_token, *, client=None):
            if path == "/api/enrollments":
                return [{"courseTitle": "Intro to Fiqh", "status": "active"}]
            if path == "/api/pledges":
                raise httpx.ConnectError("backend down")
            return []

        async def no_purchases(auth_token, *, client=None):
            return []

        monkeypatch.setattr(personal_context, "_fetch_json", fake_fetch_json)
        monkeypatch.setattr(personal_context, "fetch_user_transactions", no_purchases)

        records = await ingest_personal_records("alice", "jwt")
        types = {r.record_type for r in records}
        assert "enrollment" in types  # survived
        assert "pledge" not in types  # raised, simply absent
        # Every record is stamped with the caller.
        assert all(r.user_id == "alice" for r in records)


# ---------------------------------------------------------------------------
# Incremental refresh: a stale bundle is re-ingested on read
# ---------------------------------------------------------------------------


class TestIncrementalRefresh:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    async def test_stale_bundle_triggers_reingest(self, monkeypatch, offline_embeddings):
        monkeypatch.setattr(personal_context, "PERSONAL_CONTEXT_RELEVANCE_FLOOR", 0.05)
        monkeypatch.setattr(personal_context, "PERSONAL_CONTEXT_TTL_SECONDS", 60)
        store = InMemoryMemoryStore()
        # A bundle older than the TTL must be refreshed from the fetchers.
        stale = PersonalContextBundle(
            user_id="alice",
            records=[record("alice", "old stale alpha record", rid="old")],
            fetched_at=time.time() - 3600,
        )
        await store.save_personal_context("alice", stale)

        async def fresh_ingest(user_id, auth_token, *, client=None):
            return [
                record(
                    user_id, "fresh alpha enrollment record", record_type="enrollment", rid="new", source="enrollments"
                )
            ]

        monkeypatch.setattr(personal_context, "ingest_personal_records", fresh_ingest)
        ctx = await build_personal_context("alpha", user_id="alice", auth_token="jwt", store=store)
        assert ctx is not None
        assert ctx.info.stale_refetched is True
        assert "fresh alpha enrollment record" in ctx.prompt_block
        assert "stale" not in ctx.prompt_block


# ---------------------------------------------------------------------------
# Integration through the real /chat handler (TestClient + fake generator)
# ---------------------------------------------------------------------------

client = TestClient(app)


def mock_model_capturing(sink: dict):
    """A Gemini stand-in that records the full prompt it is sent."""
    session = MagicMock()

    async def send_message_async(message, **kwargs):
        sink["prompt"] = message
        response = MagicMock()
        response.text = "Here is your answer."
        response.candidates = [MagicMock(finish_reason="STOP")]
        response.prompt_feedback = None
        return response

    session.send_message_async = send_message_async
    session.history = []
    model = MagicMock()
    model.start_chat.return_value = session
    return model


@pytest.fixture
def chat_offline(monkeypatch):
    """Disable safety, use a fresh user-scoped store, and the offline embedder."""
    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "false")
    monkeypatch.setattr(personal_context, "embed_text", fake_embed)
    monkeypatch.setattr(personal_context, "PERSONAL_CONTEXT_RELEVANCE_FLOOR", 0.05)
    store = InMemoryMemoryStore()
    monkeypatch.setattr(main, "memory_store", store)
    main.active_chats.clear()
    yield store
    main.active_chats.clear()


class TestChatIntegration:
    def test_personal_block_reaches_the_prompt(self, monkeypatch, chat_offline):
        store = chat_offline
        recs = [
            record("alice", "Your progress in the Alpha course is fifty percent complete", rid="p"),
            record("alice", "brewing coffee beans", rid="u", record_type="saved_item", source="saved-items"),
        ]
        seed(store, "alice", recs)

        sink: dict = {}
        monkeypatch.setattr(main, "get_model", lambda: mock_model_capturing(sink))
        resp = client.post(
            "/chat",
            json={
                "prompt": "what is my progress",
                "chat_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "c-personal")),
                "user_id": "alice",
                "auth_token": "jwt-alice",
                "remember": False,
            },
        )
        assert resp.status_code == 200, resp.text
        # The retriever ran and its cited block reached the assembled prompt.
        assert "YOUR DEEN BRIDGE ACTIVITY" in sink["prompt"]
        assert "fifty percent complete" in sink["prompt"]
        # Retrieve, don't stuff: the unrelated record did not land in the prompt.
        assert "coffee" not in sink["prompt"]

    def test_cross_user_isolation_through_chat(self, monkeypatch, chat_offline):
        store = chat_offline
        seed(store, "alice", [record("alice", "Your progress in the Alpha course is halfway", rid="pa")])
        seed(store, "bob", [record("bob", "Your progress in the Beta course is starting", rid="pb")])

        sink: dict = {}
        monkeypatch.setattr(main, "get_model", lambda: mock_model_capturing(sink))

        # Bob's identical query must surface only Bob's records.
        resp = client.post(
            "/chat",
            json={
                "prompt": "what is my progress",
                "chat_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "c-bob")),
                "user_id": "bob",
                "auth_token": "jwt-bob",
                "remember": False,
            },
        )
        assert resp.status_code == 200, resp.text
        assert "Beta course" in sink["prompt"]
        assert "Alpha course" not in sink["prompt"]

        # And Alice's the same query surfaces only Alice's.
        resp = client.post(
            "/chat",
            json={
                "prompt": "what is my progress",
                "chat_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "c-alice")),
                "user_id": "alice",
                "auth_token": "jwt-alice",
                "remember": False,
            },
        )
        assert resp.status_code == 200, resp.text
        assert "Alpha course" in sink["prompt"]
        assert "Beta course" not in sink["prompt"]

    def test_unauthenticated_turn_attaches_no_personal_records(self, monkeypatch, chat_offline):
        store = chat_offline
        seed(store, "alice", [record("alice", "Your progress in the Alpha course is halfway", rid="pa")])

        sink: dict = {}
        monkeypatch.setattr(main, "get_model", lambda: mock_model_capturing(sink))
        # No user_id / auth_token → deny by default.
        resp = client.post(
            "/chat",
            json={"prompt": "what is my progress", "chat_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "c-anon"))},
        )
        assert resp.status_code == 200, resp.text
        assert "YOUR DEEN BRIDGE ACTIVITY" not in sink["prompt"]
        assert "Alpha course" not in sink["prompt"]
