"""Tests for the answer-feedback loop: store, rate limiter, endpoints, and export.

Everything runs offline against a temporary SQLite store — no Redis, no live
model calls. The FastAPI app is exercised through httpx's ASGI transport with
Gemini stubbed, matching the existing endpoint tests.
"""

import asyncio
import json
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

import feedback
from feedback import (
    FeedbackRecord,
    RateLimiter,
    SQLiteFeedbackStore,
    build_store,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_record(**overrides) -> FeedbackRecord:
    fields = dict(
        feedback_id=str(uuid.uuid4()),
        chat_id="chat-1",
        message_id="msg-1",
        rating="down",
        categories=["incorrect_information"],
        comment="wrong ayah number",
        prompt="What does Surah al-Asr say?",
        answer="An incorrect paraphrase.",
        model_name="gemini-test",
        generation_config={"temperature": 0.7},
        created_at="2026-07-25T10:00:00+00:00",
    )
    fields.update(overrides)
    return FeedbackRecord(**fields)


@pytest.fixture()
def store(tmp_path):
    return SQLiteFeedbackStore(db_path=str(tmp_path / "feedback.db"))


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


class TestSQLiteStore:
    def test_upsert_and_get_roundtrip(self, store):
        record = make_record()
        store.upsert(record)
        loaded = store.get("chat-1", "msg-1")
        assert loaded is not None
        assert loaded.rating == "down"
        assert loaded.categories == ["incorrect_information"]
        assert loaded.generation_config == {"temperature": 0.7}

    def test_get_missing_returns_none(self, store):
        assert store.get("nope", "nope") is None

    def test_upsert_is_idempotent_per_message(self, store):
        """Resubmitting for the same (chat_id, message_id) overwrites, not appends."""
        store.upsert(make_record(rating="down"))
        store.upsert(make_record(rating="up", comment="changed my mind"))
        records = store.list_records()
        assert len(records) == 1
        assert records[0].rating == "up"
        assert records[0].comment == "changed my mind"

    def test_list_filters_by_rating(self, store):
        store.upsert(make_record(message_id="a", rating="down"))
        store.upsert(make_record(message_id="b", rating="up"))
        down = store.list_records(rating="down")
        assert [r.message_id for r in down] == ["a"]

    def test_list_filters_by_category(self, store):
        store.upsert(make_record(message_id="a", categories=["too_long"]))
        store.upsert(make_record(message_id="b", categories=["poor_adab"]))
        assert [r.message_id for r in store.list_records(category="poor_adab")] == ["b"]

    def test_category_filter_is_exact_not_substring(self, store):
        """A LIKE on the quoted token must not match a different category."""
        store.upsert(make_record(message_id="a", categories=["wrong_language"]))
        assert store.list_records(category="language") == []

    def test_list_orders_newest_first(self, store):
        store.upsert(make_record(message_id="old", created_at="2026-07-01T00:00:00+00:00"))
        store.upsert(make_record(message_id="new", created_at="2026-07-25T00:00:00+00:00"))
        assert [r.message_id for r in store.list_records()] == ["new", "old"]

    def test_stats_counts_and_ratio(self, store):
        for i in range(3):
            store.upsert(make_record(chat_id=f"c{i}", rating="down", categories=["too_vague"]))
        store.upsert(make_record(chat_id="up1", rating="up", categories=[]))
        stats = store.stats()
        assert stats["total"] == 4
        assert stats["down"] == 3
        assert stats["up"] == 1
        assert stats["up_ratio"] == 0.25
        assert stats["by_category"]["too_vague"]["down"] == 3

    def test_stats_empty_store(self, store):
        stats = store.stats()
        assert stats["total"] == 0
        assert stats["up_ratio"] is None


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_up_to_the_limit_then_blocks(self):
        limiter = RateLimiter(max_calls=3, window_seconds=60)
        assert [limiter.is_allowed("ip") for _ in range(4)] == [True, True, True, False]

    def test_buckets_are_per_ip(self):
        limiter = RateLimiter(max_calls=1, window_seconds=60)
        assert limiter.is_allowed("a") is True
        assert limiter.is_allowed("b") is True
        assert limiter.is_allowed("a") is False

    def test_reset_clears_state(self):
        limiter = RateLimiter(max_calls=1, window_seconds=60)
        assert limiter.is_allowed("a") is True
        assert limiter.is_allowed("a") is False
        limiter.reset()
        assert limiter.is_allowed("a") is True

    def test_window_expiry_frees_slots(self):
        limiter = RateLimiter(max_calls=1, window_seconds=0.05)
        assert limiter.is_allowed("a") is True
        assert limiter.is_allowed("a") is False
        import time

        time.sleep(0.06)
        assert limiter.is_allowed("a") is True


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_build_store_is_sqlite_without_redis_url(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        assert isinstance(build_store(), SQLiteFeedbackStore)

    def test_build_store_falls_back_when_redis_unreachable(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6390/0")  # nothing listening
        assert isinstance(build_store(), SQLiteFeedbackStore)


# ---------------------------------------------------------------------------
# Export script
# ---------------------------------------------------------------------------


class TestExport:
    def _export_module(self):
        import importlib
        import scripts.export_eval_candidates as exp

        return importlib.reload(exp)

    def test_down_rated_records_become_candidates(self, store):
        store.upsert(make_record(message_id="a", categories=["incorrect_information"]))
        exp = self._export_module()
        candidates = exp.build_candidates(store.list_records(rating="down"))
        assert len(candidates) == 1
        c = candidates[0]
        assert c["needs_review"] is True
        assert c["source"] == "user_feedback"
        assert c["category"] == "factual_accuracy"  # taxonomy -> harness mapping
        assert c["answer_draft"] == "An incorrect paraphrase."
        assert "expected" not in c  # never fabricates a ground-truth answer

    def test_records_without_a_prompt_are_skipped(self, store):
        store.upsert(make_record(prompt=None))
        exp = self._export_module()
        assert exp.build_candidates(store.list_records(rating="down")) == []

    def test_near_duplicate_prompts_are_deduplicated(self, store):
        store.upsert(make_record(message_id="a", prompt="What is zakat?"))
        store.upsert(make_record(message_id="b", prompt="what   is   ZAKAT?"))
        exp = self._export_module()
        candidates = exp.build_candidates(store.list_records(rating="down"))
        assert len(candidates) == 1

    def test_min_categories_filter(self, store):
        store.upsert(make_record(message_id="a", categories=["too_long"]))
        store.upsert(make_record(message_id="b", categories=["too_long", "too_vague"]))
        exp = self._export_module()
        records = store.list_records(rating="down")
        assert len(exp.build_candidates(records, min_categories=2)) == 1

    def test_export_honors_db_path(self, store, tmp_path):
        store.upsert(make_record())
        exp = self._export_module()
        out = tmp_path / "candidates.jsonl"
        count = exp.export(output_path=str(out), db_path=store._db_path)
        assert count == 1
        line = json.loads(out.read_text().strip())
        assert line["question"] == "What does Surah al-Asr say?"

    def test_export_without_db_uses_configured_backend(self, tmp_path, monkeypatch):
        """No --db must read the live store, not a hardcoded feedback.db."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.setenv("FEEDBACK_DB_PATH", str(tmp_path / "live.db"))
        import feedback as fb

        live = SQLiteFeedbackStore(db_path=str(tmp_path / "live.db"))
        live.upsert(make_record())
        monkeypatch.setattr(fb, "build_store", lambda: live)
        exp = self._export_module()
        monkeypatch.setattr(exp, "build_store", lambda: live)
        assert exp.export(output_path=None, db_path=None) == 1


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

ADMIN_TOKEN = "test-admin-token"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient with feedback pointed at a temp DB and Gemini stubbed."""
    from unittest.mock import AsyncMock, MagicMock

    import main

    temp_store = SQLiteFeedbackStore(db_path=str(tmp_path / "feedback.db"))
    monkeypatch.setattr(main, "feedback_store", temp_store)
    monkeypatch.setattr(feedback, "store", temp_store)
    monkeypatch.setattr(main, "ADMIN_TOKEN", ADMIN_TOKEN)

    # Reset the shared rate limiter so counts never leak between tests.
    main.rate_limiter.reset()

    # Isolate chat state and neutralize the heavy model pipeline.
    monkeypatch.setattr(main, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main, "active_chats", {})
    monkeypatch.setattr(main, "chat_message_ids", {})
    monkeypatch.setattr(main, "answer_snapshots", main.OrderedDict())
    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "false")
    monkeypatch.setattr(main, "SEMANTIC_CACHE_ENABLED", False)
    monkeypatch.setattr(main, "zakat_retriever", AsyncMock(return_value=None))
    monkeypatch.setattr(main, "tafsir_retriever", AsyncMock(return_value=None))
    monkeypatch.setattr(main, "enqueue_for_review", AsyncMock())

    def fake_model(*args, **kwargs):
        session = MagicMock()
        session.history = []

        async def send_message_async(message, **kw):
            resp = MagicMock()
            resp.text = "A model answer."
            resp.candidates = [MagicMock(finish_reason="STOP")]
            resp.prompt_feedback = None  # avoid the safety-block false positive
            session.history = [
                MagicMock(role="user", parts=[MagicMock(text=message)]),
                MagicMock(role="model", parts=[MagicMock(text="A model answer.")]),
            ]
            return resp

        session.send_message_async = send_message_async
        session.send_message = MagicMock()
        model = MagicMock()
        model.start_chat.return_value = session
        return model

    monkeypatch.setattr(main.genai, "GenerativeModel", fake_model)
    monkeypatch.setattr(main, "get_model", lambda *a, **k: fake_model())

    transport = ASGITransport(app=main.app)
    return AsyncClient(transport=transport, base_url="http://test")


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.mark.asyncio
class TestFeedbackEndpoint:
    async def _one_answer(self, client):
        """Drive a chat turn and return (chat_id, message_id)."""
        resp = await client.post("/chat", json={"prompt": "What is zakat?"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        return body["chat_id"], body["message_id"]

    async def test_chat_response_carries_message_id(self, client):
        async with client:
            chat_id, message_id = await self._one_answer(client)
            assert message_id
            # And the model turn in history is tagged with it.
            resp = await client.post(
                "/chat", json={"prompt": "again", "chat_id": chat_id}
            )
            history = resp.json()["history"]
            model_turns = [m for m in history if m["role"] == "model"]
            assert all(m["message_id"] for m in model_turns)

    async def test_submit_down_feedback_from_live_snapshot(self, client):
        async with client:
            chat_id, message_id = await self._one_answer(client)
            resp = await client.post("/feedback", json={
                "chat_id": chat_id,
                "message_id": message_id,
                "rating": "down",
                "categories": ["incorrect_information"],
                "comment": "wrong",
            })
            assert resp.status_code == 200
            fid = resp.json()["feedback_id"]

            import main
            stored = main.feedback_store.get(chat_id, message_id)
            assert stored.rating == "down"
            assert stored.feedback_id == fid
            # Snapshot was resolved server-side, not left to the client, and it
            # captures the *displayed* answer — including any confidence or
            # hadith note appended after generation, which is what the user saw.
            assert stored.prompt == "What is zakat?"
            assert stored.answer.startswith("A model answer.")

    async def test_up_feedback_needs_no_categories(self, client):
        async with client:
            chat_id, message_id = await self._one_answer(client)
            resp = await client.post("/feedback", json={
                "chat_id": chat_id, "message_id": message_id, "rating": "up",
            })
            assert resp.status_code == 200

    @pytest.mark.parametrize("payload,field", [
        ({"rating": "sideways"}, "rating"),
        ({"rating": "down", "categories": ["not_a_category"]}, "categories"),
        ({"rating": "down", "comment": "x" * 1001}, "comment"),
    ])
    async def test_invalid_body_is_422(self, client, payload, field):
        async with client:
            chat_id, message_id = await self._one_answer(client)
            payload = {"chat_id": chat_id, "message_id": message_id, **payload}
            resp = await client.post("/feedback", json=payload)
            assert resp.status_code == 422

    async def test_missing_snapshot_requires_client_prompt_and_answer(self, client):
        async with client:
            resp = await client.post("/feedback", json={
                "chat_id": "gone", "message_id": "gone", "rating": "down",
            })
            assert resp.status_code == 422

    async def test_missing_snapshot_accepts_client_supplied_pair(self, client):
        async with client:
            resp = await client.post("/feedback", json={
                "chat_id": "gone", "message_id": "gone", "rating": "down",
                "prompt": "client prompt", "answer": "client answer",
            })
            assert resp.status_code == 200
            import main
            stored = main.feedback_store.get("gone", "gone")
            assert stored.prompt == "client prompt"
            assert stored.answer == "client answer"

    async def test_resubmission_overwrites(self, client):
        async with client:
            chat_id, message_id = await self._one_answer(client)
            base = {"chat_id": chat_id, "message_id": message_id}
            await client.post("/feedback", json={**base, "rating": "down"})
            await client.post("/feedback", json={**base, "rating": "up"})
            import main
            assert main.feedback_store.get(chat_id, message_id).rating == "up"
            assert len(main.feedback_store.list_records()) == 1

    async def test_rate_limit_blocks_a_flood(self, client, monkeypatch):
        import main
        monkeypatch.setattr(main.rate_limiter, "_max", 3)
        async with client:
            chat_id, message_id = await self._one_answer(client)
            body = {"chat_id": chat_id, "message_id": message_id, "rating": "up"}
            statuses = [(await client.post("/feedback", json=body)).status_code for _ in range(5)]
            assert 429 in statuses


@pytest.mark.asyncio
class TestAdminEndpoints:
    async def test_stats_requires_a_token(self, client):
        async with client:
            assert (await client.get("/feedback/stats")).status_code == 403

    async def test_records_requires_a_token(self, client):
        async with client:
            assert (await client.get("/feedback/records")).status_code == 403

    async def test_wrong_token_is_403(self, client):
        async with client:
            resp = await client.get("/feedback/stats", headers={"X-Admin-Token": "wrong"})
            assert resp.status_code == 403

    async def test_unconfigured_token_disables_admin(self, client, monkeypatch):
        import main
        monkeypatch.setattr(main, "ADMIN_TOKEN", "")
        async with client:
            resp = await client.get("/feedback/stats", headers={"X-Admin-Token": "anything"})
            assert resp.status_code == 503

    async def test_stats_with_token(self, client):
        async with client:
            resp = await client.get("/feedback/stats", headers={"X-Admin-Token": ADMIN_TOKEN})
            assert resp.status_code == 200
            assert "total" in resp.json()

    async def test_records_filter_validation(self, client):
        async with client:
            headers = {"X-Admin-Token": ADMIN_TOKEN}
            assert (await client.get("/feedback/records?rating=bogus", headers=headers)).status_code == 422
            assert (await client.get("/feedback/records?category=bogus", headers=headers)).status_code == 422
            assert (await client.get("/feedback/records?limit=0", headers=headers)).status_code == 422

    async def test_records_returns_flagged_items(self, client):
        async with client:
            resp = await client.post("/chat", json={"prompt": "What is zakat?"})
            chat_id, message_id = resp.json()["chat_id"], resp.json()["message_id"]
            await client.post("/feedback", json={
                "chat_id": chat_id, "message_id": message_id,
                "rating": "down", "categories": ["too_vague"],
            })
            resp = await client.get(
                "/feedback/records?rating=down", headers={"X-Admin-Token": ADMIN_TOKEN}
            )
            assert resp.status_code == 200
            records = resp.json()["records"]
            assert len(records) == 1
            assert records[0]["rating"] == "down"
