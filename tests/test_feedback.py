"""Tests for the feedback system.

All tests run fully offline against the SQLite fallback store — no Gemini
API key, no Redis, no network required.  The FastAPI app is loaded with a
mocked Gemini model so CI stays green without real credentials.

Test coverage:
  - feedback.py: SQLiteFeedbackStore upsert/get, idempotent overwrite, stats,
    category validation, comment-length enforcement, rate limiting
  - main.py endpoints: /feedback (happy path, bad category, oversized comment,
    unknown message_id, session-gone fallback), /feedback/stats,
    /feedback/records (admin token required, filters work)
  - scripts/export_eval_candidates.py: deduplication, needs_review flag,
    no fabricated expected_answer, format correctness
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
import types
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch google.generativeai BEFORE importing main so CI works without a key
# ---------------------------------------------------------------------------

def _make_mock_genai():
    """Return a minimal mock of the google.generativeai module."""
    mock = types.ModuleType("google.generativeai")

    class _Part:
        def __init__(self, text):
            self.text = text

    class _Message:
        def __init__(self, role, text):
            self.role = role
            self.parts = [_Part(text)]

    class _Response:
        def __init__(self, text):
            self.text = text

    class _ChatSession:
        def __init__(self):
            self.history = []

        def send_message(self, prompt, generation_config=None):
            self.history.append(_Message("user", prompt))
            answer = f"Mock answer to: {prompt[:40]}"
            self.history.append(_Message("model", answer))
            return _Response(answer)

    class _Model:
        def __init__(self, *args, **kwargs):
            pass

        def start_chat(self, history=None):
            return _ChatSession()

    class _GenerativeModel:
        def __new__(cls, *args, **kwargs):
            return _Model()

    mock.configure = MagicMock()
    mock.GenerativeModel = _GenerativeModel
    return mock


# Install mock before any project import
sys.modules["google.generativeai"] = _make_mock_genai()
sys.modules["google"] = types.ModuleType("google")
sys.modules["google"].generativeai = sys.modules["google.generativeai"]

# Ensure stellar_sdk mock so stellar.py imports work
_stellar_sdk = types.ModuleType("stellar_sdk")
_stellar_sdk.Server = MagicMock()
_stellar_sdk.exceptions = types.SimpleNamespace(NotFoundError=Exception)
_stellar_sdk.strkey = types.SimpleNamespace(StrKey=MagicMock())
sys.modules.setdefault("stellar_sdk", _stellar_sdk)
sys.modules.setdefault("stellar_sdk.exceptions", _stellar_sdk.exceptions)
sys.modules.setdefault("stellar_sdk.strkey", _stellar_sdk.strkey)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path):
    """Return path to a fresh temporary SQLite DB."""
    return str(tmp_path / "test_feedback.db")


@pytest.fixture()
def sqlite_store(tmp_db):
    from feedback import SQLiteFeedbackStore
    return SQLiteFeedbackStore(db_path=tmp_db)


@pytest.fixture()
def sample_record():
    from feedback import FeedbackRecord
    return FeedbackRecord(
        feedback_id=str(uuid.uuid4()),
        chat_id="chat-001",
        message_id="msg-001",
        rating="down",
        categories=["incorrect_information", "too_vague"],
        comment="The hadith citation was wrong.",
        prompt="What does Islam say about patience?",
        answer="Islam says be patient sometimes.",
        model_name="gemini-2.5-flash-preview-05-20",
        generation_config={"temperature": 0.7},
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture()
def app_client(tmp_db, monkeypatch):
    """Return a TestClient for the FastAPI app with a patched feedback store."""
    # Point feedback store at the temp DB before app loads
    monkeypatch.setenv("FEEDBACK_DB_PATH", tmp_db)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")

    # Force re-import so the store picks up the new env var
    import feedback as fb_module
    from feedback import SQLiteFeedbackStore
    fb_module.store = SQLiteFeedbackStore(db_path=tmp_db)

    import main as main_module
    importlib.reload(main_module)
    main_module.feedback_store = fb_module.store

    from fastapi.testclient import TestClient
    return TestClient(main_module.app)


# ---------------------------------------------------------------------------
# feedback.py — SQLiteFeedbackStore unit tests
# ---------------------------------------------------------------------------

class TestSQLiteFeedbackStore:

    def test_upsert_and_get(self, sqlite_store, sample_record):
        sqlite_store.upsert(sample_record)
        retrieved = sqlite_store.get(sample_record.chat_id, sample_record.message_id)
        assert retrieved is not None
        assert retrieved.rating == "down"
        assert "incorrect_information" in retrieved.categories
        assert retrieved.model_name == "gemini-2.5-flash-preview-05-20"

    def test_idempotent_overwrite(self, sqlite_store, sample_record):
        sqlite_store.upsert(sample_record)
        # Overwrite with updated rating
        sample_record.rating = "up"
        sample_record.categories = []
        sqlite_store.upsert(sample_record)

        retrieved = sqlite_store.get(sample_record.chat_id, sample_record.message_id)
        assert retrieved.rating == "up"
        assert retrieved.categories == []

        # Only one record should exist
        records = sqlite_store.list_records()
        assert len(records) == 1

    def test_list_filter_by_rating(self, sqlite_store, sample_record):
        sqlite_store.upsert(sample_record)

        up_rec = FeedbackRecord_from(sample_record, message_id="msg-002", rating="up", categories=[])
        sqlite_store.upsert(up_rec)

        downs = sqlite_store.list_records(rating="down")
        ups = sqlite_store.list_records(rating="up")
        assert len(downs) == 1
        assert len(ups) == 1

    def test_list_filter_by_category(self, sqlite_store, sample_record):
        sqlite_store.upsert(sample_record)
        results = sqlite_store.list_records(category="incorrect_information")
        assert len(results) == 1
        results_none = sqlite_store.list_records(category="too_long")
        assert len(results_none) == 0

    def test_stats_aggregation(self, sqlite_store, sample_record):
        sqlite_store.upsert(sample_record)
        up_rec = FeedbackRecord_from(
            sample_record, message_id="msg-002", rating="up", categories=["too_long"]
        )
        sqlite_store.upsert(up_rec)

        stats = sqlite_store.stats()
        assert stats["total"] == 2
        assert stats["down"] == 1
        assert stats["up"] == 1
        assert stats["up_ratio"] == 0.5
        assert "incorrect_information" in stats["by_category"]

    def test_generation_config_roundtrip(self, sqlite_store, sample_record):
        sqlite_store.upsert(sample_record)
        r = sqlite_store.get(sample_record.chat_id, sample_record.message_id)
        assert r.generation_config == {"temperature": 0.7}


def FeedbackRecord_from(base, **overrides):
    """Helper: clone a FeedbackRecord with field overrides."""
    from feedback import FeedbackRecord
    d = base.to_dict()
    d.update(overrides)
    d["feedback_id"] = str(uuid.uuid4())
    return FeedbackRecord.from_dict(d)


# ---------------------------------------------------------------------------
# feedback.py — RateLimiter unit tests
# ---------------------------------------------------------------------------

class TestRateLimiter:

    def test_allows_within_limit(self):
        from feedback import RateLimiter
        rl = RateLimiter(max_calls=5, window_seconds=60)
        for _ in range(5):
            assert rl.is_allowed("127.0.0.1") is True

    def test_blocks_when_limit_exceeded(self):
        from feedback import RateLimiter
        rl = RateLimiter(max_calls=3, window_seconds=60)
        for _ in range(3):
            rl.is_allowed("10.0.0.1")
        assert rl.is_allowed("10.0.0.1") is False

    def test_different_ips_are_independent(self):
        from feedback import RateLimiter
        rl = RateLimiter(max_calls=2, window_seconds=60)
        rl.is_allowed("1.1.1.1")
        rl.is_allowed("1.1.1.1")
        # IP 1 is blocked, IP 2 should still pass
        assert rl.is_allowed("1.1.1.1") is False
        assert rl.is_allowed("2.2.2.2") is True

    def test_window_expiry_allows_again(self):
        from feedback import RateLimiter
        rl = RateLimiter(max_calls=2, window_seconds=0.1)
        rl.is_allowed("3.3.3.3")
        rl.is_allowed("3.3.3.3")
        assert rl.is_allowed("3.3.3.3") is False
        time.sleep(0.15)
        # Window has expired — should be allowed again
        assert rl.is_allowed("3.3.3.3") is True


# ---------------------------------------------------------------------------
# main.py — /chat endpoint tests (message_id propagation)
# ---------------------------------------------------------------------------

class TestChatEndpoint:

    def test_chat_response_has_message_id(self, app_client):
        resp = app_client.post("/chat", json={"prompt": "What is Tawakkul?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "message_id" in data
        assert isinstance(data["message_id"], str)
        assert len(data["message_id"]) == 36  # UUID

    def test_existing_fields_unchanged(self, app_client):
        resp = app_client.post("/chat", json={"prompt": "What is Sabr?"})
        data = resp.json()
        assert "response" in data
        assert "chat_id" in data
        assert "history" in data

    def test_history_model_turns_have_message_id(self, app_client):
        resp1 = app_client.post("/chat", json={"prompt": "Turn 1"})
        chat_id = resp1.json()["chat_id"]
        resp2 = app_client.post("/chat", json={"prompt": "Turn 2", "chat_id": chat_id})
        history = resp2.json()["history"]
        model_turns = [m for m in history if m["role"] == "model"]
        assert all(m["message_id"] is not None for m in model_turns)

    def test_user_turns_have_no_message_id(self, app_client):
        resp = app_client.post("/chat", json={"prompt": "Hello"})
        chat_id = resp.json()["chat_id"]
        history = resp.json()["history"]
        user_turns = [m for m in history if m["role"] == "user"]
        # message_id on user turns should be None or absent
        for turn in user_turns:
            assert turn.get("message_id") is None


# ---------------------------------------------------------------------------
# main.py — /feedback endpoint tests
# ---------------------------------------------------------------------------

class TestFeedbackEndpoint:

    def _do_chat(self, client, prompt="Test prompt"):
        resp = client.post("/chat", json={"prompt": prompt})
        assert resp.status_code == 200
        return resp.json()

    def test_happy_path_up_rating(self, app_client):
        chat = self._do_chat(app_client)
        resp = app_client.post("/feedback", json={
            "chat_id": chat["chat_id"],
            "message_id": chat["message_id"],
            "rating": "up",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert "feedback_id" in resp.json()

    def test_happy_path_down_with_categories(self, app_client):
        chat = self._do_chat(app_client)
        resp = app_client.post("/feedback", json={
            "chat_id": chat["chat_id"],
            "message_id": chat["message_id"],
            "rating": "down",
            "categories": ["incorrect_information", "too_vague"],
            "comment": "The answer was not sourced properly.",
        })
        assert resp.status_code == 200

    def test_invalid_rating_rejected(self, app_client):
        resp = app_client.post("/feedback", json={
            "chat_id": "x",
            "message_id": "y",
            "rating": "meh",
        })
        assert resp.status_code == 422

    def test_invalid_category_rejected(self, app_client):
        chat = self._do_chat(app_client)
        resp = app_client.post("/feedback", json={
            "chat_id": chat["chat_id"],
            "message_id": chat["message_id"],
            "rating": "down",
            "categories": ["not_a_real_category"],
        })
        assert resp.status_code == 422

    def test_oversized_comment_rejected(self, app_client):
        from feedback import COMMENT_MAX_CHARS
        chat = self._do_chat(app_client)
        resp = app_client.post("/feedback", json={
            "chat_id": chat["chat_id"],
            "message_id": chat["message_id"],
            "rating": "down",
            "comment": "x" * (COMMENT_MAX_CHARS + 1),
        })
        assert resp.status_code == 422

    def test_unknown_message_id_session_gone_without_fallback(self, app_client):
        """When session is gone and client doesn't supply prompt/answer → 422."""
        resp = app_client.post("/feedback", json={
            "chat_id": "nonexistent-chat",
            "message_id": str(uuid.uuid4()),
            "rating": "down",
        })
        assert resp.status_code == 422

    def test_session_gone_with_client_supplied_text(self, app_client):
        """When session is gone but client supplies prompt/answer → accepted."""
        resp = app_client.post("/feedback", json={
            "chat_id": "nonexistent-chat",
            "message_id": str(uuid.uuid4()),
            "rating": "down",
            "prompt": "What is the ruling on music?",
            "answer": "Scholars differ on this topic.",
            "categories": ["one_sided_fiqh_answer"],
        })
        assert resp.status_code == 200

    def test_idempotent_resubmission(self, app_client):
        chat = self._do_chat(app_client)
        payload = {
            "chat_id": chat["chat_id"],
            "message_id": chat["message_id"],
            "rating": "down",
            "categories": ["too_vague"],
        }
        app_client.post("/feedback", json=payload)
        # Change rating on resubmit
        payload["rating"] = "up"
        payload["categories"] = []
        resp2 = app_client.post("/feedback", json=payload)
        assert resp2.status_code == 200

        # Only one record should exist
        from feedback import store as fb_store
        records = fb_store.list_records()
        matching = [r for r in records if r.message_id == chat["message_id"]]
        assert len(matching) == 1
        assert matching[0].rating == "up"

    def test_rate_limiting_blocks_flood(self, app_client, monkeypatch):
        from feedback import RateLimiter
        tight = RateLimiter(max_calls=2, window_seconds=60)
        monkeypatch.setattr("feedback.rate_limiter", tight)
        monkeypatch.setattr("main.rate_limiter", tight)

        chat = self._do_chat(app_client)
        payload = {
            "chat_id": chat["chat_id"],
            "message_id": chat["message_id"],
            "rating": "down",
            "prompt": "q",
            "answer": "a",
        }
        r1 = app_client.post("/feedback", json=payload, headers={"X-Forwarded-For": "9.9.9.9"})
        # Use new message_id for second request (idempotent key differs)
        payload["message_id"] = str(uuid.uuid4())
        r2 = app_client.post("/feedback", json=payload, headers={"X-Forwarded-For": "9.9.9.9"})
        payload["message_id"] = str(uuid.uuid4())
        r3 = app_client.post("/feedback", json=payload, headers={"X-Forwarded-For": "9.9.9.9"})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 429

    def test_feedback_record_captures_model_and_config(self, app_client):
        chat = self._do_chat(app_client)
        app_client.post("/feedback", json={
            "chat_id": chat["chat_id"],
            "message_id": chat["message_id"],
            "rating": "down",
        })
        from feedback import store as fb_store
        records = fb_store.list_records(rating="down")
        r = next((x for x in records if x.message_id == chat["message_id"]), None)
        assert r is not None
        assert r.model_name is not None
        assert r.generation_config is not None


# ---------------------------------------------------------------------------
# main.py — admin endpoints (/feedback/stats, /feedback/records)
# ---------------------------------------------------------------------------

class TestAdminEndpoints:

    ADMIN = {"X-Admin-Token": "test-admin-token"}
    WRONG = {"X-Admin-Token": "wrong"}

    def _seed(self, store, n_down=3, n_up=1):
        from feedback import FeedbackRecord
        from datetime import datetime, timezone
        for i in range(n_down):
            store.upsert(FeedbackRecord(
                feedback_id=str(uuid.uuid4()),
                chat_id=f"chat-{i}",
                message_id=f"msg-{i}",
                rating="down",
                categories=["too_vague"],
                prompt=f"question {i}",
                answer=f"answer {i}",
                model_name="gemini-2.5-flash-preview-05-20",
                generation_config={"temperature": 0.7},
                created_at=datetime.now(timezone.utc).isoformat(),
            ))
        for i in range(n_up):
            store.upsert(FeedbackRecord(
                feedback_id=str(uuid.uuid4()),
                chat_id=f"chat-up-{i}",
                message_id=f"msg-up-{i}",
                rating="up",
                categories=[],
                prompt="good question",
                answer="good answer",
                model_name="gemini-2.5-flash-preview-05-20",
                generation_config={"temperature": 0.7},
                created_at=datetime.now(timezone.utc).isoformat(),
            ))

    def test_stats_requires_admin_token(self, app_client):
        assert app_client.get("/feedback/stats").status_code == 403

    def test_stats_wrong_token(self, app_client):
        assert app_client.get("/feedback/stats", headers=self.WRONG).status_code == 403

    def test_stats_correct_aggregation(self, app_client):
        from feedback import store as fb_store
        self._seed(fb_store)
        resp = app_client.get("/feedback/stats", headers=self.ADMIN)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 4
        assert data["down"] == 3
        assert data["up"] == 1
        assert data["up_ratio"] == 0.25

    def test_records_requires_admin_token(self, app_client):
        assert app_client.get("/feedback/records").status_code == 403

    def test_records_filter_by_rating(self, app_client):
        from feedback import store as fb_store
        self._seed(fb_store)
        resp = app_client.get("/feedback/records?rating=down", headers=self.ADMIN)
        assert resp.status_code == 200
        records = resp.json()["records"]
        assert all(r["rating"] == "down" for r in records)
        assert len(records) == 3

    def test_records_filter_by_category(self, app_client):
        from feedback import store as fb_store
        self._seed(fb_store)
        resp = app_client.get("/feedback/records?category=too_vague", headers=self.ADMIN)
        assert resp.status_code == 200
        records = resp.json()["records"]
        assert len(records) == 3

    def test_records_invalid_rating_rejected(self, app_client):
        resp = app_client.get("/feedback/records?rating=meh", headers=self.ADMIN)
        assert resp.status_code == 422

    def test_records_invalid_category_rejected(self, app_client):
        resp = app_client.get("/feedback/records?category=fake", headers=self.ADMIN)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# scripts/export_eval_candidates.py — export script tests
# ---------------------------------------------------------------------------

class TestExportEvalCandidates:

    def _seed_db(self, db_path: str):
        from feedback import SQLiteFeedbackStore, FeedbackRecord
        from datetime import datetime, timezone
        store = SQLiteFeedbackStore(db_path=db_path)
        records = [
            FeedbackRecord(
                feedback_id=str(uuid.uuid4()),
                chat_id="c1", message_id="m1", rating="down",
                categories=["incorrect_information"],
                prompt="What is the ruling on music?",
                answer="Music is always halal.",
                model_name="gemini-2.5-flash-preview-05-20",
                generation_config={"temperature": 0.7},
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
            FeedbackRecord(
                feedback_id=str(uuid.uuid4()),
                chat_id="c2", message_id="m2", rating="down",
                categories=["too_vague"],
                prompt="Explain the pillars of Islam.",
                answer="There are some pillars.",
                model_name="gemini-2.5-flash-preview-05-20",
                generation_config={"temperature": 0.7},
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
            # Near-duplicate of first prompt
            FeedbackRecord(
                feedback_id=str(uuid.uuid4()),
                chat_id="c3", message_id="m3", rating="down",
                categories=["incorrect_information"],
                prompt="what is the ruling on music?",  # normalises the same
                answer="Music is forbidden.",
                model_name="gemini-2.5-flash-preview-05-20",
                generation_config={"temperature": 0.7},
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
            # Up-rated — must NOT appear in export
            FeedbackRecord(
                feedback_id=str(uuid.uuid4()),
                chat_id="c4", message_id="m4", rating="up",
                categories=[],
                prompt="What is Zakat?",
                answer="Zakat is the third pillar.",
                model_name="gemini-2.5-flash-preview-05-20",
                generation_config={"temperature": 0.7},
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
        ]
        for r in records:
            store.upsert(r)
        return store

    def test_only_down_rated_exported(self, tmp_db):
        self._seed_db(tmp_db)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.export_eval_candidates import export
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            out_path = f.name
        export(db_path=tmp_db, output_path=out_path, min_categories=0, limit=200)
        with open(out_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        ratings = {c.get("rating") for c in lines}
        # No 'rating' field in export — but no up-rated records should appear
        # (verify via question content instead)
        questions = [c["question"] for c in lines]
        assert "What is Zakat?" not in questions

    def test_near_duplicate_deduplication(self, tmp_db):
        self._seed_db(tmp_db)
        from scripts.export_eval_candidates import export
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            out_path = f.name
        count = export(db_path=tmp_db, output_path=out_path, min_categories=0, limit=200)
        # 3 down-rated but 2 are near-duplicates → 2 unique candidates
        assert count == 2

    def test_needs_review_always_true(self, tmp_db):
        self._seed_db(tmp_db)
        from scripts.export_eval_candidates import export
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            out_path = f.name
        export(db_path=tmp_db, output_path=out_path, min_categories=0, limit=200)
        with open(out_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert all(c["needs_review"] is True for c in lines)

    def test_no_expected_answer_field(self, tmp_db):
        """The export must never fabricate expected answers."""
        self._seed_db(tmp_db)
        from scripts.export_eval_candidates import export
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            out_path = f.name
        export(db_path=tmp_db, output_path=out_path, min_categories=0, limit=200)
        with open(out_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for c in lines:
            assert "expected_answer" not in c

    def test_output_format_fields(self, tmp_db):
        self._seed_db(tmp_db)
        from scripts.export_eval_candidates import export
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            out_path = f.name
        export(db_path=tmp_db, output_path=out_path, min_categories=0, limit=200)
        with open(out_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        required = {"question", "category", "categories", "needs_review", "source",
                    "feedback_id", "model_name", "answer_draft"}
        for c in lines:
            assert required.issubset(set(c.keys()))

    def test_valid_jsonl_output(self, tmp_db):
        self._seed_db(tmp_db)
        from scripts.export_eval_candidates import export
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            out_path = f.name
        export(db_path=tmp_db, output_path=out_path, min_categories=0, limit=200)
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    json.loads(line)  # must not raise
