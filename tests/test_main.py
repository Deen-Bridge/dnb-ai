"""Smoke tests for the routes declared in ``main.py``.

These exercise the HTTP surface — status codes, request validation, the
rate-limit handler, and the streaming event sequence — with the model stubbed
out. Nothing here touches the network or a real Gemini key.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main

FIXED_ANSWER = "Patience is encouraged in the Quran and in the Prophetic tradition."


class _FakeChatSession:
    """Minimal stand-in for a Gemini chat session (sync and stream calls)."""

    def __init__(self) -> None:
        self.history: list[SimpleNamespace] = []

    async def send_message_async(self, message: str, **kwargs: object) -> object:
        self.history.append(SimpleNamespace(role="user", parts=[SimpleNamespace(text=message)]))
        self.history.append(SimpleNamespace(role="model", parts=[SimpleNamespace(text=FIXED_ANSWER)]))
        if kwargs.get("stream"):
            return self._stream()
        return SimpleNamespace(text=FIXED_ANSWER, candidates=[], prompt_feedback=None)

    async def _stream(self):
        for piece in ("Patience ", "is encouraged ", "in the Quran."):
            yield SimpleNamespace(text=piece)


class _FakeModel:
    def start_chat(self, history=None) -> _FakeChatSession:
        session = _FakeChatSession()
        session.history = list(history or [])
        return session


class _FakeFeedbackStore:
    def __init__(self) -> None:
        self.records: list[object] = []

    def upsert(self, record: object) -> None:
        self.records.append(record)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "0")
    monkeypatch.setattr(main, "get_model", lambda: _FakeModel())
    monkeypatch.setattr(main, "feedback_store", _FakeFeedbackStore())
    main.active_chats.clear()
    main.limiter.reset()
    yield TestClient(main.app)
    main.active_chats.clear()
    main.limiter.reset()


def test_health_reports_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["checks"]["gemini_api_key_configured"] == "ok"


def test_health_deep_reports_reachability_check(client):
    response = client.get("/health?deep=true")

    assert response.status_code == 200
    assert response.json()["checks"]["gemini_api_reachable"] == "skipped"


def test_adhkar_recommend_returns_matches(client):
    response = client.post("/adhkar/recommend", json={"category": "morning"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert "authenticated supplication" in payload["message"]


def test_adhkar_recommend_reports_no_match(client):
    response = client.post("/adhkar/recommend", json={"category": "travel", "query": "nothing here"})

    assert response.status_code == 200
    assert response.json()["matches"] == []


def test_chat_returns_a_chat_id_on_success(client):
    response = client.post("/chat", json={"prompt": "How can I be more patient?", "chat_id": "smoke-chat"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["chat_id"] == "smoke-chat"
    assert isinstance(payload["response"], str) and payload["response"]
    assert response.headers["X-Trace-Id"]


def test_chat_rejects_a_body_without_a_prompt(client):
    response = client.post("/chat", json={"chat_id": "missing-prompt"})

    assert response.status_code == 422
    payload = response.json()
    assert "hint" in payload
    assert "Validation failed" in payload["hint"]


def test_chat_stream_emits_metadata_content_and_done_events(client):
    with client.stream("POST", "/chat/stream", json={"prompt": "How can I be more patient?"}) as response:
        assert response.status_code == 200
        events = [json.loads(line[len("data: ") :]) for line in response.iter_lines() if line.startswith("data: ")]

    kinds = [event["type"] for event in events]
    assert kinds[0] == "metadata"
    assert "content" in kinds
    assert kinds[-1] == "done"
    assert events[-1]["text"]


def test_evidence_verify_returns_a_structured_audit(client):
    response = client.post(
        "/evidence/verify",
        json={"text": "Patience is mentioned in the Quran. Quran 2:153 advises patience."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "claims" in payload
    assert "overall_score" in payload
    assert "audit_trail" in payload


def test_evidence_verify_rejects_empty_text(client):
    response = client.post("/evidence/verify", json={"text": ""})

    assert response.status_code == 422
    assert "patience" not in response.text


def test_feedback_returns_a_feedback_id(client):
    response = client.post(
        "/feedback",
        json={
            "chat_id": "chat-1",
            "message_id": "msg-1",
            "rating": "up",
            "prompt": "How can I be more patient?",
            "answer": FIXED_ANSWER,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["feedback_id"]


def test_feedback_rejects_an_unknown_rating(client):
    response = client.post("/feedback", json={"chat_id": "chat-1", "message_id": "msg-1", "rating": "sideways"})

    assert response.status_code == 422
    assert "rating must be 'up' or 'down'" in response.text


def test_feedback_rate_limit_returns_429_with_a_hint(client, monkeypatch):
    monkeypatch.setattr(main.rate_limiter, "is_allowed", lambda _ip: False)

    response = client.post(
        "/feedback",
        json={
            "chat_id": "chat-1",
            "message_id": "msg-1",
            "rating": "up",
            "prompt": "How can I be more patient?",
            "answer": FIXED_ANSWER,
        },
    )

    assert response.status_code == 429
    payload = response.json()
    assert "Too many feedback submissions" in payload["detail"]
    assert "rate-limited" in payload["hint"]


def test_chat_rate_limit_returns_429_with_retry_after(client):
    headers = {"X-Forwarded-For": "203.0.113.50"}

    statuses = [
        client.post("/chat", json={"prompt": "hello", "chat_id": f"rate-{index}"}, headers=headers).status_code
        for index in range(61)
    ]

    assert statuses[-1] == 429
    last = client.post("/chat", json={"prompt": "hello", "chat_id": "rate-final"}, headers=headers)
    assert last.status_code == 429
    assert last.headers["Retry-After"]
    assert "Rate limit exceeded" in last.json()["detail"]
    assert "Too many requests" in last.json()["hint"]
