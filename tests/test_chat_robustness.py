import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.api_core.exceptions import (
    DeadlineExceeded,
    InvalidArgument,
    ResourceExhausted,
    ServiceUnavailable,
)
from httpx import ASGITransport, AsyncClient

import main
from main import app


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setattr(main, "GEMINI_API_KEY", "test-key")
    main.genai.configure(api_key="test-key")
    monkeypatch.setattr(main, "sessions", {})
    monkeypatch.setattr(main, "active_chats", {})
    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "false")
    monkeypatch.setattr(main, "SEMANTIC_CACHE_ENABLED", False)
    monkeypatch.setattr(main, "zakat_retriever", AsyncMock(return_value=None))
    monkeypatch.setattr(main, "purchase_retriever", AsyncMock(return_value=None))
    monkeypatch.setattr(main, "tafsir_retriever", AsyncMock(return_value=None))
    monkeypatch.setattr(main, "enqueue_for_review", AsyncMock())


@pytest.mark.asyncio
async def test_concurrent_chat_requests_do_not_block_event_loop(monkeypatch):
    """Verify that two concurrent /chat requests execute in parallel on the event loop."""
    mock_session = MagicMock()

    async def slow_send_message_async(message, **kwargs):
        await asyncio.sleep(0.4)
        mock_resp = MagicMock()
        mock_resp.text = f"Response to {message}"
        mock_resp.candidates = [MagicMock(finish_reason="STOP")]
        mock_resp.prompt_feedback = None  # Avoid MagicMock auto-proxy triggering prompt_feedback check
        return mock_resp

    mock_session.send_message_async = slow_send_message_async
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        start_time = time.monotonic()
        req1 = client.post("/chat", json={"prompt": "Hello 1", "chat_id": str(uuid.uuid4())})
        req2 = client.post("/chat", json={"prompt": "Hello 2", "chat_id": str(uuid.uuid4())})

        res1, res2 = await asyncio.gather(req1, req2)
        elapsed = time.monotonic() - start_time

    assert res1.status_code == 200
    assert res2.status_code == 200
    # Two 0.4s calls in parallel should finish in ~0.4s (well under 0.7s total)
    assert elapsed < 0.7


@pytest.mark.asyncio
async def test_quota_exceeded_returns_429(monkeypatch):
    """ResourceExhausted should map to HTTP 429 with generic detail."""
    mock_session = MagicMock()
    mock_session.send_message_async = AsyncMock(side_effect=ResourceExhausted("429 Quota exceeded for project 12345"))
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/chat", json={"prompt": "Hi"})

    assert res.status_code == 429
    data = res.json()
    assert data["detail"] == "Rate limit exceeded. Please try again later."
    assert "Quota exceeded" not in data["detail"]


@pytest.mark.asyncio
async def test_timeout_returns_504(monkeypatch):
    """DeadlineExceeded should map to HTTP 504 with generic detail."""
    mock_session = MagicMock()
    mock_session.send_message_async = AsyncMock(side_effect=DeadlineExceeded("Deadline exceeded during RPC"))
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/chat", json={"prompt": "Hi"})

    assert res.status_code == 504
    data = res.json()
    assert data["detail"] == "AI service timed out."
    assert "Deadline exceeded" not in data["detail"]


@pytest.mark.asyncio
async def test_invalid_argument_returns_400(monkeypatch):
    """InvalidArgument should map to HTTP 400 with generic detail."""
    mock_session = MagicMock()
    mock_session.send_message_async = AsyncMock(side_effect=InvalidArgument("Invalid field value in payload"))
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/chat", json={"prompt": "Hi"})

    assert res.status_code == 400
    data = res.json()
    assert data["detail"] == "Invalid request parameters."
    assert "Invalid field value" not in data["detail"]


@pytest.mark.asyncio
async def test_generic_exception_returns_500_without_leaking_details(monkeypatch):
    """Unexpected exception should map to HTTP 500 without leaking raw exception text or emojis."""
    mock_session = MagicMock()
    mock_session.send_message_async = AsyncMock(
        side_effect=RuntimeError("Secret internal database connection string: db://pass@localhost")
    )
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/chat", json={"prompt": "Hi"})

    assert res.status_code == 500
    data = res.json()
    assert data["detail"] == "AI service error"
    assert "db://" not in data["detail"]
    assert "❌" not in data["detail"]


@pytest.mark.xfail(
    reason="Safety-blocked responses currently return 500. Graceful safety handling is a separate enhancement.",
    strict=False,
)
@pytest.mark.asyncio
async def test_safety_blocked_response_returns_graceful_200(monkeypatch):
    """Safety-blocked response raising ValueError on response.text returns 200 with respectful prompt."""
    mock_session = MagicMock()
    fake_response = MagicMock()
    # Accessing .text on safety block raises ValueError in google-generativeai
    type(fake_response).text = property(
        fget=MagicMock(
            side_effect=ValueError("Quick accessor for response.text is invalid. The Response has no candidate...")
        )
    )
    fake_response.candidates = [MagicMock(finish_reason="SAFETY")]

    mock_session.send_message_async = AsyncMock(return_value=fake_response)
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/chat", json={"prompt": "Harmful prompt"})

    assert res.status_code == 200
    data = res.json()
    assert data["text"] == "I cannot fulfill this request due to safety guidelines."
    assert data["citations_verified"] is True


@pytest.mark.asyncio
async def test_retry_on_transient_failure_and_history_integrity(monkeypatch):
    """Transient ServiceUnavailable error is retried and chat history integrity is maintained."""
    mock_session = MagicMock()
    fake_success = MagicMock()
    fake_success.text = "Hello back"
    fake_success.candidates = [MagicMock(finish_reason="STOP")]
    fake_success.prompt_feedback = None  # Avoid MagicMock auto-proxy triggering prompt_feedback check

    call_count = 0

    async def side_effect_func(msg, **kwargs):
        nonlocal call_count
        call_count += 1
        # Simulate SDK appending user prompt to history before request
        mock_session.history.append({"role": "user", "text": msg})
        if call_count == 1:
            raise ServiceUnavailable("503 Service Unavailable")
        mock_session.history.append({"role": "model", "text": "Hello back"})
        return fake_success

    mock_session.send_message_async = AsyncMock(side_effect=side_effect_func)
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/chat", json={"prompt": "Hi"})

    assert res.status_code == 200
    assert call_count == 2
    # Verify history integrity: exactly 1 user turn and 1 model turn (no duplicate user turn from failed attempt)
    assert len(mock_session.history) == 2
    assert mock_session.history[0]["role"] == "user"
    assert mock_session.history[1]["role"] == "model"
