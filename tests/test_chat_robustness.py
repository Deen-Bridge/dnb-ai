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
from semantic_cache import get_cache, get_chat_exact_cache, get_token_quota_tracker


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
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

    # Reset caches and quota tracker
    get_cache().clear()
    get_chat_exact_cache().clear()
    get_token_quota_tracker().reset()

    # Mock the get_model function to return a mock model
    mock_model = MagicMock()
    mock_session = MagicMock()
    mock_session.send_message_async = AsyncMock(
        return_value=MagicMock(text="Test response", candidates=[MagicMock(finish_reason="STOP")], prompt_feedback=None)
    )
    mock_session.history = []
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)


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


@pytest.mark.asyncio
async def test_oversize_prompt_rejected_pre_llm(monkeypatch):
    """Oversized prompts should be rejected with 422 before any LLM call."""
    from semantic_cache import CHAT_PROMPT_MAX_LENGTH

    mock_session = MagicMock()
    mock_session.send_message_async = AsyncMock(
        side_effect=AssertionError("LLM should not be called for oversized prompt")
    )
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a prompt that exceeds the max length
        oversize_prompt = "a" * (CHAT_PROMPT_MAX_LENGTH + 1)
        res = await client.post("/chat", json={"prompt": oversize_prompt})

    assert res.status_code == 422
    data = res.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_oversize_context_rejected_pre_llm(monkeypatch):
    """Oversized context should be rejected with 422 before any LLM call."""
    from semantic_cache import CHAT_CONTEXT_MAX_LENGTH

    mock_session = MagicMock()
    mock_session.send_message_async = AsyncMock(
        side_effect=AssertionError("LLM should not be called for oversized context")
    )
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a context that exceeds the max length
        oversize_context = "a" * (CHAT_CONTEXT_MAX_LENGTH + 1)
        res = await client.post("/chat", json={"prompt": "What is zakat?", "context": oversize_context})

    assert res.status_code == 422
    data = res.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_quota_exceeded_returns_429_with_retry_after(monkeypatch):
    """Token quota exceeded should return 429 with Retry-After header."""
    mock_session = MagicMock()
    mock_session.send_message_async = AsyncMock(
        side_effect=AssertionError("LLM should not be called when quota exceeded")
    )
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    # Set a very low quota for testing
    quota_tracker = get_token_quota_tracker()
    orig_quota = quota_tracker._quota
    quota_tracker._quota = 3

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First request should use up the quota
            await client.post("/chat", json={"prompt": "test"})

            # Second request should be rate limited
            res2 = await client.post("/chat", json={"prompt": "test"})

        assert res2.status_code == 429
        assert "Retry-After" in res2.headers
        retry_after = int(res2.headers["Retry-After"])
        assert retry_after > 0
    finally:
        quota_tracker._quota = orig_quota


@pytest.mark.asyncio
async def test_cache_hit_for_signed_in_user(monkeypatch):
    """A signed-in user should get cache hits from their scoped cache without LLM calls."""
    mock_session = MagicMock()
    call_count = 0

    async def send_message_async(message, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.text = f"Response {call_count}"
        mock_resp.candidates = [MagicMock(finish_reason="STOP")]
        mock_resp.prompt_feedback = None
        return mock_resp

    mock_session.send_message_async = send_message_async
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    # Enable cache for this test
    monkeypatch.setattr(main, "SEMANTIC_CACHE_ENABLED", True)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First request from user A - should call LLM
        res1 = await client.post("/chat", json={"prompt": "What is zakat?", "user_id": "user123"})
        assert res1.status_code == 200
        assert call_count == 1

        # Second identical request from user A - should hit cache
        res2 = await client.post("/chat", json={"prompt": "What is zakat?", "user_id": "user123"})
        assert res2.status_code == 200
        # LLM should not be called again due to cache hit
        assert call_count == 1


@pytest.mark.asyncio
async def test_cache_scope_isolation_between_users(monkeypatch):
    """User A's cached answer should not be served to user B."""
    mock_session = MagicMock()
    call_count = 0

    async def send_message_async(message, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.text = f"Response for user {call_count}"
        mock_resp.candidates = [MagicMock(finish_reason="STOP")]
        mock_resp.prompt_feedback = None
        return mock_resp

    mock_session.send_message_async = send_message_async
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    # Enable cache for this test
    monkeypatch.setattr(main, "SEMANTIC_CACHE_ENABLED", True)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Request from user A
        res1 = await client.post("/chat", json={"prompt": "What is zakat?", "user_id": "userA"})
        assert res1.status_code == 200
        assert call_count == 1

        # Same request from user B - should NOT hit user A's cache
        res2 = await client.post("/chat", json={"prompt": "What is zakat?", "user_id": "userB"})
        assert res2.status_code == 200
        # LLM should be called again due to scope isolation
        assert call_count == 2


@pytest.mark.asyncio
async def test_exact_cache_hit_before_semantic(monkeypatch):
    """Exact cache should be checked before semantic cache."""
    mock_session = MagicMock()
    call_count = 0

    async def send_message_async(message, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.text = f"Response {call_count}"
        mock_resp.candidates = [MagicMock(finish_reason="STOP")]
        mock_resp.prompt_feedback = None
        return mock_resp

    mock_session.send_message_async = send_message_async
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    monkeypatch.setattr(main, "get_model", lambda: mock_model)

    # Enable cache for this test
    monkeypatch.setattr(main, "SEMANTIC_CACHE_ENABLED", True)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First request
        res1 = await client.post("/chat", json={"prompt": "What is zakat?"})
        assert res1.status_code == 200
        assert call_count == 1
        assert res1.headers.get("X-Cache-Tier") == "miss"

        # Second identical request - should hit exact cache
        res2 = await client.post("/chat", json={"prompt": "What is zakat?"})
        assert res2.status_code == 200
        assert call_count == 1  # No additional LLM call
        assert res2.headers.get("X-Cache-Tier") == "exact"
