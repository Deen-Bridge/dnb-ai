"""Tests verifying that API error responses include actionable guidance, hints, and examples."""

from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import (
    DeadlineExceeded,
    InvalidArgument,
    ResourceExhausted,
    ServiceUnavailable,
)
from httpx import ASGITransport, AsyncClient
from stellar_sdk.exceptions import NotFoundError

import main
import review
import stellar
from main import app
from review_store import ReviewItem, Verdict, get_review_store


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ===========================================================================
# 400 Bad Request — Validation with Actionable Hints
# ===========================================================================


@pytest.mark.asyncio
async def test_tafsir_invalid_reference_includes_hint_and_examples(client):
    async with client:
        resp = await client.post("/tafsir", json={"reference": "999:1"})
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data
        assert "hint" in data
        hint = data["hint"]
        assert "surah:ayah" in hint
        assert "2:255" in hint
        assert "103:1-3" in hint
        assert "114" in hint


@pytest.mark.asyncio
async def test_tafsir_unparseable_reference_includes_hint(client):
    async with client:
        resp = await client.post("/tafsir", json={"reference": "invalid_ref"})
        assert resp.status_code == 400
        data = resp.json()
        assert "hint" in data
        assert "surah:ayah" in data["hint"]


@pytest.mark.asyncio
async def test_stellar_invalid_public_key_includes_hint(client):
    async with client:
        resp = await client.post("/zakat", json={"public_key": "not-a-valid-key"})
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data
        assert "hint" in data
        assert "56-character" in data["hint"]
        assert "GBBD" in data["hint"]
        assert "secret keys" in data["hint"].lower()


@pytest.mark.asyncio
async def test_chat_invalid_argument_from_gemini_includes_hint(client, monkeypatch):
    monkeypatch.setattr(main, "AUTH_DISABLED", True)

    def mock_send(*args, **kwargs):
        raise InvalidArgument("Prompt exceeds token limit")

    monkeypatch.setattr(main, "send_message_with_retry", mock_send)

    async with client:
        resp = await client.post("/chat", json={"prompt": "test"})
        assert resp.status_code == 400
        data = resp.json()
        assert "hint" in data
        assert "characters" in data["hint"]


# ===========================================================================
# 401 & 403 Authentication / Authorization Errors
# ===========================================================================


@pytest.mark.asyncio
async def test_missing_api_key_includes_hint(monkeypatch):
    from errors import APIException

    monkeypatch.setattr(main, "AUTH_DISABLED", False)
    monkeypatch.setattr(main, "SERVICE_API_KEY", "secret-test-key")

    mock_request = MagicMock()
    with pytest.raises(APIException) as exc:
        await main.verify_api_key(mock_request, api_key="wrong-key")

    assert exc.value.status_code == 401
    assert exc.value.detail == "Missing or invalid X-API-Key"
    assert exc.value.hint is not None
    assert "X-API-Key" in exc.value.hint
    assert "AUTH_DISABLED=true" in exc.value.hint


@pytest.mark.asyncio
async def test_reviewer_missing_token_includes_hint(client, monkeypatch):
    monkeypatch.setattr(review, "SCHOLAR_REVIEW_TOKEN", "configured-token")

    async with client:
        resp = await client.get("/review/pending")
        assert resp.status_code == 401
        data = resp.json()
        assert "hint" in data
        assert "X-Review-Token" in data["hint"]


@pytest.mark.asyncio
async def test_invalid_admin_token_includes_hint(client, monkeypatch):
    monkeypatch.setattr(main, "ADMIN_TOKEN", "valid-admin-token")

    async with client:
        resp = await client.get("/feedback/stats", headers={"X-Admin-Token": "wrong-token"})
        assert resp.status_code == 403
        data = resp.json()
        assert "hint" in data
        assert "X-Admin-Token" in data["hint"]


# ===========================================================================
# 404 Not Found & 409 Conflict Errors
# ===========================================================================


@pytest.mark.asyncio
async def test_memory_not_found_includes_hint(client):
    async with client:
        resp = await client.get("/memory/nonexistent_user_12345")
        assert resp.status_code == 404
        data = resp.json()
        assert data["detail"] == "Memory not found"
        assert "hint" in data
        assert "user profile" in data["hint"].lower()


@pytest.mark.asyncio
async def test_stellar_account_not_found_includes_hint(client):
    valid_key = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
    server = MagicMock()
    server.accounts.return_value.account_id.return_value.call.side_effect = NotFoundError(MagicMock())

    with patch.object(stellar, "Server", return_value=server):
        async with client:
            resp = await client.post("/zakat", json={"public_key": valid_key})
            assert resp.status_code == 404
            data = resp.json()
            assert "hint" in data
            assert "funded" in data["hint"].lower()
            assert "Horizon" in data["hint"]


@pytest.mark.asyncio
async def test_review_item_not_found_includes_hint(client, monkeypatch):
    monkeypatch.setattr(review, "SCHOLAR_REVIEW_TOKEN", "test-token")
    headers = {"X-Review-Token": "test-token"}

    async with client:
        resp = await client.get("/review/nonexistent-item-id", headers=headers)
        assert resp.status_code == 404
        data = resp.json()
        assert "hint" in data
        assert "GET /review/pending" in data["hint"]


@pytest.mark.asyncio
async def test_review_item_already_decided_includes_hint(client, monkeypatch):
    monkeypatch.setattr(review, "SCHOLAR_REVIEW_TOKEN", "test-token")
    headers = {"X-Review-Token": "test-token"}
    store = get_review_store()

    item = ReviewItem(id="item-already-vetted", question="Q", answer="A", confidence=0.5, band="low")
    await store.add(item)
    await store.record_verdict("item-already-vetted", Verdict.APPROVE, reviewer="Scholar")

    async with client:
        resp = await client.post(
            "/review/item-already-vetted/verdict",
            headers=headers,
            json={"verdict": "reject"},
        )
        assert resp.status_code == 409
        data = resp.json()
        assert "hint" in data
        assert "GET /review/reviewed" in data["hint"]


# ===========================================================================
# 422 Unprocessable Entity — Validation Guidance & Formats
# ===========================================================================


@pytest.mark.asyncio
async def test_pydantic_request_validation_error_includes_hint(client):
    async with client:
        resp = await client.post("/tafsir", json={})
        assert resp.status_code == 422
        data = resp.json()
        assert "detail" in data
        assert "hint" in data
        assert "Validation failed" in data["hint"]
        assert "reference" in data["hint"]


@pytest.mark.asyncio
async def test_feedback_records_rating_validation_includes_hint(client, monkeypatch):
    monkeypatch.setattr(main, "ADMIN_TOKEN", "test-admin")
    headers = {"X-Admin-Token": "test-admin"}

    async with client:
        resp = await client.get("/feedback/records?rating=invalid", headers=headers)
        assert resp.status_code == 422
        data = resp.json()
        assert "hint" in data
        assert "'up' or 'down'" in data["hint"]


@pytest.mark.asyncio
async def test_feedback_records_category_validation_includes_hint(client, monkeypatch):
    monkeypatch.setattr(main, "ADMIN_TOKEN", "test-admin")
    headers = {"X-Admin-Token": "test-admin"}

    async with client:
        resp = await client.get("/feedback/records?category=unknown_category", headers=headers)
        assert resp.status_code == 422
        data = resp.json()
        assert "hint" in data
        assert "allowed taxonomy categories" in data["hint"]


@pytest.mark.asyncio
async def test_feedback_records_limit_validation_includes_hint(client, monkeypatch):
    monkeypatch.setattr(main, "ADMIN_TOKEN", "test-admin")
    headers = {"X-Admin-Token": "test-admin"}

    async with client:
        resp = await client.get("/feedback/records?limit=0", headers=headers)
        assert resp.status_code == 422
        data = resp.json()
        assert "hint" in data
        assert "between 1 and 500" in data["hint"]


@pytest.mark.asyncio
async def test_feedback_missing_snapshot_includes_payload_example_hint(client):
    async with client:
        resp = await client.post(
            "/feedback",
            json={"chat_id": "nonexistent", "message_id": "nonexistent", "rating": "up"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "hint" in data
        assert "'prompt'" in data["hint"]
        assert "'answer'" in data["hint"]


@pytest.mark.asyncio
async def test_study_lesson_text_length_exceeded_includes_hint(client):
    long_text = "x" * 20001
    async with client:
        resp = await client.post(
            "/study/generate",
            json={
                "source": {"type": "lesson_text", "lesson_text": long_text},
                "kind": "quiz",
            },
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "hint" in data
        assert "Validation failed" in data["hint"]
        assert "20000" in data["hint"]


# ===========================================================================
# 429 Rate Limits / Quotas & 5xx Upstream Guidance
# ===========================================================================


@pytest.mark.asyncio
async def test_chat_token_quota_exceeded_includes_retry_hint(client, monkeypatch):
    monkeypatch.setattr(main, "AUTH_DISABLED", True)
    tracker = main.token_quota_tracker
    monkeypatch.setattr(tracker, "is_allowed", lambda key, tokens: (False, 120))

    async with client:
        resp = await client.post("/chat", json={"prompt": "What is zakat?"})
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "120"
        data = resp.json()
        assert "hint" in data
        assert "120 seconds" in data["hint"]


@pytest.mark.asyncio
async def test_gemini_resource_exhausted_includes_retry_hint(client, monkeypatch):
    monkeypatch.setattr(main, "AUTH_DISABLED", True)

    def mock_send(*args, **kwargs):
        raise ResourceExhausted("Rate limit hit")

    monkeypatch.setattr(main, "send_message_with_retry", mock_send)

    async with client:
        resp = await client.post("/chat", json={"prompt": "What is zakat?"})
        assert resp.status_code == 429
        data = resp.json()
        assert "hint" in data
        assert "10-30 seconds" in data["hint"]


@pytest.mark.asyncio
async def test_study_generator_retry_exhaustion_502_includes_hint(client):
    # Pass an unanswerable prompt to study generator where fake model or mocked model fails
    with patch("study.GeminiGenerator.generate", side_effect=ValueError("Invalid JSON from LLM")):
        async with client:
            resp = await client.post(
                "/study/generate",
                json={"source": {"type": "topic", "topic": "Prayer"}, "kind": "quiz"},
            )
            assert resp.status_code == 502
            data = resp.json()
            assert "hint" in data
            assert "simplifying" in data["hint"].lower()
            assert "difficulty" in data["hint"].lower()


@pytest.mark.asyncio
async def test_admin_token_unconfigured_503_includes_hint(client, monkeypatch):
    monkeypatch.setattr(main, "ADMIN_TOKEN", "")

    async with client:
        resp = await client.get("/feedback/stats", headers={"X-Admin-Token": "anything"})
        assert resp.status_code == 503
        data = resp.json()
        assert "hint" in data
        assert "ADMIN_TOKEN" in data["hint"]


@pytest.mark.asyncio
async def test_reviewer_unconfigured_503_includes_hint(client, monkeypatch):
    monkeypatch.setattr(review, "SCHOLAR_REVIEW_TOKEN", "")

    async with client:
        resp = await client.get("/review/pending", headers={"X-Review-Token": "anything"})
        assert resp.status_code == 503
        data = resp.json()
        assert "hint" in data
        assert "SCHOLAR_REVIEW_TOKEN" in data["hint"]


@pytest.mark.asyncio
async def test_gemini_service_unavailable_503_includes_hint(client, monkeypatch):
    monkeypatch.setattr(main, "AUTH_DISABLED", True)

    def mock_send(*args, **kwargs):
        raise ServiceUnavailable("Service is down")

    monkeypatch.setattr(main, "send_message_with_retry", mock_send)

    async with client:
        resp = await client.post("/chat", json={"prompt": "What is zakat?"})
        assert resp.status_code == 503
        data = resp.json()
        assert "hint" in data
        assert "30-60 seconds" in data["hint"]


@pytest.mark.asyncio
async def test_gemini_timeout_504_includes_hint(client, monkeypatch):
    monkeypatch.setattr(main, "AUTH_DISABLED", True)

    def mock_send(*args, **kwargs):
        raise DeadlineExceeded("Timeout")

    monkeypatch.setattr(main, "send_message_with_retry", mock_send)

    async with client:
        resp = await client.post("/chat", json={"prompt": "What is zakat?"})
        assert resp.status_code == 504
        data = resp.json()
        assert "hint" in data
        assert "retry" in data["hint"].lower()


@pytest.mark.asyncio
async def test_health_unhealthy_503_includes_hint(client, monkeypatch):
    monkeypatch.setattr(main, "GEMINI_API_KEY", "")

    async with client:
        resp = await client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert "hint" in data
        assert "GEMINI_API_KEY" in data["hint"]
