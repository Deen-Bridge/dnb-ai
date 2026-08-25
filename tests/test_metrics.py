"""Tests for Prometheus metrics endpoint and custom domain telemetry (#116).

Verifies standard HTTP instrumentation, custom metric exports (model calls,
latency, tokens, cache hits/misses, confidence scores, scholar queue depth),
token and IP allowlist protection, and content safety invariants.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import metrics
import telemetry
from main import app
from review_store import ReviewItem, get_review_store


@pytest.fixture
def client() -> TestClient:
    """Test client for FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def clean_metrics_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure clean metric auth and allowlist environment variables for tests."""
    monkeypatch.delenv("METRICS_TOKEN", raising=False)
    monkeypatch.delenv("METRICS_IP_ALLOWLIST", raising=False)


class TestPrometheusEndpoint:
    def test_metrics_endpoint_returns_prometheus_format(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        content = response.text
        assert "dnb_ai_model_calls_total" in content
        assert "dnb_ai_model_latency_seconds" in content
        assert "dnb_ai_tokens_total" in content
        assert "dnb_ai_cache_hits_total" in content
        assert "dnb_ai_cache_misses_total" in content
        assert "dnb_ai_confidence_score" in content
        assert "dnb_ai_scholar_queue_depth" in content

    def test_metrics_json_endpoint(self, client: TestClient) -> None:
        response = client.get("/metrics/json")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
        data = response.json()
        assert "requests" in data
        assert "tokens" in data
        assert "latency_ms" in data

    def test_metrics_content_negotiation(self, client: TestClient) -> None:
        # Requesting application/json on /metrics returns JSON snapshot
        response = client.get("/metrics", headers={"Accept": "application/json"})
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
        data = response.json()
        assert "requests" in data

        # Query param format=json also returns JSON
        response_param = client.get("/metrics?format=json")
        assert response_param.status_code == 200
        assert "application/json" in response_param.headers.get("content-type", "")


class TestCustomMetrics:
    def test_model_calls_and_tokens_recording(self, client: TestClient) -> None:
        metrics.record_model_call_metrics(
            model="gemini-2.5-flash",
            stage="test_generation",
            latency_ms=350.0,
            input_tokens=120,
            output_tokens=45,
        )

        response = client.get("/metrics")
        content = response.text

        # Verify model call counter
        assert 'dnb_ai_model_calls_total{model="gemini-2.5-flash",stage="test_generation"}' in content
        # Verify tokens counter
        assert 'dnb_ai_tokens_total{direction="input"}' in content
        assert 'dnb_ai_tokens_total{direction="output"}' in content
        # Verify latency histogram observation
        assert "dnb_ai_model_latency_seconds_count" in content

    def test_cache_hits_and_misses_recording(self, client: TestClient) -> None:
        metrics.record_cache_hit(cache_type="semantic")
        metrics.record_cache_miss(cache_type="semantic")
        metrics.record_cache_hit(cache_type="exact")

        response = client.get("/metrics")
        content = response.text

        assert 'dnb_ai_cache_hits_total{cache_type="semantic"}' in content
        assert 'dnb_ai_cache_misses_total{cache_type="semantic"}' in content
        assert 'dnb_ai_cache_hits_total{cache_type="exact"}' in content

    def test_confidence_score_recording(self, client: TestClient) -> None:
        metrics.record_confidence_score(0.85)
        metrics.record_confidence_score(0.42)

        response = client.get("/metrics")
        content = response.text
        assert "dnb_ai_confidence_score_count" in content
        assert "dnb_ai_confidence_score_bucket" in content

    @pytest.mark.asyncio
    async def test_scholar_queue_depth_gauge(self, client: TestClient) -> None:
        store = get_review_store()
        await store.clear()

        item = ReviewItem(
            question="Is fasting without intention valid?",
            answer="Niyyah is required before dawn.",
            confidence=0.35,
            band="uncertain",
        )
        await store.add(item)
        response = client.get("/metrics")
        assert "dnb_ai_scholar_queue_depth 1.0" in response.text

        await store.clear()
        response = client.get("/metrics")
        assert "dnb_ai_scholar_queue_depth 0.0" in response.text

    @pytest.mark.asyncio
    async def test_scholar_queue_depth_dynamic_refresh(self) -> None:
        store = get_review_store()
        await store.clear()
        assert metrics.SCHOLAR_QUEUE_DEPTH._value.get() == 0.0

        item = ReviewItem(
            question="Is prayer valid without wudu?",
            answer="Wudu is a prerequisite for prayer.",
            confidence=0.4,
            band="uncertain",
        )
        await store.add(item)
        await metrics.refresh_scholar_queue_depth()
        assert metrics.SCHOLAR_QUEUE_DEPTH._value.get() == 1.0


class TestEndpointProtection:
    def test_token_protection_bearer(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("METRICS_TOKEN", "secret-token-123")

        # Missing token -> 401
        res_unauth = client.get("/metrics")
        assert res_unauth.status_code == 401
        assert "WWW-Authenticate" in res_unauth.headers

        # Invalid token -> 401
        res_invalid = client.get("/metrics", headers={"Authorization": "Bearer wrong-token"})
        assert res_invalid.status_code == 401

        # Valid Bearer token -> 200
        res_valid = client.get("/metrics", headers={"Authorization": "Bearer secret-token-123"})
        assert res_valid.status_code == 200

    def test_token_protection_header(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("METRICS_TOKEN", "secret-token-123")

        # Valid X-Metrics-Token header -> 200
        res_valid = client.get("/metrics", headers={"X-Metrics-Token": "secret-token-123"})
        assert res_valid.status_code == 200

    def test_ip_allowlist_protection(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("METRICS_IP_ALLOWLIST", "10.0.0.0/8, 192.168.1.50, 127.0.0.1")

        # Allowed loopback -> 200
        res_allowed = client.get("/metrics", headers={"X-Forwarded-For": "127.0.0.1"})
        assert res_allowed.status_code == 200

        # Allowed subnet -> 200
        res_subnet = client.get("/metrics", headers={"X-Forwarded-For": "10.1.2.3"})
        assert res_subnet.status_code == 200

        # Disallowed IP -> 403
        res_denied = client.get("/metrics", headers={"X-Forwarded-For": "203.0.113.19"})
        assert res_denied.status_code == 403

    def test_invalid_ip_allowlist_syntax_is_handled_gracefully(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Invalid entry should be skipped without crashing
        monkeypatch.setenv("METRICS_IP_ALLOWLIST", "invalid-ip-string, 127.0.0.1")
        res = client.get("/metrics", headers={"X-Forwarded-For": "127.0.0.1"})
        assert res.status_code == 200


class TestContentSafetyInvariant:
    def test_no_sensitive_text_leaks_in_prometheus_output(self, client: TestClient) -> None:
        secret_query = "SUPER_SECRET_USER_INPUT_PASSWORD_XYZ987"
        secret_answer = "SECRET_GENERATED_MODEL_ANSWER_BODY_ABC123"

        class FakeUsage:
            prompt_token_count = 50
            candidates_token_count = 25
            total_token_count = 75

        class FakeResp:
            usage_metadata = FakeUsage()
            text = secret_answer

        telemetry.record_model_call(
            response=FakeResp(),
            model="gemini-2.5-flash",
            latency_ms=150.0,
            stage="generation",
        )

        response = client.get("/metrics")
        content = response.text

        assert secret_query not in content
        assert secret_answer not in content
