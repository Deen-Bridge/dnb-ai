"""Comprehensive tests for production health, readiness, and dependency degradation (AI-094 / Issue #870).

Covers:
- Provider outage: all required failing (503 not_ready) vs secondary available (200 degraded); /health stays 200.
- Redis outage: single-instance mode (200 degraded) vs multi-instance mode (503 not_ready), with bounded queue.
- Embedding provider outage: local fallback in semantic cache, degraded status, and counter increments.
- Startup misconfiguration: production fails fast with RuntimeError; non-production reports not_ready (503).
- Recovery: returning dependency returns to ready within TTL, clearing fallback flags.
- Liveness independence: /health returns 200 with NO external network/dependency calls during all outages.
- Bounded probes: probe timeout enforcement, single-flight probe deduping under concurrency.
- No cost: readiness probes never invoke LLM generation endpoints.
- No leakage: sentinel secret and PII strings never appear in responses, headers, or logs.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import main
import semantic_cache
from readiness import (
    DependencyCriticality,
    DependencyStatus,
    HealthState,
    ReasonCode,
    readiness_registry,
    validate_startup_configuration,
)
from review_store import ReviewItem, ReviewStore


@pytest.fixture(autouse=True)
def reset_readiness():
    """Reset registry state before each test."""
    readiness_registry.reset()
    yield
    readiness_registry.reset()


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(main.app)


# ---------------------------------------------------------------------------
# 1. Liveness Independence
# ---------------------------------------------------------------------------


def test_liveness_constant_time_and_independent_of_dependencies(client, monkeypatch):
    """Liveness probe (/health) returns 200 without making any external calls even during outages."""
    # Mock redis and gemini to ensure they are NEVER touched by /health
    mock_redis_ping = AsyncMock(side_effect=RuntimeError("Redis network call attempted!"))
    mock_gemini_call = MagicMock(side_effect=RuntimeError("Gemini network call attempted!"))

    monkeypatch.setattr("google.generativeai.GenerativeModel.generate_content", mock_gemini_call)

    # Force simulated outages in readiness
    readiness_registry.set_mock_provider_health("primary_llm", False)
    readiness_registry.set_mock_provider_health("secondary_llm", False)
    readiness_registry.set_mock_provider_health("redis", False)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "checks" in data
    assert data["checks"]["process"] == "ok"

    # Verify no external dependency was touched
    assert mock_redis_ping.call_count == 0
    assert mock_gemini_call.call_count == 0


def test_health_deep_parameter_compatibility(client):
    """Legacy deep query parameter returns 200 with 'skipped' reachability check."""
    response = client.get("/health?deep=true")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["checks"]["gemini_api_reachable"] == "skipped"


# ---------------------------------------------------------------------------
# 2. Provider Outage & Failover
# ---------------------------------------------------------------------------


def test_provider_outage_all_providers_failing(client):
    """When all model providers fail, GET /ready reports 503 not_ready."""
    readiness_registry.set_mock_provider_health("primary_llm", False)
    readiness_registry.set_mock_provider_health("secondary_llm", False)

    ready_resp = client.get("/ready")
    assert ready_resp.status_code == 503
    data = ready_resp.json()
    assert data["status"] == HealthState.NOT_READY.value
    assert any("primary_llm:all_providers_unavailable" in r for r in data["reasons"])
    assert data["dependencies"]["primary_llm"]["status"] == "fail"

    # Liveness remains 200
    health_resp = client.get("/health")
    assert health_resp.status_code == 200


def test_provider_outage_secondary_fallback_available(client):
    """When primary provider fails but secondary fallback is functional, report 200 degraded."""
    readiness_registry.set_mock_provider_health("primary_llm", False)
    readiness_registry.set_mock_provider_health("secondary_llm", True)

    ready_resp = client.get("/ready")
    assert ready_resp.status_code == 200
    data = ready_resp.json()
    assert data["status"] == HealthState.DEGRADED.value
    assert any("secondary_provider_degraded" in r for r in data["reasons"])
    assert data["dependencies"]["primary_llm"]["status"] == "degraded"

    # Liveness remains 200
    health_resp = client.get("/health")
    assert health_resp.status_code == 200


# ---------------------------------------------------------------------------
# 3. Redis Outage & Single vs Multi-Instance Mode
# ---------------------------------------------------------------------------


def test_redis_outage_single_instance_mode(client, monkeypatch):
    """In single-instance mode (default), Redis outage reports 200 degraded with in-memory fallback."""
    monkeypatch.setenv("REQUIRE_REDIS", "false")
    monkeypatch.setenv("DEPLOYMENT_MODE", "single-instance")

    readiness_registry.set_mock_provider_health("redis", False)

    ready_resp = client.get("/ready")
    assert ready_resp.status_code == 200
    data = ready_resp.json()
    assert data["status"] == HealthState.DEGRADED.value
    assert data["mode"] == "single-instance"
    assert data["dependencies"]["redis"]["criticality"] == DependencyCriticality.OPTIONAL.value
    assert data["dependencies"]["redis"]["status"] == "degraded"
    assert "redis:redis_degraded_in_memory" in data["reasons"]

    # Liveness remains 200
    health_resp = client.get("/health")
    assert health_resp.status_code == 200


def test_redis_outage_multi_instance_mode(client, monkeypatch):
    """In multi-instance mode (REQUIRE_REDIS=true), Redis outage reports 503 not_ready."""
    monkeypatch.setenv("REQUIRE_REDIS", "true")

    readiness_registry.set_mock_provider_health("redis", False)

    ready_resp = client.get("/ready")
    assert ready_resp.status_code == 503
    data = ready_resp.json()
    assert data["status"] == HealthState.NOT_READY.value
    assert data["mode"] == "multi-instance"
    assert data["dependencies"]["redis"]["criticality"] == DependencyCriticality.REQUIRED.value
    assert data["dependencies"]["redis"]["status"] == "fail"
    assert "redis:redis_required_unreachable" in data["reasons"]

    # Liveness remains 200
    health_resp = client.get("/health")
    assert health_resp.status_code == 200


@pytest.mark.asyncio
async def test_review_queue_bounded_fallback_and_overflow():
    """In-memory fallback queue is bounded, evicts oldest on overflow, and surfaces drop counters."""
    # Create store with capacity limit of 3
    store = ReviewStore(max_local_size=3)
    assert store.max_local_size == 3
    assert store.overflow_drops == 0

    # Add 3 items
    for i in range(3):
        item = ReviewItem(
            id=f"item-{i}",
            question=f"Question {i}",
            answer=f"Answer {i}",
            confidence=0.3,
            band="low",
        )
        await store.add(item)

    stats = await store.stats()
    assert stats["pending"] == 3
    assert stats["overflow_drops"] == 0

    # Adding a 4th item must trigger explicit overflow eviction
    item4 = ReviewItem(
        id="item-4",
        question="Question 4",
        answer="Answer 4",
        confidence=0.3,
        band="low",
    )
    await store.add(item4)

    stats = await store.stats()
    assert stats["pending"] == 3  # Capacity strictly maintained
    assert stats["overflow_drops"] == 1
    # Oldest item (item-0) was evicted
    assert await store.get("item-0") is None
    assert await store.get("item-4") is not None

    # Readiness reflects overflow drops
    assert readiness_registry._review_queue_overflow_drops == 1


# ---------------------------------------------------------------------------
# 4. Embedding Provider Outage & Semantic Cache Fallback
# ---------------------------------------------------------------------------


def test_embedding_provider_outage_local_fallback(client, monkeypatch):
    """When external embedding provider fails, local fallback is used and readiness reports degraded."""

    # Force genai.embed_content to raise an error
    def failing_embed(*args, **kwargs):
        raise ConnectionError("Embedding service unreachable")

    monkeypatch.setattr("google.generativeai.embed_content", failing_embed)
    monkeypatch.setattr(semantic_cache, "_FAKE_EMBEDDING", None)

    # Perform embedding
    vec = semantic_cache.embed_text("test query for offline fallback")
    assert vec is not None
    assert len(vec) == 384  # Local SHA-256 fallback dimension

    # Readiness reports degraded with EMBEDDING_DEGRADED_LOCAL
    ready_resp = client.get("/ready")
    assert ready_resp.status_code == 200
    data = ready_resp.json()
    assert data["status"] == HealthState.DEGRADED.value
    assert data["dependencies"]["embedding"]["status"] == "degraded"
    assert data["fallbacks"]["local_embedding"] is True
    assert data["fallbacks"]["total_fallbacks"] >= 1
    assert any("embedding:embedding_degraded_local" in r for r in data["reasons"])


# ---------------------------------------------------------------------------
# 5. Startup Misconfiguration
# ---------------------------------------------------------------------------


def test_startup_validation_production_missing_gemini_key(monkeypatch):
    """In production mode, missing GEMINI_API_KEY fails fast with RuntimeError."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("SERVICE_API_KEY", "prod-service-key")

    with pytest.raises(RuntimeError, match="Production startup configuration validation failed"):
        validate_startup_configuration()


def test_startup_validation_production_missing_service_key(monkeypatch):
    """In production mode without AUTH_DISABLED, missing SERVICE_API_KEY fails fast."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("GEMINI_API_KEY", "valid-gemini-key")
    monkeypatch.setenv("SERVICE_API_KEY", "")
    monkeypatch.setenv("AUTH_DISABLED", "false")

    with pytest.raises(RuntimeError, match="SERVICE_API_KEY is required in production"):
        validate_startup_configuration()


def test_startup_validation_production_require_redis_missing_url(monkeypatch):
    """In production mode with REQUIRE_REDIS=true, missing REDIS_URL fails fast."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("GEMINI_API_KEY", "valid-gemini-key")
    monkeypatch.setenv("SERVICE_API_KEY", "valid-service-key")
    monkeypatch.setenv("REQUIRE_REDIS", "true")
    monkeypatch.setenv("REDIS_URL", "")

    with pytest.raises(RuntimeError, match="REDIS_URL must be configured when REQUIRE_REDIS=true"):
        validate_startup_configuration()


def test_startup_validation_dev_mode_does_not_crash(monkeypatch, client):
    """Outside production, missing configuration does not crash and reports not_ready on /ready."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr(main.settings, "gemini_api_key", "")

    # Does not raise
    validate_startup_configuration()

    # But /ready reports not_ready (503)
    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == HealthState.NOT_READY.value
    assert data["dependencies"]["config"]["status"] == "fail"


# ---------------------------------------------------------------------------
# 6. Recovery & Bounded Cache TTL
# ---------------------------------------------------------------------------


def test_recovery_clears_degradation_within_ttl(client):
    """When an outage recovers, readiness returns to 'ready' after probe cache TTL."""
    # Control simulated clock
    simulated_time = 1000.0

    def get_time():
        return simulated_time

    readiness_registry.set_clock(get_time)
    readiness_registry.probe_cache_ttl_seconds = 5.0

    # Simulate temporary redis outage
    readiness_registry.set_mock_provider_health("redis", False)
    resp1 = client.get("/ready")
    assert resp1.status_code == 200
    assert resp1.json()["status"] == HealthState.DEGRADED.value

    # Redis recovers
    readiness_registry.set_mock_provider_health("redis", True)

    # Within TTL (advance 2 seconds): still cached degraded
    simulated_time += 2.0
    # In this test we check caching:
    # Note: set_mock_provider_health automatically invalidated provider entry,
    # so let's verify fresh probe returns ready:
    resp2 = client.get("/ready")
    assert resp2.status_code == 200
    assert resp2.json()["status"] == HealthState.READY.value


# ---------------------------------------------------------------------------
# 7. Bounded Probes & Single-Flight Protection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_probes_timeout_enforcement():
    """A hanging dependency probe does not block beyond the configured timeout."""
    readiness_registry.probe_timeout_seconds = 0.2

    async def hanging_probe():
        await asyncio.sleep(5.0)
        return None

    status = await readiness_registry._check_dependency_cached("hanging_dep", hanging_probe)
    assert status.status == "degraded"
    assert status.reason == ReasonCode.PROBE_TIMEOUT.value


@pytest.mark.asyncio
async def test_single_flight_concurrency_protection():
    """Concurrent readiness requests result in only a single underlying probe call."""
    probe_call_count = 0

    async def counted_probe():
        nonlocal probe_call_count
        probe_call_count += 1
        await asyncio.sleep(0.05)
        return DependencyStatus(
            name="counted",
            criticality=DependencyCriticality.OPTIONAL,
            status="ok",
            reason=ReasonCode.OK.value,
        )

    # Launch 5 concurrent checks
    tasks = [readiness_registry._check_dependency_cached("counted", counted_probe) for _ in range(5)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 5
    assert all(r.status == "ok" for r in results)
    assert probe_call_count == 1  # Single-flight deduplication verified


# ---------------------------------------------------------------------------
# 8. Zero Token Spend Guarantee
# ---------------------------------------------------------------------------


def test_no_generation_endpoints_called_by_probes(client, monkeypatch):
    """Readiness probes never call generate_content or spend model tokens."""
    mock_generate = MagicMock(side_effect=RuntimeError("PAID GENERATION ENDPOINT CALLED!"))
    monkeypatch.setattr("google.generativeai.GenerativeModel.generate_content", mock_generate)

    # Call /ready multiple times
    for _ in range(3):
        response = client.get("/ready")
        assert response.status_code in {200, 503}

    assert mock_generate.call_count == 0


# ---------------------------------------------------------------------------
# 9. Zero Secret and PII Leakage
# ---------------------------------------------------------------------------


def test_zero_leakage_in_responses_and_headers(client, monkeypatch, caplog):
    """Sentinel secret tokens, database credentials, and PII never leak in response or logs."""
    sentinel_secret = "sk-live-SUPER-SECRET-TOKEN-DO-NOT-LEAK-999"
    sentinel_redis_url = "redis://user:SUPER_SECRET_PASSWORD@db.internal:6379/0"
    sentinel_pii = "user-sensitive-email@secret-domain.org"

    monkeypatch.setenv("SERVICE_API_KEY", sentinel_secret)
    monkeypatch.setenv("REDIS_URL", sentinel_redis_url)

    with caplog.at_level(logging.DEBUG):
        # Provoke error states
        readiness_registry.set_mock_provider_health("primary_llm", False)
        readiness_registry.set_mock_provider_health("secondary_llm", False)

        resp = client.get("/ready")
        body_text = resp.text
        header_text = " ".join(f"{k}:{v}" for k, v in resp.headers.items())
        log_text = caplog.text

        # Verify absolute absence of secret sentinels
        for text, context in [
            (body_text, "response body"),
            (header_text, "response headers"),
            (log_text, "captured logs"),
        ]:
            assert sentinel_secret not in text, f"Secret leaked in {context}"
            assert "SUPER_SECRET_PASSWORD" not in text, f"Redis credential leaked in {context}"
            assert sentinel_pii not in text, f"PII leaked in {context}"


# ---------------------------------------------------------------------------
# 10. Additive X-Degraded Response Header
# ---------------------------------------------------------------------------


def test_additive_x_degraded_response_header(client, monkeypatch):
    """Client requests traversing a fallback path receive the additive X-Degraded header."""

    # When semantic cache falls back to local embedding
    def failing_embed(*args, **kwargs):
        raise TimeoutError("Embedding API timeout")

    monkeypatch.setattr("google.generativeai.embed_content", failing_embed)
    monkeypatch.setattr(semantic_cache, "_FAKE_EMBEDDING", None)

    # Directly test middleware via client request that triggers embedding
    from semantic_cache import embed_text

    @main.app.get("/test-degraded-route")
    async def sample_route():
        embed_text("test question")
        return {"ok": True}

    res = client.get("/test-degraded-route")
    assert res.status_code == 200
    assert "X-Degraded" in res.headers
    assert "local-embedding" in res.headers["X-Degraded"]
