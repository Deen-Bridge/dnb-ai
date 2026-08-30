"""Tests for Agent Fallback and Recovery Mechanisms (#225)."""

import asyncio
import pytest

from agent_fallback import (
    AgentHealth,
    AgentRegistry,
    CircuitBreakerConfig,
    CircuitState,
    RetryConfig,
)


@pytest.mark.asyncio
async def test_agent_success_first_try():
    registry = AgentRegistry()

    async def success_agent(payload):
        return f"Hello {payload['name']}"

    registry.register_agent("primary", success_agent)

    result = await registry.execute_with_resilience("primary", {"name": "DeenBridge"})
    assert result.success is True
    assert result.data == "Hello DeenBridge"
    assert "success" in result.stages_executed
    assert registry.get_health("primary") == AgentHealth.HEALTHY


@pytest.mark.asyncio
async def test_agent_retry_recovery():
    registry = AgentRegistry()
    attempts = 0

    async def flaky_agent(payload):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("Transient timeout")
        return "Recovered Success"

    registry.register_agent("flaky", flaky_agent)

    result = await registry.execute_with_resilience(
        "flaky",
        {},
        retry_config=RetryConfig(max_retries=3, initial_delay=0.01),
    )
    assert result.success is True
    assert result.data == "Recovered Success"
    assert attempts == 3


@pytest.mark.asyncio
async def test_alternative_agent_routing():
    registry = AgentRegistry()

    async def failing_primary(payload):
        raise RuntimeError("Primary crashed")

    async def backup_agent(payload):
        return "Backup Result"

    registry.register_agent("primary", failing_primary, fallback_agent="backup")
    registry.register_agent("backup", backup_agent)

    result = await registry.execute_with_resilience(
        "primary",
        {},
        retry_config=RetryConfig(max_retries=1, initial_delay=0.01),
    )
    assert result.success is True
    assert result.data == "Backup Result"
    assert result.fallback_used is True
    assert any("routed_to_alternative_backup" in s for s in result.stages_executed)


@pytest.mark.asyncio
async def test_deterministic_fallback():
    registry = AgentRegistry()

    async def failing_agent(payload):
        raise ValueError("Total failure")

    def rule_based_fallback(payload):
        return "Deterministic Rule Output"

    registry.register_agent(
        "ai_agent",
        failing_agent,
        deterministic_fallback=rule_based_fallback,
    )

    result = await registry.execute_with_resilience(
        "ai_agent",
        {},
        retry_config=RetryConfig(max_retries=0),
    )
    assert result.success is True
    assert result.data == "Deterministic Rule Output"
    assert "deterministic_fallback_used" in result.stages_executed


@pytest.mark.asyncio
async def test_circuit_breaker_tripping():
    registry = AgentRegistry()

    async def always_fail(payload):
        raise TimeoutError("Timeout")

    registry.register_agent(
        "cb_agent",
        always_fail,
        circuit_config=CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=60.0),
        deterministic_fallback=lambda p: "Fallback Rule",
    )

    # First failure
    r1 = await registry.execute_with_resilience("cb_agent", {}, retry_config=RetryConfig(max_retries=0))
    assert r1.success is True  # fallback caught it

    # Second failure -> trips circuit breaker
    r2 = await registry.execute_with_resilience("cb_agent", {}, retry_config=RetryConfig(max_retries=0))
    assert r2.success is True

    cb = registry._circuit_breakers["cb_agent"]
    assert cb.state == CircuitState.OPEN
    assert registry.get_health("cb_agent") == AgentHealth.UNHEALTHY

    # Third call should immediately hit open circuit breaker without invoking always_fail
    r3 = await registry.execute_with_resilience("cb_agent", {}, retry_config=RetryConfig(max_retries=0))
    assert r3.success is True
    assert "circuit_breaker_open" in r3.stages_executed


@pytest.mark.asyncio
async def test_partial_result_preservation():
    registry = AgentRegistry()

    async def failing_with_partial(payload):
        raise RuntimeError("Failed mid-stream")

    def preserve_partial(payload, exc):
        return {"partial_text": "partial tokens gathered", "error": str(exc)}

    registry.register_agent("stream_agent", failing_with_partial)

    result = await registry.execute_with_resilience(
        "stream_agent",
        {},
        retry_config=RetryConfig(max_retries=0),
        partial_preserver=preserve_partial,
    )
    assert result.success is False
    assert result.partial_result is not None
    assert result.partial_result["partial_text"] == "partial tokens gathered"
    assert "partial_result_preserved" in result.stages_executed


@pytest.mark.asyncio
async def test_result_caching_redundancy():
    registry = AgentRegistry()
    call_count = 0

    async def cached_agent(payload):
        nonlocal call_count
        call_count += 1
        return "Expensive Compute Result"

    registry.register_agent("compute", cached_agent)

    # First call
    res1 = await registry.execute_with_resilience("compute", {}, cache_key="test_key")
    assert res1.success is True
    assert res1.data == "Expensive Compute Result"
    assert call_count == 1

    # Second call uses cache
    res2 = await registry.execute_with_resilience("compute", {}, cache_key="test_key")
    assert res2.success is True
    assert res2.data == "Expensive Compute Result"
    assert call_count == 1  # count did not increase
    assert "cache_hit" in res2.stages_executed
