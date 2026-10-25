"""Failover, retry, timeout, circuit-breaker and degradation coverage for agent_fallback (#438).

Complements tests/test_agent_fallback.py. Every agent here is a local stub: no real
model, network or clock-dependent behaviour is used, so the suite stays deterministic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

import agent_fallback
from agent_fallback import (
    AgentExecutionResult,
    AgentHealth,
    AgentRegistry,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryConfig,
    default_agent_registry,
)


def _failing(exc: Exception) -> Callable[[Any], Awaitable[Any]]:
    async def agent(payload: Any) -> Any:
        raise exc

    return agent


def _returning(value: Any) -> Callable[[Any], Awaitable[Any]]:
    async def agent(payload: Any) -> Any:
        return value

    return agent


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff delays instead of actually sleeping."""
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(agent_fallback.asyncio, "sleep", fake_sleep)
    return recorded


# ---------------------------------------------------------------------------
# Unknown agents and result defaults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_agent_reports_not_found() -> None:
    registry = AgentRegistry()

    result = await registry.execute_with_resilience("ghost", {})

    assert result.success is False
    assert result.error == "Agent 'ghost' not found"
    assert result.data is None
    assert result.stages_executed == []


def test_execution_result_defaults() -> None:
    result = AgentExecutionResult(success=True)

    assert result.data is None
    assert result.error is None
    assert result.partial_result is None
    assert result.stages_executed == []
    assert result.fallback_used is False
    assert result.duration_ms == 0.0


def test_default_registry_is_an_agent_registry() -> None:
    assert isinstance(default_agent_registry, AgentRegistry)


# ---------------------------------------------------------------------------
# Retry behaviour and exponential backoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_retries_exhausted_without_fallback_reports_last_error() -> None:
    registry = AgentRegistry()
    attempts = 0

    async def always_down(payload: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise RuntimeError(f"boom {attempts}")

    registry.register_agent("solo", always_down)

    result = await registry.execute_with_resilience(
        "solo", {}, retry_config=RetryConfig(max_retries=2, initial_delay=0.0, jitter=False)
    )

    assert attempts == 3  # first try + 2 retries
    assert result.success is False
    assert result.error == "boom 3"
    assert result.data is None
    assert result.fallback_used is True
    assert result.stages_executed == ["attempt_0", "attempt_1", "attempt_2", "graceful_degradation_failed"]
    assert result.duration_ms >= 0.0
    # A failed execution counts once, no matter how many attempts it took.
    assert registry._failure_counts["solo"] == 1


@pytest.mark.asyncio
async def test_zero_retries_means_a_single_attempt_and_no_sleep(sleeps: list[float]) -> None:
    registry = AgentRegistry()
    attempts = 0

    async def failing(payload: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise ConnectionError("down")

    registry.register_agent("once", failing)

    result = await registry.execute_with_resilience("once", {}, retry_config=RetryConfig(max_retries=0))

    assert attempts == 1
    assert result.success is False
    assert sleeps == []


@pytest.mark.asyncio
async def test_backoff_delay_doubles_between_attempts(sleeps: list[float]) -> None:
    registry = AgentRegistry()
    registry.register_agent("flaky", _failing(RuntimeError("nope")))

    await registry.execute_with_resilience(
        "flaky",
        {},
        retry_config=RetryConfig(max_retries=3, initial_delay=0.1, backoff_factor=2.0, max_delay=5.0, jitter=False),
    )

    # No sleep after the final attempt, so 3 delays for 4 attempts.
    assert sleeps == pytest.approx([0.1, 0.2, 0.4])


@pytest.mark.asyncio
async def test_backoff_delay_is_capped_at_max_delay(sleeps: list[float]) -> None:
    registry = AgentRegistry()
    registry.register_agent("flaky", _failing(RuntimeError("nope")))

    await registry.execute_with_resilience(
        "flaky",
        {},
        retry_config=RetryConfig(max_retries=3, initial_delay=1.0, backoff_factor=10.0, max_delay=2.0, jitter=False),
    )

    assert sleeps == pytest.approx([1.0, 2.0, 2.0])


@pytest.mark.asyncio
@pytest.mark.parametrize(("random_value", "expected_delay"), [(0.0, 0.1), (1.0, 0.3)])
async def test_jitter_scales_delay_between_half_and_one_and_a_half(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], random_value: float, expected_delay: float
) -> None:
    monkeypatch.setattr(agent_fallback.random, "random", lambda: random_value)
    registry = AgentRegistry()
    registry.register_agent("flaky", _failing(RuntimeError("nope")))

    await registry.execute_with_resilience(
        "flaky", {}, retry_config=RetryConfig(max_retries=1, initial_delay=0.2, jitter=True)
    )

    assert sleeps == pytest.approx([expected_delay])


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_in_primary_fails_over_to_alternative_agent() -> None:
    registry = AgentRegistry()

    async def hangs(payload: Any) -> Any:
        await asyncio.wait_for(asyncio.sleep(30), timeout=0.01)

    registry.register_agent("slow", hangs, fallback_agent="fast")
    registry.register_agent("fast", _returning("fast answer"))

    result = await registry.execute_with_resilience("slow", {}, retry_config=RetryConfig(max_retries=0))

    assert result.success is True
    assert result.data == "fast answer"
    assert result.fallback_used is True
    assert result.stages_executed == ["attempt_0", "routed_to_alternative_fast"]


@pytest.mark.asyncio
async def test_timeout_then_success_on_retry_is_not_a_fallback() -> None:
    registry = AgentRegistry()
    attempts = 0

    async def slow_once(payload: Any) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("upstream timed out")
        return "ok"

    registry.register_agent("slow_once", slow_once)

    result = await registry.execute_with_resilience(
        "slow_once", {}, retry_config=RetryConfig(max_retries=2, initial_delay=0.0, jitter=False)
    )

    assert result.success is True
    assert result.data == "ok"
    assert result.fallback_used is False
    assert result.stages_executed == ["attempt_0", "attempt_1", "success"]


@pytest.mark.asyncio
async def test_persistent_timeout_with_no_fallback_fails_gracefully() -> None:
    registry = AgentRegistry()

    async def hangs(payload: Any) -> Any:
        await asyncio.wait_for(asyncio.sleep(30), timeout=0.01)

    registry.register_agent("slow", hangs)

    result = await registry.execute_with_resilience("slow", {}, retry_config=RetryConfig(max_retries=0))

    assert result.success is False
    assert result.stages_executed[-1] == "graceful_degradation_failed"
    assert result.error is not None


# ---------------------------------------------------------------------------
# Fallback chain: alternative agent -> deterministic fallback -> graceful failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alternative_agent_success_updates_health_counters() -> None:
    registry = AgentRegistry()
    registry.register_agent("primary", _failing(RuntimeError("down")), fallback_agent="backup")
    registry.register_agent("backup", _returning("backup ok"))

    result = await registry.execute_with_resilience("primary", {}, retry_config=RetryConfig(max_retries=0))

    assert result.data == "backup ok"
    assert registry._failure_counts["primary"] == 1
    assert registry._success_counts["backup"] == 1
    assert registry._success_counts["primary"] == 0


@pytest.mark.asyncio
async def test_failed_alternative_falls_through_to_deterministic_fallback() -> None:
    registry = AgentRegistry()
    registry.register_agent(
        "primary",
        _failing(RuntimeError("primary down")),
        fallback_agent="backup",
        deterministic_fallback=lambda payload: "rule output",
    )
    registry.register_agent("backup", _failing(RuntimeError("backup down")))

    result = await registry.execute_with_resilience("primary", {}, retry_config=RetryConfig(max_retries=0))

    assert result.success is True
    assert result.data == "rule output"
    assert result.fallback_used is True
    assert result.stages_executed == ["attempt_0", "routed_to_alternative_backup", "deterministic_fallback_used"]


@pytest.mark.asyncio
async def test_every_fallback_failing_returns_the_primary_error() -> None:
    registry = AgentRegistry()

    def broken_rule(payload: Any) -> Any:
        raise RuntimeError("rule engine down")

    registry.register_agent(
        "primary",
        _failing(ValueError("primary down")),
        fallback_agent="backup",
        deterministic_fallback=broken_rule,
    )
    registry.register_agent("backup", _failing(RuntimeError("backup down")))

    result = await registry.execute_with_resilience("primary", {}, retry_config=RetryConfig(max_retries=0))

    assert result.success is False
    assert result.data is None
    assert result.error == "primary down"
    assert result.fallback_used is True
    assert result.stages_executed == [
        "attempt_0",
        "routed_to_alternative_backup",
        "deterministic_fallback_used",
        "graceful_degradation_failed",
    ]


@pytest.mark.asyncio
async def test_unregistered_alternative_agent_is_skipped() -> None:
    registry = AgentRegistry()
    registry.register_agent(
        "primary",
        _failing(RuntimeError("down")),
        fallback_agent="never_registered",
        deterministic_fallback=lambda payload: "rule output",
    )

    result = await registry.execute_with_resilience("primary", {}, retry_config=RetryConfig(max_retries=0))

    assert result.success is True
    assert result.data == "rule output"
    assert not any(stage.startswith("routed_to_alternative") for stage in result.stages_executed)


@pytest.mark.asyncio
async def test_async_deterministic_fallback_is_awaited() -> None:
    registry = AgentRegistry()

    async def async_rule(payload: Any) -> str:
        return f"async rule for {payload['topic']}"

    registry.register_agent("primary", _failing(RuntimeError("down")), deterministic_fallback=async_rule)

    result = await registry.execute_with_resilience(
        "primary", {"topic": "salah"}, retry_config=RetryConfig(max_retries=0)
    )

    assert result.success is True
    assert result.data == "async rule for salah"


@pytest.mark.asyncio
async def test_fallbacks_receive_the_original_payload() -> None:
    registry = AgentRegistry()
    seen: list[Any] = []

    async def backup(payload: Any) -> str:
        seen.append(payload)
        return "ok"

    registry.register_agent("primary", _failing(RuntimeError("down")), fallback_agent="backup")
    registry.register_agent("backup", backup)

    payload = {"question": "What is zakat?"}
    await registry.execute_with_resilience("primary", payload, retry_config=RetryConfig(max_retries=0))

    assert seen == [payload]


# ---------------------------------------------------------------------------
# Partial result preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_result_is_kept_alongside_a_successful_fallback() -> None:
    registry = AgentRegistry()
    boom = RuntimeError("mid-stream failure")
    received: list[tuple[Any, Exception]] = []

    def preserver(payload: Any, exc: Exception) -> dict[str, str]:
        received.append((payload, exc))
        return {"partial_text": "half an answer"}

    registry.register_agent("stream", _failing(boom), deterministic_fallback=lambda payload: "complete rule answer")

    result = await registry.execute_with_resilience(
        "stream", {"q": 1}, retry_config=RetryConfig(max_retries=0), partial_preserver=preserver
    )

    assert result.success is True
    assert result.data == "complete rule answer"
    assert result.partial_result == {"partial_text": "half an answer"}
    assert result.stages_executed[:2] == ["attempt_0", "partial_result_preserved"]
    assert received == [({"q": 1}, boom)]


@pytest.mark.asyncio
async def test_failing_partial_preserver_does_not_break_the_fallback_chain() -> None:
    registry = AgentRegistry()

    def bad_preserver(payload: Any, exc: Exception) -> Any:
        raise RuntimeError("preserver exploded")

    registry.register_agent(
        "stream", _failing(RuntimeError("down")), deterministic_fallback=lambda payload: "rule answer"
    )

    result = await registry.execute_with_resilience(
        "stream", {}, retry_config=RetryConfig(max_retries=0), partial_preserver=bad_preserver
    )

    assert result.success is True
    assert result.data == "rule answer"
    assert result.partial_result is None
    assert "partial_result_preserved" not in result.stages_executed


# ---------------------------------------------------------------------------
# Circuit breaker wired through the registry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_circuit_without_fallback_fails_fast_without_calling_agent() -> None:
    registry = AgentRegistry()
    calls = 0

    async def failing(payload: Any) -> Any:
        nonlocal calls
        calls += 1
        raise RuntimeError("down")

    registry.register_agent("fragile", failing, circuit_config=CircuitBreakerConfig(failure_threshold=1))

    first = await registry.execute_with_resilience("fragile", {}, retry_config=RetryConfig(max_retries=0))
    second = await registry.execute_with_resilience("fragile", {}, retry_config=RetryConfig(max_retries=0))

    assert first.success is False
    assert calls == 1  # the second call never reached the agent
    assert second.success is False
    assert second.error == "Circuit breaker is open"
    assert second.stages_executed == ["circuit_breaker_open", "graceful_degradation_failed"]


@pytest.mark.asyncio
async def test_open_circuit_routes_straight_to_alternative_agent() -> None:
    registry = AgentRegistry()
    primary_calls = 0

    async def primary(payload: Any) -> Any:
        nonlocal primary_calls
        primary_calls += 1
        raise RuntimeError("down")

    registry.register_agent(
        "primary", primary, fallback_agent="backup", circuit_config=CircuitBreakerConfig(failure_threshold=1)
    )
    registry.register_agent("backup", _returning("backup ok"))

    await registry.execute_with_resilience("primary", {}, retry_config=RetryConfig(max_retries=0))
    second = await registry.execute_with_resilience("primary", {}, retry_config=RetryConfig(max_retries=0))

    assert primary_calls == 1
    assert second.success is True
    assert second.data == "backup ok"
    assert second.stages_executed == ["circuit_breaker_open", "routed_to_alternative_backup"]


@pytest.mark.asyncio
async def test_circuit_recovers_through_half_open_after_timeout() -> None:
    registry = AgentRegistry()
    calls = 0

    async def recovers(payload: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("first call fails")
        return "healed"

    registry.register_agent(
        "healer",
        recovers,
        circuit_config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.0, half_open_max_calls=1),
    )
    breaker = registry._circuit_breakers["healer"]

    await registry.execute_with_resilience("healer", {}, retry_config=RetryConfig(max_retries=0))
    assert breaker.state == CircuitState.OPEN

    result = await registry.execute_with_resilience("healer", {}, retry_config=RetryConfig(max_retries=0))

    assert result.success is True
    assert result.data == "healed"
    assert breaker.state == CircuitState.CLOSED
    assert registry.get_health("healer") == AgentHealth.HEALTHY


@pytest.mark.asyncio
async def test_failure_during_half_open_trips_the_circuit_again() -> None:
    registry = AgentRegistry()
    registry.register_agent(
        "stubborn",
        _failing(RuntimeError("still down")),
        deterministic_fallback=lambda payload: "rule",
        circuit_config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.0),
    )
    breaker = registry._circuit_breakers["stubborn"]

    await registry.execute_with_resilience("stubborn", {}, retry_config=RetryConfig(max_retries=0))
    assert breaker.state == CircuitState.OPEN

    # recovery_timeout is 0, so this call is admitted as a half-open probe and fails.
    result = await registry.execute_with_resilience("stubborn", {}, retry_config=RetryConfig(max_retries=0))

    assert result.success is True  # served by the deterministic fallback
    assert breaker.state == CircuitState.OPEN
    assert registry._failure_counts["stubborn"] == 2


# ---------------------------------------------------------------------------
# CircuitBreaker state machine (pure unit tests, no registry)
# ---------------------------------------------------------------------------


def test_circuit_breaker_starts_closed_and_allows_requests() -> None:
    breaker = CircuitBreaker("unit")

    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_circuit_breaker_opens_only_at_the_failure_threshold() -> None:
    breaker = CircuitBreaker("unit", CircuitBreakerConfig(failure_threshold=3))

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED

    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False


def test_success_in_closed_state_resets_the_failure_streak() -> None:
    breaker = CircuitBreaker("unit", CircuitBreakerConfig(failure_threshold=3))

    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 2


def test_open_circuit_stays_blocked_until_recovery_timeout_elapses() -> None:
    breaker = CircuitBreaker("unit", CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=60.0))
    breaker.record_failure()
    assert breaker.allow_request() is False

    breaker.last_failure_time -= 61  # pretend the recovery window has passed

    assert breaker.allow_request() is True
    assert breaker.state == CircuitState.HALF_OPEN


def test_half_open_needs_configured_number_of_successes_to_close() -> None:
    breaker = CircuitBreaker(
        "unit", CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.0, half_open_max_calls=2)
    )
    breaker.record_failure()
    assert breaker.allow_request() is True
    assert breaker.state == CircuitState.HALF_OPEN

    breaker.record_success()
    assert breaker.state == CircuitState.HALF_OPEN

    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0
    assert breaker.success_count == 0


def test_half_open_failure_reopens_and_clears_probe_successes() -> None:
    breaker = CircuitBreaker(
        "unit", CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.0, half_open_max_calls=3)
    )
    breaker.record_failure()
    breaker.allow_request()
    breaker.record_success()
    assert breaker.success_count == 1

    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN
    assert breaker.success_count == 0


# ---------------------------------------------------------------------------
# Health reporting
# ---------------------------------------------------------------------------


def test_fresh_agent_with_no_traffic_is_healthy() -> None:
    # Regression: this used to raise ZeroDivisionError (0 failures / 0 total calls).
    registry = AgentRegistry()
    registry.register_agent("new", _returning("ok"))

    assert registry.get_health("new") == AgentHealth.HEALTHY


def test_agent_degrades_after_repeated_failures_before_circuit_opens() -> None:
    registry = AgentRegistry()
    registry.register_agent("wobbly", _returning("ok"), circuit_config=CircuitBreakerConfig(failure_threshold=5))

    registry.record_failure("wobbly")
    assert registry.get_health("wobbly") == AgentHealth.HEALTHY  # one failure is not enough

    registry.record_failure("wobbly")
    assert registry.get_health("wobbly") == AgentHealth.DEGRADED


def test_open_circuit_reports_unhealthy_and_get_all_health_lists_every_agent() -> None:
    registry = AgentRegistry()
    registry.register_agent("down", _returning("ok"), circuit_config=CircuitBreakerConfig(failure_threshold=1))
    registry.register_agent("up", _returning("ok"))
    registry.record_failure("down")

    assert registry.get_all_health() == {"down": "unhealthy", "up": "healthy"}


# ---------------------------------------------------------------------------
# Result caching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_results_are_not_cached() -> None:
    registry = AgentRegistry()
    calls = 0

    async def recovers_later(payload: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("down")
        return "fresh"

    registry.register_agent("svc", recovers_later, deterministic_fallback=lambda payload: "stale rule")
    no_retry = RetryConfig(max_retries=0)

    first = await registry.execute_with_resilience("svc", {}, retry_config=no_retry, cache_key="k")
    second = await registry.execute_with_resilience("svc", {}, retry_config=no_retry, cache_key="k")
    third = await registry.execute_with_resilience("svc", {}, retry_config=no_retry, cache_key="k")

    assert first.data == "stale rule"
    assert second.data == "fresh"  # the fallback answer was not served from cache
    assert "cache_hit" not in second.stages_executed
    assert third.data == "fresh"
    assert third.stages_executed == ["cache_hit"]
    assert calls == 2


@pytest.mark.asyncio
async def test_expired_cache_entry_triggers_a_fresh_call() -> None:
    registry = AgentRegistry()
    registry.cache_ttl = -1.0  # every entry is already stale
    calls = 0

    async def counting(payload: Any) -> int:
        nonlocal calls
        calls += 1
        return calls

    registry.register_agent("svc", counting)

    first = await registry.execute_with_resilience("svc", {}, cache_key="k")
    second = await registry.execute_with_resilience("svc", {}, cache_key="k")

    assert (first.data, second.data) == (1, 2)
    assert "cache_hit" not in second.stages_executed


@pytest.mark.asyncio
async def test_different_cache_keys_do_not_share_results() -> None:
    registry = AgentRegistry()

    async def echo(payload: Any) -> Any:
        return payload

    registry.register_agent("echo", echo)

    a = await registry.execute_with_resilience("echo", "alpha", cache_key="a")
    b = await registry.execute_with_resilience("echo", "beta", cache_key="b")

    assert (a.data, b.data) == ("alpha", "beta")
    assert registry.get_cached_result("a") == "alpha"
    assert registry.get_cached_result("b") == "beta"
    assert registry.get_cached_result("missing") is None
