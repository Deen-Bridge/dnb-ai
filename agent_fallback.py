"""Agent Fallback and Recovery Mechanisms (#225).

Provides robust resilience patterns for agentic workflows:
- Circuit Breaker pattern to prevent cascading failures
- Exponential backoff retry logic with jitter
- Health check and monitoring for agent types
- Alternative agent routing and graceful degradation
- Partial result preservation and caching
- Deterministic / rule-based fallbacks
- Failure analytics and telemetry logging
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class AgentHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 2


@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 0.1
    max_delay: float = 5.0
    backoff_factor: float = 2.0
    jitter: bool = True


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0

    def record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' recovered and is now CLOSED.")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning(f"Circuit breaker '{self.name}' failed in HALF_OPEN, returning to OPEN.")
        elif self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' threshold reached ({self.failure_count}), now OPEN.")

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            now = time.time()
            if now - self.last_failure_time >= self.config.recovery_timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' recovery timeout elapsed, testing in HALF_OPEN.")
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return True
        return True


@dataclass
class AgentExecutionResult:
    success: bool
    data: Any | None = None
    error: str | None = None
    partial_result: Any | None = None
    stages_executed: list[str] = field(default_factory=list)
    fallback_used: bool = False
    duration_ms: float = 0.0


class AgentRegistry:
    """Registry for agents, fallback routes, health status, and failure metrics."""

    def __init__(self) -> None:
        self._agents: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._fallbacks: dict[str, str] = {}  # primary -> backup agent name
        self._deterministic_fallbacks: dict[str, Callable[..., Any]] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._health_statuses: dict[str, AgentHealth] = {}
        self._failure_counts: dict[str, int] = {}
        self._success_counts: dict[str, int] = {}
        self._cached_results: dict[str, tuple[float, Any]] = {}
        self.cache_ttl: float = 300.0

    def register_agent(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        fallback_agent: str | None = None,
        deterministic_fallback: Callable[..., Any] | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
    ) -> None:
        self._agents[name] = handler
        if fallback_agent:
            self._fallbacks[name] = fallback_agent
        if deterministic_fallback:
            self._deterministic_fallbacks[name] = deterministic_fallback
        self._circuit_breakers[name] = CircuitBreaker(name, circuit_config)
        self._health_statuses[name] = AgentHealth.HEALTHY
        self._failure_counts[name] = 0
        self._success_counts[name] = 0

    def get_health(self, name: str) -> AgentHealth:
        cb = self._circuit_breakers.get(name)
        if cb and cb.state == CircuitState.OPEN:
            return AgentHealth.UNHEALTHY
        fails = self._failure_counts.get(name, 0)
        succs = self._success_counts.get(name, 1)
        if fails / (fails + succs) > 0.3 and fails >= 2:
            return AgentHealth.DEGRADED
        return AgentHealth.HEALTHY

    def get_all_health(self) -> dict[str, str]:
        return {name: self.get_health(name).value for name in self._agents}

    def record_failure(self, name: str) -> None:
        self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
        if name in self._circuit_breakers:
            self._circuit_breakers[name].record_failure()
        logger.error(f"Agent '{name}' recorded failure. Total failures: {self._failure_counts[name]}")

    def record_success(self, name: str) -> None:
        self._success_counts[name] = self._success_counts.get(name, 0) + 1
        if name in self._circuit_breakers:
            self._circuit_breakers[name].record_success()

    def cache_result(self, key: str, data: Any) -> None:
        self._cached_results[key] = (time.time(), data)

    def get_cached_result(self, key: str) -> Any | None:
        if key in self._cached_results:
            ts, data = self._cached_results[key]
            if time.time() - ts <= self.cache_ttl:
                return data
        return None

    async def execute_with_resilience(
        self,
        agent_name: str,
        payload: Any,
        retry_config: RetryConfig | None = None,
        cache_key: str | None = None,
        partial_preserver: Callable[[Any, Exception], Any] | None = None,
    ) -> AgentExecutionResult:
        start_time = time.time()
        stages = []
        retry_cfg = retry_config or RetryConfig()

        if cache_key:
            cached = self.get_cached_result(cache_key)
            if cached is not None:
                stages.append("cache_hit")
                return AgentExecutionResult(
                    success=True,
                    data=cached,
                    stages_executed=stages,
                    duration_ms=(time.time() - start_time) * 1000,
                )

        cb = self._circuit_breakers.get(agent_name)
        if cb and not cb.allow_request():
            stages.append("circuit_breaker_open")
            logger.warning(f"Circuit breaker for '{agent_name}' is OPEN. Attempting fallback routing.")
            return await self._execute_fallback_chain(
                agent_name, payload, start_time, stages, Exception("Circuit breaker is open"), partial_preserver
            )

        handler = self._agents.get(agent_name)
        if not handler:
            return AgentExecutionResult(
                success=False,
                error=f"Agent '{agent_name}' not found",
                stages_executed=stages,
                duration_ms=(time.time() - start_time) * 1000,
            )

        last_exception = None
        for attempt in range(retry_cfg.max_retries + 1):
            try:
                stages.append(f"attempt_{attempt}")
                res = await handler(payload)
                self.record_success(agent_name)
                if cache_key:
                    self.cache_result(cache_key, res)
                stages.append("success")
                return AgentExecutionResult(
                    success=True,
                    data=res,
                    stages_executed=stages,
                    duration_ms=(time.time() - start_time) * 1000,
                )
            except Exception as e:
                last_exception = e
                logger.warning(f"Agent '{agent_name}' attempt {attempt} failed: {e}")
                if attempt < retry_cfg.max_retries:
                    delay = retry_cfg.initial_delay * (retry_cfg.backoff_factor ** attempt)
                    delay = min(delay, retry_cfg.max_delay)
                    if retry_cfg.jitter:
                        delay *= 0.5 + random.random()
                    await asyncio.sleep(delay)

        self.record_failure(agent_name)
        return await self._execute_fallback_chain(
            agent_name, payload, start_time, stages, last_exception, partial_preserver
        )

    async def _execute_fallback_chain(
        self,
        agent_name: str,
        payload: Any,
        start_time: float,
        stages: list[str],
        last_exception: Exception | None,
        partial_preserver: Callable[[Any, Exception], Any] | None,
    ) -> AgentExecutionResult:
        partial = None
        if partial_preserver and last_exception:
            try:
                partial = partial_preserver(payload, last_exception)
                stages.append("partial_result_preserved")
            except Exception as pe:
                logger.error(f"Partial preserver failed: {pe}")

        # 1. Try alternative agent route
        alt_agent = self._fallbacks.get(agent_name)
        if alt_agent and alt_agent in self._agents:
            stages.append(f"routed_to_alternative_{alt_agent}")
            logger.info(f"Routing failed agent '{agent_name}' to alternative agent '{alt_agent}'.")
            try:
                alt_handler = self._agents[alt_agent]
                res = await alt_handler(payload)
                self.record_success(alt_agent)
                return AgentExecutionResult(
                    success=True,
                    data=res,
                    partial_result=partial,
                    stages_executed=stages,
                    fallback_used=True,
                    duration_ms=(time.time() - start_time) * 1000,
                )
            except Exception as ae:
                logger.error(f"Alternative agent '{alt_agent}' also failed: {ae}")

        # 2. Try deterministic / rule-based fallback
        det_fallback = self._deterministic_fallbacks.get(agent_name)
        if det_fallback:
            stages.append("deterministic_fallback_used")
            logger.info(f"Falling back to deterministic method for agent '{agent_name}'.")
            try:
                if asyncio.iscoroutinefunction(det_fallback):
                    res = await det_fallback(payload)
                else:
                    res = det_fallback(payload)
                return AgentExecutionResult(
                    success=True,
                    data=res,
                    partial_result=partial,
                    stages_executed=stages,
                    fallback_used=True,
                    duration_ms=(time.time() - start_time) * 1000,
                )
            except Exception as de:
                logger.error(f"Deterministic fallback failed: {de}")

        # 3. Graceful degradation failure
        stages.append("graceful_degradation_failed")
        err_msg = str(last_exception) if last_exception else "Unknown failure"
        return AgentExecutionResult(
            success=False,
            error=err_msg,
            partial_result=partial,
            stages_executed=stages,
            fallback_used=True,
            duration_ms=(time.time() - start_time) * 1000,
        )


# Global singleton registry instance
default_agent_registry = AgentRegistry()
