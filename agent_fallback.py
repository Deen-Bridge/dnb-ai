"""Agent Fallback and Recovery Mechanisms (#225, #884).

Provides robust resilience patterns and auditable fallback decisions for agentic workflows:
- Circuit Breaker pattern to prevent cascading failures
- Exponential backoff retry logic with jitter
- Health check and monitoring for agent types
- Alternative agent routing and graceful degradation
- Auditable fallback reason codes and privacy-safe telemetry (#884)
- Truthful user-facing messaging tailored to risk levels (#884)
- Non-bypassable controls for safety, citation, auth, and cost (#884)
- Disabling thresholds and operator review/rollback controls (#884)
- Partial result preservation and caching
- Deterministic / rule-based fallbacks
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import telemetry

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class AgentHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class FallbackReason(str, Enum):
    """Stable reason codes for external-provider and agent fallback decisions (#884)."""

    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_HALF_OPEN_FAILED = "circuit_half_open_failed"
    SAFETY_BLOCK = "safety_block"
    STALE_DATA = "stale_data"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"
    LOCAL_DATA_FALLBACK = "local_data_fallback"
    HANDLER_FAILURE = "handler_failure"


class FallbackRiskLevel(str, Enum):
    """Risk levels classifying the severity of answer degradation (#884)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FallbackReviewStatus(str, Enum):
    """Review and operational status of a fallback path (#884)."""

    HEALTHY = "healthy"
    REVIEW_REQUIRED = "review_required"
    DISABLED = "disabled"
    ROLLBACK = "rollback"


# ---------------------------------------------------------------------------
# Exceptions (#884)
# ---------------------------------------------------------------------------
class SafetyBlockError(Exception):
    """Raised when an agent or query is blocked by safety policy."""


class SafetyBypassError(Exception):
    """Raised when a fallback execution attempts to bypass content safety controls."""


class ControlBypassError(Exception):
    """Raised when a fallback execution violates citation, auth, or cost controls."""


class FallbackDisabledError(Exception):
    """Raised when a fallback execution is requested but disabled by policy or operator."""


# ---------------------------------------------------------------------------
# Truthful User Messaging (#884)
# ---------------------------------------------------------------------------
DEFAULT_USER_MESSAGES: dict[FallbackReason, str] = {
    FallbackReason.PROVIDER_TIMEOUT: "The primary provider timed out. An alternative response was generated.",
    FallbackReason.PROVIDER_UNAVAILABLE: "The primary service is currently unavailable. A fallback system handled this request.",
    FallbackReason.PROVIDER_RATE_LIMIT: "Service capacity limit reached. An alternative model was used.",
    FallbackReason.CIRCUIT_OPEN: "The service is experiencing connectivity issues with the primary provider. A degraded fallback response was generated.",
    FallbackReason.CIRCUIT_HALF_OPEN_FAILED: "Provider recovery check failed. Degraded fallback response provided.",
    FallbackReason.SAFETY_BLOCK: "This response could not be generated due to safety policy constraints.",
    FallbackReason.STALE_DATA: "This response is served from cached reference data and may not reflect recent updates.",
    FallbackReason.DETERMINISTIC_FALLBACK: "A rule-based deterministic response was provided due to temporary service degradation.",
    FallbackReason.LOCAL_DATA_FALLBACK: "This response was answered using local reference records.",
    FallbackReason.HANDLER_FAILURE: "An unexpected processing error occurred. A fallback response was generated.",
}

DEFAULT_RISK_LEVELS: dict[FallbackReason, FallbackRiskLevel] = {
    FallbackReason.LOCAL_DATA_FALLBACK: FallbackRiskLevel.LOW,
    FallbackReason.STALE_DATA: FallbackRiskLevel.LOW,
    FallbackReason.PROVIDER_TIMEOUT: FallbackRiskLevel.MEDIUM,
    FallbackReason.PROVIDER_RATE_LIMIT: FallbackRiskLevel.MEDIUM,
    FallbackReason.DETERMINISTIC_FALLBACK: FallbackRiskLevel.MEDIUM,
    FallbackReason.CIRCUIT_OPEN: FallbackRiskLevel.HIGH,
    FallbackReason.CIRCUIT_HALF_OPEN_FAILED: FallbackRiskLevel.HIGH,
    FallbackReason.PROVIDER_UNAVAILABLE: FallbackRiskLevel.HIGH,
    FallbackReason.SAFETY_BLOCK: FallbackRiskLevel.HIGH,
    FallbackReason.HANDLER_FAILURE: FallbackRiskLevel.HIGH,
}


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


@dataclass
class FallbackControlPolicy:
    """Policy governing whether safety, citations, auth, and cost must be enforced on fallbacks (#884)."""

    enforce_safety: bool = True
    enforce_citations: bool = True
    enforce_auth: bool = True
    enforce_cost: bool = True
    max_cost_usd: float | None = None


@dataclass
class FallbackThresholdConfig:
    """Threshold configuration for review triggers and auto-disabling fallbacks (#884)."""

    max_fallback_rate: float = 0.5
    min_requests_for_threshold: int = 5
    auto_disable_on_threshold: bool = False


@dataclass(frozen=True)
class FallbackMetadata:
    """Audit metadata emitted by any fallback path without exposing prompts or secrets (#884)."""

    reason: FallbackReason | str
    component: str
    version: str = "1.0.0"
    duration_ms: float = 0.0
    risk_level: FallbackRiskLevel | str = FallbackRiskLevel.MEDIUM
    user_message: str = ""
    fallback_type: str = "none"
    target: str | None = None
    audit_timestamp: float = field(default_factory=time.time)
    trace_id: str | None = None
    operator_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """A content-free dictionary safe for client responses and operator audit."""
        reason_val = self.reason.value if isinstance(self.reason, Enum) else str(self.reason)
        risk_val = self.risk_level.value if isinstance(self.risk_level, Enum) else str(self.risk_level)
        return {
            "reason": reason_val,
            "component": self.component,
            "version": self.version,
            "duration_ms": round(self.duration_ms, 2),
            "risk_level": risk_val,
            "user_message": self.user_message,
            "fallback_type": self.fallback_type,
            "target": self.target or "",
            "trace_id": self.trace_id or "-",
            "audit_timestamp": self.audit_timestamp,
            "operator_detail": self.operator_detail or "",
        }


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
    # Auditable metadata fields (#884)
    fallback_reason: FallbackReason | str | None = None
    fallback_metadata: FallbackMetadata | None = None
    component: str | None = None
    version: str = "1.0.0"
    user_message: str | None = None


def _determine_fallback_reason(
    stages: list[str],
    last_exception: Exception | None,
    was_half_open: bool = False,
    is_deterministic: bool = False,
) -> FallbackReason:
    """Map execution circumstances to a stable FallbackReason code (#884)."""
    if was_half_open:
        return FallbackReason.CIRCUIT_HALF_OPEN_FAILED
    if "circuit_breaker_open" in stages:
        return FallbackReason.CIRCUIT_OPEN
    if last_exception is not None:
        if isinstance(last_exception, (SafetyBlockError, SafetyBypassError)):
            return FallbackReason.SAFETY_BLOCK
        exc_name = type(last_exception).__name__.lower()
        exc_str = str(last_exception).lower()
        if "safety" in exc_name or "safety" in exc_str:
            return FallbackReason.SAFETY_BLOCK
        if (
            isinstance(last_exception, (TimeoutError, asyncio.TimeoutError))
            or "timeout" in exc_name
            or "timeout" in exc_str
            or "deadline" in exc_str
        ):
            return FallbackReason.PROVIDER_TIMEOUT
        if "resourceexhausted" in exc_name or ("rate" in exc_str and "limit" in exc_str) or "quota" in exc_str:
            return FallbackReason.PROVIDER_RATE_LIMIT
        if "serviceunavailable" in exc_name or "connection" in exc_str or "unavailable" in exc_str:
            return FallbackReason.PROVIDER_UNAVAILABLE
    if is_deterministic:
        return FallbackReason.DETERMINISTIC_FALLBACK
    return FallbackReason.HANDLER_FAILURE


class AgentRegistry:
    """Registry for agents, fallback routes, health status, and failure metrics (#225, #884)."""

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

        # Auditable fallback enhancements (#884)
        self._providers: dict[str, str] = {}
        self._features: dict[str, str] = {}
        self._versions: dict[str, str] = {}
        self._control_policies: dict[str, FallbackControlPolicy] = {}
        self._output_safety_validators: dict[str, Callable[[Any], bool]] = {}
        self._citation_validators: dict[str, Callable[[Any], bool]] = {}
        self._auth_validators: dict[str, Callable[[Any], bool]] = {}
        self._disabled_fallbacks: set[str] = set()
        self._disabled_reasons: dict[str, str] = {}
        self._fallback_counts: dict[str, int] = {}
        self._total_calls: dict[str, int] = {}
        self._threshold_configs: dict[str, FallbackThresholdConfig] = {}
        self.default_threshold_config = FallbackThresholdConfig()

    def register_agent(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        fallback_agent: str | None = None,
        deterministic_fallback: Callable[..., Any] | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
        provider: str = "primary_llm",
        feature: str = "general",
        version: str = "1.0.0",
        control_policy: FallbackControlPolicy | None = None,
        output_safety_validator: Callable[[Any], bool] | None = None,
        citation_validator: Callable[[Any], bool] | None = None,
        auth_validator: Callable[[Any], bool] | None = None,
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
        self._providers[name] = provider
        self._features[name] = feature
        self._versions[name] = version
        self._control_policies[name] = control_policy or FallbackControlPolicy()
        if output_safety_validator:
            self._output_safety_validators[name] = output_safety_validator
        if citation_validator:
            self._citation_validators[name] = citation_validator
        if auth_validator:
            self._auth_validators[name] = auth_validator
        self._fallback_counts[name] = 0
        self._total_calls[name] = 0

    def get_health(self, name: str) -> AgentHealth:
        cb = self._circuit_breakers.get(name)
        if cb and cb.state == CircuitState.OPEN:
            return AgentHealth.UNHEALTHY
        fails = self._failure_counts.get(name, 0)
        succs = self._success_counts.get(name, 1)
        # Check `fails >= 2` first: a freshly registered agent has 0 failures and 0 successes,
        # and dividing 0 / (0 + 0) would raise ZeroDivisionError.
        if fails >= 2 and fails / (fails + succs) > 0.3:
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

    # -----------------------------------------------------------------------
    # Fallback review, threshold, and rollback controls (#884)
    # -----------------------------------------------------------------------
    def disable_fallback(self, name_or_all: str = "*", reason: str = "operator_rollback") -> None:
        """Disable fallback routing globally or for a specific agent."""
        self._disabled_fallbacks.add(name_or_all)
        self._disabled_reasons[name_or_all] = reason
        logger.warning("Fallback routing disabled for '%s': %s", name_or_all, reason)

    def enable_fallback(self, name_or_all: str = "*") -> None:
        """Re-enable fallback routing globally or for a specific agent."""
        self._disabled_fallbacks.discard(name_or_all)
        self._disabled_reasons.pop(name_or_all, None)
        logger.info("Fallback routing re-enabled for '%s'", name_or_all)

    def is_fallback_enabled(self, agent_name: str) -> bool:
        """Check whether fallback routing is active for the given agent."""
        if "*" in self._disabled_fallbacks or agent_name in self._disabled_fallbacks:
            return False
        return True

    def rollback_fallback(self, agent_name: str, reason: str = "quality_regression") -> None:
        """Operator rollback action: instantly disable broken fallback path."""
        self.disable_fallback(agent_name, reason=reason)

    def set_threshold_config(self, agent_name: str, config: FallbackThresholdConfig) -> None:
        self._threshold_configs[agent_name] = config

    def get_fallback_status(self, agent_name: str) -> dict[str, Any]:
        """Return operational and review status for an agent's fallbacks (#884)."""
        calls = self._total_calls.get(agent_name, 0)
        fallbacks = self._fallback_counts.get(agent_name, 0)
        rate = round(fallbacks / calls, 4) if calls else 0.0
        cfg = self._threshold_configs.get(agent_name, self.default_threshold_config)
        is_disabled = not self.is_fallback_enabled(agent_name)

        if is_disabled:
            status = FallbackReviewStatus.DISABLED
        elif calls >= cfg.min_requests_for_threshold and rate > cfg.max_fallback_rate:
            status = FallbackReviewStatus.REVIEW_REQUIRED
        else:
            status = FallbackReviewStatus.HEALTHY

        return {
            "agent_name": agent_name,
            "total_calls": calls,
            "fallback_count": fallbacks,
            "fallback_rate": rate,
            "threshold": cfg.max_fallback_rate,
            "status": status.value,
            "fallback_enabled": not is_disabled,
            "disabled_reason": self._disabled_reasons.get(agent_name) or self._disabled_reasons.get("*") or "",
        }

    def _record_fallback_metric(self, agent_name: str) -> None:
        """Track fallback counts and check auto-disabling thresholds."""
        self._fallback_counts[agent_name] = self._fallback_counts.get(agent_name, 0) + 1
        cfg = self._threshold_configs.get(agent_name, self.default_threshold_config)
        calls = self._total_calls.get(agent_name, 0)
        fallbacks = self._fallback_counts.get(agent_name, 0)
        if cfg.auto_disable_on_threshold and calls >= cfg.min_requests_for_threshold:
            rate = fallbacks / calls
            if rate > cfg.max_fallback_rate:
                self.disable_fallback(
                    agent_name,
                    reason=f"Auto-disabled: fallback rate {rate:.2f} exceeded threshold {cfg.max_fallback_rate:.2f}",
                )

    async def execute_with_resilience(
        self,
        agent_name: str,
        payload: Any,
        retry_config: RetryConfig | None = None,
        cache_key: str | None = None,
        partial_preserver: Callable[[Any, Exception], Any] | None = None,
        context: dict[str, Any] | None = None,
        trace_id: str | None = None,
        output_safety_validator: Callable[[Any], bool] | None = None,
        citation_validator: Callable[[Any], bool] | None = None,
        auth_validator: Callable[[Any], bool] | None = None,
    ) -> AgentExecutionResult:
        start_time = time.time()
        stages: list[str] = []
        retry_cfg = retry_config or RetryConfig()
        ctx = context or {}
        self._total_calls[agent_name] = self._total_calls.get(agent_name, 0) + 1

        policy = self._control_policies.get(agent_name, FallbackControlPolicy())
        version = self._versions.get(agent_name, "1.0.0")

        # -------------------------------------------------------------------
        # 1. Non-bypassable Pre-Execution Controls: Auth, Cost, Safety (#884)
        # -------------------------------------------------------------------
        if policy.enforce_auth:
            active_auth_check = auth_validator or self._auth_validators.get(agent_name)
            if active_auth_check and not active_auth_check(payload if ctx is None else {**ctx, "payload": payload}):
                raise ControlBypassError(f"Authentication required for '{agent_name}'; fallbacks cannot bypass auth.")

        if policy.enforce_cost and policy.max_cost_usd is not None:
            cost_usd = ctx.get("cost_usd", 0.0)
            if cost_usd > policy.max_cost_usd:
                raise ControlBypassError(
                    f"Cost limit ${policy.max_cost_usd} exceeded (${cost_usd}); fallbacks cannot bypass cost controls."
                )

        if policy.enforce_safety and (ctx.get("safety_blocked") or ctx.get("safety_violation")):
            stages.append("safety_blocked_pre_check")
            reason = FallbackReason.SAFETY_BLOCK
            self._record_fallback_metric(agent_name)
            user_msg = DEFAULT_USER_MESSAGES[reason]
            meta = FallbackMetadata(
                reason=reason,
                component=agent_name,
                version=version,
                duration_ms=(time.time() - start_time) * 1000,
                risk_level=FallbackRiskLevel.HIGH,
                user_message=user_msg,
                fallback_type="safety_terminal",
                trace_id=trace_id,
                operator_detail="Input flagged as safety policy violation; fallbacks cannot bypass safety",
            )
            telemetry.record_fallback_event(
                provider=self._providers.get(agent_name, "primary_llm"),
                feature=self._features.get(agent_name, "general"),
                reason=reason.value,
                component=agent_name,
                version=version,
                duration_ms=meta.duration_ms,
                risk_level=FallbackRiskLevel.HIGH.value,
                trace_id=trace_id,
            )
            return AgentExecutionResult(
                success=False,
                error="Request blocked by safety policy",
                stages_executed=stages,
                fallback_used=True,
                fallback_reason=reason,
                fallback_metadata=meta,
                component=agent_name,
                version=version,
                user_message=user_msg,
                duration_ms=meta.duration_ms,
            )

        # -------------------------------------------------------------------
        # 2. Caching
        # -------------------------------------------------------------------
        if cache_key:
            cached = self.get_cached_result(cache_key)
            if cached is not None:
                stages.append("cache_hit")
                return AgentExecutionResult(
                    success=True,
                    data=cached,
                    stages_executed=stages,
                    duration_ms=(time.time() - start_time) * 1000,
                    component=agent_name,
                    version=version,
                )

        # -------------------------------------------------------------------
        # 3. Circuit Breaker Evaluation
        # -------------------------------------------------------------------
        cb = self._circuit_breakers.get(agent_name)
        was_half_open = False
        if cb:
            if not cb.allow_request():
                stages.append("circuit_breaker_open")
                logger.warning(f"Circuit breaker for '{agent_name}' is OPEN. Attempting fallback routing.")
                return await self._execute_fallback_chain(
                    agent_name=agent_name,
                    payload=payload,
                    start_time=start_time,
                    stages=stages,
                    last_exception=Exception("Circuit breaker is open"),
                    partial_preserver=partial_preserver,
                    was_half_open=False,
                    trace_id=trace_id,
                    output_safety_validator=output_safety_validator,
                    citation_validator=citation_validator,
                )
            if cb.state == CircuitState.HALF_OPEN:
                was_half_open = True

        handler = self._agents.get(agent_name)
        if not handler:
            return AgentExecutionResult(
                success=False,
                error=f"Agent '{agent_name}' not found",
                stages_executed=stages,
                duration_ms=(time.time() - start_time) * 1000,
                component=agent_name,
                version=version,
            )

        # -------------------------------------------------------------------
        # 4. Handler Execution & Retries
        # -------------------------------------------------------------------
        last_exception: Exception | None = None
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
                    component=agent_name,
                    version=version,
                )
            except Exception as e:
                last_exception = e
                logger.warning(f"Agent '{agent_name}' attempt {attempt} failed: {e}")

                # If this is a terminal safety block, do not retry and do not bypass!
                if isinstance(e, (SafetyBlockError, SafetyBypassError)) or "safety" in type(e).__name__.lower():
                    break

                if attempt < retry_cfg.max_retries:
                    delay = retry_cfg.initial_delay * (retry_cfg.backoff_factor**attempt)
                    delay = min(delay, retry_cfg.max_delay)
                    if retry_cfg.jitter:
                        delay *= 0.5 + random.random()
                    await asyncio.sleep(delay)

        self.record_failure(agent_name)
        return await self._execute_fallback_chain(
            agent_name=agent_name,
            payload=payload,
            start_time=start_time,
            stages=stages,
            last_exception=last_exception,
            partial_preserver=partial_preserver,
            was_half_open=was_half_open,
            trace_id=trace_id,
            output_safety_validator=output_safety_validator,
            citation_validator=citation_validator,
        )

    async def _execute_fallback_chain(
        self,
        agent_name: str,
        payload: Any,
        start_time: float,
        stages: list[str],
        last_exception: Exception | None,
        partial_preserver: Callable[[Any, Exception], Any] | None,
        was_half_open: bool = False,
        trace_id: str | None = None,
        output_safety_validator: Callable[[Any], bool] | None = None,
        citation_validator: Callable[[Any], bool] | None = None,
    ) -> AgentExecutionResult:
        version = self._versions.get(agent_name, "1.0.0")
        provider = self._providers.get(agent_name, "primary_llm")
        feature = self._features.get(agent_name, "general")
        policy = self._control_policies.get(agent_name, FallbackControlPolicy())

        partial = None
        if partial_preserver and last_exception:
            try:
                partial = partial_preserver(payload, last_exception)
                stages.append("partial_result_preserved")
            except Exception as pe:
                logger.error(f"Partial preserver failed: {pe}")

        # Determine primary fallback reason
        determined_reason = _determine_fallback_reason(
            stages=stages,
            last_exception=last_exception,
            was_half_open=was_half_open,
        )

        # -------------------------------------------------------------------
        # Terminal Safety Policy: Fallbacks cannot bypass safety refusals (#884)
        # -------------------------------------------------------------------
        if policy.enforce_safety and determined_reason == FallbackReason.SAFETY_BLOCK:
            stages.append("safety_block_terminal")
            self._record_fallback_metric(agent_name)
            user_msg = DEFAULT_USER_MESSAGES[FallbackReason.SAFETY_BLOCK]
            meta = FallbackMetadata(
                reason=FallbackReason.SAFETY_BLOCK,
                component=agent_name,
                version=version,
                duration_ms=(time.time() - start_time) * 1000,
                risk_level=FallbackRiskLevel.HIGH,
                user_message=user_msg,
                fallback_type="safety_terminal",
                trace_id=trace_id,
                operator_detail="Primary request triggered safety block; fallbacks cannot bypass safety",
            )
            telemetry.record_fallback_event(
                provider=provider,
                feature=feature,
                reason=FallbackReason.SAFETY_BLOCK.value,
                component=agent_name,
                version=version,
                duration_ms=meta.duration_ms,
                risk_level=FallbackRiskLevel.HIGH.value,
                trace_id=trace_id,
            )
            return AgentExecutionResult(
                success=False,
                error=str(last_exception) if last_exception else "Request blocked by safety policy",
                partial_result=partial,
                stages_executed=stages,
                fallback_used=True,
                fallback_reason=FallbackReason.SAFETY_BLOCK,
                fallback_metadata=meta,
                component=agent_name,
                version=version,
                user_message=user_msg,
                duration_ms=(time.time() - start_time) * 1000,
            )

        # -------------------------------------------------------------------
        # Check if fallbacks are disabled by operator or policy (#884)
        # -------------------------------------------------------------------
        if not self.is_fallback_enabled(agent_name):
            stages.append("fallback_disabled")
            reason_disabled = (
                self._disabled_reasons.get(agent_name) or self._disabled_reasons.get("*") or "operator_disabled"
            )
            err_msg = f"Fallback disabled for '{agent_name}': {reason_disabled}"
            logger.warning(err_msg)
            return AgentExecutionResult(
                success=False,
                error=err_msg,
                partial_result=partial,
                stages_executed=stages,
                fallback_used=False,
                fallback_reason=determined_reason,
                component=agent_name,
                version=version,
                user_message=DEFAULT_USER_MESSAGES.get(determined_reason, ""),
                duration_ms=(time.time() - start_time) * 1000,
            )

        self._record_fallback_metric(agent_name)
        effective_output_safety = output_safety_validator or self._output_safety_validators.get(agent_name)
        effective_citation_check = citation_validator or self._citation_validators.get(agent_name)

        # -------------------------------------------------------------------
        # 1. Alternative Agent Route
        # -------------------------------------------------------------------
        alt_agent = self._fallbacks.get(agent_name)
        if alt_agent and alt_agent in self._agents:
            stages.append(f"routed_to_alternative_{alt_agent}")
            logger.info(f"Routing failed agent '{agent_name}' to alternative agent '{alt_agent}'.")
            try:
                alt_handler = self._agents[alt_agent]
                res = await alt_handler(payload)

                # Validate output safety on alternative agent result (#884)
                if policy.enforce_safety and effective_output_safety and not effective_output_safety(res):
                    stages.append("alternative_agent_output_safety_violation")
                    raise SafetyBypassError(f"Alternative agent '{alt_agent}' output violated safety policy.")

                # Validate citations on alternative agent result (#884)
                if policy.enforce_citations and effective_citation_check and not effective_citation_check(res):
                    stages.append("alternative_agent_citation_violation")
                    raise ControlBypassError(f"Alternative agent '{alt_agent}' output lacks required citations.")

                self.record_success(alt_agent)
                duration_ms = (time.time() - start_time) * 1000
                risk = DEFAULT_RISK_LEVELS.get(determined_reason, FallbackRiskLevel.MEDIUM)
                user_msg = DEFAULT_USER_MESSAGES.get(determined_reason, "")
                meta = FallbackMetadata(
                    reason=determined_reason,
                    component=agent_name,
                    version=version,
                    duration_ms=duration_ms,
                    risk_level=risk,
                    user_message=user_msg,
                    fallback_type="alternative_agent",
                    target=alt_agent,
                    trace_id=trace_id,
                )
                telemetry.record_fallback_event(
                    provider=provider,
                    feature=feature,
                    reason=determined_reason.value,
                    component=agent_name,
                    version=version,
                    duration_ms=duration_ms,
                    risk_level=risk.value,
                    trace_id=trace_id,
                )
                return AgentExecutionResult(
                    success=True,
                    data=res,
                    partial_result=partial,
                    stages_executed=stages,
                    fallback_used=True,
                    duration_ms=duration_ms,
                    fallback_reason=determined_reason,
                    fallback_metadata=meta,
                    component=agent_name,
                    version=version,
                    user_message=user_msg,
                )
            except (SafetyBypassError, ControlBypassError):
                raise
            except Exception as ae:
                logger.error(f"Alternative agent '{alt_agent}' also failed: {ae}")

        # -------------------------------------------------------------------
        # 2. Deterministic / Rule-based Fallback
        # -------------------------------------------------------------------
        det_fallback = self._deterministic_fallbacks.get(agent_name)
        if det_fallback:
            stages.append("deterministic_fallback_used")
            logger.info(f"Falling back to deterministic method for agent '{agent_name}'.")
            try:
                if inspect.iscoroutinefunction(det_fallback):
                    res = await det_fallback(payload)
                else:
                    res = det_fallback(payload)

                # Validate output safety on deterministic result (#884)
                if policy.enforce_safety and effective_output_safety and not effective_output_safety(res):
                    stages.append("deterministic_output_safety_violation")
                    raise SafetyBypassError("Deterministic fallback output violated safety policy.")

                # Validate citations on deterministic result (#884)
                if policy.enforce_citations and effective_citation_check and not effective_citation_check(res):
                    stages.append("deterministic_citation_violation")
                    raise ControlBypassError("Deterministic fallback output lacks required citations.")

                duration_ms = (time.time() - start_time) * 1000
                det_reason = _determine_fallback_reason(
                    stages=stages,
                    last_exception=last_exception,
                    was_half_open=was_half_open,
                    is_deterministic=True,
                )
                risk = DEFAULT_RISK_LEVELS.get(det_reason, FallbackRiskLevel.MEDIUM)
                user_msg = DEFAULT_USER_MESSAGES.get(det_reason, "")
                meta = FallbackMetadata(
                    reason=det_reason,
                    component=agent_name,
                    version=version,
                    duration_ms=duration_ms,
                    risk_level=risk,
                    user_message=user_msg,
                    fallback_type="deterministic",
                    target="deterministic_rule",
                    trace_id=trace_id,
                )
                telemetry.record_fallback_event(
                    provider=provider,
                    feature=feature,
                    reason=det_reason.value,
                    component=agent_name,
                    version=version,
                    duration_ms=duration_ms,
                    risk_level=risk.value,
                    trace_id=trace_id,
                )
                return AgentExecutionResult(
                    success=True,
                    data=res,
                    partial_result=partial,
                    stages_executed=stages,
                    fallback_used=True,
                    duration_ms=duration_ms,
                    fallback_reason=det_reason,
                    fallback_metadata=meta,
                    component=agent_name,
                    version=version,
                    user_message=user_msg,
                )
            except (SafetyBypassError, ControlBypassError):
                raise
            except Exception as de:
                logger.error(f"Deterministic fallback failed: {de}")

        # -------------------------------------------------------------------
        # 3. Graceful Degradation Failure
        # -------------------------------------------------------------------
        stages.append("graceful_degradation_failed")
        err_msg = str(last_exception) if last_exception else "Unknown failure"
        duration_ms = (time.time() - start_time) * 1000
        risk = DEFAULT_RISK_LEVELS.get(determined_reason, FallbackRiskLevel.HIGH)
        user_msg = DEFAULT_USER_MESSAGES.get(determined_reason, "")
        meta = FallbackMetadata(
            reason=determined_reason,
            component=agent_name,
            version=version,
            duration_ms=duration_ms,
            risk_level=risk,
            user_message=user_msg,
            fallback_type="graceful_degradation_failed",
            trace_id=trace_id,
            operator_detail=err_msg,
        )
        telemetry.record_fallback_event(
            provider=provider,
            feature=feature,
            reason=determined_reason.value,
            component=agent_name,
            version=version,
            duration_ms=duration_ms,
            risk_level=risk.value,
            trace_id=trace_id,
        )
        return AgentExecutionResult(
            success=False,
            error=err_msg,
            partial_result=partial,
            stages_executed=stages,
            fallback_used=True,
            duration_ms=duration_ms,
            fallback_reason=determined_reason,
            fallback_metadata=meta,
            component=agent_name,
            version=version,
            user_message=user_msg,
        )


# Global singleton registry instance
default_agent_registry = AgentRegistry()
