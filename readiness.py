"""Production health, readiness, and dependency degradation management (AI-094 / Issue #870).

This module defines the explicit dependency criticality registry, safe and bounded
readiness probes with single-flight protection and caching, fallback tracking, and
startup configuration validation.

Design Invariants:
1. Liveness (GET /health) is strictly constant-time and makes NO external network calls.
2. Readiness (GET /ready) reflects whether the instance can serve production traffic.
   - Status 'ready' (200): All required dependencies healthy, no active fallbacks.
   - Status 'degraded' (200): Required dependencies healthy, but an optional dependency is
     down or a fallback is active (e.g. local embeddings, single-instance in-memory buffers).
   - Status 'not_ready' (503): A required dependency or required configuration is missing/failing.
3. Probes are bounded: hard timeout per probe, single-flight concurrency lock, and cached
   results with short TTL so frequent probes never stampede dependencies.
4. Zero token spend: Probes NEVER call paid LLM generation endpoints; provider health is
   derived passively from real request outcomes or lightweight metadata.
5. Zero secret/PII leakage: Diagnostics use machine-readable reason codes and coarse counters;
   prompts, secrets, connection strings with credentials, and raw exception text are never surfaced.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Taxonomy
# ---------------------------------------------------------------------------


class DependencyCriticality(str, Enum):
    """Criticality level of an external or platform dependency."""

    REQUIRED = "required"
    OPTIONAL = "optional"


class HealthState(str, Enum):
    """Aggregated health state of the service."""

    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"


class ReasonCode(str, Enum):
    """Canonical, machine-readable reason codes."""

    OK = "ok"
    CONFIG_VALID = "config_valid"
    CONFIG_MISSING = "config_missing"
    PROVIDER_OK = "provider_ok"
    PROVIDER_OUTAGE = "provider_outage"
    PROVIDER_DEGRADED = "provider_degraded"
    ALL_PROVIDERS_UNAVAILABLE = "all_providers_unavailable"
    SECONDARY_PROVIDER_DEGRADED = "secondary_provider_degraded"
    REDIS_OK = "redis_ok"
    REDIS_REQUIRED_UNREACHABLE = "redis_required_unreachable"
    REDIS_DEGRADED_IN_MEMORY = "redis_degraded_in_memory"
    EMBEDDING_OK = "embedding_ok"
    EMBEDDING_DEGRADED_LOCAL = "embedding_degraded_local"
    QUEUE_OVERFLOW = "queue_overflow"
    PROBE_TIMEOUT = "probe_timeout"
    PROBE_FAILED = "probe_failed"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DependencyStatus(BaseModel):
    """Health evaluation of a single registered dependency."""

    name: str = Field(..., description="Unique dependency identifier.")
    criticality: DependencyCriticality = Field(..., description="Required vs optional.")
    status: str = Field(..., description="'ok', 'degraded', or 'fail'.")
    reason: str = Field(..., description="Canonical machine-readable reason code.")
    latency_ms: float | None = Field(None, description="Probe latency in milliseconds if probed.")


class FallbackStats(BaseModel):
    """Active and historic fallback counters."""

    active_count: int = Field(0, description="Number of currently active fallbacks.")
    local_embedding: bool = Field(False, description="True if semantic cache is using local embeddings.")
    redis_in_memory: bool = Field(False, description="True if Redis fallback to in-memory store is active.")
    review_queue_overflow_drops: int = Field(0, description="Items dropped due to in-memory queue overflow.")
    total_fallbacks: int = Field(0, description="Total fallback events recorded.")


class ReadinessResponse(BaseModel):
    """Payload returned by GET /ready."""

    status: str = Field(..., description="'ready' (200), 'degraded' (200), or 'not_ready' (503).")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp.")
    version: str | None = Field(None, description="Application version.")
    mode: str = Field(..., description="'single-instance' or 'multi-instance'.")
    reasons: list[str] = Field(default_factory=list, description="Reason codes describing active conditions.")
    dependencies: dict[str, DependencyStatus] = Field(
        default_factory=dict, description="Per-dependency health evaluation."
    )
    fallbacks: FallbackStats = Field(default_factory=FallbackStats, description="Diagnostics on fallback mechanisms.")


# ---------------------------------------------------------------------------
# Request Context for Fallback Marking
# ---------------------------------------------------------------------------

_current_degraded_reasons: ContextVar[list[str] | None] = ContextVar("current_degraded_reasons", default=None)


def mark_degraded(reason: str) -> None:
    """Mark the current request as having traversed a degraded fallback path.

    This causes the additive response header `X-Degraded: <reason>` to be emitted.
    """
    reasons = _current_degraded_reasons.get()
    if reasons is not None and reason not in reasons:
        reasons.append(reason)


def get_current_degraded_reasons() -> list[str]:
    """Retrieve the reasons recorded during the current request."""
    reasons = _current_degraded_reasons.get()
    return list(reasons) if reasons is not None else []


# ---------------------------------------------------------------------------
# Registry and Probe Engine
# ---------------------------------------------------------------------------


class ReadinessRegistry:
    """Central registry and executor for dependency readiness probes."""

    def __init__(
        self,
        probe_timeout_seconds: float = 1.0,
        probe_cache_ttl_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.probe_timeout_seconds = probe_timeout_seconds
        self.probe_cache_ttl_seconds = probe_cache_ttl_seconds
        self._clock = clock

        # Cache: name -> (DependencyStatus, timestamp)
        self._cache: dict[str, tuple[DependencyStatus, float]] = {}
        # Concurrency locks: name -> asyncio.Lock
        self._locks: dict[str, asyncio.Lock] = {}

        # LLM Passive health tracking (sliding window of recent request outcomes)
        self._llm_outcomes: deque[bool] = deque(maxlen=30)
        self._mock_provider_health: dict[str, bool | None] = {}

        # Active fallback states
        self._embedding_fallback_active: bool = False
        self._embedding_fallback_count: int = 0
        self._redis_fallback_active: bool = False
        self._review_queue_overflow_drops: int = 0

    def set_clock(self, clock: Callable[[], float]) -> None:
        """Inject a custom clock function for deterministic time-controlled tests."""
        self._clock = clock

    def record_llm_outcome(self, success: bool, provider: str = "primary") -> None:
        """Record the outcome of a real LLM request to track passive health without paid probes."""
        self._llm_outcomes.append(success)

    def set_mock_provider_health(self, provider: str, healthy: bool | None) -> None:
        """Force a mock provider health status for test failure injection."""
        self._mock_provider_health[provider] = healthy
        # Invalidate cache for provider immediately
        self._cache.pop(provider, None)

    def record_embedding_fallback(self, exc: Exception | None = None) -> None:
        """Record that the embedding provider failed and local fallback was used."""
        self._embedding_fallback_active = True
        self._embedding_fallback_count += 1
        self._cache.pop("embedding", None)

        try:
            import telemetry

            telemetry.record_fallback_event(
                provider="google",
                feature="semantic_cache",
                reason="provider_unavailable",
                component="text-embedding-004",
                version="1.0.0",
                risk_level="medium",
            )
        except Exception:
            pass

    def record_embedding_success(self) -> None:
        """Record that an external embedding call succeeded, clearing active fallback."""
        if self._embedding_fallback_active:
            self._embedding_fallback_active = False
            self._cache.pop("embedding", None)

    def record_redis_fallback(self, active: bool = True) -> None:
        """Record that Redis is unreachable and local fallback is active."""
        self._redis_fallback_active = active
        self._cache.pop("redis", None)

    def record_review_queue_overflow(self) -> None:
        """Increment count of review queue items dropped due to in-memory buffer limit."""
        self._review_queue_overflow_drops += 1

    def reset(self) -> None:
        """Reset all cached states, fallback counters, and mock overrides (for tests)."""
        self._clock = time.monotonic
        self.probe_timeout_seconds = 1.0
        self.probe_cache_ttl_seconds = 5.0
        self._cache.clear()
        self._locks.clear()
        self._llm_outcomes.clear()
        self._mock_provider_health.clear()
        self._embedding_fallback_active = False
        self._embedding_fallback_count = 0
        self._redis_fallback_active = False
        self._review_queue_overflow_drops = 0

    # -- Probe Implementations ------------------------------------------------

    async def _probe_config(self) -> DependencyStatus:
        """Check required configuration settings."""
        from config import get_settings

        settings = get_settings()
        gemini_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")

        is_configured = bool(gemini_key and gemini_key != "your_api_key_here")
        if is_configured:
            return DependencyStatus(
                name="config",
                criticality=DependencyCriticality.REQUIRED,
                status="ok",
                reason=ReasonCode.CONFIG_VALID.value,
            )
        return DependencyStatus(
            name="config",
            criticality=DependencyCriticality.REQUIRED,
            status="fail",
            reason=ReasonCode.CONFIG_MISSING.value,
        )

    async def _probe_primary_llm(self) -> DependencyStatus:
        """Evaluate primary LLM health passively without calling paid generation endpoints."""
        mock_val = self._mock_provider_health.get("primary_llm")
        if mock_val is not None:
            if mock_val:
                return DependencyStatus(
                    name="primary_llm",
                    criticality=DependencyCriticality.REQUIRED,
                    status="ok",
                    reason=ReasonCode.PROVIDER_OK.value,
                )
            # Check if secondary provider is healthy
            sec_val = self._mock_provider_health.get("secondary_llm")
            if sec_val is True:
                return DependencyStatus(
                    name="primary_llm",
                    criticality=DependencyCriticality.REQUIRED,
                    status="degraded",
                    reason=ReasonCode.SECONDARY_PROVIDER_DEGRADED.value,
                )
            return DependencyStatus(
                name="primary_llm",
                criticality=DependencyCriticality.REQUIRED,
                status="fail",
                reason=ReasonCode.ALL_PROVIDERS_UNAVAILABLE.value,
            )

        # Passive health: evaluate recent request outcomes if available
        if self._llm_outcomes:
            # If the last 5 outcomes are all failures, report provider outage
            recent = list(self._llm_outcomes)[-5:]
            if len(recent) >= 3 and not any(recent):
                return DependencyStatus(
                    name="primary_llm",
                    criticality=DependencyCriticality.REQUIRED,
                    status="fail",
                    reason=ReasonCode.PROVIDER_OUTAGE.value,
                )

        return DependencyStatus(
            name="primary_llm",
            criticality=DependencyCriticality.REQUIRED,
            status="ok",
            reason=ReasonCode.PROVIDER_OK.value,
        )

    async def _probe_redis(self) -> DependencyStatus:
        """Probe Redis health with a bounded PING if configured."""
        from config import get_settings

        settings = get_settings()
        require_redis = getattr(settings, "require_redis", False) or (
            os.getenv("REQUIRE_REDIS", "false").lower() in {"1", "true", "yes"}
            or os.getenv("DEPLOYMENT_MODE", "").lower() == "multi-instance"
        )
        criticality = DependencyCriticality.REQUIRED if require_redis else DependencyCriticality.OPTIONAL

        # Check mock override first
        mock_val = self._mock_provider_health.get("redis")
        if mock_val is not None:
            if mock_val:
                return DependencyStatus(
                    name="redis",
                    criticality=criticality,
                    status="ok",
                    reason=ReasonCode.REDIS_OK.value,
                )
            if criticality == DependencyCriticality.REQUIRED:
                return DependencyStatus(
                    name="redis",
                    criticality=criticality,
                    status="fail",
                    reason=ReasonCode.REDIS_REQUIRED_UNREACHABLE.value,
                )
            return DependencyStatus(
                name="redis",
                criticality=criticality,
                status="degraded",
                reason=ReasonCode.REDIS_DEGRADED_IN_MEMORY.value,
            )

        # If fallback is actively flagged by store operations
        if self._redis_fallback_active:
            if criticality == DependencyCriticality.REQUIRED:
                return DependencyStatus(
                    name="redis",
                    criticality=criticality,
                    status="fail",
                    reason=ReasonCode.REDIS_REQUIRED_UNREACHABLE.value,
                )
            return DependencyStatus(
                name="redis",
                criticality=criticality,
                status="degraded",
                reason=ReasonCode.REDIS_DEGRADED_IN_MEMORY.value,
            )

        redis_url = os.getenv("REDIS_URL", "")
        if not redis_url:
            if criticality == DependencyCriticality.REQUIRED:
                return DependencyStatus(
                    name="redis",
                    criticality=criticality,
                    status="fail",
                    reason=ReasonCode.REDIS_REQUIRED_UNREACHABLE.value,
                )
            return DependencyStatus(
                name="redis",
                criticality=criticality,
                status="ok",
                reason="in_memory_mode",
            )

        # Active bounded ping to live Redis
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(redis_url, socket_timeout=self.probe_timeout_seconds)
            try:
                start = self._clock()
                await client.ping()
                latency = round((self._clock() - start) * 1000.0, 2)
                return DependencyStatus(
                    name="redis",
                    criticality=criticality,
                    status="ok",
                    reason=ReasonCode.REDIS_OK.value,
                    latency_ms=latency,
                )
            finally:
                await client.aclose()
        except TimeoutError:
            if criticality == DependencyCriticality.REQUIRED:
                return DependencyStatus(
                    name="redis",
                    criticality=criticality,
                    status="fail",
                    reason=ReasonCode.PROBE_TIMEOUT.value,
                )
            return DependencyStatus(
                name="redis",
                criticality=criticality,
                status="degraded",
                reason=ReasonCode.REDIS_DEGRADED_IN_MEMORY.value,
            )
        except Exception:
            if criticality == DependencyCriticality.REQUIRED:
                return DependencyStatus(
                    name="redis",
                    criticality=criticality,
                    status="fail",
                    reason=ReasonCode.REDIS_REQUIRED_UNREACHABLE.value,
                )
            return DependencyStatus(
                name="redis",
                criticality=criticality,
                status="degraded",
                reason=ReasonCode.REDIS_DEGRADED_IN_MEMORY.value,
            )

    async def _probe_embedding(self) -> DependencyStatus:
        """Check embedding provider health based on active local fallback status."""
        mock_val = self._mock_provider_health.get("embedding")
        if mock_val is not None:
            if mock_val:
                return DependencyStatus(
                    name="embedding",
                    criticality=DependencyCriticality.OPTIONAL,
                    status="ok",
                    reason=ReasonCode.EMBEDDING_OK.value,
                )
            return DependencyStatus(
                name="embedding",
                criticality=DependencyCriticality.OPTIONAL,
                status="degraded",
                reason=ReasonCode.EMBEDDING_DEGRADED_LOCAL.value,
            )

        if self._embedding_fallback_active:
            return DependencyStatus(
                name="embedding",
                criticality=DependencyCriticality.OPTIONAL,
                status="degraded",
                reason=ReasonCode.EMBEDDING_DEGRADED_LOCAL.value,
            )
        return DependencyStatus(
            name="embedding",
            criticality=DependencyCriticality.OPTIONAL,
            status="ok",
            reason=ReasonCode.EMBEDDING_OK.value,
        )

    # -- Safe Single-Flight Execution -----------------------------------------

    async def _check_dependency_cached(
        self,
        name: str,
        probe_coro_fn: Callable[[], Any],
    ) -> DependencyStatus:
        now = self._clock()
        cached = self._cache.get(name)
        if cached is not None and (now - cached[1]) < self.probe_cache_ttl_seconds:
            return cached[0]

        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            # Double-check cache inside lock
            now = self._clock()
            cached = self._cache.get(name)
            if cached is not None and (now - cached[1]) < self.probe_cache_ttl_seconds:
                return cached[0]

            try:
                status = await asyncio.wait_for(
                    probe_coro_fn(),
                    timeout=self.probe_timeout_seconds,
                )
            except TimeoutError:
                status = DependencyStatus(
                    name=name,
                    criticality=DependencyCriticality.REQUIRED
                    if name in {"config", "primary_llm"}
                    else DependencyCriticality.OPTIONAL,
                    status="fail" if name in {"config", "primary_llm"} else "degraded",
                    reason=ReasonCode.PROBE_TIMEOUT.value,
                    latency_ms=self.probe_timeout_seconds * 1000.0,
                )
            except Exception:
                status = DependencyStatus(
                    name=name,
                    criticality=DependencyCriticality.REQUIRED
                    if name in {"config", "primary_llm"}
                    else DependencyCriticality.OPTIONAL,
                    status="fail" if name in {"config", "primary_llm"} else "degraded",
                    reason=ReasonCode.PROBE_FAILED.value,
                )

            self._cache[name] = (status, self._clock())
            return status

    # -- Public Readiness Evaluation ------------------------------------------

    async def get_readiness(self) -> tuple[ReadinessResponse, int]:
        """Evaluate overall service readiness across all registered dependencies."""
        from config import get_settings

        settings = get_settings()
        require_redis = getattr(settings, "require_redis", False) or (
            os.getenv("REQUIRE_REDIS", "false").lower() in {"1", "true", "yes"}
            or os.getenv("DEPLOYMENT_MODE", "").lower() == "multi-instance"
        )
        mode = "multi-instance" if require_redis else "single-instance"

        # Execute all probes safely with caching & single-flight protection
        dep_statuses: dict[str, DependencyStatus] = {}
        dep_statuses["config"] = await self._check_dependency_cached("config", self._probe_config)
        dep_statuses["primary_llm"] = await self._check_dependency_cached("primary_llm", self._probe_primary_llm)
        dep_statuses["redis"] = await self._check_dependency_cached("redis", self._probe_redis)
        dep_statuses["embedding"] = await self._check_dependency_cached("embedding", self._probe_embedding)

        # Aggregate overall status
        not_ready_reasons: list[str] = []
        degraded_reasons: list[str] = []

        for name, dep in dep_statuses.items():
            if dep.criticality == DependencyCriticality.REQUIRED and dep.status == "fail":
                not_ready_reasons.append(f"{name}:{dep.reason}")
            elif dep.status == "degraded" or (
                dep.criticality == DependencyCriticality.OPTIONAL and dep.status == "fail"
            ):
                degraded_reasons.append(f"{name}:{dep.reason}")

        # Review queue overflow check
        if self._review_queue_overflow_drops > 0:
            degraded_reasons.append(ReasonCode.QUEUE_OVERFLOW.value)

        # Classify state
        if not_ready_reasons:
            overall_status = HealthState.NOT_READY.value
            http_status = 503
            reasons = not_ready_reasons + degraded_reasons
        elif degraded_reasons:
            overall_status = HealthState.DEGRADED.value
            http_status = 200
            reasons = degraded_reasons
        else:
            overall_status = HealthState.READY.value
            http_status = 200
            reasons = [ReasonCode.OK.value]

        fallbacks = FallbackStats(
            active_count=len(degraded_reasons),
            local_embedding=self._embedding_fallback_active,
            redis_in_memory=self._redis_fallback_active,
            review_queue_overflow_drops=self._review_queue_overflow_drops,
            total_fallbacks=self._embedding_fallback_count + (1 if self._redis_fallback_active else 0),
        )

        response = ReadinessResponse(
            status=overall_status,
            timestamp=datetime.now(UTC).isoformat(),
            version="0.1.0",
            mode=mode,
            reasons=reasons,
            dependencies=dep_statuses,
            fallbacks=fallbacks,
        )
        return response, http_status


# Global process singleton
readiness_registry = ReadinessRegistry()


# ---------------------------------------------------------------------------
# Startup Configuration Validation
# ---------------------------------------------------------------------------


def validate_startup_configuration() -> None:
    """Validate required configuration settings at startup.

    Rationale:
    In production mode (`ENVIRONMENT=production`), failure to configure required secrets
    (e.g. GEMINI_API_KEY, SERVICE_API_KEY) must fail fast with a non-zero exit code.
    Because process liveness (/health) returns 200 independently of external dependencies,
    a bad deploy would otherwise be promoted by Render's healthCheckPath.
    Outside production (dev/test), missing configuration does not crash the server,
    allowing developers to run tests and observe 'not_ready' status via GET /ready.
    """
    is_prod = os.getenv("ENVIRONMENT", "").lower() == "production"
    errors: list[str] = []

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key or gemini_key in {"your_api_key_here", "test-key"}:
        if is_prod:
            errors.append("GEMINI_API_KEY is not configured or using default test key")

    auth_disabled = os.getenv("AUTH_DISABLED", "false").lower() in {"1", "true", "yes"}
    service_key = os.getenv("SERVICE_API_KEY", "")
    if is_prod and not auth_disabled and not service_key:
        errors.append("SERVICE_API_KEY is required in production when AUTH_DISABLED=false")

    require_redis = os.getenv("REQUIRE_REDIS", "false").lower() in {"1", "true", "yes"}
    redis_url = os.getenv("REDIS_URL", "")
    if is_prod and require_redis and not redis_url:
        errors.append("REDIS_URL must be configured when REQUIRE_REDIS=true")

    if errors:
        if is_prod:
            msg = f"Production startup configuration validation failed: {'; '.join(errors)}"
            logger.critical("FATAL: %s", msg)
            raise RuntimeError(msg)
        logger.warning("WARNING: Startup configuration issues: %s", "; ".join(errors))
