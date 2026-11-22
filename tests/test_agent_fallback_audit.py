"""Audit and compliance tests for external-provider fallback decisions (#884).

Covers:
- Stable fallback reasons (timeout, circuit open, circuit half-open, safety block, deterministic)
- Truthful user-facing messages matching risk levels (low, medium, high)
- Non-bypassable controls for safety, citation, auth, and cost
- Dashboards and telemetry distinguishing fallback rate by provider and feature
- Thresholds for disabling fallbacks, review status, and operator rollback
- Privacy invariants: strictly zero prompts, secrets, or PII in telemetry
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import telemetry
from agent_fallback import (
    AgentRegistry,
    CircuitBreakerConfig,
    CircuitState,
    ControlBypassError,
    FallbackControlPolicy,
    FallbackMetadata,
    FallbackReason,
    FallbackReviewStatus,
    FallbackRiskLevel,
    FallbackThresholdConfig,
    RetryConfig,
    SafetyBlockError,
    SafetyBypassError,
)

SECRET_PROMPT = "Explain surah Al-Baqarah verse 255 with secret context XYZ_PROMPT_SECRET"
SECRET_API_KEY = "AIzaSyFakeSecretApiKey-12345-NoLeak"


@pytest.fixture(autouse=True)
def fresh_telemetry_registry(monkeypatch: pytest.MonkeyPatch) -> telemetry.MetricsRegistry:
    reg = telemetry.MetricsRegistry()
    monkeypatch.setattr(telemetry, "registry", reg)
    return reg


# ---------------------------------------------------------------------------
# 1. Provider Timeout Fallback
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_provider_timeout_fallback_emits_audit_metadata() -> None:
    registry = AgentRegistry()

    async def timeout_handler(payload: Any) -> Any:
        raise TimeoutError("External provider timed out after 5000ms")

    async def backup_handler(payload: Any) -> Any:
        return {"answer": "Backup model answer"}

    registry.register_agent(
        "primary_gemini",
        timeout_handler,
        fallback_agent="backup_claude",
        provider="google",
        feature="tafsir",
        version="2.1.0",
    )
    registry.register_agent(
        "backup_claude",
        backup_handler,
        provider="anthropic",
        feature="tafsir",
        version="1.0.0",
    )

    result = await registry.execute_with_resilience(
        "primary_gemini",
        {"query": "tafsir request"},
        retry_config=RetryConfig(max_retries=0),
        trace_id="trace-timeout-1",
    )

    assert result.success is True
    assert result.fallback_used is True
    assert result.fallback_reason == FallbackReason.PROVIDER_TIMEOUT
    assert result.component == "primary_gemini"
    assert result.version == "2.1.0"
    assert result.duration_ms > 0
    assert result.user_message is not None
    assert "The primary provider timed out" in result.user_message

    meta = result.fallback_metadata
    assert isinstance(meta, FallbackMetadata)
    assert meta.reason == FallbackReason.PROVIDER_TIMEOUT
    assert meta.component == "primary_gemini"
    assert meta.version == "2.1.0"
    assert meta.risk_level == FallbackRiskLevel.MEDIUM
    assert meta.target == "backup_claude"
    assert meta.fallback_type == "alternative_agent"
    assert meta.trace_id == "trace-timeout-1"

    # Telemetry snapshot check
    snap = telemetry.registry.snapshot()
    assert snap["fallbacks"]["total"] == 1
    assert snap["fallbacks"]["by_reason"]["provider_timeout"] == 1
    assert snap["fallbacks"]["by_provider"]["google"] == 1
    assert snap["fallbacks"]["by_feature"]["tafsir"] == 1


# ---------------------------------------------------------------------------
# 2. Circuit Breaker: Open and Half-Open Failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_circuit_open_fallback_emits_audit_metadata() -> None:
    registry = AgentRegistry()

    async def failing_agent(payload: Any) -> Any:
        raise ConnectionError("Connection refused by upstream provider")

    def local_fallback(payload: Any) -> Any:
        return "Offline deterministic knowledge"

    registry.register_agent(
        "fragile_provider",
        failing_agent,
        deterministic_fallback=local_fallback,
        circuit_config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=60.0),
        provider="remote_hadith",
        feature="hadith",
        version="1.5.0",
    )

    # First call fails and opens the circuit
    r1 = await registry.execute_with_resilience("fragile_provider", {}, retry_config=RetryConfig(max_retries=0))
    assert r1.success is True
    cb = registry._circuit_breakers["fragile_provider"]
    assert cb.state == CircuitState.OPEN

    # Second call hits circuit breaker immediately without calling primary
    r2 = await registry.execute_with_resilience("fragile_provider", {}, retry_config=RetryConfig(max_retries=0))
    assert r2.success is True
    assert r2.fallback_used is True
    assert r2.fallback_reason == FallbackReason.CIRCUIT_OPEN
    assert r2.component == "fragile_provider"
    assert r2.version == "1.5.0"
    assert "circuit_breaker_open" in r2.stages_executed
    assert r2.fallback_metadata is not None
    assert r2.fallback_metadata.risk_level == FallbackRiskLevel.HIGH
    assert r2.user_message is not None
    assert "experiencing connectivity issues" in r2.user_message


@pytest.mark.asyncio
async def test_circuit_half_open_failure_emits_audit_metadata() -> None:
    registry = AgentRegistry()

    async def flaky_agent(payload: Any) -> Any:
        raise ConnectionResetError("Reset by peer")

    registry.register_agent(
        "probe_agent",
        flaky_agent,
        circuit_config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.0, half_open_max_calls=1),
        deterministic_fallback=lambda p: "Rule output",
        provider="unstable_llm",
        feature="chat",
    )

    # 1. First failure trips circuit to OPEN
    await registry.execute_with_resilience("probe_agent", {}, retry_config=RetryConfig(max_retries=0))
    cb = registry._circuit_breakers["probe_agent"]
    assert cb.state == CircuitState.OPEN

    # 2. Second call allows request under recovery timeout=0.0, entering HALF_OPEN.
    # When the call fails again, it records a half-open trial failure.
    r2 = await registry.execute_with_resilience("probe_agent", {}, retry_config=RetryConfig(max_retries=0))
    assert r2.success is True
    assert r2.fallback_used is True
    assert r2.fallback_reason == FallbackReason.CIRCUIT_HALF_OPEN_FAILED
    assert r2.fallback_metadata is not None
    assert r2.fallback_metadata.risk_level == FallbackRiskLevel.HIGH
    assert r2.user_message is not None
    assert "recovery check failed" in r2.user_message.lower()
    assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# 3. Content Safety Invariant: Fallbacks CANNOT Bypass Safety (#884)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_safety_block_terminal_cannot_bypass_safety() -> None:
    registry = AgentRegistry()
    alt_called = False
    det_called = False

    async def unsafe_agent(payload: Any) -> Any:
        raise SafetyBlockError("Prompt violated hate speech policy")

    async def backup_agent(payload: Any) -> Any:
        nonlocal alt_called
        alt_called = True
        return "Bypassed content"

    def deterministic_fallback(payload: Any) -> Any:
        nonlocal det_called
        det_called = True
        return "Bypassed rule"

    registry.register_agent(
        "safety_guarded_agent",
        unsafe_agent,
        fallback_agent="backup_agent",
        deterministic_fallback=deterministic_fallback,
        provider="gemini",
        feature="chat",
    )
    registry.register_agent("backup_agent", backup_agent)

    result = await registry.execute_with_resilience(
        "safety_guarded_agent",
        {"query": "unsafe query"},
        retry_config=RetryConfig(max_retries=0),
    )

    # Safety refusals must terminate immediately and cannot route to fallbacks!
    assert result.success is False
    assert result.fallback_used is True
    assert result.fallback_reason == FallbackReason.SAFETY_BLOCK
    assert result.user_message is not None
    assert "safety policy constraints" in result.user_message
    assert result.fallback_metadata is not None
    assert result.fallback_metadata.risk_level == FallbackRiskLevel.HIGH
    assert "safety_block_terminal" in result.stages_executed
    assert alt_called is False, "Alternative agent was called despite safety block!"
    assert det_called is False, "Deterministic fallback was called despite safety block!"


@pytest.mark.asyncio
async def test_output_safety_guard_blocks_unsafe_fallback_response() -> None:
    registry = AgentRegistry()

    async def timeout_primary(payload: Any) -> Any:
        raise TimeoutError("Timeout")

    async def toxic_fallback(payload: Any) -> Any:
        return "Harmful unmoderated advice"

    def safety_filter(text: Any) -> bool:
        # Rejects harmful content
        return "harmful" not in str(text).lower()

    registry.register_agent(
        "primary",
        timeout_primary,
        fallback_agent="backup",
        output_safety_validator=safety_filter,
    )
    registry.register_agent("backup", toxic_fallback)

    # When fallback produces output that fails safety, it must not be served
    with pytest.raises(SafetyBypassError) as exc_info:
        await registry.execute_with_resilience("primary", {}, retry_config=RetryConfig(max_retries=0))

    assert "violated safety policy" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. Deterministic Fallback Path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_deterministic_fallback_emits_audit_metadata() -> None:
    registry = AgentRegistry()

    async def failing_agent(payload: Any) -> Any:
        raise ValueError("General model parsing error")

    def rule_based_math(payload: Any) -> Any:
        return {"calculation": "2.5% zakat standard rate"}

    registry.register_agent(
        "zakat_agent",
        failing_agent,
        deterministic_fallback=rule_based_math,
        provider="local_rule_engine",
        feature="zakat",
        version="3.0.0",
    )

    result = await registry.execute_with_resilience("zakat_agent", {}, retry_config=RetryConfig(max_retries=0))
    assert result.success is True
    assert result.fallback_used is True
    assert result.fallback_reason == FallbackReason.DETERMINISTIC_FALLBACK
    assert result.component == "zakat_agent"
    assert result.version == "3.0.0"
    assert result.fallback_metadata is not None
    assert result.fallback_metadata.fallback_type == "deterministic"
    assert result.user_message is not None
    assert "rule-based deterministic response" in result.user_message
    assert result.fallback_metadata.risk_level == FallbackRiskLevel.MEDIUM
    assert result.data == {"calculation": "2.5% zakat standard rate"}


# ---------------------------------------------------------------------------
# 5. Citation, Auth, and Cost Controls Enforcement (#884)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_citation_control_enforcement_on_fallback() -> None:
    registry = AgentRegistry()

    async def primary_fail(payload: Any) -> Any:
        raise TimeoutError("Down")

    def unverified_fallback(payload: Any) -> Any:
        return "Claim with no citation"

    def requires_hadith_citation(data: Any) -> bool:
        return "Bukhari" in str(data) or "Muslim" in str(data)

    registry.register_agent(
        "hadith_research",
        primary_fail,
        deterministic_fallback=unverified_fallback,
        citation_validator=requires_hadith_citation,
        control_policy=FallbackControlPolicy(enforce_citations=True),
    )

    with pytest.raises(ControlBypassError) as exc_info:
        await registry.execute_with_resilience("hadith_research", {}, retry_config=RetryConfig(max_retries=0))

    assert "lacks required citations" in str(exc_info.value)


@pytest.mark.asyncio
async def test_auth_control_enforcement() -> None:
    registry = AgentRegistry()

    def check_auth(ctx: Any) -> bool:
        if isinstance(ctx, dict):
            return ctx.get("auth_token") == "valid_token"
        return False

    registry.register_agent(
        "secure_agent",
        lambda p: asyncio.sleep(0.01),
        auth_validator=check_auth,
        control_policy=FallbackControlPolicy(enforce_auth=True),
    )

    with pytest.raises(ControlBypassError) as exc_info:
        await registry.execute_with_resilience(
            "secure_agent",
            {"query": "unauthorized"},
            context={"auth_token": "expired_or_missing"},
        )

    assert "Authentication required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cost_control_enforcement() -> None:
    registry = AgentRegistry()

    registry.register_agent(
        "expensive_agent",
        lambda p: asyncio.sleep(0.01),
        control_policy=FallbackControlPolicy(enforce_cost=True, max_cost_usd=0.05),
    )

    with pytest.raises(ControlBypassError) as exc_info:
        await registry.execute_with_resilience(
            "expensive_agent",
            {"query": "expensive"},
            context={"cost_usd": 0.12},
        )

    assert "Cost limit $0.05 exceeded" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Fallback Disabling Thresholds and Operator Rollback (#884)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_threshold_review_trigger_and_auto_disable() -> None:
    registry = AgentRegistry()

    async def flaky(p: Any) -> Any:
        raise ConnectionError("Unstable")

    def fallback(p: Any) -> Any:
        return "fallback"

    registry.register_agent(
        "fragile",
        flaky,
        deterministic_fallback=fallback,
    )
    registry.set_threshold_config(
        "fragile",
        FallbackThresholdConfig(
            max_fallback_rate=0.4,
            min_requests_for_threshold=3,
            auto_disable_on_threshold=True,
        ),
    )

    # Call 1 & 2: fallbacks occur, but min_requests (3) not reached yet
    await registry.execute_with_resilience("fragile", {}, retry_config=RetryConfig(max_retries=0))
    await registry.execute_with_resilience("fragile", {}, retry_config=RetryConfig(max_retries=0))
    status = registry.get_fallback_status("fragile")
    assert status["status"] == FallbackReviewStatus.HEALTHY.value

    # Call 3: total 3, fallbacks 3 (rate 1.0 > 0.4) -> triggers auto-disable
    await registry.execute_with_resilience("fragile", {}, retry_config=RetryConfig(max_retries=0))
    status = registry.get_fallback_status("fragile")
    assert status["status"] == FallbackReviewStatus.DISABLED.value
    assert status["fallback_enabled"] is False

    # Call 4: fallback is now disabled, call fails without routing to fallback
    r4 = await registry.execute_with_resilience("fragile", {}, retry_config=RetryConfig(max_retries=0))
    assert r4.success is False
    assert r4.fallback_used is False
    assert r4.error is not None
    assert "Fallback disabled" in r4.error


@pytest.mark.asyncio
async def test_operator_manual_rollback_and_reenable() -> None:
    registry = AgentRegistry()

    async def primary(p: Any) -> Any:
        raise RuntimeError("Primary crashed")

    def fallback(p: Any) -> Any:
        return "Degraded answer"

    registry.register_agent("search_agent", primary, deterministic_fallback=fallback)

    # Works before rollback
    res1 = await registry.execute_with_resilience("search_agent", {}, retry_config=RetryConfig(max_retries=0))
    assert res1.success is True
    assert res1.fallback_used is True

    # Operator performs emergency rollback of the fallback
    registry.rollback_fallback("search_agent", reason="Degraded answer hallucination detected")
    assert registry.is_fallback_enabled("search_agent") is False
    assert registry.get_fallback_status("search_agent")["status"] == "disabled"

    # Execution during rollback stops fallback
    res2 = await registry.execute_with_resilience("search_agent", {}, retry_config=RetryConfig(max_retries=0))
    assert res2.success is False
    assert res2.fallback_used is False
    assert res2.error is not None
    assert "Fallback disabled for 'search_agent'" in res2.error

    # Operator re-enables fallback
    registry.enable_fallback("search_agent")
    assert registry.is_fallback_enabled("search_agent") is True

    res3 = await registry.execute_with_resilience("search_agent", {}, retry_config=RetryConfig(max_retries=0))
    assert res3.success is True
    assert res3.fallback_used is True


# ---------------------------------------------------------------------------
# 7. Telemetry & Privacy: Zero Prompts or Secrets Exposed (#884)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_telemetry_and_metadata_never_leak_prompts_or_secrets() -> None:
    registry = AgentRegistry()

    async def leaking_primary(payload: Any) -> Any:
        # Simulates error that echoes sensitive prompt/secret
        raise TimeoutError(f"Upstream provider timeout with secret {SECRET_API_KEY} for query {SECRET_PROMPT}")

    def safe_fallback(payload: Any) -> Any:
        return "Safe generic answer"

    registry.register_agent(
        "secret_agent",
        leaking_primary,
        deterministic_fallback=safe_fallback,
        provider="google",
        feature="chat",
    )

    result = await registry.execute_with_resilience(
        "secret_agent",
        {"prompt": SECRET_PROMPT, "api_key": SECRET_API_KEY},
        retry_config=RetryConfig(max_retries=0),
        trace_id="trace-sec-1",
    )

    assert result.success is True
    assert result.fallback_used is True

    # Inspect fallback metadata
    assert result.fallback_metadata is not None
    metadata_json = json.dumps(result.fallback_metadata.to_dict())
    assert SECRET_PROMPT not in metadata_json
    assert SECRET_API_KEY not in metadata_json

    # Inspect telemetry snapshot
    snap_json = json.dumps(telemetry.registry.snapshot())
    assert SECRET_PROMPT not in snap_json
    assert SECRET_API_KEY not in snap_json


# ---------------------------------------------------------------------------
# 8. Multi-Provider and Multi-Feature Telemetry Aggregation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_telemetry_distinguishes_fallback_rate_by_provider_and_feature() -> None:
    registry = AgentRegistry()

    async def fail(p: Any) -> Any:
        raise TimeoutError("timeout")

    registry.register_agent(
        "gemini_chat",
        fail,
        deterministic_fallback=lambda p: "fb",
        provider="gemini",
        feature="chat",
    )
    registry.register_agent(
        "gemini_hadith",
        fail,
        deterministic_fallback=lambda p: "fb",
        provider="gemini",
        feature="hadith",
    )
    registry.register_agent(
        "anthropic_fiqh",
        fail,
        deterministic_fallback=lambda p: "fb",
        provider="anthropic",
        feature="fiqh",
    )

    await registry.execute_with_resilience("gemini_chat", {}, retry_config=RetryConfig(max_retries=0))
    await registry.execute_with_resilience("gemini_chat", {}, retry_config=RetryConfig(max_retries=0))
    await registry.execute_with_resilience("gemini_hadith", {}, retry_config=RetryConfig(max_retries=0))
    await registry.execute_with_resilience("anthropic_fiqh", {}, retry_config=RetryConfig(max_retries=0))

    snap = telemetry.registry.snapshot()
    fb = snap["fallbacks"]

    assert fb["total"] == 4
    assert fb["by_provider"]["gemini"] == 3
    assert fb["by_provider"]["anthropic"] == 1
    assert fb["by_feature"]["chat"] == 2
    assert fb["by_feature"]["hadith"] == 1
    assert fb["by_feature"]["fiqh"] == 1
    assert fb["by_provider_feature"]["gemini"]["chat"] == 2
    assert fb["by_provider_feature"]["gemini"]["hadith"] == 1
    assert fb["by_provider_feature"]["anthropic"]["fiqh"] == 1
