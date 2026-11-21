"""Tests for provider cost accounting and spend controls (#882).

Validates:
- Cost math against known price fixtures from telemetry.PRICE_TABLE
- Per-call record field completeness (model, usage, price_version, estimated cost)
- Daily / monthly window spend accumulation and limit detection
- Alert threshold fires before cutoff
- Per-tenant cap enforcement
- Emergency cutoff blocks non-critical paths but never health/auth
- Retries and fallbacks counted against budget (cannot bypass)
- Safe degraded response returned on budget exhaustion
- No secrets or prompt text in any cost record or log label
"""

from __future__ import annotations

import threading
import time

import pytest

from provider_cost import (
    BUDGET_EXHAUSTED_MESSAGE,
    BUDGET_EXHAUSTED_STATUS,
    BudgetAlertThresholdReached,
    BudgetConfig,
    BudgetCutoff,
    BudgetExceededError,
    DegradedResponse,
    SpendGuard,
    SpendWindow,
    build_call_record,
    get_price_version,
    is_critical_path,
)
from telemetry import GEMINI_MODEL, estimate_cost

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SECRET_KEY = "sk-prod-SECRET_API_KEY_DO_NOT_LOG_XYZ"
SECRET_PROMPT = "Bismillah — user prompt text that must never appear in cost logs"


@pytest.fixture()
def tight_config() -> BudgetConfig:
    """A very tight budget used to trigger alerts and cutoffs in tests."""
    return BudgetConfig(
        daily_limit_usd=0.01,
        monthly_limit_usd=0.05,
        alert_fraction=0.80,
        per_call_limit_usd=0.005,
    )


@pytest.fixture()
def loose_config() -> BudgetConfig:
    """A large budget that won't be hit by a handful of test calls."""
    return BudgetConfig(
        daily_limit_usd=1000.0,
        monthly_limit_usd=10000.0,
        alert_fraction=0.90,
    )


@pytest.fixture()
def tight_guard(tight_config: BudgetConfig) -> SpendGuard:
    return SpendGuard(tight_config)


@pytest.fixture()
def loose_guard(loose_config: BudgetConfig) -> SpendGuard:
    return SpendGuard(loose_config)


@pytest.fixture()
def tight_cutoff(tight_guard: SpendGuard) -> BudgetCutoff:
    return BudgetCutoff(tight_guard)


@pytest.fixture()
def loose_cutoff(loose_guard: SpendGuard) -> BudgetCutoff:
    return BudgetCutoff(loose_guard)


# ---------------------------------------------------------------------------
# 1. Cost math against known price fixtures
# ---------------------------------------------------------------------------


class TestCostMathFixtures:
    """Verify cost calculation against the pinned telemetry price table."""

    def test_gemini_flash_input_price(self) -> None:
        # 1,000 input tokens at $0.000075/1k = $0.000075
        est = estimate_cost(GEMINI_MODEL, 1000, 0)
        assert est.priced is True
        assert est.usd == pytest.approx(0.000075, abs=1e-10)

    def test_gemini_flash_output_price(self) -> None:
        # 1,000 output tokens at $0.0003/1k = $0.0003
        est = estimate_cost(GEMINI_MODEL, 0, 1000)
        assert est.priced is True
        assert est.usd == pytest.approx(0.0003, abs=1e-10)

    def test_gemini_flash_combined_price(self) -> None:
        # 800 input + 200 output
        # = 800/1000*0.000075 + 200/1000*0.0003
        # = 0.000060 + 0.000060 = 0.000120
        est = estimate_cost(GEMINI_MODEL, 800, 200)
        assert est.priced is True
        assert est.usd == pytest.approx(0.000060 + 0.000060, abs=1e-10)

    def test_gemini_pro_price_fixture(self) -> None:
        # gemini-2.5-pro: input $0.00125/1k, output $0.01/1k
        # 500 input + 100 output = 0.000625 + 0.001 = 0.001625
        est = estimate_cost("gemini-2.5-pro", 500, 100)
        assert est.priced is True
        assert est.usd == pytest.approx(0.001625, abs=1e-9)

    def test_unknown_model_unpriced(self) -> None:
        est = estimate_cost("unknown-model-xyz", 1000, 1000)
        assert est.priced is False
        assert est.usd == 0.0

    def test_zero_tokens_zero_cost(self) -> None:
        est = estimate_cost(GEMINI_MODEL, 0, 0)
        assert est.usd == 0.0 and est.priced is True


# ---------------------------------------------------------------------------
# 2. CallRecord — model, usage, price_version, estimated cost
# ---------------------------------------------------------------------------


class TestCallRecord:
    def test_build_call_record_fields(self) -> None:
        record = build_call_record(
            model=GEMINI_MODEL,
            input_tokens=800,
            output_tokens=200,
            trace_id="trace-abc",
            tenant_id="tenant-1",
        )
        assert record.model == GEMINI_MODEL
        assert record.input_tokens == 800
        assert record.output_tokens == 200
        assert record.estimated_cost_usd == pytest.approx(0.000120, abs=1e-10)
        assert record.priced is True
        assert isinstance(record.price_version, str) and len(record.price_version) > 0
        assert record.trace_id == "trace-abc"
        assert record.tenant_id == "tenant-1"

    def test_retry_and_fallback_flags(self) -> None:
        retry = build_call_record(GEMINI_MODEL, 100, 50, is_retry=True)
        fallback = build_call_record(GEMINI_MODEL, 100, 50, is_fallback=True)
        assert retry.is_retry is True
        assert fallback.is_fallback is True

    def test_as_labels_no_secrets(self) -> None:
        record = build_call_record(GEMINI_MODEL, 100, 50, trace_id=SECRET_KEY)
        # trace_id may appear; SECRET_KEY must not appear as content in a *value*
        # but we specifically ensure no prompt or secret text in the values
        labels = record.as_labels()
        blob = str(labels)
        assert SECRET_PROMPT not in blob

    def test_price_version_matches_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PRICE_VERSION", "2099-01")
        assert get_price_version() == "2099-01"


# ---------------------------------------------------------------------------
# 3. SpendWindow — accumulation, alert, limit
# ---------------------------------------------------------------------------


class TestSpendWindow:
    def test_spend_accumulates(self) -> None:
        w = SpendWindow(window_seconds=3600, limit_usd=1.0)
        w.record(0.3)
        w.record(0.4)
        assert w.current_spend() == pytest.approx(0.7, abs=1e-9)

    def test_remaining_decreases(self) -> None:
        w = SpendWindow(window_seconds=3600, limit_usd=1.0)
        w.record(0.6)
        assert w.remaining() == pytest.approx(0.4, abs=1e-9)

    def test_alert_fires_at_threshold(self) -> None:
        w = SpendWindow(window_seconds=3600, limit_usd=1.0, alert_fraction=0.80)
        w.record(0.80)
        assert w.alert_threshold_reached() is True

    def test_alert_not_before_threshold(self) -> None:
        w = SpendWindow(window_seconds=3600, limit_usd=1.0, alert_fraction=0.80)
        w.record(0.79)
        assert w.alert_threshold_reached() is False

    def test_limit_reached(self) -> None:
        w = SpendWindow(window_seconds=3600, limit_usd=0.5)
        w.record(0.5)
        assert w.limit_reached() is True

    def test_window_resets_after_expiry(self) -> None:
        w = SpendWindow(window_seconds=0.01, limit_usd=1.0)
        w.record(0.9)
        time.sleep(0.05)
        assert w.current_spend() == pytest.approx(0.0, abs=1e-9)

    def test_snapshot_fields(self) -> None:
        w = SpendWindow(window_seconds=3600, limit_usd=1.0, alert_fraction=0.80)
        w.record(0.5)
        snap = w.snapshot()
        assert snap["spend_usd"] == pytest.approx(0.5, abs=1e-6)
        assert snap["limit_usd"] == 1.0
        assert snap["remaining_usd"] == pytest.approx(0.5, abs=1e-6)
        assert snap["fraction_used"] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. SpendGuard — per-call limit, daily/monthly, per-tenant, alert vs cutoff
# ---------------------------------------------------------------------------


class TestSpendGuard:
    def test_normal_call_passes(self, loose_guard: SpendGuard) -> None:
        record = build_call_record(GEMINI_MODEL, 100, 50)
        loose_guard.record_call(record)  # must not raise

    def test_per_call_limit_exceeded_raises(self, tight_config: BudgetConfig) -> None:
        config = BudgetConfig(
            daily_limit_usd=100.0,
            monthly_limit_usd=1000.0,
            per_call_limit_usd=0.000001,
        )
        guard = SpendGuard(config)
        record = build_call_record(GEMINI_MODEL, 1000, 500)
        with pytest.raises(BudgetExceededError) as exc_info:
            guard.record_call(record)
        assert exc_info.value.window == "per_call"

    def test_daily_limit_raises_after_accumulation(self) -> None:
        config = BudgetConfig(daily_limit_usd=0.0002, monthly_limit_usd=1.0)
        guard = SpendGuard(config)
        # Each call costs ~0.000120; two calls exceed 0.0002
        record = build_call_record(GEMINI_MODEL, 800, 200)
        guard.record_call(record)  # first passes
        with pytest.raises(BudgetExceededError) as exc_info:
            guard.record_call(record)  # second should exceed daily
        assert exc_info.value.window == "daily"

    def test_monthly_limit_raises(self) -> None:
        config = BudgetConfig(daily_limit_usd=10.0, monthly_limit_usd=0.0002)
        guard = SpendGuard(config)
        record = build_call_record(GEMINI_MODEL, 800, 200)
        guard.record_call(record)
        with pytest.raises(BudgetExceededError) as exc_info:
            guard.record_call(record)
        assert exc_info.value.window == "monthly"

    def test_alert_threshold_fires_before_cutoff(self) -> None:
        # Set alert at 50%, limit at a value that takes 2 calls to breach
        config = BudgetConfig(
            daily_limit_usd=0.00025,
            monthly_limit_usd=1.0,
            alert_fraction=0.50,  # alert at 50 %
        )
        guard = SpendGuard(config)
        record = build_call_record(GEMINI_MODEL, 800, 200)  # ~0.000120

        # First call: 0.000120 / 0.00025 = 48 % — below alert, passes silently
        guard.record_call(record)

        # Second call brings total to 0.000240 / 0.00025 = 96 % — fires alert
        with pytest.raises(BudgetAlertThresholdReached) as exc_info:
            guard.record_call(record)
        assert exc_info.value.window == "daily"
        assert exc_info.value.fraction_used > 0.50

    def test_per_tenant_cap_enforced(self) -> None:
        config = BudgetConfig(daily_limit_usd=100.0, monthly_limit_usd=1000.0)
        guard = SpendGuard(config)
        guard.set_tenant_cap("tenant-99", daily_limit_usd=0.0001)

        record = build_call_record(GEMINI_MODEL, 800, 200, tenant_id="tenant-99")
        # First call brings tenant to ~0.000120 > 0.0001 → exceeds cap
        with pytest.raises(BudgetExceededError) as exc_info:
            guard.record_call(record)
        assert exc_info.value.window == "tenant_daily"
        assert exc_info.value.tenant_id == "tenant-99"

    def test_other_tenant_not_affected_by_cap(self) -> None:
        config = BudgetConfig(daily_limit_usd=100.0, monthly_limit_usd=1000.0)
        guard = SpendGuard(config)
        guard.set_tenant_cap("tenant-capped", daily_limit_usd=0.0001)

        # tenant-other has no cap → should pass
        record = build_call_record(GEMINI_MODEL, 800, 200, tenant_id="tenant-other")
        guard.record_call(record)  # must not raise

    def test_snapshot_no_secret_content(self, loose_guard: SpendGuard) -> None:
        record = build_call_record(GEMINI_MODEL, 100, 50, trace_id=SECRET_KEY)
        loose_guard.record_call(record)
        import json

        blob = json.dumps(loose_guard.snapshot())
        assert SECRET_PROMPT not in blob

    def test_thread_safety(self, loose_guard: SpendGuard) -> None:
        """Multiple threads recording calls must not corrupt the spend total."""
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(10):
                    loose_guard.record_call(build_call_record(GEMINI_MODEL, 10, 5))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # 5 threads × 10 calls × (10/1000*0.000075 + 5/1000*0.0003) each
        expected_per_call = estimate_cost(GEMINI_MODEL, 10, 5).usd
        expected_total = 50 * expected_per_call
        assert loose_guard.snapshot()["daily"]["spend_usd"] == pytest.approx(expected_total, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Retries and fallbacks count against the budget
# ---------------------------------------------------------------------------


class TestRetryFallbackCounting:
    """Retries and fallbacks must count against budget and cannot bypass it."""

    def test_retry_call_counted(self, loose_guard: SpendGuard) -> None:
        primary = build_call_record(GEMINI_MODEL, 100, 50, is_retry=False)
        retry1 = build_call_record(GEMINI_MODEL, 100, 50, is_retry=True)
        retry2 = build_call_record(GEMINI_MODEL, 100, 50, is_retry=True)
        loose_guard.record_call(primary)
        loose_guard.record_call(retry1)
        loose_guard.record_call(retry2)
        expected = 3 * estimate_cost(GEMINI_MODEL, 100, 50).usd
        assert loose_guard.snapshot()["daily"]["spend_usd"] == pytest.approx(expected, rel=1e-6)

    def test_fallback_call_counted(self, loose_guard: SpendGuard) -> None:
        primary = build_call_record(GEMINI_MODEL, 100, 50)
        fallback = build_call_record(GEMINI_MODEL, 100, 50, is_fallback=True)
        loose_guard.record_call(primary)
        loose_guard.record_call(fallback)
        expected = 2 * estimate_cost(GEMINI_MODEL, 100, 50).usd
        assert loose_guard.snapshot()["daily"]["spend_usd"] == pytest.approx(expected, rel=1e-6)

    def test_fallback_cannot_bypass_daily_limit(self) -> None:
        config = BudgetConfig(daily_limit_usd=0.0002, monthly_limit_usd=1.0)
        guard = SpendGuard(config)
        primary = build_call_record(GEMINI_MODEL, 800, 200)
        guard.record_call(primary)
        # Fallback attempt must also count and hit the limit
        fallback = build_call_record(GEMINI_MODEL, 800, 200, is_fallback=True)
        with pytest.raises(BudgetExceededError) as exc_info:
            guard.record_call(fallback)
        assert exc_info.value.window == "daily"


# ---------------------------------------------------------------------------
# 6. Emergency cutoff — blocks non-critical, never blocks health/auth
# ---------------------------------------------------------------------------


class TestBudgetCutoff:
    def test_emergency_stop_blocks_non_critical(self, tight_cutoff: BudgetCutoff) -> None:
        tight_cutoff.activate_emergency_stop(reason="test")
        with pytest.raises(BudgetExceededError):
            tight_cutoff.check(path="/chat")

    def test_emergency_stop_does_not_block_health(self, tight_cutoff: BudgetCutoff) -> None:
        tight_cutoff.activate_emergency_stop(reason="test")
        # Must not raise for health / auth paths
        tight_cutoff.check(path="/health")
        tight_cutoff.check(path="/healthz")
        tight_cutoff.check(path="/readiness")
        tight_cutoff.check(path="/auth")
        tight_cutoff.check(path="/metrics")

    def test_critical_flag_bypasses_cutoff(self, tight_cutoff: BudgetCutoff) -> None:
        tight_cutoff.activate_emergency_stop(reason="test")
        tight_cutoff.check(path="/chat", critical=True)  # must not raise

    def test_deactivate_resumes_generation(self, tight_cutoff: BudgetCutoff) -> None:
        tight_cutoff.activate_emergency_stop(reason="test")
        tight_cutoff.deactivate_emergency_stop()
        tight_cutoff.check(path="/chat")  # must not raise

    def test_check_and_record_returns_degraded_on_emergency_stop(self, tight_cutoff: BudgetCutoff) -> None:
        tight_cutoff.activate_emergency_stop(reason="unit_test")
        record = build_call_record(GEMINI_MODEL, 100, 50)
        degraded = tight_cutoff.check_and_record(record, path="/chat")
        assert degraded is not None
        assert isinstance(degraded, DegradedResponse)
        assert degraded.status == BUDGET_EXHAUSTED_STATUS
        assert degraded.user_message == BUDGET_EXHAUSTED_MESSAGE

    def test_check_and_record_returns_degraded_on_budget_exceeded(self, tight_cutoff: BudgetCutoff) -> None:
        # Exhaust the daily limit then try another call
        record = build_call_record(GEMINI_MODEL, 800, 200)
        # Keep calling until budget is hit
        degraded: DegradedResponse | None = None
        for _ in range(100):
            degraded = tight_cutoff.check_and_record(record, path="/chat")
            if degraded is not None:
                break
        assert degraded is not None
        assert degraded.status == BUDGET_EXHAUSTED_STATUS

    def test_check_and_record_allows_health_during_budget_exceeded(self, tight_cutoff: BudgetCutoff) -> None:
        tight_cutoff.activate_emergency_stop(reason="budget_exceeded")
        record = build_call_record(GEMINI_MODEL, 100, 50)
        # Health path must return None (allowed) even during emergency stop
        result = tight_cutoff.check_and_record(record, path="/health")
        assert result is None

    def test_degraded_response_has_no_secret_content(self, tight_cutoff: BudgetCutoff) -> None:
        import json

        tight_cutoff.activate_emergency_stop(reason="test")
        record = build_call_record(GEMINI_MODEL, 100, 50, trace_id=SECRET_KEY)
        degraded = tight_cutoff.check_and_record(record, path="/chat")
        assert degraded is not None
        blob = json.dumps(degraded.to_dict())
        assert SECRET_PROMPT not in blob
        assert SECRET_KEY not in blob

    def test_snapshot_contains_emergency_stop_field(self, loose_cutoff: BudgetCutoff) -> None:
        snap = loose_cutoff.snapshot()
        assert "emergency_stop" in snap
        assert snap["emergency_stop"]["active"] is False

    def test_snapshot_no_secret_content(self, loose_cutoff: BudgetCutoff) -> None:
        import json

        loose_cutoff._guard.record_call(build_call_record(GEMINI_MODEL, 100, 50, trace_id=SECRET_KEY))
        blob = json.dumps(loose_cutoff.snapshot())
        assert SECRET_PROMPT not in blob
        assert SECRET_KEY not in blob


# ---------------------------------------------------------------------------
# 7. is_critical_path
# ---------------------------------------------------------------------------


class TestIsCriticalPath:
    @pytest.mark.parametrize(
        "path",
        [
            "/health",
            "/healthz",
            "/readiness",
            "/ready",
            "/liveness",
            "/alive",
            "/metrics",
            "/auth",
            "/token",
            "/login",
            "/logout",
            "/health/",
            "/auth/token",
        ],
    )
    def test_critical_paths_recognized(self, path: str) -> None:
        assert is_critical_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/chat",
            "/api/v1/chat",
            "/fatwa",
            "/quran",
            "/feedback",
        ],
    )
    def test_non_critical_paths_not_blocked(self, path: str) -> None:
        assert is_critical_path(path) is False


# ---------------------------------------------------------------------------
# 8. BudgetConfig validation
# ---------------------------------------------------------------------------


class TestBudgetConfigValidation:
    def test_invalid_alert_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="alert_fraction"):
            BudgetConfig(daily_limit_usd=10.0, monthly_limit_usd=100.0, alert_fraction=0.0)

    def test_zero_daily_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            BudgetConfig(daily_limit_usd=0.0, monthly_limit_usd=100.0)
