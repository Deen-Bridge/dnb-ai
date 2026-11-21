"""Provider cost accounting and spend controls (#882).

Adds production-wide spend budgets, per-tenant caps, alert thresholds, and an
emergency cutoff on top of the per-call cost invariant already defined in
telemetry.py and enforced in agent_fallback.FallbackControlPolicy.

Design principles
-----------------
* **Reuse, don't duplicate**: versioned model prices are built on top of
  ``telemetry.PRICE_TABLE`` / ``telemetry.estimate_cost``.  The price version
  string is derived from the same environment variable (``LLM_PRICE_TABLE``).
* **Zero-content invariant**: prompt text and secrets never enter this module.
  Only token counts, model names, tenant IDs, and cost figures are recorded.
* **Never block health / auth**: ``BudgetCutoff.check()`` raises
  ``BudgetExceededError`` only for non-critical generation paths.  Callers
  that handle health or auth endpoints must pass ``critical=True``.
* **Retries and fallbacks count**: every attempt – primary, retry, and
  fallback – must call ``SpendGuard.record_call()`` so the budget cannot be
  circumvented by routing through a cheaper path.
* **Thread-safe**: all mutable state is protected by ``threading.Lock``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from telemetry import CostEstimate, estimate_cost

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Sentinel returned as the user-facing message when the budget is exhausted.
BUDGET_EXHAUSTED_MESSAGE = (
    "Service is temporarily unavailable while provider spend limits are enforced. Please try again later."
)

#: HTTP-compatible status string callers may surface in responses.
BUDGET_EXHAUSTED_STATUS = "budget_exhausted"

# ---------------------------------------------------------------------------
# Price versioning
# ---------------------------------------------------------------------------

# The price version is a human-readable label that tracks which price schedule
# is in force.  Incrementing it whenever ``LLM_PRICE_TABLE`` changes lets
# operators correlate cost spikes with provider price updates in audit logs.
PRICE_VERSION_DEFAULT = "2026-05"


def get_price_version() -> str:
    """Return the active price version label.

    Defaults to ``PRICE_VERSION_DEFAULT``.  Override at deploy time with the
    ``LLM_PRICE_VERSION`` environment variable, e.g. ``"2026-07"`` after a
    provider price change.
    """
    import os

    return os.getenv("LLM_PRICE_VERSION", PRICE_VERSION_DEFAULT)


# ---------------------------------------------------------------------------
# Per-call record
# ---------------------------------------------------------------------------


@dataclass
class CallRecord:
    """Telemetry for a single provider call.  Contains no prompt or answer text."""

    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    priced: bool
    price_version: str
    trace_id: str = "-"
    tenant_id: str | None = None
    is_retry: bool = False
    is_fallback: bool = False
    timestamp: float = field(default_factory=time.time)

    def as_labels(self) -> dict[str, Any]:
        """Content-free dict safe to log or surface in metrics."""
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "priced": self.priced,
            "price_version": self.price_version,
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id or "",
            "is_retry": self.is_retry,
            "is_fallback": self.is_fallback,
            "timestamp": self.timestamp,
        }


def build_call_record(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    trace_id: str = "-",
    tenant_id: str | None = None,
    is_retry: bool = False,
    is_fallback: bool = False,
) -> CallRecord:
    """Build a ``CallRecord`` for a provider call, computing cost from the
    shared ``telemetry.PRICE_TABLE``.

    This is the canonical factory used by every call site — primary, retry, and
    fallback — so all paths count against the same budget.
    """
    cost: CostEstimate = estimate_cost(model, input_tokens, output_tokens)
    return CallRecord(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost.usd,
        priced=cost.priced,
        price_version=get_price_version(),
        trace_id=trace_id,
        tenant_id=tenant_id,
        is_retry=is_retry,
        is_fallback=is_fallback,
    )


# ---------------------------------------------------------------------------
# Budget windows
# ---------------------------------------------------------------------------


@dataclass
class BudgetConfig:
    """Configurable budget thresholds for a deployment or tenant.

    All dollar amounts are in USD.  ``alert_fraction`` controls at what fraction
    of the ceiling an alert fires before the cutoff takes effect.
    """

    daily_limit_usd: float
    monthly_limit_usd: float
    alert_fraction: float = 0.80  # alert when spend >= 80 % of limit
    per_call_limit_usd: float | None = None  # optional hard cap per call

    def __post_init__(self) -> None:
        if not (0.0 < self.alert_fraction <= 1.0):
            raise ValueError("alert_fraction must be in (0.0, 1.0]")
        if self.daily_limit_usd <= 0 or self.monthly_limit_usd <= 0:
            raise ValueError("Budget limits must be positive")


@dataclass
class SpendWindow:
    """Accumulates spend for a rolling calendar window.

    ``window_seconds`` is the duration of the window; the spend is reset to 0
    when the window expires.  For daily windows use ``86400``; for monthly use
    ``2592000`` (30 days).
    """

    window_seconds: float
    limit_usd: float
    alert_fraction: float = 0.80

    _spend_usd: float = field(default=0.0, init=False, repr=False)
    _window_start: float = field(default_factory=time.time, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _rotate_if_expired(self) -> None:
        """Reset the window if the duration has elapsed."""
        now = time.time()
        if now - self._window_start >= self.window_seconds:
            self._spend_usd = 0.0
            self._window_start = now

    def record(self, amount_usd: float) -> None:
        """Add ``amount_usd`` to the current window's spend."""
        with self._lock:
            self._rotate_if_expired()
            self._spend_usd = round(self._spend_usd + amount_usd, 8)

    def current_spend(self) -> float:
        with self._lock:
            self._rotate_if_expired()
            return self._spend_usd

    def remaining(self) -> float:
        return max(0.0, self.limit_usd - self.current_spend())

    def fraction_used(self) -> float:
        spend = self.current_spend()
        if self.limit_usd <= 0:
            return 1.0
        return round(spend / self.limit_usd, 6)

    def alert_threshold_reached(self) -> bool:
        return self.fraction_used() >= self.alert_fraction

    def limit_reached(self) -> bool:
        return self.current_spend() >= self.limit_usd

    def snapshot(self) -> dict[str, Any]:
        spend = self.current_spend()
        return {
            "spend_usd": round(spend, 8),
            "limit_usd": self.limit_usd,
            "remaining_usd": round(max(0.0, self.limit_usd - spend), 8),
            "fraction_used": self.fraction_used(),
            "alert_threshold_reached": self.alert_threshold_reached(),
            "limit_reached": self.limit_reached(),
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    """Raised when a provider call would exceed a spend budget."""

    def __init__(
        self,
        message: str,
        window: str = "unknown",
        spend_usd: float = 0.0,
        limit_usd: float = 0.0,
        tenant_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.window = window
        self.spend_usd = spend_usd
        self.limit_usd = limit_usd
        self.tenant_id = tenant_id


class BudgetAlertThresholdReached(Exception):
    """Raised when spend crosses the alert threshold (not yet at hard limit)."""

    def __init__(
        self,
        message: str,
        window: str = "unknown",
        fraction_used: float = 0.0,
        alert_fraction: float = 0.80,
        tenant_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.window = window
        self.fraction_used = fraction_used
        self.alert_fraction = alert_fraction
        self.tenant_id = tenant_id


# ---------------------------------------------------------------------------
# Per-call limit check
# ---------------------------------------------------------------------------


def check_per_call_limit(record: CallRecord, limit_usd: float) -> None:
    """Raise ``BudgetExceededError`` if a single call exceeds ``limit_usd``.

    This reuses the cost invariant from ``agent_fallback.FallbackControlPolicy``
    but applies it at the aggregate spend-guard layer for defence-in-depth.
    """
    if record.estimated_cost_usd > limit_usd:
        raise BudgetExceededError(
            f"Single call cost ${record.estimated_cost_usd:.6f} exceeds per-call "
            f"limit ${limit_usd:.6f} for model '{record.model}'.",
            window="per_call",
            spend_usd=record.estimated_cost_usd,
            limit_usd=limit_usd,
            tenant_id=record.tenant_id,
        )


# ---------------------------------------------------------------------------
# SpendGuard — global and per-tenant budget enforcement
# ---------------------------------------------------------------------------


class SpendGuard:
    """Thread-safe spend accounting with daily/monthly windows and per-tenant caps.

    Usage
    -----
    ::

        guard = SpendGuard(BudgetConfig(daily_limit_usd=10.0, monthly_limit_usd=200.0))

        # On every provider call (primary, retry, or fallback):
        record = build_call_record(model="gemini-2.5-flash", input_tokens=800,
                                   output_tokens=200, tenant_id="tenant-42")
        guard.record_call(record)   # raises BudgetExceededError if over limit
                                    # raises BudgetAlertThresholdReached if at alert %
    """

    def __init__(self, config: BudgetConfig) -> None:
        self._config = config
        self._daily = SpendWindow(
            window_seconds=86400.0,
            limit_usd=config.daily_limit_usd,
            alert_fraction=config.alert_fraction,
        )
        self._monthly = SpendWindow(
            window_seconds=2592000.0,  # 30 days
            limit_usd=config.monthly_limit_usd,
            alert_fraction=config.alert_fraction,
        )
        # Per-tenant windows keyed by tenant_id
        self._tenant_windows: dict[str, SpendWindow] = {}
        self._tenant_limits: dict[str, float] = {}
        self._lock = threading.Lock()

    def set_tenant_cap(self, tenant_id: str, daily_limit_usd: float) -> None:
        """Register a daily spend cap for a specific tenant."""
        with self._lock:
            self._tenant_limits[tenant_id] = daily_limit_usd
            self._tenant_windows[tenant_id] = SpendWindow(
                window_seconds=86400.0,
                limit_usd=daily_limit_usd,
                alert_fraction=self._config.alert_fraction,
            )

    def _get_tenant_window(self, tenant_id: str) -> SpendWindow | None:
        with self._lock:
            return self._tenant_windows.get(tenant_id)

    def record_call(self, record: CallRecord) -> None:
        """Record a provider call and check all budget constraints.

        Raises
        ------
        BudgetExceededError
            When the daily or monthly limit is reached, or the per-call limit
            is exceeded.
        BudgetAlertThresholdReached
            When spend crosses the alert fraction but has not yet hit the hard
            limit.  Callers *may* catch this to emit an alert without blocking
            the request; if not caught, it propagates like a budget breach.
        """
        cost = record.estimated_cost_usd

        # 1. Per-call limit
        if self._config.per_call_limit_usd is not None:
            check_per_call_limit(record, self._config.per_call_limit_usd)

        # 2. Per-tenant cap
        if record.tenant_id:
            tenant_window = self._get_tenant_window(record.tenant_id)
            if tenant_window is not None:
                tenant_window.record(cost)
                if tenant_window.limit_reached():
                    raise BudgetExceededError(
                        f"Daily spend cap ${tenant_window.limit_usd:.2f} reached for "
                        f"tenant '{record.tenant_id}' "
                        f"(spend: ${tenant_window.current_spend():.6f}).",
                        window="tenant_daily",
                        spend_usd=tenant_window.current_spend(),
                        limit_usd=tenant_window.limit_usd,
                        tenant_id=record.tenant_id,
                    )
                if tenant_window.alert_threshold_reached():
                    logger.warning(
                        "cost_alert tenant=%s window=tenant_daily fraction=%.2f limit=%.2f",
                        record.tenant_id,
                        tenant_window.fraction_used(),
                        tenant_window.limit_usd,
                    )

        # 3. Global daily window
        self._daily.record(cost)
        if self._daily.limit_reached():
            raise BudgetExceededError(
                f"Daily spend limit ${self._daily.limit_usd:.2f} reached (spend: ${self._daily.current_spend():.6f}).",
                window="daily",
                spend_usd=self._daily.current_spend(),
                limit_usd=self._daily.limit_usd,
                tenant_id=record.tenant_id,
            )
        if self._daily.alert_threshold_reached():
            logger.warning(
                "cost_alert window=daily fraction=%.2f limit=%.2f",
                self._daily.fraction_used(),
                self._daily.limit_usd,
            )
            raise BudgetAlertThresholdReached(
                f"Daily spend alert: {self._daily.fraction_used() * 100:.1f}% of "
                f"${self._daily.limit_usd:.2f} limit used.",
                window="daily",
                fraction_used=self._daily.fraction_used(),
                alert_fraction=self._config.alert_fraction,
                tenant_id=record.tenant_id,
            )

        # 4. Global monthly window
        self._monthly.record(cost)
        if self._monthly.limit_reached():
            raise BudgetExceededError(
                f"Monthly spend limit ${self._monthly.limit_usd:.2f} reached "
                f"(spend: ${self._monthly.current_spend():.6f}).",
                window="monthly",
                spend_usd=self._monthly.current_spend(),
                limit_usd=self._monthly.limit_usd,
                tenant_id=record.tenant_id,
            )
        if self._monthly.alert_threshold_reached():
            logger.warning(
                "cost_alert window=monthly fraction=%.2f limit=%.2f",
                self._monthly.fraction_used(),
                self._monthly.limit_usd,
            )
            raise BudgetAlertThresholdReached(
                f"Monthly spend alert: {self._monthly.fraction_used() * 100:.1f}% of "
                f"${self._monthly.limit_usd:.2f} limit used.",
                window="monthly",
                fraction_used=self._monthly.fraction_used(),
                alert_fraction=self._config.alert_fraction,
                tenant_id=record.tenant_id,
            )

        logger.info("cost_record %s", record.as_labels())

    def snapshot(self) -> dict[str, Any]:
        """Return a content-free spend snapshot for dashboards and /metrics."""
        with self._lock:
            tenants = {tid: w.snapshot() for tid, w in self._tenant_windows.items()}
        return {
            "daily": self._daily.snapshot(),
            "monthly": self._monthly.snapshot(),
            "tenants": tenants,
            "config": {
                "daily_limit_usd": self._config.daily_limit_usd,
                "monthly_limit_usd": self._config.monthly_limit_usd,
                "alert_fraction": self._config.alert_fraction,
                "per_call_limit_usd": self._config.per_call_limit_usd,
            },
        }

    def reset(self) -> None:
        """Reset all spend windows.  For testing only."""
        self._daily._spend_usd = 0.0
        self._daily._window_start = time.time()
        self._monthly._spend_usd = 0.0
        self._monthly._window_start = time.time()
        with self._lock:
            for w in self._tenant_windows.values():
                w._spend_usd = 0.0
                w._window_start = time.time()


# ---------------------------------------------------------------------------
# Emergency budget cutoff
# ---------------------------------------------------------------------------

#: Endpoints that must never be blocked by the budget cutoff.
_CRITICAL_PATHS: frozenset[str] = frozenset(
    {
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
    }
)


def is_critical_path(path: str) -> bool:
    """Return True when ``path`` must never be blocked by budget controls."""
    p = path.rstrip("/").lower()
    return any(p == cp or p.startswith(cp + "/") for cp in _CRITICAL_PATHS)


@dataclass
class DegradedResponse:
    """Safe, content-free degraded response returned when the budget is exhausted."""

    status: str = BUDGET_EXHAUSTED_STATUS
    user_message: str = BUDGET_EXHAUSTED_MESSAGE
    fallback_reason: str = "budget_exhausted"
    retry_after_seconds: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "user_message": self.user_message,
            "fallback_reason": self.fallback_reason,
            "retry_after_seconds": self.retry_after_seconds,
            "metadata": self.metadata,
        }


class BudgetCutoff:
    """Emergency cutoff that stops non-critical generation when the budget is hit.

    Wraps a ``SpendGuard`` and adds a manual emergency-stop flag that operators
    can flip without redeploying.  Health, readiness, and auth endpoints are
    never blocked.

    Retries and fallbacks must call ``check_and_record()`` for every attempt —
    including fallback paths — so the budget cannot be bypassed by routing
    around the primary provider.
    """

    def __init__(self, guard: SpendGuard) -> None:
        self._guard = guard
        self._emergency_stop = False
        self._stop_reason: str = ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Operator controls
    # ------------------------------------------------------------------

    def activate_emergency_stop(self, reason: str = "operator_initiated") -> None:
        """Immediately stop all non-critical generation regardless of spend."""
        with self._lock:
            self._emergency_stop = True
            self._stop_reason = reason
        logger.error("EMERGENCY_CUTOFF_ACTIVATED reason=%s", reason)

    def deactivate_emergency_stop(self) -> None:
        """Resume normal operation after the budget or incident is resolved."""
        with self._lock:
            self._emergency_stop = False
            self._stop_reason = ""
        logger.info("EMERGENCY_CUTOFF_DEACTIVATED")

    @property
    def is_emergency_stop_active(self) -> bool:
        with self._lock:
            return self._emergency_stop

    # ------------------------------------------------------------------
    # Request-time check
    # ------------------------------------------------------------------

    def check(self, path: str = "/", critical: bool = False) -> None:
        """Check whether generation is currently allowed for ``path``.

        Parameters
        ----------
        path:
            The request path, used to skip the cutoff for critical endpoints.
        critical:
            If ``True``, the cutoff is bypassed entirely (use for health / auth
            handlers).

        Raises
        ------
        BudgetExceededError
            When the emergency stop is active and the path is non-critical.
        """
        if critical or is_critical_path(path):
            return
        with self._lock:
            if self._emergency_stop:
                raise BudgetExceededError(
                    f"Emergency budget cutoff is active: {self._stop_reason}. Non-critical generation is suspended.",
                    window="emergency_stop",
                )

    def check_and_record(
        self,
        record: CallRecord,
        path: str = "/",
        critical: bool = False,
    ) -> DegradedResponse | None:
        """Check the budget *and* record the call cost.

        Returns ``None`` when the call is allowed.  Returns a ``DegradedResponse``
        when the budget is exhausted (callers should return this to the user).
        Never raises for critical paths.

        Retries and fallbacks must call this method for *every* attempt.
        """
        if critical or is_critical_path(path):
            # Still record spend even for allowed paths — just don't block.
            try:
                self._guard.record_call(record)
            except (BudgetExceededError, BudgetAlertThresholdReached):
                pass
            return None

        # Check emergency stop
        try:
            self.check(path=path, critical=False)
        except BudgetExceededError as exc:
            logger.error(
                "budget_cutoff path=%s window=%s spend=%.6f limit=%.6f",
                path,
                exc.window,
                exc.spend_usd,
                exc.limit_usd,
            )
            snap = self._guard.snapshot()
            return DegradedResponse(
                metadata={
                    "window": exc.window,
                    "daily_spend_usd": snap["daily"]["spend_usd"],
                    "monthly_spend_usd": snap["monthly"]["spend_usd"],
                }
            )

        # Record the call and check spend limits
        try:
            self._guard.record_call(record)
        except BudgetExceededError as exc:
            logger.error(
                "budget_exceeded path=%s window=%s spend=%.6f limit=%.6f tenant=%s",
                path,
                exc.window,
                exc.spend_usd,
                exc.limit_usd,
                exc.tenant_id or "",
            )
            # Activate the emergency stop so subsequent requests are fast-failed
            # without redundant budget computation.
            self.activate_emergency_stop(
                reason=f"Budget exceeded: {exc.window} spend ${exc.spend_usd:.4f} >= limit ${exc.limit_usd:.4f}"
            )
            snap = self._guard.snapshot()
            return DegradedResponse(
                metadata={
                    "window": exc.window,
                    "daily_spend_usd": snap["daily"]["spend_usd"],
                    "monthly_spend_usd": snap["monthly"]["spend_usd"],
                    "tenant_id": exc.tenant_id or "",
                }
            )
        except BudgetAlertThresholdReached as exc:
            logger.warning(
                "budget_alert path=%s window=%s fraction=%.3f",
                path,
                exc.window,
                exc.fraction_used,
            )
            # Alert threshold: allow the request but don't suppress the warning.
            # Downstream alerting infrastructure reads the logs.

        return None

    def snapshot(self) -> dict[str, Any]:
        """Return a content-free snapshot for /metrics and dashboards."""
        snap = self._guard.snapshot()
        with self._lock:
            snap["emergency_stop"] = {
                "active": self._emergency_stop,
                "reason": self._stop_reason,
            }
        return snap


# ---------------------------------------------------------------------------
# Module-level singleton (application-wide)
# ---------------------------------------------------------------------------

#: Default configuration.  Override via environment variables or by calling
#: ``configure_default_guard()`` before the application starts.
_DEFAULT_BUDGET = BudgetConfig(
    daily_limit_usd=float(__import__("os").getenv("BUDGET_DAILY_USD", "50.0")),
    monthly_limit_usd=float(__import__("os").getenv("BUDGET_MONTHLY_USD", "1000.0")),
    alert_fraction=float(__import__("os").getenv("BUDGET_ALERT_FRACTION", "0.80")),
    per_call_limit_usd=(float(v) if (v := __import__("os").getenv("BUDGET_PER_CALL_USD")) else None),
)

default_guard = SpendGuard(_DEFAULT_BUDGET)
default_cutoff = BudgetCutoff(default_guard)


def configure_default_guard(config: BudgetConfig) -> None:
    """Replace the module-level guard with a new configuration.

    Call this once at application startup *before* serving requests.
    """
    global default_guard, default_cutoff  # noqa: PLW0603
    default_guard = SpendGuard(config)
    default_cutoff = BudgetCutoff(default_guard)
