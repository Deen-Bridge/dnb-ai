"""Privacy-safe Stellar provider health and latency metrics (#578)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ALLOWED_PROVIDERS: frozenset[str] = frozenset({"horizon", "soroban_rpc"})
ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {"account_summary", "transaction_lookup", "orderbook", "fee_estimate", "simulation"}
)
ALLOWED_OUTCOMES: frozenset[str] = frozenset({"success", "not_found", "timeout", "rate_limited", "provider_error"})


@dataclass(frozen=True)
class SanitizedProviderCallMetric:
    """Metric event containing only aggregate operational dimensions and zero PII."""

    provider: str
    operation_category: str
    duration_ms: float
    outcome: str
    retry_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StellarProviderMetricsCollector:
    """Collect aggregate health and latency stats without logging addresses or hashes."""

    def __init__(self) -> None:
        self._events: list[SanitizedProviderCallMetric] = []

    def record_call(
        self,
        *,
        provider: str,
        operation_category: str,
        duration_ms: float,
        outcome: str,
        retry_count: int = 0,
    ) -> SanitizedProviderCallMetric:
        prov = provider.strip().lower()
        cat = operation_category.strip().lower()
        out = outcome.strip().lower()
        if prov not in ALLOWED_PROVIDERS:
            raise ValueError(f"Disallowed provider metric dimension: {provider}")
        if cat not in ALLOWED_CATEGORIES:
            raise ValueError(f"Disallowed operation_category dimension: {operation_category}")
        if out not in ALLOWED_OUTCOMES:
            raise ValueError(f"Disallowed outcome dimension: {outcome}")

        event = SanitizedProviderCallMetric(
            provider=prov,
            operation_category=cat,
            duration_ms=round(max(0.0, float(duration_ms)), 2),
            outcome=out,
            retry_count=max(0, int(retry_count)),
        )
        self._events.append(event)
        return event

    def summary(self) -> dict[str, Any]:
        total = len(self._events)
        errors = sum(1 for e in self._events if e.outcome in {"timeout", "rate_limited", "provider_error"})
        avg_ms = round(sum(e.duration_ms for e in self._events) / total, 2) if total > 0 else 0.0
        return {
            "total_calls": total,
            "error_calls": errors,
            "average_duration_ms": avg_ms,
        }
