"""Rate-limit expensive public Stellar network lookup tools (#579)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RateLimitDecision:
    """Typed rate-limit decision with HTTP 429 guidance and zero identifier leakage."""

    allowed: bool
    status_code: int  # 200 or 429
    provider: str
    retry_after_seconds: int | None
    error_code: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StellarLookupRateLimiter:
    """Per-client and global weighted token-budget limiter for Stellar network tools."""

    def __init__(self, *, per_client_budget: int = 10, global_budget: int = 50, window_seconds: int = 60) -> None:
        self.per_client_budget = per_client_budget
        self.global_budget = global_budget
        self.window_seconds = window_seconds
        self._client_used: dict[str, int] = {}
        self._global_used: int = 0

    def check_and_consume(
        self,
        *,
        client_id: str,
        provider: str,
        operation_cost: int = 1,
        is_cache_hit: bool = False,
    ) -> RateLimitDecision:
        """Enforce client and global budgets before allowing cache or live lookup."""
        prov = provider.strip().lower()
        cost = max(1, int(operation_cost)) if not is_cache_hit else 1
        client_total = self._client_used.get(client_id, 0)

        if client_total + cost > self.per_client_budget or self._global_used + cost > self.global_budget:
            return RateLimitDecision(
                allowed=False,
                status_code=429,
                provider=prov,
                retry_after_seconds=self.window_seconds,
                error_code="STELLAR_LOOKUP_RATE_LIMITED",
                message=(
                    f"Rate limit exceeded for {prov} network lookup tools. "
                    f"Please retry after {self.window_seconds} seconds."
                ),
            )

        self._client_used[client_id] = client_total + cost
        self._global_used += cost
        return RateLimitDecision(
            allowed=True,
            status_code=200,
            provider=prov,
            retry_after_seconds=None,
        )
