"""Tests for Stellar network lookup rate limiter (#579)."""

from __future__ import annotations

from stellar_lookup_rate_limiter import StellarLookupRateLimiter


def test_enforces_weighted_limits_and_returns_typed_429_even_on_cache_hit() -> None:
    limiter = StellarLookupRateLimiter(per_client_budget=5, global_budget=10, window_seconds=30)

    d1 = limiter.check_and_consume(client_id="client-1", provider="soroban_rpc", operation_cost=4)
    assert d1.allowed is True
    assert d1.status_code == 200

    d2 = limiter.check_and_consume(client_id="client-1", provider="soroban_rpc", operation_cost=2)
    assert d2.allowed is False
    assert d2.status_code == 429
    assert d2.retry_after_seconds == 30

    limiter.check_and_consume(client_id="client-1", provider="horizon", operation_cost=1, is_cache_hit=True)
    d3 = limiter.check_and_consume(client_id="client-1", provider="horizon", operation_cost=1, is_cache_hit=True)
    assert d3.allowed is False
    assert d3.status_code == 429
