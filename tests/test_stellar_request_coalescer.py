"""Tests for concurrent Stellar provider lookup coalescing (#576)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from stellar_request_coalescer import PublicLookupCoalescer


def test_coalesces_concurrent_identical_public_lookups_and_isolates_copies() -> None:
    coalescer = PublicLookupCoalescer()
    provider_calls = 0

    def slow_fetcher() -> dict[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        time.sleep(0.05)
        return {"ledger": 100, "balances": ["10.0"]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [
            pool.submit(
                coalescer.execute_or_join,
                network="public",
                resource_type="account",
                identifier="G" + "A" * 55,
                fetcher=slow_fetcher,
            )
            for _ in range(4)
        ]
        results = [f.result() for f in futs]

    assert provider_calls == 1
    results[0]["balances"].append("mutated")
    assert results[1]["balances"] == ["10.0"]
