"""Tests for bounded retry and timeout behavior (#575)."""

from __future__ import annotations

import pytest

from stellar_retry_policy import (
    ReadOnlyTimeoutPolicy,
    execute_bounded_readonly_request,
)


def test_retries_transient_timeout_up_to_bound_and_succeeds() -> None:
    policy = ReadOnlyTimeoutPolicy(max_retries=2)
    calls = 0

    def flaky_op(conn_to: float, read_to: float) -> dict[str, str]:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise TimeoutError("slow endpoint")
        return {"status": "ok"}

    res = execute_bounded_readonly_request(is_idempotent_read=True, policy=policy, operation=flaky_op)
    assert res.ok is True
    assert res.attempts_used == 2


def test_exhausts_retries_and_forbids_non_idempotent_requests() -> None:
    policy = ReadOnlyTimeoutPolicy(max_retries=1)
    res = execute_bounded_readonly_request(
        is_idempotent_read=True,
        policy=policy,
        operation=lambda c, r: (_ for _ in ()).throw(TimeoutError("timeout")),
    )
    assert res.ok is False
    assert res.attempts_used == 2
    assert res.error_code == "READ_TIMEOUT_EXCEEDED"

    with pytest.raises(ValueError, match="idempotent read"):
        execute_bounded_readonly_request(
            is_idempotent_read=False,
            policy=policy,
            operation=lambda c, r: {},
        )
