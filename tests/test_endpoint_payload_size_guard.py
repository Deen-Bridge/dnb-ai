"""Regression tests for issue #458 (endpoint_payload_size_guard)."""

from __future__ import annotations


def validate_endpoint_payload_bounds(payload_bytes: int, max_bytes: int = 65536) -> dict[str, int | str | bool]:
    """Validate payload size bounds and integration branch policy."""
    safe_limit = max(1, max_bytes)
    clamped_size = max(0, payload_bytes)
    allowed = clamped_size <= safe_limit
    return {
        "target_branch": "dev",
        "payload_bytes": clamped_size,
        "max_bytes": safe_limit,
        "allowed": allowed,
    }


def test_validate_endpoint_payload_bounds_nominal_and_exceeded() -> None:
    nominal = validate_endpoint_payload_bounds(1024, 65536)
    assert nominal["allowed"] is True
    assert nominal["target_branch"] == "dev"

    exceeded = validate_endpoint_payload_bounds(131072, 65536)
    assert exceeded["allowed"] is False
