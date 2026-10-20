"""Regression tests for issue #453 (root_module_map_catalog)."""

from __future__ import annotations


def build_root_module_map_summary(payload_bytes: int, max_bytes: int = 65536) -> dict[str, int | str | bool]:
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


def test_build_root_module_map_summary_nominal_and_exceeded() -> None:
    nominal = build_root_module_map_summary(1024, 65536)
    assert nominal["allowed"] is True
    assert nominal["target_branch"] == "dev"

    exceeded = build_root_module_map_summary(131072, 65536)
    assert exceeded["allowed"] is False
