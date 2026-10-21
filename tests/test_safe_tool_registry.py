"""Tests for allowlisted Gemini function-calling tool registry (#25)."""

from __future__ import annotations

import pytest

from safe_tool_registry import RegisteredCapabilityTool, SafeCapabilityToolRegistry


def test_invokes_allowlisted_read_only_tool_and_blocks_unknown_or_mutating_tools() -> None:
    reg = SafeCapabilityToolRegistry()
    reg.register(
        RegisteredCapabilityTool(
            name="lookup_ayah",
            description="Look up a Quranic ayah by surah and ayah number.",
            required_args=("surah", "ayah"),
            read_only=True,
            handler=lambda a: {"ref": f"{a['surah']}:{a['ayah']}"},
        )
    )
    assert reg.invoke("lookup_ayah", {"surah": 2, "ayah": 255}) == {"ref": "2:255"}

    with pytest.raises(PermissionError):
        reg.invoke("delete_database", {})

    with pytest.raises(ValueError):
        reg.register(
            RegisteredCapabilityTool(
                name="mutating_tool",
                description="Forbidden",
                required_args=(),
                read_only=False,
                handler=lambda _: {},
            )
        )
