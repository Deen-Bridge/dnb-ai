"""Safe function-calling tool registry with allowlist and argument validation (#25)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegisteredCapabilityTool:
    name: str
    description: str
    required_args: tuple[str, ...]
    read_only: bool
    handler: Callable[[dict[str, Any]], dict[str, Any]]


class SafeCapabilityToolRegistry:
    """Allowlisted registry preventing arbitrary or unvalidated tool execution."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredCapabilityTool] = {}

    def register(self, tool: RegisteredCapabilityTool) -> None:
        if not tool.read_only:
            raise ValueError(f"Tool '{tool.name}' must be read_only=True in the safe capability registry.")
        self._tools[tool.name] = tool

    def invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise PermissionError(f"Tool '{name}' is not allowlisted.")
        tool = self._tools[name]
        missing = [k for k in tool.required_args if k not in args]
        if missing:
            raise ValueError(f"Missing required arguments for '{name}': {missing}")
        return tool.handler(args)
