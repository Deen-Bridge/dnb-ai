"""Tool definition, registry, and schema bridging for Gemini function-calling.

Safety
------
Tool handlers must be read-only against external systems.
Nothing outside the explicit allowlist is ever callable by the model.
Every invocation is logged with name, truncated args, duration, and outcome.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    args_schema: type[BaseModel]
    handler: Callable[..., dict[str, Any]]
    timeout_seconds: int = 10


class ToolCallRecord(BaseModel):
    tool_name: str
    args: dict[str, Any]
    result: str


# ---------------------------------------------------------------------------
# Schema bridging — pydantic v2 model_json_schema() -> Gemini declaration
# ---------------------------------------------------------------------------

_UNSUPPORTED_KEYS = frozenset({"$schema", "$defs", "title"})


def _clean_schema(d: Any) -> Any:
    """Recursively strip keys Gemini's FunctionDeclaration does not accept."""
    if isinstance(d, dict):
        return {k: _clean_schema(v) for k, v in d.items() if k not in _UNSUPPORTED_KEYS}
    if isinstance(d, list):
        return [_clean_schema(v) for v in d]
    return d


def get_tool_declarations(tools: list[Tool]) -> list[dict[str, Any]]:
    """Build Gemini FunctionDeclaration dicts from a list of Tool objects."""
    declarations = []
    for tool in tools:
        raw = tool.args_schema.model_json_schema()
        cleaned = _clean_schema(raw)
        declarations.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": cleaned,
        })
    return declarations


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Explicit allowlist of tools callable by the model."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    @property
    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def declarations(self) -> list[dict[str, Any]]:
        return get_tool_declarations(self.all)


# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------


def run_tool_handler(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    """Validate args through pydantic and execute the handler with timeout."""
    try:
        validated = tool.args_schema.model_validate(args)
        validated_args = validated.model_dump(exclude_unset=True)
    except Exception as exc:
        return {"error": f"Argument validation failed for '{tool.name}': {exc}"}

    started = time.perf_counter()
    logger.info("Tool call: %s args=%s", tool.name, _truncate(validated_args, 200))

    try:
        future = _EXECUTOR.submit(tool.handler, **validated_args)
        result = future.result(timeout=tool.timeout_seconds)
    except FutureTimeout:
        result = {"error": f"Tool '{tool.name}' timed out after {tool.timeout_seconds}s"}
        logger.warning("Tool timeout: %s (%ss)", tool.name, tool.timeout_seconds)
    except Exception as exc:
        result = {"error": f"Tool '{tool.name}' error: {exc}"}
        logger.error("Tool error: %s — %s", tool.name, exc)

    duration = time.perf_counter() - started
    logger.info(
        "Tool result: %s duration=%.2fs outcome=%s",
        tool.name,
        duration,
        "error" if "error" in result else "ok",
    )
    return result


def _truncate(obj: Any, limit: int = 200) -> Any:
    s = str(obj)
    return s[:limit] + "..." if len(s) > limit else s
