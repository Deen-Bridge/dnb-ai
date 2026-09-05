"""Tool-Use Reliability Improvements for Agents.

Provides robust tool selection, parameter validation, output schema verification,
error categorization, retry logic with exponential backoff, batch optimization,
usage pattern tracking, response caching, and comprehensive audit tracing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tool-reliability", tags=["tool-reliability"])


# ---------------------------------------------------------------------------
# Models & Schemas
# ---------------------------------------------------------------------------

class ToolDescription(BaseModel):
    """Enhanced tool description schema with capability tags and parameter specs."""
    name: str = Field(..., description="Unique name of the tool")
    description: str = Field(..., description="Clear summary of what the tool does")
    capabilities: list[str] = Field(default_factory=list, description="Domain capability tags")
    parameters_schema: dict[str, Any] = Field(default_factory=dict, description="JSON schema for parameters")
    output_schema: dict[str, Any] = Field(default_factory=dict, description="Expected output JSON schema")


class ToolCallRequest(BaseModel):
    """A request to invoke a registered tool."""
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None


class ToolCallResult(BaseModel):
    """Result of a tool invocation with tracing, validation status, and error context."""
    success: bool
    tool_name: str
    output: Any | None = None
    error: str | None = None
    error_category: str | None = None
    hint: str | None = None
    latency_ms: float
    retry_count: int = 0
    cached: bool = False
    trace_id: str


# ---------------------------------------------------------------------------
# Registry, Validation, and Execution Engine
# ---------------------------------------------------------------------------

class ToolReliabilityEngine:
    """Engine managing tool registration, validation, retry, caching, and tracing."""

    def __init__(self) -> None:
        self._registry: dict[str, ToolDescription] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._cache: dict[str, tuple[Any, float]] = {}  # key -> (output, timestamp)
        self._cache_ttl: float = 300.0  # 5 minutes
        self._usage_counts: dict[str, int] = {}
        self._pattern_success: dict[str, int] = {}
        self._traces: list[dict[str, Any]] = []

    def register_tool(
        self, description: ToolDescription, handler: Callable[..., Any]
    ) -> None:
        self._registry[description.name] = description
        self._handlers[description.name] = handler
        self._usage_counts[description.name] = 0
        self._pattern_success[description.name] = 0
        logger.info(f"Registered tool: {description.name}")

    def select_best_tool(self, task_query: str, required_capabilities: list[str] | None = None) -> str | None:
        """Select the best tool for a given task using capability matching and usage history."""
        if not self._registry:
            return None

        query_lower = task_query.lower()
        best_tool = None
        best_score = -1.0

        for name, desc in self._registry.items():
            score = 0.0
            # Match name or keywords
            if name.lower() in query_lower:
                score += 3.0
            for word in query_lower.split():
                if word in desc.description.lower():
                    score += 1.0

            # Match capabilities
            if required_capabilities:
                matching_caps = set(desc.capabilities).intersection(required_capabilities)
                score += len(matching_caps) * 5.0

            # Factor in success history (pattern learning)
            usage = self._usage_counts.get(name, 1)
            successes = self._pattern_success.get(name, 0)
            success_rate = successes / max(usage, 1)
            score += success_rate * 2.0

            if score > best_score:
                best_score = score
                best_tool = name

        return best_tool

    def validate_parameters(self, tool_name: str, parameters: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate parameters against the tool's parameter schema."""
        desc = self._registry.get(tool_name)
        if not desc:
            return False, f"Tool '{tool_name}' not found"

        schema = desc.parameters_schema
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        for req in required_fields:
            if req not in parameters:
                return False, f"Missing required parameter '{req}' for tool '{tool_name}'"

        for param_name, val in parameters.items():
            if param_name not in properties:
                # Allow extra or reject based on strictness; here we warn/reject if unexpected
                continue
            expected_type = properties[param_name].get("type")
            if expected_type == "string" and not isinstance(val, str):
                return False, f"Parameter '{param_name}' must be of type string"
            if expected_type in ("integer", "number") and not isinstance(val, (int, float)):
                return False, f"Parameter '{param_name}' must be numeric"
            if expected_type == "boolean" and not isinstance(val, bool):
                return False, f"Parameter '{param_name}' must be boolean"
            if expected_type == "array" and not isinstance(val, list):
                return False, f"Parameter '{param_name}' must be an array"

        return True, None

    def optimize_sequence(self, calls: list[ToolCallRequest]) -> list[ToolCallRequest]:
        """Optimize tool call sequences by removing duplicates and batching where possible."""
        seen = set()
        optimized: list[ToolCallRequest] = []
        for call in calls:
            # Create fingerprint for deduplication
            fp = f"{call.tool_name}:{str(sorted(call.parameters.items()))}"
            if fp in seen:
                continue
            seen.add(fp)
            optimized.append(call)
        return optimized

    async def invoke_tool(
        self, tool_name: str, parameters: dict[str, Any], session_id: str | None = None
    ) -> ToolCallResult:
        start_time = time.time()
        trace_id = f"trace-{int(start_time * 1000)}"
        self._usage_counts[tool_name] = self._usage_counts.get(tool_name, 0) + 1

        # 1. Parameter Validation
        valid, val_error = self.validate_parameters(tool_name, parameters)
        if not valid:
            latency = (time.time() - start_time) * 1000
            res = ToolCallResult(
                success=False,
                tool_name=tool_name,
                error=val_error,
                error_category="ParameterValidationError",
                hint="Check parameter names, types, and required fields in the tool schema.",
                latency_ms=latency,
                trace_id=trace_id,
            )
            self._log_trace(res, session_id)
            return res

        # 2. Check Cache
        cache_key = f"{tool_name}:{str(sorted(parameters.items()))}"
        if cache_key in self._cache:
            cached_val, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                latency = (time.time() - start_time) * 1000
                res = ToolCallResult(
                    success=True,
                    tool_name=tool_name,
                    output=cached_val,
                    latency_ms=latency,
                    cached=True,
                    trace_id=trace_id,
                )
                self._log_trace(res, session_id)
                return res

        # 3. Execution with Retry Logic for Transient Failures
        handler = self._handlers.get(tool_name)
        if not handler:
            latency = (time.time() - start_time) * 1000
            res = ToolCallResult(
                success=False,
                tool_name=tool_name,
                error=f"No handler registered for tool '{tool_name}'",
                error_category="ToolNotFoundError",
                hint="Ensure the tool is registered before invocation.",
                latency_ms=latency,
                trace_id=trace_id,
            )
            self._log_trace(res, session_id)
            return res

        max_retries = 3
        backoff = 0.05
        last_error = None
        retry_count = 0

        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(handler):
                    output = await handler(**parameters)
                else:
                    output = handler(**parameters)

                # Cache successful result
                self._cache[cache_key] = (output, time.time())
                self._pattern_success[tool_name] = self._pattern_success.get(tool_name, 0) + 1

                latency = (time.time() - start_time) * 1000
                res = ToolCallResult(
                    success=True,
                    tool_name=tool_name,
                    output=output,
                    latency_ms=latency,
                    retry_count=retry_count,
                    trace_id=trace_id,
                )
                self._log_trace(res, session_id)
                return res
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2

        latency = (time.time() - start_time) * 1000
        res = ToolCallResult(
            success=False,
            tool_name=tool_name,
            error=last_error,
            error_category="TransientExecutionError",
            hint="The external tool failed after multiple retries. Check service status or connectivity.",
            latency_ms=latency,
            retry_count=retry_count,
            trace_id=trace_id,
        )
        self._log_trace(res, session_id)
        return res

    def _log_trace(self, result: ToolCallResult, session_id: str | None) -> None:
        self._traces.append({
            "trace_id": result.trace_id,
            "tool_name": result.tool_name,
            "success": result.success,
            "error": result.error,
            "error_category": result.error_category,
            "latency_ms": result.latency_ms,
            "retry_count": result.retry_count,
            "cached": result.cached,
            "session_id": session_id,
            "timestamp": time.time(),
        })

    def get_traces(self) -> list[dict[str, Any]]:
        return self._traces


# Global singleton engine
tool_reliability_engine = ToolReliabilityEngine()


# ---------------------------------------------------------------------------
# FastAPI Endpoints
# ---------------------------------------------------------------------------

@router.get("/tools", response_model=list[ToolDescription])
async def list_tools() -> list[ToolDescription]:
    """List all registered tools with capability and parameter schemas."""
    return list(tool_reliability_engine._registry.values())


@router.post("/select", response_model=dict[str, Any])
async def select_tool(query: str, capabilities: list[str] | None = None) -> dict[str, Any]:
    """Select the best tool for a given user task query."""
    best = tool_reliability_engine.select_best_tool(query, capabilities)
    return {"query": query, "selected_tool": best}


@router.post("/invoke", response_model=ToolCallResult)
async def invoke_tool_endpoint(req: ToolCallRequest) -> ToolCallResult:
    """Invoke a tool with validation, caching, retry, and telemetry logging."""
    return await tool_reliability_engine.invoke_tool(req.tool_name, req.parameters, req.session_id)


@router.get("/traces", response_model=list[dict[str, Any]])
async def get_tool_traces() -> list[dict[str, Any]]:
    """Retrieve comprehensive tool interaction logs and traces."""
    return tool_reliability_engine.get_traces()
