"""Tests for tool-use reliability improvements in agents."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tool_reliability import (
    ToolDescription,
    ToolReliabilityEngine,
    router,
    tool_reliability_engine,
)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
py_engine_reset = lambda: tool_reliability_engine._registry.clear() or tool_reliability_engine._handlers.clear() or tool_reliability_engine._traces.clear() or tool_reliability_engine._cache.clear()


def test_tool_registration_and_listing(client) -> None:
    desc = ToolDescription(
        name="calculator",
        description="Performs arithmetic calculations",
        capabilities=["math", "computation"],
        parameters_schema={
            "type": "object",
            "required": ["expression"],
            "properties": {"expression": {"type": "string"}},
        },
    )
    tool_reliability_engine.register_tool(desc, lambda expression: eval(expression))

    response = client.get("/tool-reliability/tools")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "calculator"


def test_tool_selection_logic() -> None:
    desc1 = ToolDescription(
        name="hadith_lookup",
        description="Looks up hadith texts by reference",
        capabilities=["hadith", "sunnah"],
    );
    desc2 = ToolDescription(
        name="zakat_calculator",
        description="Calculates nisab and zakat obligations",
        capabilities=["finance", "zakat"],
    );
    tool_reliability_engine.register_tool(desc1, lambda ref: ref)
    tool_reliability_engine.register_tool(desc2, lambda amt: amt)

    best = tool_reliability_engine.select_best_tool("What is the zakat on my wallet balance?")
    assert best == "zakat_calculator"


def test_parameter_validation_failure() -> None:
    desc = ToolDescription(
        name="fetch_data",
        description="Fetches data",
        parameters_schema={
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "integer"}},
        },
    )
    tool_reliability_engine.register_tool(desc, lambda id: id)

    # Missing required parameter
    valid, err = tool_reliability_engine.validate_parameters("fetch_data", {})
    assert valid is False
    assert "Missing required parameter 'id'" in err

    # Invalid type
    valid2, err2 = tool_reliability_engine.validate_parameters(
        "fetch_data", {"id": "not-an-int"}
    )
    assert valid2 is False
    assert "must be numeric" in err2


@pytest.mark.asyncio
async def test_tool_invocation_success_and_caching() -> None:
    call_count = 0
    def mock_tool(x: int):
        nonlocal call_count
        call_count += 1
        return x * 2

    desc = ToolDescription(
        name="double",
        description="Doubles a number",
        parameters_schema={
            "type": "object",
            "required": ["x"],
            "properties": {"x": {"type": "integer"}},
        },
    )
    tool_reliability_engine.register_tool(desc, mock_tool)

    res1 = await tool_reliability_engine.invoke_tool("double", {"x": 21})
    assert res1.success is True
    assert res1.output == 42
    assert res1.cached is False

    # Second call should hit cache
    res2 = await tool_reliability_engine.invoke_tool("double", {"x": 21})
    assert res2.success is True
    assert res2.output == 42
    assert res2.cached is True
    assert call_count == 1


@pytest.mark.asyncio
async def test_tool_retry_and_error_handling() -> None:
    attempts = 0
    def flaky_tool():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("Temporary failure")
        return "success"

    desc = ToolDescription(
        name="flaky",
        description="A flaky tool",
        parameters_schema={"type": "object"},
    )
    tool_reliability_engine.register_tool(desc, flaky_tool)

    res = await tool_reliability_engine.invoke_tool("flaky", {})
    assert res.success is True
    assert res.output == "success"
    assert res.retry_count == 2


def test_tool_call_optimization_sequence() -> None:
    from tool_reliability import ToolCallRequest
    reqs = [
        ToolCallRequest(tool_name="t1", parameters={"a": 1}),
        ToolCallRequest(tool_name="t1", parameters={"a": 1}),  # Duplicate
        ToolCallRequest(tool_name="t2", parameters={"b": 2}),
    ]
    optimized = tool_reliability_engine.optimize_sequence(reqs)
    assert len(optimized) == 2
    assert optimized[0].tool_name == "t1"
    assert optimized[1].tool_name == "t2"
