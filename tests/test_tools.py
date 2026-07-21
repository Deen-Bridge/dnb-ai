"""Tests for the Gemini function-calling framework — no live API calls.

Uses a fake model client that emits scripted function_call / text parts.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from tools.registry import (
    MAX_TOOL_ROUNDS,
    Tool,
    ToolCallRecord,
    ToolRegistry,
    _clean_schema,
    get_tool_declarations,
    run_tool_handler,
)


# ---------------------------------------------------------------------------
# Fake model response helpers
# ---------------------------------------------------------------------------


class FakePart:
    def __init__(self, text=None, function_call=None):
        self._text = text
        self._function_call = function_call

    @property
    def text(self):
        return self._text

    @property
    def function_call(self):
        return self._function_call


class FakeCandidate:
    def __init__(self, part):
        self.content = MagicMock()
        self.content.parts = [part]


class FakeResponse:
    def __init__(self, part):
        self.candidates = [FakeCandidate(part)]


class FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


# ---------------------------------------------------------------------------
# Schema bridging tests
# ---------------------------------------------------------------------------


class SampleArgs(BaseModel):
    name: str = Field(..., description="A name")
    count: int = Field(5, description="A count")
    tags: list[str] = Field(default_factory=list, description="Tags")


def test_clean_schema_strips_unsupported_keys():
    raw = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$defs": {"Foo": {"type": "string"}},
        "title": "Sample",
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    cleaned = _clean_schema(raw)
    assert "$schema" not in cleaned
    assert "$defs" not in cleaned
    assert "title" not in cleaned
    assert cleaned["type"] == "object"
    assert cleaned["required"] == ["name"]


def test_get_tool_declarations_from_pydantic():
    tool = Tool(
        name="test_tool",
        description="A test tool",
        args_schema=SampleArgs,
        handler=lambda **kw: {"ok": True},
        timeout_seconds=5,
    )
    decls = get_tool_declarations([tool])
    assert len(decls) == 1
    decl = decls[0]
    assert decl["name"] == "test_tool"
    assert "description" in decl
    assert decl["parameters"]["type"] == "object"
    assert "name" in decl["parameters"]["properties"]
    assert "count" in decl["parameters"]["properties"]
    assert "tags" in decl["parameters"]["properties"]
    assert "name" in decl["parameters"]["required"]
    # Unsupported keys stripped
    assert "$schema" not in decl["parameters"]
    assert "$defs" not in decl["parameters"]


# ---------------------------------------------------------------------------
# Tool registry tests
# ---------------------------------------------------------------------------


def test_registry_register_and_get():
    registry = ToolRegistry()
    tool = Tool(
        name="hello",
        description="Says hello",
        args_schema=SampleArgs,
        handler=lambda **kw: {"msg": "hello"},
        timeout_seconds=1,
    )
    registry.register(tool)
    assert registry.get("hello") is tool
    assert registry.get("unknown") is None


def test_registry_duplicate_raises():
    registry = ToolRegistry()
    tool = Tool(
        name="dup",
        description="",
        args_schema=SampleArgs,
        handler=lambda **kw: {},
        timeout_seconds=1,
    )
    registry.register(tool)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_registry_all_and_declarations():
    registry = ToolRegistry()
    t1 = Tool(name="a", description="", args_schema=SampleArgs, handler=lambda **kw: {}, timeout_seconds=1)
    t2 = Tool(name="b", description="", args_schema=SampleArgs, handler=lambda **kw: {}, timeout_seconds=1)
    registry.register(t1)
    registry.register(t2)
    assert len(registry.all) == 2
    assert len(registry.declarations()) == 2


# ---------------------------------------------------------------------------
# Arg validation tests
# ---------------------------------------------------------------------------


class StrictArgs(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., gt=0)


def test_tool_handler_validates_args():
    handler_called = []

    def my_handler(name: str, age: int):
        handler_called.append((name, age))
        return {"ok": True}

    tool = Tool(
        name="strict",
        description="",
        args_schema=StrictArgs,
        handler=my_handler,
        timeout_seconds=5,
    )

    # Valid args
    result = run_tool_handler(tool, {"name": "Alice", "age": 30})
    assert result == {"ok": True}
    assert len(handler_called) == 1

    # Invalid args — fails pydantic validation
    handler_called.clear()
    result = run_tool_handler(tool, {"name": "", "age": -1})
    assert "error" in result


# ---------------------------------------------------------------------------
# Timeout enforcement tests
# ---------------------------------------------------------------------------


def slow_handler(**kw):
    time.sleep(10)
    return {"done": True}


def test_tool_timeout_is_enforced():
    tool = Tool(
        name="slow",
        description="",
        args_schema=SampleArgs,
        handler=slow_handler,
        timeout_seconds=1,
    )
    started = time.time()
    result = run_tool_handler(tool, {"name": "test", "count": 1})
    elapsed = time.time() - started
    assert "error" in result
    assert "timed out" in result["error"]
    assert elapsed < 5  # Should not wait the full 10s


# ---------------------------------------------------------------------------
# Loop bound tests — fake chat that emits function_calls
# ---------------------------------------------------------------------------


def _make_fake_chat(responses, final_text="Final answer"):
    """Create a fake chat session that yields scripted function_call responses."""
    chat = MagicMock()
    reply_index = [0]

    def side_effect(content, **kw):
        idx = reply_index[0]
        reply_index[0] += 1
        if idx < len(responses):
            fc_data = responses[idx]
            fc = FakeFunctionCall(name=fc_data["name"], args=fc_data.get("args", {}))
            return FakeResponse(FakePart(function_call=fc))
        return FakeResponse(FakePart(text=final_text))

    chat.send_message = side_effect
    return chat


@patch("tools.registry.run_tool_handler", return_value={"ok": True})
def test_loop_terminates_on_text(mock_handler):
    """When the model returns text (no function_call), loop exits immediately."""
    chat = _make_fake_chat([], "Direct text answer")
    from main import _run_with_tools

    # Need a real registry for _run_with_tools to work
    registry = ToolRegistry()
    tool = Tool(
        name="dummy",
        description="",
        args_schema=SampleArgs,
        handler=lambda **kw: {},
        timeout_seconds=5,
    )
    registry.register(tool)
    # Temp patch the module-level registry
    import main as main_module
    original_reg = main_module.tool_registry
    main_module.tool_registry = registry
    original_decls = main_module.tool_declarations
    main_module.tool_declarations = registry.declarations()

    try:
        text, calls = _run_with_tools(chat, "test prompt")
        assert text == "Direct text answer"
        assert calls == []
    finally:
        main_module.tool_registry = original_reg
        main_module.tool_declarations = original_decls


@patch("tools.registry.run_tool_handler", return_value={"ok": True})
def test_loop_bounded(mock_handler):
    """Model that keeps returning function_calls cannot loop forever."""
    # Script MAX_TOOL_ROUNDS function_call responses — after that the forced
    # text message will get the final_text from the mock.
    responses = [{"name": "dummy", "args": {"name": "x", "count": 1}} for _ in range(MAX_TOOL_ROUNDS)]
    chat = _make_fake_chat(responses, "Final after loop")
    from main import _run_with_tools

    registry = ToolRegistry()
    tool = Tool(
        name="dummy",
        description="",
        args_schema=SampleArgs,
        handler=lambda **kw: {},
        timeout_seconds=5,
    )
    registry.register(tool)
    import main as main_module
    original_reg = main_module.tool_registry
    main_module.tool_registry = registry
    original_decls = main_module.tool_declarations
    main_module.tool_declarations = registry.declarations()

    try:
        text, calls = _run_with_tools(chat, "test")
        # Should have MAX_TOOL_ROUNDS calls, then forced text
        assert len(calls) == MAX_TOOL_ROUNDS
        assert text == "Final after loop"
    finally:
        main_module.tool_registry = original_reg
        main_module.tool_declarations = original_decls


# ---------------------------------------------------------------------------
# Tool execution tests
# ---------------------------------------------------------------------------


def test_tool_handler_called_with_validated_args():
    recorded = {}

    def my_handler(name: str, count: int = 5, **kwargs):
        recorded["name"] = name
        recorded["count"] = count
        return {"result": f"Hello {name}"}

    tool = Tool(
        name="greet",
        description="Greets someone",
        args_schema=SampleArgs,
        handler=my_handler,
        timeout_seconds=5,
    )
    result = run_tool_handler(tool, {"name": "Alice", "count": 3})
    assert result == {"result": "Hello Alice"}
    assert recorded == {"name": "Alice", "count": 3}


def test_tool_handler_defaults():
    recorded = {}

    def my_handler(name: str, count: int = 5, **kwargs):
        recorded["count"] = count
        return {}

    tool = Tool(
        name="defaults",
        description="",
        args_schema=SampleArgs,
        handler=my_handler,
        timeout_seconds=5,
    )
    run_tool_handler(tool, {"name": "Bob"})
    assert recorded["count"] == 5


def test_tool_handler_exception_becomes_error():
    def broken(**kw):
        raise ValueError("Something broke")

    tool = Tool(
        name="broken",
        description="",
        args_schema=SampleArgs,
        handler=broken,
        timeout_seconds=5,
    )
    result = run_tool_handler(tool, {"name": "test"})
    assert "error" in result
    assert "Something broke" in result["error"]


# ---------------------------------------------------------------------------
# ToolCallRecord model
# ---------------------------------------------------------------------------


def test_tool_call_record_creation():
    rec = ToolCallRecord(
        tool_name="zakat",
        args={"public_key": "GABCDEF"},
        result='{"zakat_due": "10.5"}',
    )
    assert rec.tool_name == "zakat"
    assert rec.args["public_key"] == "GABCDEF"
    assert json.loads(rec.result)["zakat_due"] == "10.5"


# ---------------------------------------------------------------------------
# Clean schema edge cases
# ---------------------------------------------------------------------------


def test_clean_schema_non_dict():
    assert _clean_schema("string") == "string"
    assert _clean_schema(42) == 42
    assert _clean_schema([1, {"$schema": "x"}, 3]) == [1, {}, 3]
