"""Tests for the tool registry and built-in tools."""

import pytest

from apforge.models import TraceStatus
from apforge.tools import (
    Tool,
    ToolRegistry,
    UnreliableDemoTool,
    default_tools,
    safe_evaluate,
)


def test_math_evaluate_basic():
    assert safe_evaluate("2 + 3 * 4") == 14
    assert safe_evaluate("(2 + 3) * 4") == 20
    assert safe_evaluate("10 / 4") == 2.5
    assert safe_evaluate("2 ** 3") == 8


def test_math_evaluate_rejects_unsafe():
    for expr in (
        "__import__('os').system('dir')",
        "open('/etc/passwd')",
        "1;2",
        "x + 1",
        "len([])",
        "",
    ):
        with pytest.raises(ValueError):
            safe_evaluate(expr)


def test_math_evaluate_division_by_zero_is_captured():
    registry = ToolRegistry()
    registry.register_func(
        "math.evaluate", "safe math", lambda arg: safe_evaluate(arg)
    )
    result = registry.execute("math.evaluate", "1 / 0")
    assert result.status == TraceStatus.ERROR
    assert "ZeroDivisionError" in result.error


def test_registry_executes_and_captures_errors():
    registry = ToolRegistry()

    def boom(_):
        raise RuntimeError("always fails")

    registry.register_func("tools.fail", "fails", boom)
    registry.register_func("tools.ok", "succeeds", lambda arg: {"echo": arg})

    ok = registry.execute("tools.ok", {"value": "hi"})
    assert ok.status == TraceStatus.SUCCESS
    assert ok.result == {"echo": {"value": "hi"}}
    assert ok.duration_ms >= 0

    bad = registry.execute("tools.fail", None)
    assert bad.status == TraceStatus.ERROR
    assert "always fails" in bad.error

    missing = registry.execute("nope.tool", None)
    assert missing.status == TraceStatus.ERROR
    assert "unknown tool" in missing.error


def test_registry_rejects_duplicate():
    registry = ToolRegistry()
    registry.register_func("a", "a", lambda x: x)
    with pytest.raises(ValueError):
        registry.register_func("a", "b", lambda x: x)


def test_unreliable_demo_tool_is_deterministic():
    tool = UnreliableDemoTool(failure_window=2)
    registry = ToolRegistry()
    registry.register(Tool("demo.unreliable", "demo", tool))

    r1 = registry.execute("demo.unreliable", 1)
    r2 = registry.execute("demo.unreliable", 2)
    r3 = registry.execute("demo.unreliable", 3)
    assert (r1.status, r2.status) == (TraceStatus.ERROR, TraceStatus.ERROR)
    assert r3.status == TraceStatus.SUCCESS
    assert r3.result == {"attempt": 3, "value": 3}


def test_default_tools_present():
    registry = default_tools()
    names = {tool.name for tool in registry.list()}
    assert {"math.evaluate", "string.reverse", "store.get", "demo.unreliable"} <= names

    rev = registry.execute("string.reverse", "abc")
    assert rev.status == TraceStatus.SUCCESS
    assert rev.result == "cba"

    lookup = registry.execute("store.get", "city")
    assert lookup.result == "Tokyo"

    missing = registry.execute("store.get", "nope")
    assert missing.status == TraceStatus.ERROR
    assert "unknown key" in missing.error