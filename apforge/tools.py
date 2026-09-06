"""Tool registry and execution for the APforge learning engine.

A tool is a named callable. Execution always produces a structured
`ToolResult` (success or error) so failures can be traced, evaluated,
and reflected upon instead of raising out of the loop.
"""

from __future__ import annotations

import ast
import operator
from time import perf_counter
from typing import Any, Callable, Optional

from .models import ToolDef, ToolResult, TraceStatus


class Tool:
    """A registered tool the agent may invoke."""

    def __init__(self, name: str, description: str, func: Callable[[Any], Any], version: int = 1):
        self.name = name
        self.description = description
        self.func = func
        self.version = version

    def to_def(self) -> ToolDef:
        return ToolDef(name=self.name, description=self.description, version=self.version)


class ToolRegistry:
    """Ordered registry of tools. Execution never raises: errors become results."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._order: list[str] = []

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        self._order.append(tool.name)
        return tool

    def register_func(
        self,
        name: str,
        description: str,
        func: Callable[[Any], Any],
        version: int = 1,
    ) -> Tool:
        return self.register(Tool(name, description, func, version=version))

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return [self._tools[name] for name in self._order]

    def execute(self, name: str, arguments: Any) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                status=TraceStatus.ERROR,
                error=f"unknown tool: {name}",
                duration_ms=0,
            )
        start = perf_counter()
        try:
            result = tool.func(arguments)
            return ToolResult(
                tool_name=name,
                status=TraceStatus.SUCCESS,
                result=result,
                duration_ms=int((perf_counter() - start) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 - failures are data for the loop
            return ToolResult(
                tool_name=name,
                status=TraceStatus.ERROR,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=int((perf_counter() - start) * 1000),
            )


def _single_value(arg: Any) -> Any:
    """Unwrap a bare value that callers may pass either directly or as {'value': x}."""
    if isinstance(arg, dict) and "value" in arg and set(arg) <= {"value"}:
        return arg["value"]
    return arg


def _expression_value(arg: Any) -> str:
    if isinstance(arg, dict):
        return arg.get("expression", "")
    return arg if isinstance(arg, str) else ""


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS: dict[type, Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_evaluate(expression: str) -> float:
    """Evaluate a basic arithmetic expression without using eval().

    Only numeric literals and +, -, *, /, //, %, ** are accepted; anything
    else (function calls, attribute access, names) is rejected.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"malformed expression: {exc}") from exc
    return _eval_math_node(tree.body)


def _eval_math_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](
            _eval_math_node(node.left), _eval_math_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_math_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def _math_evaluate(arg: Any) -> float:
    return safe_evaluate(_expression_value(arg))


def _string_reverse(arg: Any) -> str:
    value = _single_value(arg)
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {type(value).__name__}")
    return value[::-1]


_KNOWLEDGE_STORE: dict[str, Any] = {
    "city": "Tokyo",
    "language": "python",
    "pi": 3.14159,
}


def _store_get(arg: Any) -> Any:
    key = _single_value(arg)
    if not isinstance(key, str):
        raise TypeError(f"expected a string key, got {type(key).__name__}")
    if key not in _KNOWLEDGE_STORE:
        raise KeyError(f"unknown key: {key}")
    return _KNOWLEDGE_STORE[key]


class UnreliableDemoTool:
    """Deterministic demo tool: fails for the first N calls, then succeeds."""

    def __init__(self, failure_window: int = 3):
        self.failure_window = failure_window
        self._calls = 0

    def __call__(self, arg: Any) -> dict[str, Any]:
        self._calls += 1
        if self._calls <= self.failure_window:
            raise RuntimeError("upstream API unavailable (demo failure)")
        return {"attempt": self._calls, "value": _single_value(arg)}

    def reset(self) -> None:
        self._calls = 0


def default_tools() -> ToolRegistry:
    """The built-in toolset shipped with the engine."""
    registry = ToolRegistry()
    registry.register_func(
        "math.evaluate",
        "Safely evaluate a basic arithmetic expression (numbers and + - * / // % **).",
        _math_evaluate,
    )
    registry.register_func(
        "string.reverse",
        "Reverse a string.",
        _string_reverse,
    )
    registry.register_func(
        "store.get",
        "Look up a value by key in the built-in knowledge store (city, language, pi).",
        _store_get,
    )
    registry.register(
        Tool(
            "demo.unreliable",
            "Simulates an upstream API that fails for the first 3 calls, then succeeds.",
            UnreliableDemoTool(failure_window=3),
        )
    )
    return registry