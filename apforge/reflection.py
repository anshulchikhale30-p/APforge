"""The REFLECT step: failure analysis and lessons.

Given an evaluation and the raw observation, classifies the failure and
suggests the best alternative tool according to the current strategy engine,
so the CHANGE STRATEGY step has something concrete to reinforce.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from .models import (
    Evaluation,
    FailureCategory,
    Observation,
    Reflection,
)

if TYPE_CHECKING:
    from .strategy import StrategyEngine
    from .tools import Tool


class ReflectionEngine:
    """Produces a Reflection (failure category, analysis, lesson, suggestion)."""

    def __init__(self) -> None:
        pass

    def analyze(
        self,
        *,
        evaluation: Evaluation,
        observation: Observation,
        agent_id: str,
        task_type: str,
        used_tool: str,
        available_tools: list["Tool"],
        strategy_engine: "StrategyEngine",
    ) -> Reflection:
        if evaluation.success:
            return Reflection(
                id=uuid.uuid4().hex,
                trace_id=observation.trace_id,
                evaluation_id=evaluation.id,
                failure_category=FailureCategory.NONE,
                analysis=(
                    f"{used_tool} completed successfully for task type '{task_type}'."
                ),
                lesson=f"{used_tool} is a reliable choice for '{task_type}'.",
                suggested_tool=None,
            )

        category, analysis = self._classify(observation)
        suggested = self._suggest_alternative(
            agent_id=agent_id,
            task_type=task_type,
            used_tool=used_tool,
            available_tools=available_tools,
            strategy_engine=strategy_engine,
        )
        if suggested:
            lesson = (
                f"For '{task_type}', {used_tool} failed ({category.value}); "
                f"try {suggested} instead."
            )
        else:
            lesson = (
                f"For '{task_type}', {used_tool} failed ({category.value}); "
                "no alternative tool available."
            )
        return Reflection(
            id=uuid.uuid4().hex,
            trace_id=observation.trace_id,
            evaluation_id=evaluation.id,
            failure_category=category,
            analysis=analysis,
            lesson=lesson,
            suggested_tool=suggested,
        )

    @staticmethod
    def _classify(observation: Observation) -> tuple[FailureCategory, str]:
        """Classify a failure from the raw error text."""
        error = (observation.error or "").lower()
        if not error:
            return (
                FailureCategory.LOGIC_ERROR,
                "tool failed without an error message.",
            )
        if any(
            marker in error
            for marker in ("unknown tool", "not registered", "unavailable")
        ):
            return (
                FailureCategory.TOOL_UNAVAILABLE,
                f"the tool could not be invoked: {observation.error}",
            )
        if any(
            marker in error
            for marker in ("timed out", "timeout", "connection", "network", "dns")
        ):
            return (
                FailureCategory.ENVIRONMENT_ERROR,
                f"the failure looks environmental: {observation.error}",
            )
        if any(
            marker in error
            for marker in (
                "parameter",
                "argument",
                "invalid",
                "unsupported",
                "typeerror",
                "valueerror",
                "keyerror",
            )
        ):
            return (
                FailureCategory.PARAMETER_ERROR,
                f"the tool was called with bad parameters: {observation.error}",
            )
        return (
            FailureCategory.LOGIC_ERROR,
            f"unclassified failure: {observation.error}",
        )

    def _suggest_alternative(
        self,
        *,
        agent_id: str,
        task_type: str,
        used_tool: str,
        available_tools: list["Tool"],
        strategy_engine: "StrategyEngine",
    ) -> Optional[str]:
        candidates = [tool for tool in available_tools if tool.name != used_tool]
        if not candidates:
            return None
        selection = strategy_engine.select_tool(agent_id, task_type, candidates)
        return selection.tool_name