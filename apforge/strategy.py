"""The CHANGE STRATEGY step: learning mechanism that influences tool choice.

The strategy engine owns two things:

1. ``select_tool`` -- picks the next tool for a (agent, task type) by scoring
   every candidate as ``prior + learned delta + empirical memory factor``.
   This is the "USE AGAIN" half: the strategy state directly changes which
   tool gets used next.

2. ``update`` -- turns an evaluation + reflection into concrete strategy
   deltas (reward successes, penalize failures, boost suggested tools).
   Negative/positive deltas are clamped and persisted to SQLite; the engine
   bumps the agent version whenever a material update is applied.
"""

from __future__ import annotations

from typing import Optional

from .db import Database
from .memory import ToolUseMemory
from .models import (
    Evaluation,
    ExecutionTrace,
    Reflection,
    Strategy,
    StrategyUpdate,
    ToolSelection,
)
from .tools import Tool


class StrategyEngine:
    def __init__(
        self,
        db: Database,
        memory: ToolUseMemory,
        *,
        default_prior: float = 1.0,
        reward_step: float = 0.4,
        penalty_step: float = 1.0,
        suggested_boost: float = 0.6,
        memory_weight: float = 2.0,
        smoothing: float = 2.0,
        delta_clamp: tuple[float, float] = (-5.0, 5.0),
        success_threshold: float = 0.8,
        failure_threshold: float = 0.5,
    ):
        self.db = db
        self.memory = memory
        self.default_prior = default_prior
        self.reward_step = reward_step
        self.penalty_step = penalty_step
        self.suggested_boost = suggested_boost
        self.memory_weight = memory_weight
        self.smoothing = smoothing
        self.delta_clamp = delta_clamp
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold

    # ------------------------------------------------------------------ priors

    def set_prior(
        self,
        agent_id: str,
        task_type: str,
        tool_name: str,
        prior: float,
        reason: str = "manual prior",
    ) -> Strategy:
        """Set a baseline preference, e.g. to bias a fresh agent toward one tool."""
        return self.db.set_strategy(
            agent_id, task_type, tool_name, prior=prior, reason=reason
        )

    # ------------------------------------------------------------------ scoring

    def _score_tool(
        self, agent_id: str, task_type: str, tool: Tool
    ) -> tuple[float, dict[str, float]]:
        strategy = self.db.get_strategy(agent_id, task_type, tool.name)
        prior = strategy.prior if strategy else self.default_prior
        delta = strategy.delta if strategy else 0.0

        memory_row = self.memory.get(agent_id, task_type, tool.name)
        empirical = 0.0
        if memory_row:
            empirical = (
                (memory_row.success_count - memory_row.failure_count)
                / (memory_row.total_count + self.smoothing)
                * 2.0
                * self.memory_weight
            )
        score = prior + delta + empirical
        return score, {"prior": prior, "delta": delta, "memory": empirical}

    def select_tool(
        self,
        agent_id: str,
        task_type: str,
        tools: list[Tool],
        exclude: Optional[str] = None,
    ) -> ToolSelection:
        """Choose the best tool for a task. Ties go to registration order."""
        candidates = [tool for tool in tools if tool.name != exclude]
        if not candidates:
            raise ValueError("no candidate tools to select from")

        scores: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        for tool in candidates:
            score, parts = self._score_tool(agent_id, task_type, tool)
            scores[tool.name] = round(score, 4)
            components[tool.name] = {k: round(v, 4) for k, v in parts.items()}

        best: str = candidates[0].name
        for tool in candidates[1:]:
            if scores[tool.name] > scores[best]:
                best = tool.name

        detail = ", ".join(
            f"{name}={scores[name]:.2f} "
            f"(prior {components[name]['prior']:.2f}, "
            f"delta {components[name]['delta']:+.2f}, "
            f"memory {components[name]['memory']:+.2f})"
            for name in scores
        )
        rationale = f"selected {best} for task '{task_type}': {detail}"
        return ToolSelection(
            tool_name=best, score=scores[best], rationale=rationale, scores=scores
        )

    # ------------------------------------------------------------------ updates

    def update(
        self,
        agent_id: str,
        task_type: str,
        trace: ExecutionTrace,
        evaluation: Evaluation,
        reflection: Reflection,
    ) -> list[StrategyUpdate]:
        """Apply reinforcement/penalties; return the applied updates."""
        updates: list[StrategyUpdate] = []
        used_tool = trace.tool_name

        def adjust(tool_name: str, delta_change: float, reason: str) -> None:
            strategy = self.db.get_strategy(agent_id, task_type, tool_name)
            current = strategy.delta if strategy else 0.0
            low, high = self.delta_clamp
            new_delta = round(min(high, max(low, current + delta_change)), 4)
            if new_delta != current:
                self.db.set_strategy(
                    agent_id, task_type, tool_name, delta=new_delta, reason=reason
                )
                updates.append(
                    StrategyUpdate(
                        agent_id=agent_id,
                        task_type=task_type,
                        tool_name=tool_name,
                        delta=round(new_delta - current, 4),
                        reason=reason,
                    )
                )

        if evaluation.success and evaluation.score >= self.success_threshold:
            adjust(
                used_tool,
                self.reward_step,
                f"reinforced: {used_tool} succeeded on '{task_type}'",
            )
        elif not evaluation.success and evaluation.score < self.failure_threshold:
            adjust(
                used_tool,
                -self.penalty_step,
                f"penalized: {used_tool} failed on '{task_type}'",
            )
            suggested = reflection.suggested_tool
            if suggested and suggested != used_tool:
                adjust(
                    suggested,
                    self.suggested_boost,
                    f"boosted: suggested better tool {suggested} for '{task_type}'",
                )
        return updates