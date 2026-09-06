"""The APforge learning engine.

Orchestrates one full iteration of the loop

    USE -> OBSERVE -> EVALUATE -> REFLECT -> REMEMBER -> CHANGE STRATEGY -> USE AGAIN

Each phase is a public method so individual steps can be tested and reused;
``turn`` / ``run_session`` compose them end to end.
"""

from __future__ import annotations

import uuid
from typing import Optional

from .db import Database
from .evaluation import RuleEvaluator
from .memory import ToolUseMemory
from .models import (
    ExecutionTrace,
    Observation,
    Reflection,
    TaskSpec,
    TurnResult,
    utcnow_iso,
)
from .reflection import ReflectionEngine
from .strategy import StrategyEngine
from .tools import ToolRegistry


class LearningEngine:
    def __init__(
        self,
        db: Database,
        registry: ToolRegistry,
        evaluator: Optional[RuleEvaluator] = None,
        reflector: Optional[ReflectionEngine] = None,
        strategy_engine: Optional[StrategyEngine] = None,
        memory: Optional[ToolUseMemory] = None,
    ):
        self.db = db
        self.registry = registry
        self.evaluator = evaluator or RuleEvaluator()
        self.reflector = reflector or ReflectionEngine()
        self.memory = memory or ToolUseMemory(db)
        self.strategy = strategy_engine or StrategyEngine(db, self.memory)

    # ------------------------------------------------------------------ phases

    def use(self, agent_id: str, task_type: str, tool_input: Optional[object] = None):
        """USE: select a tool via the strategy engine and execute it."""
        agent = self.db.get_agent(agent_id)
        if agent is None:
            raise KeyError(f"agent {agent_id} not found")

        selection = self.strategy.select_tool(agent_id, task_type, self.registry.list())

        started_at = utcnow_iso()
        result = self.registry.execute(selection.tool_name, tool_input)
        trace = ExecutionTrace(
            id=uuid.uuid4().hex,
            agent_id=agent.id,
            agent_version=agent.version,
            task_type=task_type,
            tool_name=selection.tool_name,
            arguments=tool_input,
            status=result.status,
            result=result.result,
            error=result.error,
            started_at=started_at,
            ended_at=utcnow_iso(),
            duration_ms=result.duration_ms,
        )
        self.db.insert_trace(trace)
        return selection, trace

    @staticmethod
    def observe(trace: ExecutionTrace) -> Observation:
        """OBSERVE: capture what actually happened."""
        return Observation(
            trace_id=trace.id,
            status=trace.status,
            result=trace.result,
            error=trace.error,
            duration_ms=trace.duration_ms,
        )

    def evaluate(
        self,
        observation: Observation,
        criteria: Optional[dict] = None,
    ) -> object:
        """EVALUATE: score the outcome against task criteria."""
        return self.evaluator.evaluate(observation, criteria=criteria)

    def reflect(
        self,
        evaluation: object,
        observation: Observation,
        agent_id: str,
        task_type: str,
        used_tool: str,
    ) -> Reflection:
        """REFLECT: analyze the failure and suggest a better tool."""
        return self.reflector.analyze(
            evaluation=evaluation,
            observation=observation,
            agent_id=agent_id,
            task_type=task_type,
            used_tool=used_tool,
            available_tools=self.registry.list(),
            strategy_engine=self.strategy,
        )

    def remember(self, trace: ExecutionTrace, evaluation: object) -> object:
        """REMEMBER: persist tool-use memory for this turn."""
        return self.memory.record(
            trace.agent_id,
            trace.task_type,
            trace.tool_name,
            success=evaluation.success,
            used_at=trace.ended_at,
        )

    def change_strategy(
        self,
        agent_id: str,
        task_type: str,
        trace: ExecutionTrace,
        evaluation: object,
        reflection: Reflection,
    ) -> tuple[list, Optional[str]]:
        """CHANGE STRATEGY: apply deltas and record a new agent version if changed."""
        updates = self.strategy.update(
            agent_id, task_type, trace, evaluation, reflection
        )
        if not updates:
            return updates, None
        reason = "; ".join(f"{u.tool_name}: {u.reason}" for u in updates)
        snapshot = self.db.snapshot_strategies(agent_id)
        self.db.bump_agent_version(agent_id, snapshot, reason)
        return updates, reason

    # ------------------------------------------------------------------ loops

    def turn(self, agent_id: str, task: TaskSpec) -> TurnResult:
        """Run one full iteration of the learning loop."""
        agent = self.db.get_agent(agent_id)
        if agent is None:
            raise KeyError(f"agent {agent_id} not found")

        before_version = agent.version

        selection, trace = self.use(agent_id, task.task_type, task.tool_input)
        observation = self.observe(trace)
        evaluation = self.evaluate(observation, task.criteria)
        reflection = self.reflect(
            evaluation, observation, agent_id, task.task_type, trace.tool_name
        )
        self.remember(trace, evaluation)
        updates, version_reason = self.change_strategy(
            agent_id, task.task_type, trace, evaluation, reflection
        )

        after = self.db.get_agent(agent_id)
        return TurnResult(
            agent_id=agent_id,
            agent_version_before=before_version,
            agent_version_after=after.version,
            selection=selection,
            trace=trace,
            observation=observation,
            evaluation=evaluation,
            reflection=reflection,
            strategy_updates=updates,
            version_reason=version_reason,
        )

    def run_session(self, agent_id: str, tasks: list[TaskSpec]) -> list[TurnResult]:
        """Feed several tasks through the loop one after another."""
        return [self.turn(agent_id, task) for task in tasks]