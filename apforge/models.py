"""Pydantic models for the APforge core learning engine.

Every piece of state that crosses the learning loop is represented here:
agents and their versions, execution traces, observations, evaluations,
reflections (failure analysis), persistent tool-use memory, strategies
(learned preferences), and the results of a learning turn.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp used throughout the SQLite storage layer."""
    return datetime.now(timezone.utc).isoformat()


class TraceStatus(str, Enum):
    """Outcome of a tool execution."""

    SUCCESS = "success"
    ERROR = "error"


class FailureCategory(str, Enum):
    """Classification produced by the failure-analysis (REFLECT) step."""

    NONE = "none"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TOOL_SELECTION_MISMATCH = "tool_selection_mismatch"
    PARAMETER_ERROR = "parameter_error"
    ENVIRONMENT_ERROR = "environment_error"
    LOGIC_ERROR = "logic_error"


class ToolDef(BaseModel):
    """A tool registered in the engine's tool registry."""

    name: str
    description: str
    version: int = 1


class ToolResult(BaseModel):
    """Raw result of executing a tool."""

    tool_name: str
    status: TraceStatus
    result: Any = None
    error: Optional[str] = None
    duration_ms: int = 0


class Agent(BaseModel):
    """A learning agent. `version` is bumped whenever its strategy changes."""

    id: str
    name: str
    version: int = 1
    created_at: str


class AgentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class AgentVersion(BaseModel):
    """An immutable snapshot of the agent's strategy set at a given version."""

    agent_id: str
    version: int
    strategies: dict[str, Any] = Field(default_factory=dict)
    reason: str
    created_at: str


class ExecutionTrace(BaseModel):
    """A full record of one tool use (the USE step)."""

    id: str
    agent_id: str
    agent_version: int
    task_type: str
    tool_name: str
    arguments: Any = None
    status: TraceStatus
    result: Any = None
    error: Optional[str] = None
    started_at: str
    ended_at: str
    duration_ms: int


class Observation(BaseModel):
    """What was observed after a tool use (the OBSERVE step)."""

    trace_id: str
    status: TraceStatus
    result: Any = None
    error: Optional[str] = None
    duration_ms: int


class Evaluation(BaseModel):
    """Score of a tool use against task criteria (the EVALUATE step)."""

    id: str
    trace_id: str
    success: bool
    score: float = Field(ge=0.0, le=1.0)
    criteria: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class Reflection(BaseModel):
    """Failure analysis + lesson produced by the REFLECT step."""

    id: str
    trace_id: str
    evaluation_id: str
    failure_category: FailureCategory
    analysis: str
    lesson: str
    suggested_tool: Optional[str] = None


class ToolMemory(BaseModel):
    """Persistent tool-use memory row (the REMEMBER step)."""

    agent_id: str
    task_type: str
    tool_name: str
    success_count: int
    failure_count: int
    total_count: int
    success_rate: float
    last_used_at: Optional[str] = None


class Strategy(BaseModel):
    """Learned preference for one (agent, task type, tool) triple.

    `prior` is the baseline preference and `delta` accumulates reinforcement
    and penalties from the learning loop.
    """

    agent_id: str
    task_type: str
    tool_name: str
    prior: float = 1.0
    delta: float = 0.0
    reason: str = "initial"
    updated_at: str


class ToolSelection(BaseModel):
    """The tool the strategy engine chose for a task and why."""

    tool_name: str
    score: float
    rationale: str
    scores: dict[str, float] = Field(default_factory=dict)


class StrategyUpdate(BaseModel):
    """One strategy change applied in the CHANGE STRATEGY step."""

    agent_id: str
    task_type: str
    tool_name: str
    delta: float
    reason: str


class TaskSpec(BaseModel):
    """A unit of work fed into the learning loop."""

    task_type: str = Field(min_length=1)
    tool_input: Any = None
    criteria: dict[str, Any] = Field(default_factory=dict)


class SessionRequest(BaseModel):
    tasks: list[TaskSpec]


class TurnResult(BaseModel):
    """Full output of one iteration of the learning loop."""

    agent_id: str
    agent_version_before: int
    agent_version_after: int
    selection: ToolSelection
    trace: ExecutionTrace
    observation: Observation
    evaluation: Evaluation
    reflection: Reflection
    strategy_updates: list[StrategyUpdate] = Field(default_factory=list)
    version_reason: Optional[str] = None


class TraceDetail(BaseModel):
    """An execution trace enriched with its evaluation and reflection."""

    trace: ExecutionTrace
    evaluation: Optional[Evaluation] = None
    reflection: Optional[Reflection] = None