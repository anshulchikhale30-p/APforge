"""Tests for the CHANGE STRATEGY step: scoring, updates, and behavior change."""

from apforge.db import Database
from apforge.memory import ToolUseMemory
from apforge.models import (
    Evaluation,
    ExecutionTrace,
    FailureCategory,
    Reflection,
    TraceStatus,
    utcnow_iso,
)
from apforge.strategy import StrategyEngine
from apforge.tools import ToolRegistry


def make_registry(bad_first=True):
    registry = ToolRegistry()

    def fail(_):
        raise RuntimeError("always fails")

    def ok(arg):
        return {"status": "ok", "echo": arg}

    if bad_first:
        registry.register_func("api.a", "unreliable api", fail)
        registry.register_func("api.b", "reliable api", ok)
    else:
        registry.register_func("api.b", "reliable api", ok)
        registry.register_func("api.a", "unreliable api", fail)
    return registry


def make_trace(agent_id, tool_name, task_type="api_call", status=TraceStatus.ERROR):
    now = utcnow_iso()
    return ExecutionTrace(
        id=f"trace-{tool_name}-{status.value}",
        agent_id=agent_id,
        agent_version=1,
        task_type=task_type,
        tool_name=tool_name,
        arguments=None,
        status=status,
        error="boom" if status == TraceStatus.ERROR else None,
        started_at=now,
        ended_at=now,
        duration_ms=1,
    )


def make_evaluation(trace, success):
    return Evaluation(
        id=f"ev-{trace.id}",
        trace_id=trace.id,
        success=success,
        score=1.0 if success else 0.0,
        notes="ok" if success else "failed",
    )


def make_reflection(trace, evaluation, suggested=None):
    return Reflection(
        id=f"re-{trace.id}",
        trace_id=trace.id,
        evaluation_id=evaluation.id,
        failure_category=(
            FailureCategory.NONE if evaluation.success else FailureCategory.LOGIC_ERROR
        ),
        analysis="x",
        lesson="y",
        suggested_tool=suggested,
    )


def test_selection_uses_prior_bias(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)
    registry = make_registry()

    strategy.set_prior(agent.id, "api_call", "api.a", prior=3.0)
    selection = strategy.select_tool(agent.id, "api_call", registry.list())
    assert selection.tool_name == "api.a"
    assert selection.rationale
    db.close()


def test_ties_go_to_registration_order(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)
    registry = make_registry()

    selection = strategy.select_tool(agent.id, "api_call", registry.list())
    assert selection.tool_name == "api.a"  # registered first, no learned info yet
    db.close()


def test_exclude_skips_tool(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)
    registry = make_registry()

    selection = strategy.select_tool(
        agent.id, "api_call", registry.list(), exclude="api.a"
    )
    assert selection.tool_name == "api.b"
    db.close()


def test_failure_penalizes_used_tool_and_boosts_suggestion(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)

    trace = make_trace(agent.id, "api.a", status=TraceStatus.ERROR)
    evaluation = make_evaluation(trace, success=False)
    reflection = make_reflection(trace, evaluation, suggested="api.b")

    updates = strategy.update(
        agent.id, trace.task_type, trace, evaluation, reflection
    )
    names = {u.tool_name for u in updates}
    assert names == {"api.a", "api.b"}

    a = db.get_strategy(agent.id, "api_call", "api.a")
    b = db.get_strategy(agent.id, "api_call", "api.b")
    assert a.delta <= -strategy.penalty_step
    assert b.delta >= strategy.suggested_boost
    assert "penalized" in a.reason
    assert "boosted" in b.reason
    db.close()


def test_success_reinforces_used_tool(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)

    trace = make_trace(agent.id, "api.b", status=TraceStatus.SUCCESS)
    evaluation = make_evaluation(trace, success=True)
    reflection = make_reflection(trace, evaluation)

    updates = strategy.update(
        agent.id, trace.task_type, trace, evaluation, reflection
    )
    assert [u.tool_name for u in updates] == ["api.b"]
    assert db.get_strategy(agent.id, "api_call", "api.b").delta >= strategy.reward_step
    db.close()


def test_delta_clamping(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)

    trace = make_trace(agent.id, "api.a", status=TraceStatus.ERROR)
    evaluation = make_evaluation(trace, success=False)
    reflection = make_reflection(trace, evaluation)

    for _ in range(50):
        strategy.update(agent.id, trace.task_type, trace, evaluation, reflection)

    a = db.get_strategy(agent.id, "api_call", "api.a")
    assert a.delta == strategy.delta_clamp[0]
    db.close()


def test_empirical_memory_changes_selection_without_strategy_rows(tmp_path):
    """Pure memory (no strategy deltas) must change future tool selection."""
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)
    registry = make_registry()

    # Initially the first-registered bad tool wins.
    assert strategy.select_tool(agent.id, "api_call", registry.list()).tool_name == "api.a"

    # Record several failures for api.a and successes for api.b in memory.
    for _ in range(3):
        memory.record(agent.id, "api_call", "api.a", success=False)
    for _ in range(3):
        memory.record(agent.id, "api_call", "api.b", success=True)

    selection = strategy.select_tool(agent.id, "api_call", registry.list())
    assert selection.tool_name == "api.b"
    assert selection.scores["api.a"] < selection.scores["api.b"]
    db.close()


def test_strategy_and_memory_together_flip_selection(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)
    registry = make_registry()

    # Bias the agent toward the wrong tool.
    strategy.set_prior(agent.id, "api_call", "api.a", prior=5.0)
    assert strategy.select_tool(agent.id, "api_call", registry.list()).tool_name == "api.a"

    # Simulate learning: api.a fails, api.b succeeds.
    trace_a = make_trace(agent.id, "api.a", status=TraceStatus.ERROR)
    ev_a = make_evaluation(trace_a, success=False)
    ref_a = make_reflection(trace_a, ev_a, suggested="api.b")
    strategy.update(agent.id, "api_call", trace_a, ev_a, ref_a)
    memory.record(agent.id, "api_call", "api.a", success=False)

    trace_b = make_trace(agent.id, "api.b", status=TraceStatus.SUCCESS)
    ev_b = make_evaluation(trace_b, success=True)
    ref_b = make_reflection(trace_b, ev_b)
    strategy.update(agent.id, "api_call", trace_b, ev_b, ref_b)
    memory.record(agent.id, "api_call", "api.b", success=True)

    # The learned deltas + memory now dominate the prior bias.
    selection = strategy.select_tool(agent.id, "api_call", registry.list())
    assert selection.tool_name == "api.b"
    db.close()