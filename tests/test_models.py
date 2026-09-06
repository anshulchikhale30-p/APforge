"""Model sanity tests."""

import pytest

from apforge.models import (
    Evaluation,
    ExecutionTrace,
    FailureCategory,
    Observation,
    Reflection,
    TaskSpec,
    ToolMemory,
    ToolResult,
    ToolSelection,
    TraceStatus,
    TurnResult,
    utcnow_iso,
)


def test_trace_status_serializes_to_string():
    assert TraceStatus.SUCCESS == "success"
    assert TraceStatus.ERROR == "error"
    assert TraceStatus("success") is TraceStatus.SUCCESS


def test_failure_category_values():
    assert FailureCategory.NONE == "none"
    assert FailureCategory.TOOL_SELECTION_MISMATCH == "tool_selection_mismatch"
    assert FailureCategory.ENVIRONMENT_ERROR == "environment_error"


def test_evaluation_score_is_clamped():
    with pytest.raises(ValueError):
        Evaluation(id="e1", trace_id="t1", success=True, score=1.5, notes="")
    with pytest.raises(ValueError):
        Evaluation(id="e1", trace_id="t1", success=True, score=-0.1, notes="")


def test_trace_roundtrips_complex_json():
    trace = ExecutionTrace(
        id="t1",
        agent_id="a1",
        agent_version=2,
        task_type="math",
        tool_name="math.evaluate",
        arguments={"expression": "2+2"},
        status=TraceStatus.SUCCESS,
        result=4,
        error=None,
        started_at=utcnow_iso(),
        ended_at=utcnow_iso(),
        duration_ms=3,
    )
    dumped = trace.model_dump()
    restored = ExecutionTrace.model_validate(dumped)
    assert restored == trace
    assert restored.result == 4


def test_task_spec_defaults():
    spec = TaskSpec(task_type="api_call")
    assert spec.tool_input is None
    assert spec.criteria == {}


def test_tool_result_error_status():
    result = ToolResult(tool_name="x", status=TraceStatus.ERROR, error="boom")
    assert result.status == TraceStatus.ERROR
    assert result.duration_ms == 0


def test_reflection_defaults():
    reflection = Reflection(
        id="r1",
        trace_id="t1",
        evaluation_id="e1",
        failure_category=FailureCategory.LOGIC_ERROR,
        analysis="failed",
        lesson="try another tool",
    )
    assert reflection.suggested_tool is None


def test_tool_memory_counts():
    memory = ToolMemory(
        agent_id="a1",
        task_type="t",
        tool_name="x",
        success_count=3,
        failure_count=1,
        total_count=4,
        success_rate=0.75,
    )
    assert memory.total_count == 4
    assert memory.success_rate == 0.75


def test_turn_result_serializes():
    now = utcnow_iso()
    trace = ExecutionTrace(
        id="t1",
        agent_id="a1",
        agent_version=1,
        task_type="t",
        tool_name="x",
        status=TraceStatus.SUCCESS,
        started_at=now,
        ended_at=now,
        duration_ms=1,
    )
    observation = Observation(
        trace_id="t1", status=TraceStatus.SUCCESS, duration_ms=1
    )
    evaluation = Evaluation(
        id="e1", trace_id="t1", success=True, score=1.0, notes="ok"
    )
    reflection = Reflection(
        id="r1",
        trace_id="t1",
        evaluation_id="e1",
        failure_category=FailureCategory.NONE,
        analysis="ok",
        lesson="ok",
    )
    result = TurnResult(
        agent_id="a1",
        agent_version_before=1,
        agent_version_after=2,
        selection=ToolSelection(tool_name="x", score=1.0, rationale="picked"),
        trace=trace,
        observation=observation,
        evaluation=evaluation,
        reflection=reflection,
    )
    data = result.model_dump()
    assert data["agent_version_after"] == 2
    assert data["strategy_updates"] == []
    assert data["selection"]["rationale"] == "picked"
    assert data["evaluation"]["score"] == 1.0