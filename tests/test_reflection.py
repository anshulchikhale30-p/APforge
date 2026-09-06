"""Tests for the REFLECT step (failure analysis)."""

from apforge.db import Database
from apforge.evaluation import RuleEvaluator
from apforge.memory import ToolUseMemory
from apforge.models import FailureCategory, Observation, TraceStatus
from apforge.reflection import ReflectionEngine
from apforge.strategy import StrategyEngine
from apforge.tools import ToolRegistry


def make_reflector(registry, db, agent_id):
    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)
    reflector = ReflectionEngine()
    return reflector, strategy


def evaluate_failure(observation):
    evaluation = RuleEvaluator().evaluate(observation)
    assert evaluation.success is False
    return evaluation


def error_observation(error, trace_id="t1"):
    return Observation(
        trace_id=trace_id, status=TraceStatus.ERROR, error=error, duration_ms=1
    )


def test_success_produces_none_category(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    registry = ToolRegistry()
    registry.register_func("a", "a", lambda x: x)
    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)
    reflector = ReflectionEngine()

    observation = Observation(
        trace_id="t1", status=TraceStatus.SUCCESS, result="ok", duration_ms=1
    )
    evaluation = RuleEvaluator().evaluate(observation)
    reflection = reflector.analyze(
        evaluation=evaluation,
        observation=observation,
        agent_id=agent.id,
        task_type="math",
        used_tool="a",
        available_tools=registry.list(),
        strategy_engine=strategy,
    )
    assert reflection.failure_category == FailureCategory.NONE
    assert reflection.suggested_tool is None
    db.close()


def test_classifies_environment_timeout(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    registry = ToolRegistry()
    registry.register_func("network.fetch", "fetch", lambda x: x)
    reflector, strategy = make_reflector(registry, db, agent.id)

    observation = error_observation("TimeoutError: timed out after 30s")
    reflection = reflector.analyze(
        evaluation=evaluate_failure(observation),
        observation=observation,
        agent_id=agent.id,
        task_type="fetch",
        used_tool="network.fetch",
        available_tools=registry.list(),
        strategy_engine=strategy,
    )
    assert reflection.failure_category == FailureCategory.ENVIRONMENT_ERROR
    assert "timeout" in reflection.analysis.lower()
    db.close()


def test_classifies_parameter_error(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    registry = ToolRegistry()
    registry.register_func("t", "t", lambda x: x)
    reflector, strategy = make_reflector(registry, db, agent.id)

    observation = error_observation("ValueError: invalid argument 'x'")
    reflection = reflector.analyze(
        evaluation=evaluate_failure(observation),
        observation=observation,
        agent_id=agent.id,
        task_type="t",
        used_tool="t",
        available_tools=registry.list(),
        strategy_engine=strategy,
    )
    assert reflection.failure_category == FailureCategory.PARAMETER_ERROR
    db.close()


def test_classifies_tool_unavailable(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    registry = ToolRegistry()
    registry.register_func("t", "t", lambda x: x)
    reflector, strategy = make_reflector(registry, db, agent.id)

    observation = error_observation("RuntimeError: tool not registered: plugin.x")
    reflection = reflector.analyze(
        evaluation=evaluate_failure(observation),
        observation=observation,
        agent_id=agent.id,
        task_type="t",
        used_tool="t",
        available_tools=registry.list(),
        strategy_engine=strategy,
    )
    assert reflection.failure_category == FailureCategory.TOOL_UNAVAILABLE
    db.close()


def test_defaults_to_logic_error(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    registry = ToolRegistry()
    registry.register_func("t", "t", lambda x: x)
    reflector, strategy = make_reflector(registry, db, agent.id)

    observation = error_observation("RuntimeError: always fails")
    reflection = reflector.analyze(
        evaluation=evaluate_failure(observation),
        observation=observation,
        agent_id=agent.id,
        task_type="t",
        used_tool="t",
        available_tools=registry.list(),
        strategy_engine=strategy,
    )
    assert reflection.failure_category == FailureCategory.LOGIC_ERROR
    db.close()


def test_suggests_alternative_tool_on_failure(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    registry = ToolRegistry()
    registry.register_func("api.a", "unreliable api", lambda x: (_ for _ in ()).throw(RuntimeError("api a down")))
    registry.register_func("api.b", "reliable api", lambda x: "ok")

    reflector, strategy = make_reflector(registry, db, agent.id)
    observation = error_observation("RuntimeError: api a down")
    reflection = reflector.analyze(
        evaluation=evaluate_failure(observation),
        observation=observation,
        agent_id=agent.id,
        task_type="api_call",
        used_tool="api.a",
        available_tools=registry.list(),
        strategy_engine=strategy,
    )
    assert reflection.suggested_tool == "api.b"
    assert "try api.b instead" in reflection.lesson
    db.close()


def test_no_suggestion_when_only_one_tool(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    registry = ToolRegistry()
    registry.register_func("only.tool", "only", lambda x: x)
    reflector, strategy = make_reflector(registry, db, agent.id)
    observation = error_observation("RuntimeError: always fails")
    reflection = reflector.analyze(
        evaluation=evaluate_failure(observation),
        observation=observation,
        agent_id=agent.id,
        task_type="t",
        used_tool="only.tool",
        available_tools=registry.list(),
        strategy_engine=strategy,
    )
    assert reflection.suggested_tool is None
    assert "no alternative" in reflection.lesson
    db.close()