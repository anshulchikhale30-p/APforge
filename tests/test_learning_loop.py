"""End-to-end learning loop tests.

The flagship assertion: learned memory + strategy changes alter the agent's
future tool selection and planning, persist across a database reopen, and are
scoped per agent.
"""

from apforge.db import Database
from apforge.evaluation import RuleEvaluator
from apforge.engine import LearningEngine
from apforge.memory import ToolUseMemory
from apforge.models import TaskSpec, TraceStatus
from apforge.reflection import ReflectionEngine
from apforge.strategy import StrategyEngine
from apforge.tools import ToolRegistry


def make_learning_world(tmp_path):
    """Bad tool biased as the initial choice + good tool available."""
    db = Database(str(tmp_path / "learn.db"))
    registry = ToolRegistry()

    def bad(_):
        raise RuntimeError("upstream api is down")

    def good(arg):
        return {"status": "ok", "echo": arg}

    registry.register_func("api.flaky", "unreliable endpoint", bad)
    registry.register_func("api.stable", "reliable endpoint", good)

    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)
    engine = LearningEngine(
        db,
        registry,
        evaluator=RuleEvaluator(),
        reflector=ReflectionEngine(),
        strategy_engine=strategy,
        memory=memory,
    )
    return db, engine, strategy, registry


def api_tasks(n):
    return [
        TaskSpec(task_type="api_call", tool_input={"endpoint": f"/x{i}"}, criteria={"output_contains": "ok"})
        for i in range(n)
    ]


def test_loop_learns_and_changes_future_behavior(tmp_path):
    db, engine, strategy, _ = make_learning_world(tmp_path)
    agent = db.create_agent("learner")

    # Explicitly bias the fresh agent toward the WRONG tool, so any behavior
    # change afterwards is clearly caused by learning, not by priors.
    strategy.set_prior(agent.id, "api_call", "api.flaky", prior=3.0)
    assert strategy.select_tool(agent.id, "api_call", engine.registry.list()).tool_name == "api.flaky"

    results = engine.run_session(agent.id, api_tasks(6))

    # Turn 1 respects the initial bias and fails.
    assert results[0].selection.tool_name == "api.flaky"
    assert results[0].trace.status == TraceStatus.ERROR
    assert results[0].evaluation.success is False
    assert results[0].reflection.suggested_tool == "api.stable"

    # Every later turn must consistently pick the learned-good tool.
    for result in results[1:]:
        assert result.selection.tool_name == "api.stable", result.selection.rationale
        assert result.trace.status == TraceStatus.SUCCESS
        assert result.evaluation.success is True

    # CHANGE STRATEGY: deltas were applied and agent versions were bumped.
    flaky = db.get_strategy(agent.id, "api_call", "api.flaky")
    stable = db.get_strategy(agent.id, "api_call", "api.stable")
    assert flaky.delta < 0
    assert stable.delta > 0

    versions = db.list_agent_versions(agent.id)
    assert len(versions) >= 3  # initial + change(s)

    # REMEMBER: persistent tool-use memory was written.
    memory_rows = db.list_memory(agent.id)
    by_tool = {row.tool_name: row for row in memory_rows}
    assert by_tool["api.flaky"].failure_count >= 1
    assert by_tool["api.stable"].success_count >= 1

    # Agent versioning is recorded in each trace.
    assert results[0].trace.agent_version == results[0].agent_version_before
    db.close()


def test_learned_memory_persists_across_reopen(tmp_path):
    db, engine, strategy, _ = make_learning_world(tmp_path)
    agent = db.create_agent("learner")
    strategy.set_prior(agent.id, "api_call", "api.flaky", prior=3.0)
    engine.run_session(agent.id, api_tasks(4))
    db.close()

    # Reopen the same SQLite file with fresh engine objects.
    db2 = Database(str(tmp_path / "learn.db"))
    registry2 = engine.registry
    memory2 = ToolUseMemory(db2)
    strategy2 = StrategyEngine(db2, memory2)
    engine2 = LearningEngine(
        db2,
        registry2,
        evaluator=RuleEvaluator(),
        reflector=ReflectionEngine(),
        strategy_engine=strategy2,
        memory=memory2,
    )

    # The agent's stored strategy + memory still prefer the stable tool.
    selection = strategy2.select_tool(agent.id, "api_call", engine2.registry.list())
    assert selection.tool_name == "api.stable"

    # And new turns keep using it.
    result = engine2.turn(agent.id, api_tasks(1)[0])
    assert result.selection.tool_name == "api.stable"
    assert result.trace.status == TraceStatus.SUCCESS
    db2.close()


def test_fresh_agent_does_not_inherit_learning(tmp_path):
    """Learning is per-agent: a new agent starts naive again."""
    db = Database(str(tmp_path / "learn.db"))
    registry = ToolRegistry()

    def bad(_):
        raise RuntimeError("down")

    def good(_):
        return "ok"

    registry.register_func("api.flaky", "a", bad)
    registry.register_func("api.stable", "b", good)

    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)
    engine = LearningEngine(
        db,
        registry,
        evaluator=RuleEvaluator(),
        reflector=ReflectionEngine(),
        strategy_engine=strategy,
        memory=memory,
    )

    learner = db.create_agent("learner")
    strategy.set_prior(learner.id, "api_call", "api.flaky", prior=3.0)
    engine.run_session(learner.id, api_tasks(4))
    # learner has now learned
    assert strategy.select_tool(learner.id, "api_call", registry.list()).tool_name == "api.stable"

    newcomer = db.create_agent("newcomer")
    # The newcomer has no inherited strategy rows at all.
    assert db.get_strategy(newcomer.id, "api_call", "api.flaky") is None
    assert db.get_strategy(newcomer.id, "api_call", "api.stable") is None
    newcomer_result = engine.turn(newcomer.id, api_tasks(1)[0])
    # ...and starts from the registration-order default, not the learner's learning.
    assert newcomer_result.selection.tool_name == "api.flaky"
    db.close()


def test_recovery_when_bad_tool_becomes_healthy(tmp_path):
    """The agent can return to a tool once it stops failing."""
    db = Database(str(tmp_path / "learn.db"))
    registry = ToolRegistry()

    state = {"healthy": False}

    def flaky(_):
        if not state["healthy"]:
            raise RuntimeError("still warming up")
        return "ok"

    registry.register_func("api.warming", "warming endpoint", flaky)
    registry.register_func("api.alternative", "fallback", lambda x: "ok(alt)")

    memory = ToolUseMemory(db)
    strategy = StrategyEngine(db, memory)
    engine = LearningEngine(
        db,
        registry,
        evaluator=RuleEvaluator(),
        reflector=ReflectionEngine(),
        strategy_engine=strategy,
        memory=memory,
    )
    agent = db.create_agent("learner")
    strategy.set_prior(agent.id, "api_call", "api.warming", prior=3.0)

    # Failing phase: agent learns to avoid api.warming.
    engine.run_session(agent.id, api_tasks(3))
    assert strategy.select_tool(agent.id, "api_call", registry.list()).tool_name == "api.alternative"

    # The tool heals. Simulate operator-forced manual successes; the memory
    # factor must overcome the accumulated penalty (delta -1.0, one failure)
    # before the strategy engine will route back to api.warming.
    state["healthy"] = True
    for _ in range(8):
        memory.record(agent.id, "api_call", "api.warming", success=True)
    selection = strategy.select_tool(agent.id, "api_call", registry.list())
    assert selection.tool_name == "api.warming"
    db.close()


def test_turn_unknown_agent_raises(tmp_path):
    db, engine, _, _ = make_learning_world(tmp_path)
    try:
        engine.turn("missing-agent", TaskSpec(task_type="api_call"))
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
    db.close()