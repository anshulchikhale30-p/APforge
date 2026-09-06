"""Storage-layer tests: schema, agents, versioning, traces, memory, strategies."""

import pytest

from apforge.db import Database
from apforge.models import (
    ExecutionTrace,
    FailureCategory,
    Reflection,
    TraceStatus,
    utcnow_iso,
)


def completion_trace(**overrides):
    now = utcnow_iso()
    defaults = dict(
        id="trace-1",
        agent_id="agent-1",
        agent_version=1,
        task_type="math",
        tool_name="math.evaluate",
        arguments={"expression": "2+2"},
        status=TraceStatus.SUCCESS,
        result=4,
        error=None,
        started_at=now,
        ended_at=now,
        duration_ms=2,
    )
    defaults.update(overrides)
    return ExecutionTrace(**defaults)


def test_schema_created_on_open(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    tables = {
        row["name"]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "agents",
        "agent_versions",
        "execution_traces",
        "evaluations",
        "reflections",
        "tool_memory",
        "strategies",
    } <= tables
    db.close()


def test_create_agent_seeds_version_one(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    assert agent.version == 1
    assert agent.name == "learner"
    versions = db.list_agent_versions(agent.id)
    assert len(versions) == 1
    assert versions[0].version == 1
    assert versions[0].strategies == {}
    assert versions[0].reason == "initial"
    db.close()


def test_bump_version_snapshots_strategies(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    db.set_strategy(
        agent.id, "api_call", "api.a", prior=1.0, delta=-1.0, reason="penalty"
    )
    snapshot = db.snapshot_strategies(agent.id)
    assert snapshot == {"api_call::api.a": {"prior": 1.0, "delta": -1.0}}

    new_version = db.bump_agent_version(agent.id, snapshot, "learned")
    assert new_version == 2
    agent2 = db.get_agent(agent.id)
    assert agent2.version == 2

    versions = db.list_agent_versions(agent.id)
    assert [v.version for v in versions] == [2, 1]
    assert versions[0].strategies == {"api_call::api.a": {"prior": 1.0, "delta": -1.0}}
    assert versions[0].reason == "learned"
    db.close()


def test_bump_version_unknown_agent_raises(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    with pytest.raises(KeyError):
        db.bump_agent_version("nope", {}, "x")
    db.close()


def test_trace_roundtrip_and_listing(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    trace = completion_trace(
        id="t1", agent_id=agent.id, agent_version=1, result={"answer": 42}
    )
    db.insert_trace(trace)

    loaded = db.get_trace("t1")
    assert loaded is not None
    assert loaded.result == {"answer": 42}
    assert loaded.status == TraceStatus.SUCCESS

    traces = db.list_traces(agent_id=agent.id)
    assert len(traces) == 1
    assert traces[0].id == "t1"
    db.close()


def test_memory_upsert_accumulates(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")

    m1 = db.upsert_memory(agent.id, "api_call", "api.a", success=True)
    assert (m1.success_count, m1.failure_count, m1.total_count) == (1, 0, 1)
    assert m1.success_rate == 1.0

    m2 = db.upsert_memory(agent.id, "api_call", "api.a", success=False)
    assert (m2.success_count, m2.failure_count, m2.total_count) == (1, 1, 2)
    assert m2.success_rate == 0.5

    rows = db.list_memory(agent.id)
    assert len(rows) == 1
    db.close()


def test_memory_persists_across_reopen(tmp_path):
    path = str(tmp_path / "t.db")
    db = Database(path)
    agent = db.create_agent("learner")
    db.upsert_memory(agent.id, "api_call", "api.a", success=True)
    db.close()

    db2 = Database(path)
    loaded = db2.get_memory(agent.id, "api_call", "api.a")
    assert loaded is not None
    assert loaded.success_count == 1
    assert loaded.success_rate == 1.0
    db2.close()


def test_strategy_set_get_and_defaults(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")

    created = db.set_strategy(
        agent.id, "api_call", "api.a", delta=-1.0, reason="penalty"
    )
    assert created.prior == 1.0
    assert created.delta == -1.0

    got = db.get_strategy(agent.id, "api_call", "api.a")
    assert got is not None and got.delta == -1.0

    # updating only delta keeps prior
    updated = db.set_strategy(
        agent.id, "api_call", "api.a", delta=0.5, reason="reward"
    )
    assert updated.prior == 1.0
    assert updated.delta == 0.5

    missing = db.get_strategy(agent.id, "api_call", "api.missing")
    assert missing is None
    db.close()


def test_strategies_are_scoped_per_agent(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    a1 = db.create_agent("one")
    a2 = db.create_agent("two")
    db.set_strategy(a1.id, "api_call", "api.a", delta=-1.0, reason="penalty")
    assert db.get_strategy(a1.id, "api_call", "api.a").delta == -1.0
    assert db.get_strategy(a2.id, "api_call", "api.a") is None
    assert db.snapshot_strategies(a2.id) == {}
    db.close()


def test_evaluation_and_reflection_roundtrip(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    trace = completion_trace(id="t1", agent_id=agent.id)
    db.insert_trace(trace)

    evaluation = db.get_evaluation_by_trace("t1")
    assert evaluation is None

    from apforge.models import Evaluation

    ev = Evaluation(
        id="e1",
        trace_id="t1",
        success=False,
        score=0.0,
        criteria={"expected_output": "ok"},
        notes="failed",
    )
    db.insert_evaluation(ev)
    loaded_ev = db.get_evaluation("e1")
    assert loaded_ev is not None
    assert loaded_ev.criteria == {"expected_output": "ok"}
    assert loaded_ev.success is False

    reflection = Reflection(
        id="r1",
        trace_id="t1",
        evaluation_id="e1",
        failure_category=FailureCategory.LOGIC_ERROR,
        analysis="bad tool",
        lesson="try other tool",
        suggested_tool="api.b",
    )
    db.insert_reflection(reflection)
    loaded_ref = db.get_reflection_by_trace("t1")
    assert loaded_ref is not None
    assert loaded_ref.failure_category == FailureCategory.LOGIC_ERROR
    assert loaded_ref.suggested_tool == "api.b"
    assert len(db.list_reflections()) == 1
    db.close()