"""Tests for the REMEMBER step (persistent tool-use memory)."""

from apforge.db import Database
from apforge.memory import ToolUseMemory


def test_memory_records_success_and_failure(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    memory = ToolUseMemory(db)

    first = memory.record(agent.id, "api_call", "api.a", success=True)
    assert (first.success_count, first.failure_count, first.total_count) == (1, 0, 1)
    assert first.success_rate == 1.0

    second = memory.record(agent.id, "api_call", "api.a", success=False)
    assert (second.success_count, second.failure_count, second.total_count) == (1, 1, 2)
    assert second.success_rate == 0.5

    stats = memory.stats(agent.id)
    assert len(stats) == 1
    assert stats[0].tool_name == "api.a"
    db.close()


def test_memory_is_scoped_per_agent_and_task_type(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    a1 = db.create_agent("one")
    a2 = db.create_agent("two")
    memory = ToolUseMemory(db)

    memory.record(a1.id, "api_call", "api.a", success=True)
    memory.record(a1.id, "math", "math.evaluate", success=False)
    memory.record(a2.id, "api_call", "api.a", success=False)

    assert len(memory.stats(a1.id)) == 2
    assert len(memory.stats(a2.id)) == 1
    assert len(memory.stats(a1.id, task_type="api_call")) == 1
    db.close()


def test_memory_survives_db_reopen(tmp_path):
    path = str(tmp_path / "t.db")
    db = Database(path)
    agent = db.create_agent("learner")
    ToolUseMemory(db).record(agent.id, "api_call", "api.a", success=True)
    db.close()

    db2 = Database(path)
    restored = ToolUseMemory(db2).get(agent.id, "api_call", "api.a")
    assert restored is not None
    assert restored.success_count == 1
    db2.close()


def test_clear_removes_memory(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    agent = db.create_agent("learner")
    memory = ToolUseMemory(db)
    memory.record(agent.id, "api_call", "api.a", success=True)
    assert len(memory.stats(agent.id)) == 1
    memory.clear(agent.id)
    assert memory.stats(agent.id) == []
    db.close()