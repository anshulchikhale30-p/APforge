"""SQLite storage layer for the APforge learning engine.

All persistent state lives in a single SQLite database:
agents, agent versions (strategy snapshots), execution traces, evaluations,
reflections (failure analysis), tool-use memory, and strategies.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from typing import Any, Optional

from .models import (
    Agent,
    AgentVersion,
    Evaluation,
    ExecutionTrace,
    Reflection,
    Strategy,
    ToolMemory,
    TraceStatus,
    utcnow_iso,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_versions (
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    strategies_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (agent_id, version)
);

CREATE TABLE IF NOT EXISTS execution_traces (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    agent_version INTEGER NOT NULL,
    task_type TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_json TEXT,
    status TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES execution_traces(id) ON DELETE CASCADE,
    success INTEGER NOT NULL,
    score REAL NOT NULL,
    criteria_json TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS reflections (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES execution_traces(id) ON DELETE CASCADE,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    failure_category TEXT NOT NULL,
    analysis TEXT NOT NULL DEFAULT '',
    lesson TEXT NOT NULL DEFAULT '',
    suggested_tool TEXT
);

CREATE TABLE IF NOT EXISTS tool_memory (
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    task_type TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    total_count INTEGER NOT NULL DEFAULT 0,
    success_rate REAL NOT NULL DEFAULT 0.0,
    last_used_at TEXT,
    PRIMARY KEY (agent_id, task_type, tool_name)
);

CREATE TABLE IF NOT EXISTS strategies (
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    task_type TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    prior REAL NOT NULL DEFAULT 1.0,
    delta REAL NOT NULL DEFAULT 0.0,
    reason TEXT NOT NULL DEFAULT 'initial',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (agent_id, task_type, tool_name)
);

CREATE INDEX IF NOT EXISTS idx_traces_agent ON execution_traces(agent_id, started_at);
CREATE INDEX IF NOT EXISTS idx_traces_tool ON execution_traces(tool_name);
CREATE INDEX IF NOT EXISTS idx_evaluations_trace ON evaluations(trace_id);
CREATE INDEX IF NOT EXISTS idx_reflections_trace ON reflections(trace_id);
"""


class Database:
    """Thread-safe wrapper around a SQLite database."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _to_json(value: Any) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, default=str)

    @staticmethod
    def _from_json(text: Optional[str]) -> Any:
        if text is None:
            return None
        return json.loads(text)

    # ------------------------------------------------------------------ agents

    def create_agent(self, name: str) -> Agent:
        agent_id = uuid.uuid4().hex
        now = utcnow_iso()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO agents (id, name, version, created_at) VALUES (?, ?, ?, ?)",
                (agent_id, name, 1, now),
            )
            self._conn.execute(
                "INSERT INTO agent_versions (agent_id, version, strategies_json, reason, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (agent_id, 1, "{}", "initial", now),
            )
        return self.get_agent(agent_id)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
        return Agent(**dict(row)) if row else None

    def list_agents(self) -> list[Agent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM agents ORDER BY created_at ASC"
            ).fetchall()
        return [Agent(**dict(row)) for row in rows]

    def bump_agent_version(
        self, agent_id: str, strategies_snapshot: dict[str, Any], reason: str
    ) -> int:
        """Record a new immutable agent version and return it."""
        now = utcnow_iso()
        with self._lock:
            row = self._conn.execute(
                "SELECT version FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"agent {agent_id} not found")
            new_version = row["version"] + 1
            with self._conn:
                self._conn.execute(
                    "INSERT INTO agent_versions"
                    " (agent_id, version, strategies_json, reason, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (agent_id, new_version, json.dumps(strategies_snapshot), reason, now),
                )
                self._conn.execute(
                    "UPDATE agents SET version = ? WHERE id = ?",
                    (new_version, agent_id),
                )
        return new_version

    def list_agent_versions(self, agent_id: str) -> list[AgentVersion]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM agent_versions WHERE agent_id = ? ORDER BY version DESC",
                (agent_id,),
            ).fetchall()
        return [
            AgentVersion(
                agent_id=row["agent_id"],
                version=row["version"],
                strategies=json.loads(row["strategies_json"] or "{}"),
                reason=row["reason"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------ traces

    def insert_trace(self, trace: ExecutionTrace) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO execution_traces"
                " (id, agent_id, agent_version, task_type, tool_name, arguments_json,"
                "  status, result_json, error, started_at, ended_at, duration_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.id,
                    trace.agent_id,
                    trace.agent_version,
                    trace.task_type,
                    trace.tool_name,
                    self._to_json(trace.arguments),
                    trace.status.value,
                    self._to_json(trace.result),
                    trace.error,
                    trace.started_at,
                    trace.ended_at,
                    trace.duration_ms,
                ),
            )

    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM execution_traces WHERE id = ?", (trace_id,)
            ).fetchone()
        return self._trace_from_row(row) if row else None

    def list_traces(self, agent_id: Optional[str] = None, limit: int = 100) -> list[ExecutionTrace]:
        sql = "SELECT * FROM execution_traces"
        params: tuple = ()
        if agent_id:
            sql += " WHERE agent_id = ?"
            params = (agent_id,)
        sql += " ORDER BY started_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, params + (limit,)).fetchall()
        return [self._trace_from_row(row) for row in rows]

    @classmethod
    def _trace_from_row(cls, row: sqlite3.Row) -> ExecutionTrace:
        return ExecutionTrace(
            id=row["id"],
            agent_id=row["agent_id"],
            agent_version=row["agent_version"],
            task_type=row["task_type"],
            tool_name=row["tool_name"],
            arguments=cls._from_json(row["arguments_json"]),
            status=TraceStatus(row["status"]),
            result=cls._from_json(row["result_json"]),
            error=row["error"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            duration_ms=row["duration_ms"],
        )

    # ------------------------------------------------------------------ evaluations

    def insert_evaluation(self, evaluation: Evaluation) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO evaluations (id, trace_id, success, score, criteria_json, notes)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    evaluation.id,
                    evaluation.trace_id,
                    1 if evaluation.success else 0,
                    evaluation.score,
                    json.dumps(evaluation.criteria),
                    evaluation.notes,
                ),
            )

    def get_evaluation(self, evaluation_id: str) -> Optional[Evaluation]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
            ).fetchone()
        return self._evaluation_from_row(row) if row else None

    def get_evaluation_by_trace(self, trace_id: str) -> Optional[Evaluation]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM evaluations WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return self._evaluation_from_row(row) if row else None

    @classmethod
    def _evaluation_from_row(cls, row: sqlite3.Row) -> Evaluation:
        return Evaluation(
            id=row["id"],
            trace_id=row["trace_id"],
            success=bool(row["success"]),
            score=row["score"],
            criteria=json.loads(row["criteria_json"] or "{}"),
            notes=row["notes"],
        )

    # ------------------------------------------------------------------ reflections

    def insert_reflection(self, reflection: Reflection) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO reflections"
                " (id, trace_id, evaluation_id, failure_category, analysis, lesson, suggested_tool)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    reflection.id,
                    reflection.trace_id,
                    reflection.evaluation_id,
                    reflection.failure_category.value,
                    reflection.analysis,
                    reflection.lesson,
                    reflection.suggested_tool,
                ),
            )

    def get_reflection_by_trace(self, trace_id: str) -> Optional[Reflection]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM reflections WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return self._reflection_from_row(row) if row else None

    def list_reflections(self, limit: int = 100) -> list[Reflection]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reflections ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._reflection_from_row(row) for row in rows]

    @classmethod
    def _reflection_from_row(cls, row: sqlite3.Row) -> Reflection:
        from .models import FailureCategory

        return Reflection(
            id=row["id"],
            trace_id=row["trace_id"],
            evaluation_id=row["evaluation_id"],
            failure_category=FailureCategory(row["failure_category"]),
            analysis=row["analysis"],
            lesson=row["lesson"],
            suggested_tool=row["suggested_tool"],
        )

    # ------------------------------------------------------------------ tool memory

    def upsert_memory(
        self,
        agent_id: str,
        task_type: str,
        tool_name: str,
        success: bool,
        used_at: Optional[str] = None,
    ) -> ToolMemory:
        used_at = used_at or utcnow_iso()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tool_memory"
                " WHERE agent_id = ? AND task_type = ? AND tool_name = ?",
                (agent_id, task_type, tool_name),
            ).fetchone()
            if row:
                success_count = row["success_count"] + (1 if success else 0)
                failure_count = row["failure_count"] + (0 if success else 1)
            else:
                success_count = 1 if success else 0
                failure_count = 0 if success else 1
            total = success_count + failure_count
            rate = success_count / total if total else 0.0
            with self._conn:
                self._conn.execute(
                    "INSERT INTO tool_memory"
                    " (agent_id, task_type, tool_name, success_count, failure_count,"
                    "  total_count, success_rate, last_used_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(agent_id, task_type, tool_name) DO UPDATE SET"
                    "  success_count = excluded.success_count,"
                    "  failure_count = excluded.failure_count,"
                    "  total_count = excluded.total_count,"
                    "  success_rate = excluded.success_rate,"
                    "  last_used_at = excluded.last_used_at",
                    (
                        agent_id,
                        task_type,
                        tool_name,
                        success_count,
                        failure_count,
                        total,
                        rate,
                        used_at,
                    ),
                )
        return ToolMemory(
            agent_id=agent_id,
            task_type=task_type,
            tool_name=tool_name,
            success_count=success_count,
            failure_count=failure_count,
            total_count=total,
            success_rate=rate,
            last_used_at=used_at,
        )

    def get_memory(
        self, agent_id: str, task_type: str, tool_name: str
    ) -> Optional[ToolMemory]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tool_memory"
                " WHERE agent_id = ? AND task_type = ? AND tool_name = ?",
                (agent_id, task_type, tool_name),
            ).fetchone()
        return self._memory_from_row(row) if row else None

    def clear_memory(self, agent_id: str) -> None:
        """Drop all memory rows for an agent."""
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM tool_memory WHERE agent_id = ?", (agent_id,)
            )

    def list_memory(
        self, agent_id: str, task_type: Optional[str] = None
    ) -> list[ToolMemory]:
        sql = "SELECT * FROM tool_memory WHERE agent_id = ?"
        params: tuple = (agent_id,)
        if task_type:
            sql += " AND task_type = ?"
            params = params + (task_type,)
        sql += " ORDER BY task_type ASC, tool_name ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._memory_from_row(row) for row in rows]

    @classmethod
    def _memory_from_row(cls, row: sqlite3.Row) -> ToolMemory:
        return ToolMemory(
            agent_id=row["agent_id"],
            task_type=row["task_type"],
            tool_name=row["tool_name"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            total_count=row["total_count"],
            success_rate=row["success_rate"],
            last_used_at=row["last_used_at"],
        )

    # ------------------------------------------------------------------ strategies

    def get_strategy(
        self, agent_id: str, task_type: str, tool_name: str
    ) -> Optional[Strategy]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM strategies"
                " WHERE agent_id = ? AND task_type = ? AND tool_name = ?",
                (agent_id, task_type, tool_name),
            ).fetchone()
        return self._strategy_from_row(row) if row else None

    def set_strategy(
        self,
        agent_id: str,
        task_type: str,
        tool_name: str,
        *,
        prior: Optional[float] = None,
        delta: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> Strategy:
        now = utcnow_iso()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM strategies"
                " WHERE agent_id = ? AND task_type = ? AND tool_name = ?",
                (agent_id, task_type, tool_name),
            ).fetchone()
            if row:
                new_prior = row["prior"] if prior is None else prior
                new_delta = row["delta"] if delta is None else delta
                new_reason = reason or row["reason"]
                with self._conn:
                    self._conn.execute(
                        "UPDATE strategies SET prior = ?, delta = ?, reason = ?, updated_at = ?"
                        " WHERE agent_id = ? AND task_type = ? AND tool_name = ?",
                        (
                            new_prior,
                            new_delta,
                            new_reason,
                            now,
                            agent_id,
                            task_type,
                            tool_name,
                        ),
                    )
            else:
                new_prior = 1.0 if prior is None else prior
                new_delta = 0.0 if delta is None else delta
                new_reason = reason or "initial"
                with self._conn:
                    self._conn.execute(
                        "INSERT INTO strategies"
                        " (agent_id, task_type, tool_name, prior, delta, reason, updated_at)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            agent_id,
                            task_type,
                            tool_name,
                            new_prior,
                            new_delta,
                            new_reason,
                            now,
                        ),
                    )
        return Strategy(
            agent_id=agent_id,
            task_type=task_type,
            tool_name=tool_name,
            prior=new_prior,
            delta=new_delta,
            reason=new_reason,
            updated_at=now,
        )

    def list_strategies(
        self, agent_id: str, task_type: Optional[str] = None
    ) -> list[Strategy]:
        sql = "SELECT * FROM strategies WHERE agent_id = ?"
        params: tuple = (agent_id,)
        if task_type:
            sql += " AND task_type = ?"
            params = params + (task_type,)
        sql += " ORDER BY task_type ASC, tool_name ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._strategy_from_row(row) for row in rows]

    def snapshot_strategies(self, agent_id: str) -> dict[str, Any]:
        return {
            f"{s.task_type}::{s.tool_name}": {"prior": s.prior, "delta": s.delta}
            for s in self.list_strategies(agent_id)
        }

    @classmethod
    def _strategy_from_row(cls, row: sqlite3.Row) -> Strategy:
        return Strategy(
            agent_id=row["agent_id"],
            task_type=row["task_type"],
            tool_name=row["tool_name"],
            prior=row["prior"],
            delta=row["delta"],
            reason=row["reason"],
            updated_at=row["updated_at"],
        )