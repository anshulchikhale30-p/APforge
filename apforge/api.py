"""FastAPI app exposing the APforge learning engine.

Read/write endpoints for agents, learning turns (the full loop), execution
traces, evaluations/reflections, tool-use memory, strategies, and agent
versions. No authentication and no UI: this is the engine's API surface.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from .db import Database
from .engine import LearningEngine
from .evaluation import RuleEvaluator
from .memory import ToolUseMemory
from .models import (
    Agent,
    AgentCreate,
    AgentVersion,
    ExecutionTrace,
    Reflection,
    SessionRequest,
    Strategy,
    TaskSpec,
    ToolDef,
    ToolMemory,
    TraceDetail,
    TurnResult,
)
from .reflection import ReflectionEngine
from .strategy import StrategyEngine
from .tools import ToolRegistry, default_tools


class EngineContext:
    """Bundle of engine collaborators for one app instance."""

    def __init__(self, db_path: str) -> None:
        self.db = Database(db_path)
        self.registry = default_tools()
        self.memory = ToolUseMemory(self.db)
        self.strategy = StrategyEngine(self.db, self.memory)
        self.reflector = ReflectionEngine()
        self.evaluator = RuleEvaluator()
        self.engine = LearningEngine(
            self.db,
            self.registry,
            evaluator=self.evaluator,
            reflector=self.reflector,
            strategy_engine=self.strategy,
            memory=self.memory,
        )


def create_app(db_path: Optional[str] = None) -> FastAPI:
    """Build the FastAPI app. `db_path` defaults to $APFORGE_DB_PATH or apforge.db."""
    if db_path is None:
        db_path = os.environ.get("APFORGE_DB_PATH", "apforge.db")
    ctx = EngineContext(db_path)

    app = FastAPI(
        title="APforge Core Learning Engine",
        description=(
            "USE -> OBSERVE -> EVALUATE -> REFLECT -> REMEMBER -> "
            "CHANGE STRATEGY -> USE AGAIN"
        ),
        version="0.1.0",
    )
    app.state.context = ctx
    engine = ctx.engine
    db = ctx.db
    registry = ctx.registry

    def get_agent_or_404(agent_id: str) -> Agent:
        agent = db.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"agent {agent_id} not found")
        return agent

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "db": db.path, "tools": len(registry.list())}

    @app.get("/api/tools", response_model=list[ToolDef])
    def list_tools() -> list[ToolDef]:
        return [tool.to_def() for tool in registry.list()]

    @app.post("/api/agents", response_model=Agent, status_code=201)
    def create_agent(body: AgentCreate) -> Agent:
        return db.create_agent(body.name)

    @app.get("/api/agents", response_model=list[Agent])
    def list_agents() -> list[Agent]:
        return db.list_agents()

    @app.get("/api/agents/{agent_id}", response_model=Agent)
    def get_agent(agent_id: str) -> Agent:
        return get_agent_or_404(agent_id)

    @app.get("/api/agents/{agent_id}/versions", response_model=list[AgentVersion])
    def list_agent_versions(agent_id: str) -> list[AgentVersion]:
        get_agent_or_404(agent_id)
        return db.list_agent_versions(agent_id)

    @app.get("/api/agents/{agent_id}/traces", response_model=list[ExecutionTrace])
    def list_agent_traces(
        agent_id: str, limit: int = Query(default=100, ge=1, le=1000)
    ) -> list[ExecutionTrace]:
        get_agent_or_404(agent_id)
        return db.list_traces(agent_id=agent_id, limit=limit)

    @app.get("/api/agents/{agent_id}/memories", response_model=list[ToolMemory])
    def list_agent_memories(
        agent_id: str, task_type: Optional[str] = None
    ) -> list[ToolMemory]:
        get_agent_or_404(agent_id)
        return db.list_memory(agent_id, task_type=task_type)

    @app.get("/api/agents/{agent_id}/strategies", response_model=list[Strategy])
    def list_agent_strategies(
        agent_id: str, task_type: Optional[str] = None
    ) -> list[Strategy]:
        get_agent_or_404(agent_id)
        return db.list_strategies(agent_id, task_type=task_type)

    @app.post(
        "/api/agents/{agent_id}/turns", response_model=TurnResult, status_code=200
    )
    def run_turn(agent_id: str, body: TaskSpec) -> TurnResult:
        get_agent_or_404(agent_id)
        try:
            return engine.turn(agent_id, body)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(
        "/api/agents/{agent_id}/session", response_model=list[TurnResult]
    )
    def run_session(agent_id: str, body: SessionRequest) -> list[TurnResult]:
        get_agent_or_404(agent_id)
        try:
            return engine.run_session(agent_id, body.tasks)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/traces/{trace_id}", response_model=TraceDetail)
    def get_trace_detail(trace_id: str) -> TraceDetail:
        trace = db.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail=f"trace {trace_id} not found")
        return TraceDetail(
            trace=trace,
            evaluation=db.get_evaluation_by_trace(trace_id),
            reflection=db.get_reflection_by_trace(trace_id),
        )

    @app.get("/api/reflections", response_model=list[Reflection])
    def list_reflections(limit: int = Query(default=100, ge=1, le=1000)) -> list[Reflection]:
        return db.list_reflections(limit=limit)

    return app


app = create_app()
"""Module-level app for `uvicorn apforge.api:app`."""


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("APFORGE_HOST", "127.0.0.1")
    port = int(os.environ.get("APFORGE_PORT", "8000"))
    uvicorn.run("apforge.api:app", host=host, port=port, reload=False)