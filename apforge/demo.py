"""APforge demo app: a polished, lightweight dashboard for the learning loop.

Serves a single-file web dashboard (``demo/index.html``) plus a small API.
Every number the dashboard displays is produced by *real* engine executions —
either the committed benchmark results under ``benchmarks/results/`` or a
fresh run triggered by the "Run Demo" button:

- ``GET  /api/demo/summary``  -> benchmarks/results/evolution_summary.json
- ``GET  /api/demo/details``  -> compact detail view derived from
                                 benchmarks/results/results.json (per-suite
                                 task definitions, per-call tool traces,
                                 methodology notes)
- ``GET  /api/demo/loop``     -> a live learning-loop walkthrough produced by
                                 the real engine on deterministic benchmark
                                 tasks (USE -> OBSERVE -> EVALUATE -> REFLECT
                                 -> REMEMBER -> CHANGE STRATEGY)
- ``POST /api/demo/run``      -> re-run the real evolution benchmark
                                 (``apforge.benchmark.runner.run_evolution``)
                                 and return the refreshed results
- ``GET  /health``            -> service + data availability check

Run locally:

    python -m apforge.demo          # then open http://127.0.0.1:8000
    # or: uvicorn apforge.demo:app
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .benchmark.runner import build_suite_world, run_evolution
from .benchmark.tasks import CI_BUILD, build_criteria, make_bundle, set_latency_scale
from .models import TaskSpec

# ---------------------------------------------------------------------------
# Paths (resolved relative to the repository root so local runs "just work")
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(_REPO_ROOT, "benchmarks", "results")
DEFAULT_HTML_PATH = os.path.join(_REPO_ROOT, "demo", "index.html")


# ---------------------------------------------------------------------------
# Data access (real committed benchmark results)
# ---------------------------------------------------------------------------


def _read_results_json(results_dir: str, filename: str) -> dict[str, Any]:
    """Read one committed result file, with a helpful 404 if absent."""
    path = os.path.join(results_dir, filename)
    if not os.path.isfile(path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"missing {filename} in {results_dir!r} - run the benchmark first "
                "(`python benchmarks/run_evolution.py`) or click \"Run Demo\"."
            ),
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _evolution_summary(results_dir: str) -> dict[str, Any]:
    """The concise evolution summary (the dashboard's headline numbers)."""
    return _read_results_json(results_dir, "evolution_summary.json")


def _derived_details(results_dir: str) -> dict[str, Any]:
    """Compact detail view derived from the full results.json.

    Adds the experiment metadata (engine, methodology, narrative) and passes
    the per-suite breakdown through verbatim: task definitions, cost model,
    per-stage metrics, per-call tool traces, and training summaries.
    """
    full = _read_results_json(results_dir, "results.json")
    return {
        "schema_version": full.get("schema_version"),
        "generated_at": full.get("generated_at"),
        "experiment": full.get("experiment"),
        "engine": full.get("engine"),
        "methodology": full.get("methodology"),
        "notes": full.get("notes"),
        "narrative": full.get("narrative"),
        "suites": full.get("per_suite", {}),
    }


# ---------------------------------------------------------------------------
# Live learning-loop walkthrough (real engine turns on benchmark tasks)
# ---------------------------------------------------------------------------

# The demo walks the *actual* learning loop the way the benchmark's phase A
# does: suite = ci.build (a realistic engineering investigation), bundles from
# the deterministic train_a instance set, full engine turns. The naive V1
# priors make the first turns fail (decoy instrument chosen), and the loop's
# REFLECT / REMEMBER / CHANGE STRATEGY steps visibly fix the behavior.
DEMO_LOOP_SUITE = CI_BUILD


def _loop_turn_payload(turn: Any, index: int, step: Any) -> dict[str, Any]:
    """Serialize one engine turn into the dashboard's loop-walkthrough shape."""
    selection = turn.selection
    trace = turn.trace
    evaluation = turn.evaluation
    reflection = turn.reflection
    return {
        "index": index,
        "task_type": step.category,
        "step_id": step.step_id,
        "step_question": step.question,
        "selection": {
            "tool_name": selection.tool_name,
            "score": selection.score,
            "scores": selection.scores,
            "rationale": selection.rationale,
        },
        "trace": {
            "tool_name": trace.tool_name,
            "status": trace.status.value,
            "duration_ms": trace.duration_ms,
            "error": trace.error,
            "result": trace.result,
        },
        "evaluation": {
            "success": evaluation.success,
            "score": evaluation.score,
            "notes": evaluation.notes,
        },
        "reflection": {
            "failure_category": reflection.failure_category.value,
            "analysis": reflection.analysis,
            "lesson": reflection.lesson,
            "suggested_tool": reflection.suggested_tool,
        },
        "strategy_updates": [
            {
                "tool_name": update.tool_name,
                "delta": update.delta,
                "reason": update.reason,
            }
            for update in turn.strategy_updates
        ],
        "agent_version_before": turn.agent_version_before,
        "agent_version_after": turn.agent_version_after,
        "version_reason": turn.version_reason,
    }


def run_live_loop() -> dict[str, Any]:
    """Run a real learning-loop session on deterministic benchmark tasks.

    Uses a fresh in-memory agent + engine (identical setup to a benchmark
    suite) and records every phase of every turn. All tool calls, failures,
    reflections, memory rows, strategy deltas, and version bumps are produced
    by the real engine on this run.
    """
    set_latency_scale(1.0)  # real measured (simulated) per-tool latency
    suite = DEMO_LOOP_SUITE
    db, engine, _strategy, agent_id = build_suite_world(suite)

    turns: list[dict[str, Any]] = []
    try:
        count = suite.instance_counts.get("train_a", 3)
        index = 0
        for bundle_idx in range(count):
            bundle = make_bundle(suite, "train_a", bundle_idx)
            for step in suite.steps:
                index += 1
                task = TaskSpec(
                    task_type=step.category,
                    tool_input={"bundle": bundle},
                    criteria=build_criteria(suite, bundle, step),
                )
                turn = engine.turn(agent_id, task)
                turns.append(_loop_turn_payload(turn, index, step))

        agent = db.get_agent(agent_id)
        memory = [row.model_dump(mode="json") for row in db.list_memory(agent_id)]
        strategies = [
            row.model_dump(mode="json") for row in db.list_strategies(agent_id)
        ]
        versions = [
            {"version": row.version, "reason": row.reason}
            for row in db.list_agent_versions(agent_id)
        ]

        successes = sum(1 for t in turns if t["evaluation"]["success"])
        return {
            "suite_id": suite.suite_id,
            "title": suite.title,
            "description": suite.description,
            "steps": [
                {
                    "step_id": step.step_id,
                    "category": step.category,
                    "question": step.question,
                }
                for step in suite.steps
            ],
            "turns": turns,
            "turns_total": len(turns),
            "successes": successes,
            "failures": len(turns) - successes,
            "agent_version_start": turns[0]["agent_version_before"] if turns else 1,
            "agent_version_end": agent.version if agent else 1,
            "memory": memory,
            "strategies": strategies,
            "versions": versions,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_demo_app(
    results_dir: Optional[str] = None, html_path: Optional[str] = None
) -> FastAPI:
    """Build the demo app. ``results_dir`` defaults to benchmarks/results.

    ``APFORGE_DEMO_RESULTS_DIR`` / ``APFORGE_DEMO_HTML`` override the defaults
    (useful for installed-package deployments or custom data locations).
    """
    if results_dir is None:
        results_dir = os.environ.get("APFORGE_DEMO_RESULTS_DIR") or DEFAULT_RESULTS_DIR
    if html_path is None:
        html_path = os.environ.get("APFORGE_DEMO_HTML") or DEFAULT_HTML_PATH

    app = FastAPI(
        title="APforge Demo",
        description=(
            "Polished dashboard for the APforge learning loop: TASK -> tool "
            "decisions/trace -> evaluation/failure -> reflection -> persistent "
            "memory -> strategy update -> V2 -> V3 -> unseen-task generalization."
        ),
        version="0.1.0",
    )
    app.state.results_dir = results_dir
    app.state.html_path = html_path
    app.state.lock = threading.Lock()

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        if not os.path.isfile(html_path):
            raise HTTPException(
                status_code=404,
                detail=f"dashboard file not found: {html_path!r}",
            )
        return FileResponse(html_path)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "results_dir": results_dir,
            "results_present": os.path.isfile(
                os.path.join(results_dir, "evolution_summary.json")
            ),
        }

    @app.get("/api/demo/summary")
    def summary() -> dict[str, Any]:
        with app.state.lock:
            return _evolution_summary(results_dir)

    @app.get("/api/demo/details")
    def details() -> dict[str, Any]:
        with app.state.lock:
            return _derived_details(results_dir)

    @app.get("/api/demo/loop")
    def loop() -> dict[str, Any]:
        return run_live_loop()

    @app.post("/api/demo/run")
    def run_demo() -> dict[str, Any]:
        """Execute the real evolution benchmark and return refreshed results.

        Runs ``run_evolution`` over all suites with real simulated latency
        (deterministically seeded, so the cohort-level numbers reproduce while
        measured wall-clock latency is freshly recorded). Writes the refreshed
        JSON results under ``results_dir``, then returns the updated payloads.
        """
        if not app.state.lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409, detail="a benchmark run is already in progress"
            )
        try:
            run_evolution(output_dir=results_dir)
            summary_body = _evolution_summary(results_dir)
            details_body = _derived_details(results_dir)
            loop_body = run_live_loop()
        finally:
            app.state.lock.release()
        return {
            "generated_at": summary_body.get("generated_at"),
            "summary": summary_body,
            "details": details_body,
            "loop": loop_body,
        }

    return app


app = create_demo_app()
"""Module-level app for `uvicorn apforge.demo:app`."""


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    host = os.environ.get("APFORGE_DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("APFORGE_DEMO_PORT", "8000"))
    uvicorn.run("apforge.demo:app", host=host, port=port, reload=False)