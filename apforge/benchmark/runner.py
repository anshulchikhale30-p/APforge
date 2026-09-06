"""Evolution benchmarking harness for the APforge learning engine.

Runs the reproducible evolution experiment

    V1 (naive) --experience--> learning --learning--> V2 --experience-->
    learning --> V3

per engineering-investigation suite, and reports every metric from *real
executions* of the harness:

- actual success / correctness (scenario solve rate + step success rate)
- total tool calls
- unnecessary tool calls (calls that did not use the correct instrument)
- failures (tool executions that errored)
- latency (wall-clock ms measured around every real tool call)
- cost (modeled per-instrument charge applied to real call counts)

Measurement phases only *use* the current learned strategy (select -> execute
-> observe -> evaluate): they never mutate strategy deltas or tool-use memory,
so each measured version reflects exactly the state reached by the preceding
learning phases. Learning phases run the full engine loop (USE -> OBSERVE ->
EVALUATE -> REFLECT -> REMEMBER -> CHANGE STRATEGY), which is what bumps the
agent version and accumulates tool-use memory.

The generalization experiment measures V1/V2/V3 on *unseen* instances
(bundles the agent has never executed) to test transfer of the learned
tool-use strategies to new instances of the same task types.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..db import Database
from ..engine import LearningEngine
from ..evaluation import RuleEvaluator
from ..memory import ToolUseMemory
from ..models import TaskSpec, TraceStatus, utcnow_iso
from ..reflection import ReflectionEngine
from ..strategy import StrategyEngine
from .tasks import (
    SUITES,
    SuiteDefinition,
    build_criteria,
    build_registry,
    make_bundle,
    scenario_steps,
    set_latency_scale,
)

# A step is attempted at most this many times before the scenario is abandoned.
MAX_ATTEMPTS_PER_STEP = 3

# Naive-agent priors: each step's decoy instrument is biased to be the initial
# selection, so V1 reliably demonstrates an unlearned (and failing) strategy.
NAIVE_DECOY_PRIOR = 3.0

MEASURED_SETS = ("eval", "gen")
STAGES = ("v1", "v2", "v3")


# ---------------------------------------------------------------------------
# Scenario execution + metrics
# ---------------------------------------------------------------------------


@dataclass
class CallRecord:
    """One real tool execution during a measurement scenario."""

    category: str
    step_id: str
    tool: str
    is_correct_tool: bool
    status: str  # "success" | "error"
    evaluation_success: bool
    unnecessary: bool
    duration_ms: int
    cost: float
    trace_id: str = ""


@dataclass
class ScenarioResult:
    """Result of solving one investigation scenario in measurement mode."""

    suite_id: str
    set_name: str
    scenario_id: str
    stage: str
    agent_version: int
    solved: bool
    steps_total: int
    steps_solved: int
    total_calls: int
    unnecessary_calls: int
    failure_calls: int
    latency_ms: int
    cost: float
    calls: list[CallRecord] = field(default_factory=list)


def measure_scenarios(
    suite: SuiteDefinition,
    engine: LearningEngine,
    agent_id: str,
    set_name: str,
    stage: str,
    bundles: list[dict[str, Any]],
    version: int,
) -> list[ScenarioResult]:
    """Measure the current strategy state on scenario instances.

    Measurement uses the engine's real ``use -> observe -> evaluate`` pipeline
    but never applies strategy updates or memory writes, so it is a pure probe
    of the current learned state.
    """
    step_sequences = scenario_steps(suite, set_name)
    results: list[ScenarioResult] = []
    for idx, bundle in enumerate(bundles):
        steps = step_sequences[idx % len(step_sequences)]
        scenario_id = f"{suite.suite_id}-{set_name}-{idx:02d}"
        result = ScenarioResult(
            suite_id=suite.suite_id,
            set_name=set_name,
            scenario_id=scenario_id,
            stage=stage,
            agent_version=version,
            solved=False,
            steps_total=len(steps),
            steps_solved=0,
            total_calls=0,
            unnecessary_calls=0,
            failure_calls=0,
            latency_ms=0,
            cost=0.0,
        )
        for step in steps:
            attempts = 0
            solved_step = False
            while attempts < MAX_ATTEMPTS_PER_STEP and not solved_step:
                attempts += 1
                tool_input = {"bundle": bundle}
                selection, trace = engine.use(agent_id, step.category, tool_input)
                observation = engine.observe(trace)
                evaluation = engine.evaluate(observation, build_criteria(suite, bundle, step))

                used_tool = trace.tool_name
                is_correct = used_tool == step.correct_tool
                failed = trace.status == TraceStatus.ERROR
                is_unnecessary = not is_correct
                cost = suite.cost_map().get(used_tool, 1.0)

                result.total_calls += 1
                result.latency_ms += trace.duration_ms
                result.cost += cost
                if is_unnecessary:
                    result.unnecessary_calls += 1
                if failed:
                    result.failure_calls += 1
                result.calls.append(
                    CallRecord(
                        category=step.category,
                        step_id=step.step_id,
                        tool=used_tool,
                        is_correct_tool=is_correct,
                        status=trace.status.value,
                        evaluation_success=evaluation.success,
                        unnecessary=is_unnecessary,
                        duration_ms=trace.duration_ms,
                        cost=cost,
                        trace_id=trace.id,
                    )
                )
                if is_correct and evaluation.success:
                    solved_step = True

            if solved_step:
                result.steps_solved += 1

        result.solved = result.steps_solved == result.steps_total
        results.append(result)
    return results


def _stage_metrics(results: list[ScenarioResult]) -> dict[str, Any]:
    """Aggregate raw scenario results into a machine-readable metrics block."""
    scenarios_total = len(results)
    scenarios_solved = sum(1 for r in results if r.solved)
    steps_total = sum(r.steps_total for r in results)
    steps_solved = sum(r.steps_solved for r in results)
    total_calls = sum(r.total_calls for r in results)
    unnecessary = sum(r.unnecessary_calls for r in results)
    failures = sum(r.failure_calls for r in results)
    latency = sum(r.latency_ms for r in results)
    cost = sum(r.cost for r in results)

    tool_breakdown: dict[str, int] = {}
    for result in results:
        for call in result.calls:
            tool_breakdown[call.tool] = tool_breakdown.get(call.tool, 0) + 1

    return {
        "scenarios_total": scenarios_total,
        "scenarios_solved": scenarios_solved,
        "solve_rate": round(scenarios_solved / scenarios_total, 4) if scenarios_total else 0.0,
        "steps_total": steps_total,
        "steps_solved": steps_solved,
        "step_success_rate": round(steps_solved / steps_total, 4) if steps_total else 0.0,
        "total_tool_calls": total_calls,
        "unnecessary_tool_calls": unnecessary,
        "unnecessary_call_rate": round(unnecessary / total_calls, 4) if total_calls else 0.0,
        "failures": failures,
        "latency_ms": latency,
        "average_latency_ms_per_call": round(latency / total_calls, 2) if total_calls else 0.0,
        "cost": round(cost, 4),
        "average_cost_per_solved_scenario": round(cost / scenarios_solved, 4) if scenarios_solved else 0.0,
        "tool_call_breakdown": dict(sorted(tool_breakdown.items())),
    }


def _scenario_summaries(results: list[ScenarioResult]) -> list[dict[str, Any]]:
    out = []
    for r in results:
        out.append(
            {
                "scenario_id": r.scenario_id,
                "solved": r.solved,
                "steps_solved": r.steps_solved,
                "steps_total": r.steps_total,
                "total_calls": r.total_calls,
                "unnecessary_calls": r.unnecessary_calls,
                "failure_calls": r.failure_calls,
                "latency_ms": r.latency_ms,
                "cost": round(r.cost, 4),
                "calls": [
                    {
                        "category": c.category,
                        "step_id": c.step_id,
                        "tool": c.tool,
                        "is_correct_tool": c.is_correct_tool,
                        "status": c.status,
                        "evaluation_success": c.evaluation_success,
                        "unnecessary": c.unnecessary,
                        "duration_ms": c.duration_ms,
                        "cost": c.cost,
                    }
                    for c in r.calls
                ],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Learning (experience) phases
# ---------------------------------------------------------------------------


@dataclass
class TrainingSummary:
    """What one learning phase actually did, with real loop numbers."""

    turns: int = 0
    successes: int = 0
    failures: int = 0
    strategy_updates: int = 0
    version_before: int = 1
    version_after: int = 1
    category_selections: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_on(
    suite: SuiteDefinition,
    engine: LearningEngine,
    agent_id: str,
    bundles: list[dict[str, Any]],
) -> TrainingSummary:
    """Run one experience phase: the full learning loop on training instances.

    Every turn runs USE -> OBSERVE -> EVALUATE -> REFLECT -> REMEMBER ->
    CHANGE STRATEGY, so strategy deltas, tool-use memory, and agent versions
    all move here (never during measurement).
    """
    summary = TrainingSummary()
    summary.version_before = _agent_version(engine, agent_id)
    for bundle in bundles:
        for step in suite.steps:
            task = TaskSpec(
                task_type=step.category,
                tool_input={"bundle": bundle},
                criteria=build_criteria(suite, bundle, step),
            )
            turn = engine.turn(agent_id, task)
            summary.turns += 1
            summary.strategy_updates += len(turn.strategy_updates)
            if turn.evaluation.success:
                summary.successes += 1
            else:
                summary.failures += 1
            selections = summary.category_selections.setdefault(step.category, {})
            selections[turn.selection.tool_name] = selections.get(turn.selection.tool_name, 0) + 1
    summary.version_after = _agent_version(engine, agent_id)
    return summary


def _agent_version(engine: LearningEngine, agent_id: str) -> int:
    agent = engine.db.get_agent(agent_id)
    return agent.version if agent else 0


# ---------------------------------------------------------------------------
# Suite-level evolution
# ---------------------------------------------------------------------------


def set_naive_priors(
    strategy: StrategyEngine, agent_id: str, suite: SuiteDefinition
) -> None:
    """Bias V1 toward the tempting-but-wrong instrument of every step."""
    for step in suite.steps:
        strategy.set_prior(
            agent_id,
            step.category,
            step.decoy_tool,
            prior=NAIVE_DECOY_PRIOR,
            reason="naive V1 bias: untested initial preference",
        )


def build_suite_world(
    suite: SuiteDefinition, db_path: Optional[str] = None
) -> tuple[Database, LearningEngine, StrategyEngine, str]:
    """Create a fresh agent + engine for one suite on an isolated database."""
    db = Database(db_path or ":memory:")
    registry = build_registry(suite)
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
    agent = db.create_agent(f"{suite.suite_id}-agent")
    set_naive_priors(strategy, agent.id, suite)
    return db, engine, strategy, agent.id


def _instance_bundles(
    suite: SuiteDefinition, set_name: str
) -> list[dict[str, Any]]:
    count = suite.instance_counts.get(set_name, 0)
    return [make_bundle(suite, set_name, idx) for idx in range(count)]


def run_suite_evolution(
    suite: SuiteDefinition,
    db_path: Optional[str] = None,
    *,
    train_a: Optional[list[dict[str, Any]]] = None,
    train_b: Optional[list[dict[str, Any]]] = None,
    eval_bundles: Optional[list[dict[str, Any]]] = None,
    gen_bundles: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run V1 -> learning -> V2 -> learning -> V3 for one suite and return its
    full machine-readable report.

    The default instance sets come from the suite definition; test callers can
    override any of them with smaller deterministic bundles.
    """
    db, engine, strategy, agent_id = build_suite_world(suite, db_path)
    try:
        # Fixed instance sets so results are reproducible.
        train_a_bundles = train_a if train_a is not None else _instance_bundles(suite, "train_a")
        train_b_bundles = train_b if train_b is not None else _instance_bundles(suite, "train_b")
        eval_set = eval_bundles if eval_bundles is not None else _instance_bundles(suite, "eval")
        gen_set = gen_bundles if gen_bundles is not None else _instance_bundles(suite, "gen")

        report: dict[str, Any] = {
            "suite_id": suite.suite_id,
            "title": suite.title,
            "description": suite.description,
            "tools": suite.tools(),
            "cost_model": suite.cost_map(),
            "steps": [
                {
                    "step_id": s.step_id,
                    "category": s.category,
                    "question": s.question,
                    "correct_tool": s.correct_tool,
                    "decoy_tool": s.decoy_tool,
                }
                for s in suite.steps
            ],
            "instance_counts": dict(suite.instance_counts),
            "stages": {},
            "training": {},
        }

        # --- V1: naive strategy (decoy priors only) -------------------------
        v1 = _agent_version(engine, agent_id)
        report["stages"]["v1"] = {}
        report["stages"]["v1"]["agent_version"] = v1
        for set_name in MEASURED_SETS:
            bundles = eval_set if set_name == "eval" else gen_set
            results = measure_scenarios(
                suite, engine, agent_id, set_name, "v1", bundles, v1
            )
            report["stages"]["v1"][set_name] = _stage_metrics(results)
            report["stages"]["v1"][f"{set_name}_scenarios"] = _scenario_summaries(results)

        # --- Learning phase A (experience -> learning -> V2) ----------------
        phase_a = train_on(suite, engine, agent_id, train_a_bundles)
        report["training"]["phase_a"] = phase_a.to_dict()
        v2 = _agent_version(engine, agent_id)
        report["stages"]["v2"] = {}
        report["stages"]["v2"]["agent_version"] = v2
        for set_name in MEASURED_SETS:
            bundles = eval_set if set_name == "eval" else gen_set
            results = measure_scenarios(
                suite, engine, agent_id, set_name, "v2", bundles, v2
            )
            report["stages"]["v2"][set_name] = _stage_metrics(results)
            report["stages"]["v2"][f"{set_name}_scenarios"] = _scenario_summaries(results)

        # --- Learning phase B (more experience -> learning -> V3) -----------
        phase_b = train_on(suite, engine, agent_id, train_b_bundles)
        report["training"]["phase_b"] = phase_b.to_dict()
        v3 = _agent_version(engine, agent_id)
        report["stages"]["v3"] = {}
        report["stages"]["v3"]["agent_version"] = v3
        for set_name in MEASURED_SETS:
            bundles = eval_set if set_name == "eval" else gen_set
            results = measure_scenarios(
                suite, engine, agent_id, set_name, "v3", bundles, v3
            )
            report["stages"]["v3"][set_name] = _stage_metrics(results)
            report["stages"]["v3"][f"{set_name}_scenarios"] = _scenario_summaries(results)

        return report
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Multi-suite evolution + output generation
# ---------------------------------------------------------------------------


def _aggregate_stage(
    reports: list[dict[str, Any]], stage: str, set_name: str
) -> dict[str, Any]:
    """Totals across all suites for one (stage, set)."""
    metrics = [r["stages"][stage][set_name] for r in reports]
    scenarios_total = sum(m["scenarios_total"] for m in metrics)
    scenarios_solved = sum(m["scenarios_solved"] for m in metrics)
    steps_total = sum(m["steps_total"] for m in metrics)
    steps_solved = sum(m["steps_solved"] for m in metrics)
    total_calls = sum(m["total_tool_calls"] for m in metrics)
    unnecessary = sum(m["unnecessary_tool_calls"] for m in metrics)
    failures = sum(m["failures"] for m in metrics)
    latency = sum(m["latency_ms"] for m in metrics)
    cost = sum(m["cost"] for m in metrics)
    return {
        "scenarios_total": scenarios_total,
        "scenarios_solved": scenarios_solved,
        "solve_rate": round(scenarios_solved / scenarios_total, 4) if scenarios_total else 0.0,
        "step_success_rate": round(steps_solved / steps_total, 4) if steps_total else 0.0,
        "total_tool_calls": total_calls,
        "unnecessary_tool_calls": unnecessary,
        "unnecessary_call_rate": round(unnecessary / total_calls, 4) if total_calls else 0.0,
        "failures": failures,
        "latency_ms": latency,
        "average_latency_ms_per_call": round(latency / total_calls, 2) if total_calls else 0.0,
        "cost": round(cost, 4),
        "average_cost_per_solved_scenario": round(cost / scenarios_solved, 4) if scenarios_solved else 0.0,
    }


def _pct_change(later: float, earlier: float) -> Optional[float]:
    if earlier == 0:
        return None
    return round((later - earlier) / earlier * 100.0, 2)


def _evolution_summary_json(report: dict[str, Any]) -> dict[str, Any]:
    """Concise, numbers-only summary derived from the full report."""
    suites = report["per_suite"]
    stages = report["evolution"]

    summary: dict[str, Any] = {
        "experiment": "V1 -> experience -> learning -> V2 -> learning -> V3",
        "generated_at": report["generated_at"],
        "schema_version": report["schema_version"],
        "notes": report["notes"],
        "headline": {},
        "improvement": {},
        "generalization": {"eval_held_out": {}, "gen_unseen": {}},
        "per_suite": {},
    }

    key_metrics = [
        "solve_rate",
        "step_success_rate",
        "total_tool_calls",
        "unnecessary_tool_calls",
        "unnecessary_call_rate",
        "failures",
        "latency_ms",
        "average_latency_ms_per_call",
        "cost",
    ]

    for set_name in MEASURED_SETS:
        summary["headline"][set_name] = {
            stage: stages[set_name][stage] for stage in STAGES
        }
        improvements: dict[str, Any] = {}
        for stage_now, stage_before in (("v2", "v1"), ("v3", "v2"), ("v3", "v1")):
            before = stages[set_name][stage_before]
            now = stages[set_name][stage_now]
            improvements[stage_now + "_vs_" + stage_before] = {
                "solve_rate": {
                    "v": now["solve_rate"],
                    "before": before["solve_rate"],
                    "pct_change": _pct_change(now["solve_rate"], before["solve_rate"]),
                },
                "unnecessary_tool_calls": {
                    "v": now["unnecessary_tool_calls"],
                    "before": before["unnecessary_tool_calls"],
                    "pct_change": _pct_change(now["unnecessary_tool_calls"], before["unnecessary_tool_calls"]),
                },
                "total_tool_calls": {
                    "v": now["total_tool_calls"],
                    "before": before["total_tool_calls"],
                    "pct_change": _pct_change(now["total_tool_calls"], before["total_tool_calls"]),
                },
                "failures": {
                    "v": now["failures"],
                    "before": before["failures"],
                    "pct_change": _pct_change(now["failures"], before["failures"]),
                },
                "latency_ms": {
                    "v": now["latency_ms"],
                    "before": before["latency_ms"],
                    "pct_change": _pct_change(now["latency_ms"], before["latency_ms"]),
                },
                "cost": {
                    "v": now["cost"],
                    "before": before["cost"],
                    "pct_change": _pct_change(now["cost"], before["cost"]),
                },
            }
        summary["improvement"][set_name] = improvements

        for stage in STAGES:
            summary["generalization"][("eval_held_out" if set_name == "eval" else "gen_unseen")][stage] = {
                k: stages[set_name][stage][k] for k in key_metrics
            }

    for suite_id, suite_report in suites.items():
        summary["per_suite"][suite_id] = {
            "title": suite_report["title"],
            "stages": {
                stage: {
                    "agent_version": suite_report["stages"][stage]["agent_version"],
                    "eval": {
                        k: suite_report["stages"][stage]["eval"][k]
                        for k in key_metrics
                    },
                    "gen": {
                        k: suite_report["stages"][stage]["gen"][k]
                        for k in key_metrics
                    },
                }
                for stage in STAGES
            },
            "training": suite_report["training"],
        }

    return summary


def _narrative(report: dict[str, Any]) -> list[str]:
    """Concise evolution narrative built only from the measured numbers."""
    stages = report["evolution"]
    lines: list[str] = []
    for set_name, label in (("eval", "held-out"), ("gen", "unseen")):
        s1, s2, s3 = (stages[set_name][s] for s in STAGES)
        lines.append(
            f"On the {label} investigation instances, scenario solve rate rose from "
            f"{s1['solve_rate'] * 100:.1f}% (V1) to {s2['solve_rate'] * 100:.1f}% (V2) "
            f"to {s3['solve_rate'] * 100:.1f}% (V3)."
        )
        lines.append(
            f"Unnecessary tool calls on {label} instances fell from {s1['unnecessary_tool_calls']} "
            f"(V1) to {s2['unnecessary_tool_calls']} (V2) to {s3['unnecessary_tool_calls']} (V3); "
            f"total tool calls from {s1['total_tool_calls']} to {s2['total_tool_calls']} to "
            f"{s3['total_tool_calls']}."
        )
        lines.append(
            f"Failures on {label} instances: V1 {s1['failures']}, V2 {s2['failures']}, V3 {s3['failures']}. "
            f"Measured latency: {s1['latency_ms']}ms -> {s2['latency_ms']}ms -> {s3['latency_ms']}ms "
            f"(avg per call {s1['average_latency_ms_per_call']}ms -> {s2['average_latency_ms_per_call']}ms "
            f"-> {s3['average_latency_ms_per_call']}ms). Modeled cost: {s1['cost']:.2f} -> "
            f"{s2['cost']:.2f} -> {s3['cost']:.2f}."
        )
    return lines


def run_evolution(
    suites: Optional[list[SuiteDefinition]] = None,
    output_dir: str = "benchmarks/results",
    latency_scale: float = 1.0,
) -> dict[str, Any]:
    """Run the full evolution benchmark across suites and write the machine-
    readable results + evolution summary files. Returns the report dict.

    All metrics in the output are computed from this run's real executions;
    nothing is hardcoded or fabricated.
    """
    set_latency_scale(latency_scale)
    suites = suites if suites is not None else SUITES
    os.makedirs(output_dir, exist_ok=True)

    per_suite: dict[str, Any] = {}
    per_suite_ordered = []
    for suite in suites:
        db_path = os.path.join(output_dir, f"{suite.suite_id}.apforge.db")
        report = run_suite_evolution(suite, db_path=db_path)
        per_suite[suite.suite_id] = report
        per_suite_ordered.append(report)

    evolution: dict[str, Any] = {}
    for set_name in MEASURED_SETS:
        evolution[set_name] = {
            stage: _aggregate_stage(per_suite_ordered, stage, set_name)
            for stage in STAGES
        }

    full_report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": utcnow_iso(),
        "experiment": "V1 -> experience -> learning -> V2 -> learning -> V3",
        "engine": {
            "name": "APforge core learning engine",
            "loop": "USE -> OBSERVE -> EVALUATE -> REFLECT -> REMEMBER -> CHANGE STRATEGY -> USE AGAIN",
        },
        "methodology": {
            "measurement": (
                "Measurement phases probe the current learned strategy via the "
                "real use/observe/evaluate pipeline and never mutate strategy "
                "deltas or tool-use memory."
            ),
            "learning": (
                "Learning phases run the full engine loop on training "
                "instances; they are what accumulate tool-use memory, apply "
                "strategy deltas, and bump agent versions."
            ),
            "correctness": (
                "A step is solved only when the correct instrument was used and "
                "its output matched the ground-truth value for that instance."
            ),
            "unnecessary_calls": (
                "A call is unnecessary when it did not use the correct "
                "instrument for the step being investigated."
            ),
            "max_attempts_per_step": MAX_ATTEMPTS_PER_STEP,
            "naive_decoy_prior": NAIVE_DECOY_PRIOR,
            "determinism": "All instance bundles are generated from fixed seeds; runs are reproducible.",
        },
        "notes": [
            "All metrics come from real executions of the harness in this run.",
            "Latency is the wall-clock ms measured around each real tool call; "
            "tools model heavyweight instrumentation with small deterministic "
            "delays.",
            "Cost is a modeled per-instrument charge (units per call) applied "
            "to the real recorded call counts.",
            "Agent versions are bumped by the engine whenever a learning phase "
            "changes the strategy; V1/V2/V3 are the measured stages at version "
            "1, after phase A, and after phase B respectively.",
        ],
        "per_suite": per_suite,
        "evolution": evolution,
    }
    full_report["summary"] = _evolution_summary_json(full_report)
    full_report["narrative"] = _narrative({"evolution": evolution})

    # Machine-readable outputs.
    full_path = os.path.join(output_dir, "results.json")
    with open(full_path, "w", encoding="utf-8") as handle:
        json.dump(full_report, handle, indent=2, sort_keys=False)

    summary_json = _evolution_summary_json(full_report)
    summary_path = os.path.join(output_dir, "evolution_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2, sort_keys=False)

    markdown_path = os.path.join(output_dir, "evolution_summary.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(render_summary_markdown(full_report, summary_json))

    return full_report


def render_summary_markdown(
    full_report: dict[str, Any], summary: dict[str, Any]
) -> str:
    """The concise evolution summary used for the final demo (Markdown)."""
    stages = full_report["evolution"]
    lines: list[str] = []
    lines.append("# APforge evaluation & benchmarking layer - evolution summary")
    lines.append("")
    lines.append("Experiment: **V1 -> experience -> learning -> V2 -> learning -> V3**")
    lines.append("")
    lines.append(
        f"Generated: {full_report['generated_at']}  |  "
        f"Suites: {len(full_report['per_suite'])}  |  "
        f"Deterministic (fixed seeds)"
    )
    lines.append("")

    lines.append("## Headline metrics (all suites, measured from real executions)")
    lines.append("")
    lines.append("| metric | V1 (naive) | V2 (after learning 1) | V3 (after learning 2) |")
    lines.append("|---|---|---|---|")
    for set_name, label in (("eval", "held-out eval"), ("gen", "unseen gen")):
        lines.append(f"| **{label} instances** |  |  |  |")
        s1, s2, s3 = (stages[set_name][s] for s in STAGES)
        lines.append(
            f"| scenario solve rate | {s1['solve_rate']*100:.1f}% | "
            f"{s2['solve_rate']*100:.1f}% | {s3['solve_rate']*100:.1f}% |"
        )
        lines.append(
            f"| step success rate | {s1['step_success_rate']*100:.1f}% | "
            f"{s2['step_success_rate']*100:.1f}% | {s3['step_success_rate']*100:.1f}% |"
        )
        lines.append(
            f"| total tool calls | {s1['total_tool_calls']} | "
            f"{s2['total_tool_calls']} | {s3['total_tool_calls']} |"
        )
        lines.append(
            f"| unnecessary tool calls | {s1['unnecessary_tool_calls']} | "
            f"{s2['unnecessary_tool_calls']} | {s3['unnecessary_tool_calls']} |"
        )
        lines.append(
            f"| failures | {s1['failures']} | {s2['failures']} | {s3['failures']} |"
        )
        lines.append(
            f"| latency (total ms) | {s1['latency_ms']} | {s2['latency_ms']} | "
            f"{s3['latency_ms']} |"
        )
        lines.append(
            f"| avg latency / call (ms) | {s1['average_latency_ms_per_call']} | "
            f"{s2['average_latency_ms_per_call']} | {s3['average_latency_ms_per_call']} |"
        )
        lines.append(
            f"| modeled cost | {s1['cost']:.2f} | {s2['cost']:.2f} | {s3['cost']:.2f} |"
        )
        lines.append("")

    lines.append("## Generalization experiment (transfer of learned strategies)")
    lines.append("")
    lines.append(
        "V1/V2/V3 were measured on **unseen instances** (bundles never executed "
        "during training). If the learned tool-use strategies transfer, the "
        "later stages must solve novel instances without any retraining."
    )
    lines.append("")
    lines.append("| suite | V1 solve rate | V2 solve rate | V3 solve rate | V1->V3 unnecessary calls |")
    lines.append("|---|---|---|---|---|")
    for suite_id, suite_report in full_report["per_suite"].items():
        g1 = suite_report["stages"]["v1"]["gen"]
        g2 = suite_report["stages"]["v2"]["gen"]
        g3 = suite_report["stages"]["v3"]["gen"]
        lines.append(
            f"| {suite_id} | {g1['solve_rate']*100:.1f}% | "
            f"{g2['solve_rate']*100:.1f}% | {g3['solve_rate']*100:.1f}% | "
            f"{g1['unnecessary_tool_calls']} -> {g3['unnecessary_tool_calls']} |"
        )
    lines.append("")

    lines.append("## Narrative")
    lines.append("")
    for paragraph in full_report["narrative"]:
        lines.append(f"- {paragraph}")
    lines.append("")

    lines.append("## Per-suite stage snapshots (agent version + tool call breakdown)")
    lines.append("")
    for suite_id, suite_report in full_report["per_suite"].items():
        lines.append(f"### {suite_id} - {suite_report['title']}")
        lines.append("")
        for phase in ("phase_a", "phase_b"):
            summary = suite_report["training"][phase]
            turns = summary["turns"]
            successes = summary["successes"]
            failures = summary["failures"]
            updates = summary["strategy_updates"]
            lines.append(
                f"Learning {phase}: {turns} turns, "
                f"{successes} {'success' if successes == 1 else 'successes'}, "
                f"{failures} {'failure' if failures == 1 else 'failures'}, "
                f"{updates} {'strategy update' if updates == 1 else 'strategy updates'} "
                f"(agent version {summary['version_before']} -> {summary['version_after']})."
            )
        lines.append("")
        lines.append(
            "| stage | agent version (engine) | eval solve rate | gen (unseen) solve rate | "
            "eval unnecessary | gen unnecessary |"
        )
        lines.append("|---|---|---|---|---|---|")
        for stage in STAGES:
            snap = suite_report["stages"][stage]
            lines.append(
                f"| {stage} | {snap['agent_version']} | "
                f"{snap['eval']['solve_rate']*100:.1f}% | {snap['gen']['solve_rate']*100:.1f}% | "
                f"{snap['eval']['unnecessary_tool_calls']} | {snap['gen']['unnecessary_tool_calls']} |"
            )
        lines.append("")
        lines.append("Tool call breakdown on unseen (gen) instances by stage:")
        for stage in STAGES:
            breakdown = suite_report["stages"][stage]["gen"]["tool_call_breakdown"]
            lines.append(
                f"- {stage}: "
                + ", ".join(f"{tool}={count}" for tool, count in sorted(breakdown.items()))
            )
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    for note in full_report["notes"]:
        lines.append(f"- {note}")
    lines.append(f"- Reproduction: `python benchmarks/run_evolution.py`")
    lines.append("")
    return "\n".join(lines)


def load_results(path: str) -> dict[str, Any]:
    """Load a previously generated results.json."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":  # pragma: no cover
    run_evolution()