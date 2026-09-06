"""Tests for the APforge evaluation & benchmarking layer.

These tests exercise the real harness (real tool executions, real engine loop,
real strategy/memory writes) with simulated latency disabled for speed. The
metrics they assert are computed from those executions — nothing is hardcoded.
"""

import json

import pytest

from apforge.benchmark.runner import (
    MAX_ATTEMPTS_PER_STEP,
    build_suite_world,
    measure_scenarios,
    run_evolution,
    run_suite_evolution,
    set_naive_priors,
    train_on,
)
from apforge.benchmark.tasks import (
    CI_BUILD,
    SUITES,
    build_registry,
    make_bundle,
    scenario_steps,
    set_latency_scale,
)
from apforge.models import TraceStatus
from apforge.strategy import StrategyEngine


@pytest.fixture(autouse=True)
def no_simulated_latency():
    """Benchmark tests run fast: zero simulated latency."""
    set_latency_scale(0)
    yield


def small_sets(suite, n_train_a=3, n_train_b=5, n_eval=3, n_gen=4):
    """Small deterministic instance bundles for fast tests."""
    return {
        "train_a": [make_bundle(suite, "train_a", i) for i in range(n_train_a)],
        "train_b": [make_bundle(suite, "train_b", i) for i in range(n_train_b)],
        "eval": [make_bundle(suite, "eval", i) for i in range(n_eval)],
        "gen": [make_bundle(suite, "gen", i) for i in range(n_gen)],
    }


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------


def test_four_engineering_suites_defined():
    assert [s.suite_id for s in SUITES] == [
        "ci.build",
        "sre.outage",
        "debug.segfault",
        "perf.regression",
    ]
    for suite in SUITES:
        assert len(suite.steps) == 4
        assert len(suite.instruments) == 9
        assert suite.title and suite.description


def test_suite_steps_reference_registered_tools():
    for suite in SUITES:
        tool_names = set(suite.tools())
        for step in suite.steps:
            assert step.correct_tool in tool_names, step
            assert step.decoy_tool in tool_names, step
            assert step.correct_tool != step.decoy_tool
        cost_map = suite.cost_map()
        assert set(cost_map) == tool_names
        assert all(c > 0 for c in cost_map.values())


def test_scenario_schedules_are_valid():
    for suite in SUITES:
        step_ids = {s.step_id for s in suite.steps}
        for set_name in ("eval", "gen"):
            sequences = scenario_steps(suite, set_name)
            assert sequences, set_name
            for sequence in sequences:
                assert len(sequence) <= len(suite.steps)
                for step in sequence:
                    assert step.step_id in step_ids


def test_bundles_are_deterministic():
    a = make_bundle(CI_BUILD, "eval", 0)
    b = make_bundle(CI_BUILD, "eval", 0)
    assert a == b
    c = make_bundle(CI_BUILD, "eval", 1)
    assert a != c  # different instances differ


def test_registry_executes_suite_tools():
    registry = build_registry(CI_BUILD)
    names = {t.name for t in registry.list()}
    assert names == set(CI_BUILD.tools())
    # Correct instrument returns the ground-truth datum.
    bundle = make_bundle(CI_BUILD, "eval", 0)
    step = CI_BUILD.steps[0]
    result = registry.execute(step.correct_tool, {"bundle": bundle})
    assert result.status == TraceStatus.SUCCESS
    assert result.result == {"source": step.correct_tool, "value": bundle[step.data_key]}


# ---------------------------------------------------------------------------
# Harness behavior
# ---------------------------------------------------------------------------


def test_naive_priors_bias_toward_decoys():
    db, engine, strategy, agent_id = build_suite_world(CI_BUILD, ":memory:")
    try:
        for step in CI_BUILD.steps:
            selection = strategy.select_tool(agent_id, step.category, engine.registry.list())
            assert selection.tool_name == step.decoy_tool
            strategy_row = db.get_strategy(agent_id, step.category, step.decoy_tool)
            assert strategy_row.prior >= 3.0
    finally:
        db.close()


def test_measurement_probes_without_mutating_learning_state():
    db, engine, strategy, agent_id = build_suite_world(CI_BUILD, ":memory:")
    try:
        bundles = small_sets(CI_BUILD)["eval"]
        results = measure_scenarios(
            CI_BUILD, engine, agent_id, "eval", "v1", bundles, version=1
        )
        # V1 (naive) never solves anything and wastes every attempt on decoys.
        assert all(not r.solved for r in results)
        assert all(
            r.total_calls == r.steps_total * MAX_ATTEMPTS_PER_STEP
            for r in results
        )
        assert all(r.unnecessary_calls == r.total_calls for r in results)
        assert sum(r.failure_calls for r in results) > 0

        # Measurement must not touch strategy or memory or versions.
        assert db.list_memory(agent_id) == []
        strategies = db.list_strategies(agent_id)
        assert strategies and all(s.delta == 0.0 for s in strategies)
        assert db.get_agent(agent_id).version == 1
    finally:
        db.close()


def test_learning_phase_accumulates_memory_and_versions():
    db, engine, strategy, agent_id = build_suite_world(CI_BUILD, ":memory:")
    try:
        sets = small_sets(CI_BUILD)
        summary = train_on(CI_BUILD, engine, agent_id, sets["train_a"])
        assert summary.turns == len(sets["train_a"]) * len(CI_BUILD.steps)
        assert summary.strategy_updates > 0
        assert summary.version_after > summary.version_before
        # Tool-use memory was written (the REMEMBER step ran).
        memory = db.list_memory(agent_id)
        assert memory, "learning phase must record tool-use memory"
        by_tool = {row.tool_name: row for row in memory}
        assert any(row.failure_count > 0 for row in memory)  # decoys failed
        assert any(row.success_count > 0 for row in memory)  # converged tools won
        assert any(by_tool[t].success_count > 0 for t in by_tool)
    finally:
        db.close()


def test_evolution_improves_v1_to_v2_to_v3(tmp_path):
    sets = small_sets(CI_BUILD)
    report = run_suite_evolution(
        CI_BUILD,
        db_path=str(tmp_path / "ci.db"),
        train_a=sets["train_a"],
        train_b=sets["train_b"],
        eval_bundles=sets["eval"],
        gen_bundles=sets["gen"],
    )
    for set_name in ("eval", "gen"):
        v1 = report["stages"]["v1"][set_name]
        v2 = report["stages"]["v2"][set_name]
        v3 = report["stages"]["v3"][set_name]
        # Genuine improvement through learned tool-use memory.
        assert v1["solve_rate"] < v2["solve_rate"] < v3["solve_rate"]
        assert v2["unnecessary_tool_calls"] < v1["unnecessary_tool_calls"]
        assert v3["unnecessary_tool_calls"] <= v2["unnecessary_tool_calls"]
        assert v2["failures"] <= v1["failures"]
        assert v3["failures"] <= v2["failures"]
        assert v3["total_tool_calls"] < v1["total_tool_calls"]
        assert v3["cost"] < v1["cost"]
    # V1 measured at version 1; later stages at bumped versions.
    assert report["stages"]["v1"]["agent_version"] == 1
    assert report["stages"]["v3"]["agent_version"] > report["stages"]["v2"]["agent_version"]
    # Both learning phases actually ran.
    assert report["training"]["phase_a"]["turns"] > 0
    assert report["training"]["phase_b"]["turns"] > 0


def test_generalization_transfer_to_unseen_instances(tmp_path):
    """Learned strategies must transfer to never-seen instances."""
    sets = small_sets(CI_BUILD)
    report = run_suite_evolution(
        CI_BUILD,
        db_path=str(tmp_path / "ci.db"),
        train_a=sets["train_a"],
        train_b=sets["train_b"],
        eval_bundles=sets["eval"],
        gen_bundles=sets["gen"],
    )
    g1 = report["stages"]["v1"]["gen"]
    g3 = report["stages"]["v3"]["gen"]
    assert g1["solve_rate"] == 0.0
    assert g3["solve_rate"] == 1.0
    assert g3["unnecessary_tool_calls"] == 0
    assert g3["failures"] == 0
    # Solved scenarios consume exactly one call per step.
    scenarios = report["stages"]["v3"]["gen_scenarios"]
    expected_calls = sum(s["steps_total"] for s in scenarios)
    assert g3["total_tool_calls"] == expected_calls


def test_metrics_are_derived_from_executions_not_fabricated(tmp_path):
    """Aggregated metrics must equal the sums of the recorded call data."""
    sets = small_sets(CI_BUILD)
    report = run_suite_evolution(
        CI_BUILD,
        db_path=str(tmp_path / "ci.db"),
        train_a=sets["train_a"],
        train_b=sets["train_b"],
        eval_bundles=sets["eval"],
        gen_bundles=sets["gen"],
    )
    for set_name in ("eval", "gen"):
        for stage in ("v1", "v2", "v3"):
            metrics = report["stages"][stage][set_name]
            scenarios = report["stages"][stage][f"{set_name}_scenarios"]
            assert metrics["scenarios_total"] == len(scenarios)
            assert metrics["scenarios_solved"] == sum(1 for s in scenarios if s["solved"])
            calls = [c for s in scenarios for c in s["calls"]]
            assert metrics["total_tool_calls"] == len(calls)
            assert metrics["unnecessary_tool_calls"] == sum(1 for c in calls if c["unnecessary"])
            assert metrics["failures"] == sum(1 for c in calls if c["status"] == "error")
            assert metrics["latency_ms"] == sum(c["duration_ms"] for c in calls)
            assert metrics["cost"] == round(sum(c["cost"] for c in calls), 4)
            # Step counts reconcile with the scenario summaries.
            assert metrics["steps_total"] == sum(s["steps_total"] for s in scenarios)
            assert metrics["steps_solved"] == sum(s["steps_solved"] for s in scenarios)


def test_runs_are_deterministic(tmp_path):
    sets = small_sets(CI_BUILD)
    def run():
        return run_suite_evolution(
            CI_BUILD,
            db_path=str(tmp_path / "run.db"),
            train_a=[dict(b) for b in sets["train_a"]],
            train_b=[dict(b) for b in sets["train_b"]],
            eval_bundles=[dict(b) for b in sets["eval"]],
            gen_bundles=[dict(b) for b in sets["gen"]],
        )

    first = run()
    second = run()
    for stage in ("v1", "v2", "v3"):
        assert first["stages"][stage] == second["stages"][stage]


def test_measurement_uses_real_agent_version_tagging(tmp_path):
    sets = small_sets(CI_BUILD)
    report = run_suite_evolution(
        CI_BUILD,
        db_path=str(tmp_path / "ci.db"),
        train_a=sets["train_a"],
        train_b=sets["train_b"],
        eval_bundles=sets["eval"],
        gen_bundles=sets["gen"],
    )
    # Scenario call records were captured from real execution traces.
    for scenario in report["stages"]["v3"]["eval_scenarios"]:
        assert all(call["evaluation_success"] for call in scenario["calls"])


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def test_run_evolution_writes_machine_readable_results(tmp_path):
    output_dir = str(tmp_path / "out")
    report = run_evolution([CI_BUILD], output_dir=output_dir, latency_scale=0)
    results_path = f"{output_dir}/results.json"
    summary_path = f"{output_dir}/evolution_summary.json"
    markdown_path = f"{output_dir}/evolution_summary.md"
    with open(results_path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(markdown_path, encoding="utf-8") as handle:
        markdown = handle.read()

    # Report shape.
    assert loaded["experiment"] == "V1 -> experience -> learning -> V2 -> learning -> V3"
    assert set(loaded["evolution"]) == {"eval", "gen"}
    assert set(loaded["evolution"]["eval"]) == {"v1", "v2", "v3"}
    assert set(loaded["per_suite"]) == {"ci.build"}
    assert loaded["summary"]["headline"]["eval"]["v1"]["solve_rate"] == 0.0
    assert "unnecessary_tool_calls" in loaded["summary"]["improvement"]["eval"]["v3_vs_v1"]
    assert "unseen instances" in markdown

    # The on-disk files match the in-memory report.
    assert loaded["evolution"] == report["evolution"]
    assert summary["headline"] == report["summary"]["headline"]
    assert summary["per_suite"] == report["summary"]["per_suite"]


def test_full_report_aggregates_multiple_suites(tmp_path):
    from apforge.benchmark.tasks import SRE_OUTAGE

    output_dir = str(tmp_path / "out")
    report = run_evolution([CI_BUILD, SRE_OUTAGE], output_dir=output_dir, latency_scale=0)
    assert len(report["per_suite"]) == 2
    agg = report["evolution"]["eval"]["v3"]
    # Aggregated totals equal the sum of the per-suite totals.
    total = sum(
        report["per_suite"][sid]["stages"]["v3"]["eval"]["total_tool_calls"]
        for sid in ("ci.build", "sre.outage")
    )
    assert agg["total_tool_calls"] == total
    assert agg["solve_rate"] == 1.0