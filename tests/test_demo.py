"""Tests for the APforge demo dashboard.

The demo must display *real* benchmark metrics read from the committed result
files and real engine output — not fabricated numbers. These tests therefore:

- compare the summary/details endpoints byte-for-byte (field-wise) with the
  committed ``benchmarks/results`` files,
- assert the live loop endpoint runs real engine turns (real reflections,
  memory rows, strategy deltas, version bumps),
- verify the "Run Demo" endpoint executes the real evolution harness
  (patched down to a single deterministic suite so the test stays fast) and
  returns refreshed results written to its own results directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apforge import demo
from apforge.benchmark.runner import run_evolution
from apforge.benchmark.tasks import CI_BUILD

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_RESULTS = REPO_ROOT / "benchmarks" / "results"

RESULT_FILES = ("results.json", "evolution_summary.json", "evolution_summary.md")


@pytest.fixture
def client(tmp_path):
    """Demo app over a copy of the committed results (repo files untouched)."""
    results = tmp_path / "results"
    results.mkdir()
    for name in RESULT_FILES:
        shutil.copy2(COMMITTED_RESULTS / name, results / name)
    app = demo.create_demo_app(results_dir=str(results))
    return TestClient(app)


def _committed(name: str) -> dict:
    with open(COMMITTED_RESULTS / name, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Page + data endpoints (real committed results)
# ---------------------------------------------------------------------------


def test_index_serves_dashboard(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "APforge" in html
    assert "Run Demo" in html
    assert "generalization" in html


def test_health_reports_results_present(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["results_present"] is True


def test_summary_matches_committed_metrics(client):
    committed = _committed("evolution_summary.json")
    body = client.get("/api/demo/summary").json()

    # identical to the committed file
    assert body == committed

    # and the headline really is the committed evolution: naive V1 solves
    # nothing, V3 solves everything, held-out eval and unseen gen both land.
    eval_v1 = body["headline"]["eval"]["v1"]
    eval_v3 = body["headline"]["eval"]["v3"]
    gen_v2 = body["headline"]["gen"]["v2"]
    assert eval_v1["solve_rate"] == 0.0
    assert eval_v3["solve_rate"] == 1.0
    assert gen_v2["solve_rate"] > 0.0
    assert gen_v2["solve_rate"] < 1.0
    assert eval_v3["unnecessary_tool_calls"] == 0
    assert "improvement" in body and "generalization" in body


def test_details_derived_from_committed_results(client):
    committed_full = _committed("results.json")
    body = client.get("/api/demo/details").json()

    assert body["generated_at"] == committed_full["generated_at"]
    assert body["engine"]["name"] == committed_full["engine"]["name"]
    assert set(body["suites"]) == set(committed_full["per_suite"])

    suite = body["suites"]["ci.build"]
    committed_suite = committed_full["per_suite"]["ci.build"]
    assert suite["title"] == committed_suite["title"]
    assert suite["steps"] == committed_suite["steps"]

    # per-call trace records must come straight from the committed file
    committed_call = committed_suite["stages"]["v1"]["eval_scenarios"][0]["calls"][0]
    served_call = suite["stages"]["v1"]["eval_scenarios"][0]["calls"][0]
    assert served_call == committed_call
    assert "unnecessary" in served_call
    assert "evaluation_success" in served_call


def test_details_include_narrative_and_notes(client):
    body = client.get("/api/demo/details").json()
    assert body["narrative"], "narrative must not be empty"
    assert body["notes"], "notes must not be empty"
    assert "real executions" in " ".join(body["notes"]).lower()


# ---------------------------------------------------------------------------
# Live learning-loop endpoint (real engine turns)
# ---------------------------------------------------------------------------


def test_loop_runs_real_engine_turns(client):
    body = client.get("/api/demo/loop").json()

    assert body["turns_total"] == len(body["turns"]) >= 1
    assert body["agent_version_start"] == 1
    assert body["agent_version_end"] > body["agent_version_start"]

    first = body["turns"][0]
    # deterministic naive V1: the decoy instrument is selected first and fails
    assert first["evaluation"]["success"] is False
    assert first["reflection"]["failure_category"] != "none"
    assert first["reflection"]["analysis"]
    assert first["reflection"]["lesson"]
    assert first["strategy_updates"], "a failing turn must produce strategy deltas"

    # every turn carries the full phase chain
    for turn in body["turns"]:
        assert set(turn) >= {
            "task_type", "step_question", "selection", "trace", "evaluation",
            "reflection", "strategy_updates",
        }
        assert "tool_name" in turn["selection"]
        assert turn["trace"]["status"] in ("success", "error")

    # the loop learns: at least one success later, memory + strategies persist
    assert any(t["evaluation"]["success"] for t in body["turns"])
    assert body["memory"], "persistent tool-use memory must be non-empty"
    assert body["strategies"], "strategy state must be non-empty"
    assert all("success_rate" in m for m in body["memory"])


def test_loop_payload_does_not_reveal_answer_key(client):
    """The live-loop walkthrough must not annotate steps with the answer key."""
    body = client.get("/api/demo/loop").json()

    steps = body.get("steps", [])
    assert steps, "live loop must describe its steps"
    for step in steps:
        assert "correct_tool" not in step
        assert "decoy_tool" not in step
        # non-answer task description is preserved for the dashboard
        assert {"step_id", "category", "question"} <= set(step)

    # no per-turn annotation names the ground-truth instrument
    for turn in body["turns"]:
        assert "correct_tool" not in turn
        assert "correct instrument" not in json.dumps(turn)


# ---------------------------------------------------------------------------
# Run Demo endpoint (real benchmark, refreshed results)
# ---------------------------------------------------------------------------


def test_run_endpoint_executes_real_benchmark_and_refreshes(client, monkeypatch):
    results_dir = Path(client.app.state.results_dir)

    # Patch only the scope (single deterministic suite, no simulated latency);
    # everything else is the real harness writing to the app's own directory.
    calls = {}

    def fast_run(output_dir):
        calls["output_dir"] = output_dir
        return run_evolution([CI_BUILD], output_dir=output_dir, latency_scale=0)

    monkeypatch.setattr(demo, "run_evolution", fast_run)

    response = client.post("/api/demo/run")
    assert response.status_code == 200
    body = response.json()

    assert calls["output_dir"] == str(results_dir)

    # refreshed result files were written into the app's results dir
    for name in RESULT_FILES:
        assert (results_dir / name).is_file()
    refreshed = json.loads((results_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    assert refreshed["generated_at"] == body["generated_at"]

    # single-suite deterministic run: naive V1 solves nothing, V3 solves all
    summary = body["summary"]
    assert set(summary["per_suite"]) == {"ci.build"}
    assert summary["headline"]["eval"]["v1"]["solve_rate"] == 0.0
    assert summary["headline"]["eval"]["v3"]["solve_rate"] == 1.0
    assert summary["headline"]["gen"]["v3"]["solve_rate"] == 1.0

    # refreshed detail payload tracks the new run
    assert set(body["details"]["suites"]) == {"ci.build"}
    assert body["loop"]["turns_total"] >= 1