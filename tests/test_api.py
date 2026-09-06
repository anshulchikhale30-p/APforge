"""API tests: the learning loop exposed over HTTP."""

import pytest
from fastapi.testclient import TestClient

from apforge.api import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(str(tmp_path / "api.db"))
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["tools"] >= 4


def test_tools_listed(client):
    response = client.get("/api/tools")
    assert response.status_code == 200
    names = {tool["name"] for tool in response.json()}
    assert "math.evaluate" in names
    assert "demo.unreliable" in names


def test_create_and_get_agent(client):
    created = client.post("/api/agents", json={"name": "learner"})
    assert created.status_code == 201
    agent = created.json()
    assert agent["version"] == 1
    assert agent["name"] == "learner"

    fetched = client.get(f"/api/agents/{agent['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == agent["id"]

    missing = client.get("/api/agents/nope")
    assert missing.status_code == 404


def test_agents_list(client):
    client.post("/api/agents", json={"name": "a"})
    client.post("/api/agents", json={"name": "b"})
    agents = client.get("/api/agents").json()
    assert len(agents) == 2


def test_run_turn_and_inspect_state(client):
    agent = client.post("/api/agents", json={"name": "learner"}).json()
    aid = agent["id"]

    turn = client.post(
        f"/api/agents/{aid}/turns",
        json={
            "task_type": "math",
            "tool_input": "2 + 2 * 3",
            "criteria": {"expected_output": 8},
        },
    )
    assert turn.status_code == 200
    body = turn.json()
    assert body["selection"]["tool_name"] == "math.evaluate"
    assert body["trace"]["status"] == "success"
    assert body["evaluation"]["success"] is True
    assert body["reflection"]["failure_category"] == "none"
    # a success reinforces the tool -> strategy changed -> version bumped
    assert body["agent_version_after"] == 2
    assert body["agent_version_before"] == 1
    assert len(body["strategy_updates"]) == 1

    traces = client.get(f"/api/agents/{aid}/traces").json()
    assert len(traces) == 1
    assert traces[0]["id"] == body["trace"]["id"]

    trace_detail = client.get(f"/api/traces/{body['trace']['id']}")
    assert trace_detail.status_code == 200
    detail = trace_detail.json()
    assert detail["evaluation"]["success"] is True
    assert detail["reflection"] is not None

    memories = client.get(f"/api/agents/{aid}/memories").json()
    assert len(memories) == 1
    assert memories[0]["tool_name"] == "math.evaluate"
    assert memories[0]["success_count"] == 1

    strategies = client.get(f"/api/agents/{aid}/strategies").json()
    assert len(strategies) == 1
    assert strategies[0]["delta"] > 0

    versions = client.get(f"/api/agents/{aid}/versions").json()
    assert [v["version"] for v in versions] == [2, 1]
    assert versions[0]["reason"].startswith("math.evaluate:")


def test_loop_learns_over_http(tmp_path):
    """Learning changes tool selection across HTTP turns, deterministically."""
    app = create_app(str(tmp_path / "api.db"))
    client = TestClient(app)
    agent = client.post("/api/agents", json={"name": "learner"}).json()
    aid = agent["id"]

    # Turn 1: ties go to registration order -> math.evaluate; it fails on
    # non-arithmetic input, so the loop penalizes it.
    turn1 = client.post(
        f"/api/agents/{aid}/turns",
        json={"task_type": "api_call", "tool_input": "boom"},
    ).json()
    assert turn1["selection"]["tool_name"] == "math.evaluate"
    assert turn1["trace"]["status"] == "error"
    assert any(u["delta"] < 0 for u in turn1["strategy_updates"])
    assert turn1["agent_version_after"] == 2

    # Turn 2: the strategy change makes the agent switch tools.
    turn2 = client.post(
        f"/api/agents/{aid}/turns",
        json={"task_type": "api_call", "tool_input": "boom"},
    ).json()
    assert turn2["selection"]["tool_name"] != "math.evaluate"

    memories = client.get(f"/api/agents/{aid}/memories").json()
    strategies = client.get(f"/api/agents/{aid}/strategies").json()
    versions = client.get(f"/api/agents/{aid}/versions").json()
    assert any(s["delta"] < 0 for s in strategies)
    assert any(m["failure_count"] > 0 for m in memories)
    assert len(versions) > 1


def test_session_runs_multiple_turns(client):
    agent = client.post("/api/agents", json={"name": "learner"}).json()
    aid = agent["id"]
    response = client.post(
        f"/api/agents/{aid}/session",
        json={
            "tasks": [
                {"task_type": "math", "tool_input": "1+1"},
                {"task_type": "math", "tool_input": "2+2"},
            ]
        },
    )
    assert response.status_code == 200
    turns = response.json()
    assert len(turns) == 2
    assert turns[0]["agent_version_after"] == turns[1]["agent_version_before"]


def test_unknown_agent_returns_404(client):
    response = client.post(
        "/api/agents/nope/turns",
        json={"task_type": "math", "tool_input": "1+1"},
    )
    assert response.status_code == 404


def test_unknown_trace_returns_404(client):
    response = client.get("/api/traces/nope")
    assert response.status_code == 404


def test_reflections_endpoint(client):
    agent = client.post("/api/agents", json={"name": "learner"}).json()
    client.post(
        f"/api/agents/{agent['id']}/turns",
        json={"task_type": "api_call", "tool_input": "boom"},
    )
    reflections = client.get("/api/reflections").json()
    assert len(reflections) == 1
    assert reflections[0]["failure_category"] == "parameter_error"