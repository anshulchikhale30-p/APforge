"""Tests for the EVALUATE step."""

from apforge.evaluation import RuleEvaluator
from apforge.models import Observation, TraceStatus


def make_observation(status=TraceStatus.SUCCESS, result=None, error=None, duration_ms=1):
    return Observation(
        trace_id="t1", status=status, result=result, error=error, duration_ms=duration_ms
    )


def test_success_without_criteria_scores_full():
    ev = RuleEvaluator().evaluate(make_observation(result="ok"))
    assert ev.success is True
    assert ev.score == 1.0


def test_error_scores_zero():
    ev = RuleEvaluator().evaluate(
        make_observation(status=TraceStatus.ERROR, error="boom")
    )
    assert ev.success is False
    assert ev.score == 0.0
    assert "boom" in ev.notes


def test_expected_output_match_and_mismatch():
    evaluator = RuleEvaluator()
    criteria = {"expected_output": "ok"}
    assert evaluator.evaluate(make_observation(result="ok"), criteria).success
    bad = evaluator.evaluate(make_observation(result="nope"), criteria)
    assert bad.success is False
    assert "output mismatch" in bad.notes


def test_output_contains():
    evaluator = RuleEvaluator()
    criteria = {"output_contains": "error_code: 0"}
    ok = evaluator.evaluate(make_observation(result={"msg": "all error_code: 0"}), criteria)
    assert ok.success
    bad = evaluator.evaluate(make_observation(result="failed"), criteria)
    assert bad.success is False


def test_max_duration():
    evaluator = RuleEvaluator()
    criteria = {"max_duration_ms": 5}
    assert evaluator.evaluate(make_observation(duration_ms=3), criteria).success
    assert not evaluator.evaluate(make_observation(duration_ms=50), criteria).success


def test_must_not_contain():
    evaluator = RuleEvaluator()
    criteria = {"must_not_contain": ["Traceback"]}
    assert evaluator.evaluate(make_observation(result="fine"), criteria).success
    assert not evaluator.evaluate(
        make_observation(result="Traceback (most recent call last)"), criteria
    ).success


def test_evaluation_records_criteria_and_trace_id():
    ev = RuleEvaluator().evaluate(
        make_observation(status=TraceStatus.ERROR, error="nope"),
        {"expected_output": "ok"},
    )
    assert ev.trace_id == "t1"
    assert ev.criteria == {"expected_output": "ok"}
    assert 0.0 <= ev.score <= 1.0