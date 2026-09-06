"""The EVALUATE step: score an observation against task criteria."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from .models import Evaluation, Observation, TraceStatus


class RuleEvaluator:
    """Deterministic rule-based evaluator.

    Scores an observation in [0, 1] against optional criteria:

    - ``expected_output``: the tool result must equal this value
    - ``output_contains``: the stringified result must contain this substring
    - ``max_duration_ms``: the tool must not take longer than this
    - ``must_not_contain``: list of substrings that must not appear in the
      result or error text
    """

    def evaluate(
        self, observation: Observation, criteria: Optional[dict[str, Any]] = None
    ) -> Evaluation:
        criteria = criteria or {}
        notes: list[str] = []
        ok = True

        if observation.status != TraceStatus.SUCCESS:
            ok = False
            notes.append(f"tool errored: {observation.error or 'unknown error'}")

        if "expected_output" in criteria:
            expected = criteria["expected_output"]
            if observation.result != expected:
                ok = False
                notes.append(
                    f"output mismatch: expected {expected!r}, got {observation.result!r}"
                )

        if "output_contains" in criteria:
            needle = criteria["output_contains"]
            if needle not in str(observation.result or ""):
                ok = False
                notes.append(f"output missing expected substring {needle!r}")

        max_duration = criteria.get("max_duration_ms")
        if max_duration is not None and observation.duration_ms > max_duration:
            ok = False
            notes.append(
                f"took {observation.duration_ms}ms, exceeded {max_duration}ms limit"
            )

        for forbidden in criteria.get("must_not_contain", []):
            if forbidden in (observation.error or "") or forbidden in str(
                observation.result or ""
            ):
                ok = False
                notes.append(f"output/error unexpectedly contained {forbidden!r}")

        return Evaluation(
            id=uuid.uuid4().hex,
            trace_id=observation.trace_id,
            success=ok,
            score=1.0 if ok else 0.0,
            criteria=criteria,
            notes="; ".join(notes) if notes else "all criteria passed",
        )