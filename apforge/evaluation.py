"""The EVALUATE step: score an observation against task criteria."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from .models import Evaluation, Observation, TraceStatus


class RuleEvaluator:
    """Deterministic rule-based evaluator.

    Scores an observation in [0, 1] against optional criteria:

    - ``expected_output``: the tool result must equal this value. If the
      criteria also set ``expected_field``, only that sub-field of the result
      dict is compared (used by the benchmark so the ground-truth *datum* is
      checked without exposing the correct tool identity in feedback).
    - ``signal_on_mismatch``: outcome signal name (e.g. ``wrong-information-
      source``) recorded in the notes when ``expected_field`` comparison fails.
    - ``signal_on_error``: outcome signal name (e.g. ``insufficient-evidence``)
      recorded in the notes when the tool errored.
    - ``output_contains``: the stringified result must contain this substring
    - ``max_duration_ms``: the tool must not take longer than this
    - ``must_not_contain``: list of substrings that must not appear in the
      result or error text

    Feedback is outcome-level: notes describe *why* an observation failed
    (success/failure, wrong-information-source, insufficient-evidence, ...)
    but never echo a ground-truth answer key.
    """

    def evaluate(
        self, observation: Observation, criteria: Optional[dict[str, Any]] = None
    ) -> Evaluation:
        criteria = criteria or {}
        notes: list[str] = []
        ok = True

        if observation.status != TraceStatus.SUCCESS:
            ok = False
            signal = criteria.get("signal_on_error")
            if signal:
                notes.append(
                    f"{signal}: tool errored: {observation.error or 'unknown error'}"
                )
            else:
                notes.append(f"tool errored: {observation.error or 'unknown error'}")

        if "expected_output" in criteria:
            expected = criteria["expected_output"]
            field = criteria.get("expected_field")
            if field:
                got = (
                    observation.result.get(field)
                    if isinstance(observation.result, dict)
                    else None
                )
                if got != expected:
                    ok = False
                    notes.append(
                        criteria.get(
                            "signal_on_mismatch", "wrong-information-source"
                        )
                        + ": the selected tool returned a different information "
                          "source than the task requires"
                    )
            elif observation.result != expected:
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