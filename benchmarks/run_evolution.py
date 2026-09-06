"""Run the full APforge evaluation & benchmarking evolution experiment.

Generates machine-readable JSON results and a concise evolution summary under
``benchmarks/results/``:

- ``results.json``          full per-call, per-scenario, per-suite data
- ``evolution_summary.json`` concise machine-readable summary
- ``evolution_summary.md``   readable demo summary

Usage:  python benchmarks/run_evolution.py
"""

from __future__ import annotations

import os
import sys

# Allow running as a plain script from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    from apforge.benchmark.runner import run_evolution

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    report = run_evolution(output_dir=output_dir)
    print(f"benchmark complete: {report['experiment']}")
    for set_name, label in (("eval", "held-out"), ("gen", "unseen")):
        stages = report["evolution"][set_name]
        v1, v2, v3 = (stages[s] for s in ("v1", "v2", "v3"))
        print(
            f"[{label}] solve rate: V1 {v1['solve_rate']*100:.1f}% -> "
            f"V2 {v2['solve_rate']*100:.1f}% -> V3 {v3['solve_rate']*100:.1f}%"
        )
        print(
            f"[{label}] total calls: V1 {v1['total_tool_calls']} -> "
            f"V2 {v2['total_tool_calls']} -> V3 {v3['total_tool_calls']} | "
            f"unnecessary: V1 {v1['unnecessary_tool_calls']} -> "
            f"V2 {v2['unnecessary_tool_calls']} -> V3 {v3['unnecessary_tool_calls']} | "
            f"failures: V1 {v1['failures']} -> V2 {v2['failures']} -> "
            f"V3 {v3['failures']}"
        )
        print(
            f"[{label}] latency: V1 {v1['latency_ms']}ms -> V2 {v2['latency_ms']}ms -> "
            f"V3 {v3['latency_ms']}ms | cost: V1 {v1['cost']:.2f} -> "
            f"V2 {v2['cost']:.2f} -> V3 {v3['cost']:.2f}"
        )
    print(f"results written under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())