"""Console entry point: ``python -m apforge.benchmark``."""

from __future__ import annotations

import sys


def main() -> int:
    from .runner import run_evolution

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results"
    report = run_evolution(output_dir=output_dir)
    print(f"benchmark complete: {report['experiment']}")
    for set_name in ("eval", "gen"):
        stages = report["evolution"][set_name]
        v1, v2, v3 = (stages[s] for s in ("v1", "v2", "v3"))
        print(
            f"[{set_name}] solve rate: V1 {v1['solve_rate']*100:.1f}% -> "
            f"V2 {v2['solve_rate']*100:.1f}% -> V3 {v3['solve_rate']*100:.1f}% | "
            f"unnecessary calls: V1 {v1['unnecessary_tool_calls']} -> "
            f"V2 {v2['unnecessary_tool_calls']} -> V3 {v3['unnecessary_tool_calls']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())