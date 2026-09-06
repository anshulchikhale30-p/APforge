"""APforge evaluation and benchmarking layer.

Runs reproducible evolution experiments on top of the core learning engine:

    V1 (naive) --experience--> learning --> V2 --experience--> learning --> V3

across realistic engineering-investigation benchmark suites, measuring actual
success/correctness, total tool calls, unnecessary tool calls, failures,
latency, and cost from real executions, and exporting machine-readable JSON
results plus a concise evolution summary.

See ``apforge.benchmark.tasks`` for the benchmark task definitions and
``apforge.benchmark.runner`` for the harness.
"""

from . import tasks

__all__ = ["tasks"]