"""Engineering-investigation benchmark tasks for the APforge learning engine.

Each benchmark *suite* models a realistic engineering investigation (flaky CI
build, production outage triage, native crash debugging, latency regression
hunt). A suite owns:

- a set of *instruments* (tools registered through the standard
  ``ToolRegistry`` architecture, so failures become structured results and the
  learning loop can reflect on them),
- a set of *steps* (the pieces of information an engineer needs to gather to
  close the investigation),
- deterministic *instances* of the investigation (scenario data bundles) split
  into ``train_a`` (experience phase 1), ``train_b`` (experience phase 2),
  ``eval`` (held-out measurement), and ``gen`` (held-out, unseen instances for
  the generalization experiment).

Ground truth is part of the fixture: for every step there is exactly one
correct instrument; every other instrument is a decoy (either it returns a
plausible-but-wrong signal or it raises, modeling an unavailable instrument).
A *degraded* instrument models infrastructure that is flaky for some
instances. Nothing is random: instance bundles are generated from fixed seeds,
so every run is reproducible and every metric comes from a real execution of
the harness.

Latency is simulated with small deterministic sleeps so that per-tool latency
is *measured* (perf_counter around the real call) rather than fabricated; the
simulation can be disabled for fast tests via :func:`set_latency_scale`.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..tools import ToolRegistry

# ---------------------------------------------------------------------------
# Latency simulation
# ---------------------------------------------------------------------------

_LATENCY_SCALE = 1.0
"""Multiplier applied to per-tool simulated latency. 0 disables sleeps."""


def set_latency_scale(scale: float) -> None:
    """Set the latency simulation multiplier (0 disables simulated sleeps)."""
    global _LATENCY_SCALE
    _LATENCY_SCALE = float(scale)


def _latency_scale() -> float:
    return _LATENCY_SCALE


# ---------------------------------------------------------------------------
# Instrument / step / suite definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instrument:
    """One tool spec in a benchmark suite."""

    name: str
    description: str
    kind: str  # "reader" | "raiser" | "degraded"
    key: str = ""  # bundle key a reader/degraded instrument reads
    source: str = ""  # label embedded in the result dict
    message: str = ""  # error text for raiser/degraded instruments
    flag: str = ""  # bundle flag that makes a degraded instrument fail
    latency_ms: int = 0  # base simulated latency for this instrument
    cost: float = 1.0  # modeled cost per call (units)


@dataclass(frozen=True)
class StepDef:
    """One piece of the investigation: which datum, and which tool finds it."""

    step_id: str
    category: str  # task_type used by the strategy engine / memory
    question: str  # the engineering question this step answers
    correct_tool: str
    data_key: str  # bundle key the correct tool reads
    decoy_tool: str  # the tempting-but-wrong instrument for this step


@dataclass
class SuiteDefinition:
    """A full engineering-investigation benchmark suite."""

    suite_id: str
    title: str
    description: str
    steps: list[StepDef]
    instruments: list[Instrument]  # registration order (correct tools first)
    instance_counts: dict[str, int] = field(
        default_factory=lambda: {"train_a": 3, "train_b": 7, "eval": 5, "gen": 6}
    )
    seed_base: int = 1000
    scenario_schedules: dict[str, list[list[str]]] = field(default_factory=dict)

    def cost_map(self) -> dict[str, float]:
        """Modeled per-call cost for every registered instrument."""
        return {inst.name: inst.cost for inst in self.instruments}

    def costs(self) -> dict[str, float]:
        return self.cost_map()

    def tools(self) -> list[str]:
        return [inst.name for inst in self.instruments]


def _mixed_schedules(step_ids: list[str]) -> dict[str, list[list[str]]]:
    """Mixed investigation scopes: some scenarios need every probe, others a
    subset. Realistic (not every incident needs every instrument) and it lets
    the V2 stage show partial solve that V3 completes.
    """
    a, b, c, d = step_ids[0], step_ids[1], step_ids[2], step_ids[3]
    return {
        "eval": [
            [a, b, c],
            [a, b, c, d],
            [b, c, d, a],
            [a, b, c],
            [a, b, c, d],
        ],
        "gen": [
            [a, b, c, d],
            [a, b, c],
            [b, c, d, a],
            [a, b, c, d],
            [a, b, c],
            [a, b, c, d],
        ],
    }


def scenario_steps(
    suite: SuiteDefinition, set_name: str
) -> list[list[StepDef]]:
    """Per-scenario step sequences for a set, derived from the schedules."""
    steps_by_id = {step.step_id: step for step in suite.steps}
    schedules = suite.scenario_schedules.get(set_name)
    if not schedules:
        return [[step for step in suite.steps]]
    return [[steps_by_id[step_id] for step_id in schedule] for schedule in schedules]


# ---------------------------------------------------------------------------
# Bundle generation (deterministic per suite / set / index)
# ---------------------------------------------------------------------------


_SET_SEED_OFFSET = {"train_a": 1, "train_b": 2, "eval": 3, "gen": 4}


def _bundle_host(suite: SuiteDefinition, set_name: str, idx: int) -> dict[str, Any]:
    """Build one deterministic investigation data bundle for an instance.

    ``bundle[step.data_key]`` is the datum the correct instrument returns; the
    decoy fields are plausible-but-wrong signals. The ``degraded_<tool>`` flags
    flip the degraded instrument into an error for some instances.
    """
    offset = _SET_SEED_OFFSET.get(set_name, len(_SET_SEED_OFFSET) + 1)
    rng = random.Random(suite.seed_base * 1000 + offset * 100 + idx)

    def pick(*choices: str) -> str:
        return rng.choice(choices)

    def num(lo: int, hi: int) -> int:
        return rng.randint(lo, hi)

    suite_id = suite.suite_id
    variant = "gen" if set_name == "gen" else set_name

    if suite_id == "ci.build":
        flavor = "gen" if set_name == "gen" else ""
        return {
            "deps": f"dep:{pick('libx','liby','libz')}@"
                    f"{num(1,9)}.{num(0,9)}.{num(0,9)}{flavor}",
            "tests": f"test:test_{pick('auth','checkout','build','deploy')}"
                     f"_{num(1,50)}{flavor}",
            "runners": f"runner:{pick('healthy','degraded')}:{num(1,20)}{flavor}",
            "config": f"cfg:drift={pick('none','minor','major')}:{num(0,9)}{flavor}",
            # decoy signals (read by the wrong instruments)
            "artifacts": f"artifact:stale-{num(1,99)}{flavor}",
            "cache": f"cache:hit-{num(0,100)}%{flavor}",
            "branch": f"branch:main:{num(1,200)}{flavor}",
            "secret": f"secret:scan-{num(1,9)}{flavor}",
            "metrics": f"metrics:cpu-{num(0,95)}%{flavor}",
            "degraded_metrics": rng.random() < 0.4,
            "latency_hint": "ci",
        }
    if suite_id == "sre.outage":
        flavor = "-gen" if set_name == "gen" else ""
        return {
            "svc": f"svc:{pick('api','web','worker')}:{pick('ok','degraded')}"
                   f":{num(0,30)}s{flavor}",
            "region": f"region:{pick('us-east','eu-west','ap-south')}"
                      f":load={num(0,100)}%{flavor}",
            "config": f"cfg:diff={pick('none','threshold','timeout')}"
                      f":{num(0,9)}{flavor}",
            "deploys": f"deploy:{pick('v2.1.0','v2.1.1','v2.2.0')}"
                       f":{num(0,9)}h{flavor}",
            "env": f"env:{pick('prod','staging')}:{num(1,9)}{flavor}",
            "host": f"host:load={num(60,100)}%{flavor}",
            "backup": f"backup:{pick('ok','stale')}:{num(0,48)}h{flavor}",
            "deploy_env": f"envcfg:{pick('prod','canary')}:{num(0,9)}{flavor}",
            "logs": f"logs:agg-{num(100,9999)}{flavor}",
            "degraded_logs": rng.random() < 0.4,
            "latency_hint": "sre",
        }
    if suite_id == "debug.segfault":
        flavor = "-gen" if set_name == "gen" else ""
        return {
            "stack": f"stack:{pick('crash_thread','gc_thread')}:"
                     f"{pick('SIGSEGV','SIGABRT')}@{hex(num(0x1000, 0xf000))}{flavor}",
            "mem": f"mem:rss={num(100,4096)}MB:heap={num(10,90)}%{flavor}",
            "binary": f"bin:{pick('app','agent')}:{pick('1.3.2','1.3.3','1.4.0')}{flavor}",
            "cores": f"core:{num(0,5)}dumps:{pick('present','truncated')}{flavor}",
            "symbols": f"syms:{pick('stripped','present')}:{num(1,999)}{flavor}",
            "gc": f"gc:pause={num(1,400)}ms{flavor}",
            "manifest": f"manifest:{pick('signed','unsigned')}:{num(1,9)}{flavor}",
            "symsrv": f"symsrv:{pick('ok','unreachable')}:{num(0,9)}{flavor}",
            "perf": f"perf:sample-{num(0,1000)}{flavor}",
            "degraded_perf": rng.random() < 0.4,
            "latency_hint": "debug",
        }
    if suite_id == "perf.regression":
        flavor = "-gen" if set_name == "gen" else ""
        return {
            "p95": f"p95={num(50,4000)}ms{flavor}",
            "conns": f"conns={num(100,50000)}{flavor}",
            "locks": f"locks:contention={num(0,100)}%:{num(0,50)}waits{flavor}",
            "queues": f"queue:depth={num(0,1000)}{flavor}",
            "latavg": f"avg={num(20,2000)}ms{flavor}",
            "handshake": f"handshake={num(1,500)}ms{flavor}",
            "waits": f"waits={num(0,200)}/s{flavor}",
            "jitter": f"jitter={num(0,50)}ms{flavor}",
            "cpu": f"cpu:busy={num(10,99)}%{flavor}",
            "degraded_cpu": rng.random() < 0.4,
            "latency_hint": "perf",
        }
    raise ValueError(f"unknown suite: {suite_id}")


def make_bundle(suite: SuiteDefinition, set_name: str, idx: int) -> dict[str, Any]:
    """Deterministic instance bundle for ``(suite, set_name, index)``."""
    return _bundle_host(suite, set_name, idx)


def build_criteria(
    suite: SuiteDefinition, bundle: dict[str, Any], step: StepDef
) -> dict[str, Any]:
    """Evaluation criteria for one step on one instance."""
    return {"expected_output": {"source": step.correct_tool, "value": bundle[step.data_key]}}


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------


def _bundle_of(arg: Any) -> dict[str, Any]:
    if isinstance(arg, dict) and isinstance(arg.get("bundle"), dict):
        return arg["bundle"]
    return {}


def _reader_func(instrument: Instrument) -> Callable[[Any], Any]:
    def read(arg: Any) -> dict[str, Any]:
        return {"source": instrument.source, "value": _bundle_of(arg).get(instrument.key)}
    return read


def _raiser_func(instrument: Instrument) -> Callable[[Any], Any]:
    def raise_(arg: Any) -> Any:
        raise RuntimeError(instrument.message)
    return raise_


def _degraded_func(instrument: Instrument) -> Callable[[Any], Any]:
    def read(arg: Any) -> dict[str, Any]:
        bundle = _bundle_of(arg)
        if bundle.get(instrument.flag):
            raise RuntimeError(instrument.message)
        return {"source": instrument.source, "value": bundle.get(instrument.key)}
    return read


def _with_latency(func: Callable[[Any], Any], ms: int) -> Callable[[Any], Any]:
    if ms <= 0:
        return func

    def wrapped(arg: Any) -> Any:
        scale = _latency_scale()
        if scale > 0:
            time.sleep((ms / 1000.0) * scale)
        return func(arg)

    return wrapped


def build_registry(suite: SuiteDefinition) -> ToolRegistry:
    """Register every instrument of a suite through the standard registry."""
    registry = ToolRegistry()
    for instrument in suite.instruments:
        if instrument.kind == "raiser":
            func: Callable[[Any], Any] = _raiser_func(instrument)
        elif instrument.kind == "degraded":
            func = _degraded_func(instrument)
        else:
            func = _reader_func(instrument)
        registry.register_func(
            instrument.name,
            instrument.description,
            _with_latency(func, instrument.latency_ms),
        )
    return registry


# ---------------------------------------------------------------------------
# Benchmark suites (the realistic engineering-investigation tasks)
# ---------------------------------------------------------------------------

CI_BUILD = SuiteDefinition(
    suite_id="ci.build",
    title="Flaky CI build investigation",
    description=(
        "Root-cause why a CI job intermittently fails: resolve effective "
        "dependency versions, identify the flaky test, check runner health, "
        "and detect CI config drift."
    ),
    steps=[
        StepDef(
            step_id="deps",
            category="investigate.ci.dependencies",
            question="Which dependency combination broke the build?",
            correct_tool="ci.deps.resolve",
            data_key="deps",
            decoy_tool="ci.artifacts.fetch",
        ),
        StepDef(
            step_id="tests",
            category="investigate.ci.test_flakiness",
            question="Which test is flaky in the last runs?",
            correct_tool="ci.tests.analyze",
            data_key="tests",
            decoy_tool="ci.cache.inspect",
        ),
        StepDef(
            step_id="runners",
            category="investigate.ci.runner_health",
            question="Are the build runners healthy or saturated?",
            correct_tool="ci.runner.status",
            data_key="runners",
            decoy_tool="ci.branch.log",
        ),
        StepDef(
            step_id="config",
            category="investigate.ci.config_drift",
            question="Did the CI configuration drift from the baseline?",
            correct_tool="ci.config.read",
            data_key="config",
            decoy_tool="ci.secret.scan",
        ),
    ],
    instruments=[
        Instrument(
            "ci.deps.resolve",
            "Resolve the effective dependency versions for the CI job from "
            "lockfile and override sources.",
            "reader",
            key="deps",
            source="ci.deps.resolve",
            latency_ms=6,
            cost=1.0,
        ),
        Instrument(
            "ci.tests.analyze",
            "Analyze test history to flag flaky tests across recent CI runs.",
            "reader",
            key="tests",
            source="ci.tests.analyze",
            latency_ms=7,
            cost=1.0,
        ),
        Instrument(
            "ci.runner.status",
            "Report health, queue depth, and saturation of the build runners.",
            "reader",
            key="runners",
            source="ci.runner.status",
            latency_ms=5,
            cost=1.0,
        ),
        Instrument(
            "ci.config.read",
            "Read the effective CI configuration and compare with the baseline.",
            "reader",
            key="config",
            source="ci.config.read",
            latency_ms=6,
            cost=1.0,
        ),
        Instrument(
            "ci.artifacts.fetch",
            "Fetch build artifacts produced by the last CI run.",
            "raiser",
            message="upstream artifact store timed out: service unavailable",
            latency_ms=14,
            cost=1.6,
        ),
        Instrument(
            "ci.cache.inspect",
            "Inspect the contents of the CI dependency cache.",
            "reader",
            key="cache",
            source="ci.cache.inspect",
            latency_ms=12,
            cost=1.4,
        ),
        Instrument(
            "ci.branch.log",
            "List the newest commits on the current branch.",
            "raiser",
            message="git remote call timed out: network timeout",
            latency_ms=16,
            cost=1.6,
        ),
        Instrument(
            "ci.secret.scan",
            "Run a secret scan over the repository workspace.",
            "reader",
            key="secret",
            source="ci.secret.scan",
            latency_ms=13,
            cost=1.5,
        ),
        Instrument(
            "ci.metrics.query",
            "Query aggregate CI metrics (queue times, failure ratios).",
            "degraded",
            key="metrics",
            source="ci.metrics.query",
            message="metrics pipeline is down: ingestion latency exceeded budget",
            flag="degraded_metrics",
            latency_ms=22,
            cost=2.2,
        ),
    ],
    instance_counts={"train_a": 3, "train_b": 7, "eval": 5, "gen": 6},
    seed_base=1000,
)

SRE_OUTAGE = SuiteDefinition(
    suite_id="sre.outage",
    title="Production outage triage",
    description=(
        "Triage a production incident: which service is degraded, which region "
        "is overloaded, whether configuration drifted, and what changed in the "
        "last deploys."
    ),
    steps=[
        StepDef(
            step_id="svc",
            category="investigate.sre.service_status",
            question="Which service is degraded right now?",
            correct_tool="svc.status",
            data_key="svc",
            decoy_tool="svc.env",
        ),
        StepDef(
            step_id="region",
            category="investigate.sre.region_load",
            question="Which region is carrying abnormal load?",
            correct_tool="metric.region",
            data_key="region",
            decoy_tool="metric.host",
        ),
        StepDef(
            step_id="config",
            category="investigate.sre.config_drift",
            question="Did configuration drift before the incident?",
            correct_tool="config.diff",
            data_key="config",
            decoy_tool="config.backup",
        ),
        StepDef(
            step_id="deploys",
            category="investigate.sre.recent_deploys",
            question="Which deploy shipped closest to the incident start?",
            correct_tool="deploy.history",
            data_key="deploys",
            decoy_tool="deploy.env",
        ),
    ],
    instruments=[
        Instrument(
            "svc.status",
            "Report the current status of every service in the fleet.",
            "reader",
            key="svc",
            source="svc.status",
            latency_ms=6,
            cost=1.0,
        ),
        Instrument(
            "metric.region",
            "Return per-region load and error-rate metrics.",
            "reader",
            key="region",
            source="metric.region",
            latency_ms=7,
            cost=1.0,
        ),
        Instrument(
            "config.diff",
            "Diff live configuration against the last-known-good baseline.",
            "reader",
            key="config",
            source="config.diff",
            latency_ms=6,
            cost=1.0,
        ),
        Instrument(
            "deploy.history",
            "List recent deployments with timestamps and owners.",
            "reader",
            key="deploys",
            source="deploy.history",
            latency_ms=8,
            cost=1.0,
        ),
        Instrument(
            "svc.env",
            "Read the environment label of the investigated services.",
            "reader",
            key="env",
            source="svc.env",
            latency_ms=11,
            cost=1.4,
        ),
        Instrument(
            "metric.host",
            "Return per-host load metrics for the fleet.",
            "raiser",
            message="host metrics endpoint timed out: upstream unavailable",
            latency_ms=15,
            cost=1.6,
        ),
        Instrument(
            "config.backup",
            "Download the most recent configuration backup snapshot.",
            "reader",
            key="backup",
            source="config.backup",
            latency_ms=13,
            cost=1.5,
        ),
        Instrument(
            "deploy.env",
            "Read the current environment configuration of the services.",
            "raiser",
            message="config service timed out: connection refused",
            latency_ms=17,
            cost=1.7,
        ),
        Instrument(
            "log.aggregate",
            "Aggregate error logs across all services.",
            "degraded",
            key="logs",
            source="log.aggregate",
            message="log shipper is down: aggregation latency exceeded budget",
            flag="degraded_logs",
            latency_ms=24,
            cost=2.3,
        ),
    ],
    instance_counts={"train_a": 3, "train_b": 7, "eval": 5, "gen": 6},
    seed_base=2000,
)

DEBUG_SEGFAULT = SuiteDefinition(
    suite_id="debug.segfault",
    title="Native crash debugging",
    description=(
        "Debug a native crash: capture the stack trace, inspect memory, check "
        "which binary version shipped, and look for core dumps."
    ),
    steps=[
        StepDef(
            step_id="stack",
            category="investigate.debug.stack_trace",
            question="Which frame does the crash stack point at?",
            correct_tool="trace.dump",
            data_key="stack",
            decoy_tool="trace.symbols",
        ),
        StepDef(
            step_id="mem",
            category="investigate.debug.memory",
            question="What is the memory profile at crash time?",
            correct_tool="mem.profile",
            data_key="mem",
            decoy_tool="mem.gc",
        ),
        StepDef(
            step_id="binary",
            category="investigate.debug.binary_version",
            question="Which binary version was deployed?",
            correct_tool="deploy.binary",
            data_key="binary",
            decoy_tool="deploy.manifest",
        ),
        StepDef(
            step_id="cores",
            category="investigate.debug.core_dumps",
            question="Are core dumps available for the crashed process?",
            correct_tool="core.files",
            data_key="cores",
            decoy_tool="core.symsrv",
        ),
    ],
    instruments=[
        Instrument(
            "trace.dump",
            "Dump the stack trace (top frames) of the crashed process.",
            "reader",
            key="stack",
            source="trace.dump",
            latency_ms=8,
            cost=1.0,
        ),
        Instrument(
            "mem.profile",
            "Profile RSS and heap usage captured at crash time.",
            "reader",
            key="mem",
            source="mem.profile",
            latency_ms=7,
            cost=1.0,
        ),
        Instrument(
            "deploy.binary",
            "Report which binary version is deployed on the host.",
            "reader",
            key="binary",
            source="deploy.binary",
            latency_ms=5,
            cost=1.0,
        ),
        Instrument(
            "core.files",
            "List core dump files for the crashed process.",
            "reader",
            key="cores",
            source="core.files",
            latency_ms=6,
            cost=1.0,
        ),
        Instrument(
            "trace.symbols",
            "Resolve symbol tables for the crashed binary.",
            "raiser",
            message="symbol server timed out: upstream unavailable",
            latency_ms=16,
            cost=1.7,
        ),
        Instrument(
            "mem.gc",
            "Read GC pause statistics for the process.",
            "reader",
            key="gc",
            source="mem.gc",
            latency_ms=12,
            cost=1.4,
        ),
        Instrument(
            "deploy.manifest",
            "Read the deployment manifest for the release.",
            "reader",
            key="manifest",
            source="deploy.manifest",
            latency_ms=11,
            cost=1.3,
        ),
        Instrument(
            "core.symsrv",
            "Query the symbol server for core dump metadata.",
            "raiser",
            message="symbol server unreachable: connection timed out",
            latency_ms=18,
            cost=1.8,
        ),
        Instrument(
            "perf.sample",
            "Sample CPU profiles from the crashed process.",
            "degraded",
            key="perf",
            source="perf.sample",
            message="perf sampling failed: kernel counters unavailable",
            flag="degraded_perf",
            latency_ms=25,
            cost=2.4,
        ),
    ],
    instance_counts={"train_a": 3, "train_b": 7, "eval": 5, "gen": 6},
    seed_base=3000,
)

PERF_REGRESSION = SuiteDefinition(
    suite_id="perf.regression",
    title="Latency regression hunt",
    description=(
        "Track down a latency regression: check p95 latency, active "
        "connections, lock contention, and queue depth."
    ),
    steps=[
        StepDef(
            step_id="p95",
            category="investigate.perf.p95_latency",
            question="What is the p95 latency right now?",
            correct_tool="latency.p95",
            data_key="p95",
            decoy_tool="latency.avg",
        ),
        StepDef(
            step_id="conns",
            category="investigate.perf.connections",
            question="How many active connections are there?",
            correct_tool="conn.count",
            data_key="conns",
            decoy_tool="conn.handshake",
        ),
        StepDef(
            step_id="locks",
            category="investigate.perf.lock_contention",
            question="Is there lock contention in the hot path?",
            correct_tool="lock.stats",
            data_key="locks",
            decoy_tool="lock.waits",
        ),
        StepDef(
            step_id="queues",
            category="investigate.perf.queue_depth",
            question="How deep is the work queue?",
            correct_tool="queue.depth",
            data_key="queues",
            decoy_tool="queue.jitter",
        ),
    ],
    instruments=[
        Instrument(
            "latency.p95",
            "Return the p95 latency percentile from the metrics store.",
            "reader",
            key="p95",
            source="latency.p95",
            latency_ms=6,
            cost=1.0,
        ),
        Instrument(
            "conn.count",
            "Return the current number of active connections.",
            "reader",
            key="conns",
            source="conn.count",
            latency_ms=5,
            cost=1.0,
        ),
        Instrument(
            "lock.stats",
            "Return lock contention statistics for the hot path.",
            "reader",
            key="locks",
            source="lock.stats",
            latency_ms=7,
            cost=1.0,
        ),
        Instrument(
            "queue.depth",
            "Return the current work queue depth.",
            "reader",
            key="queues",
            source="queue.depth",
            latency_ms=6,
            cost=1.0,
        ),
        Instrument(
            "latency.avg",
            "Return the average (mean) latency from the metrics store.",
            "reader",
            key="latavg",
            source="latency.avg",
            latency_ms=10,
            cost=1.3,
        ),
        Instrument(
            "conn.handshake",
            "Measure TCP handshake times for new connections.",
            "raiser",
            message="network probe timed out: connection refused",
            latency_ms=15,
            cost=1.6,
        ),
        Instrument(
            "lock.waits",
            "Sample per-thread lock wait events.",
            "reader",
            key="waits",
            source="lock.waits",
            latency_ms=12,
            cost=1.4,
        ),
        Instrument(
            "queue.jitter",
            "Measure queue jitter across the last minute.",
            "raiser",
            message="queue probe timed out: upstream unavailable",
            latency_ms=17,
            cost=1.7,
        ),
        Instrument(
            "cpu.profile",
            "Sample CPU profiles of the service processes.",
            "degraded",
            key="cpu",
            source="cpu.profile",
            message="cpu profiling failed: kernel counters unavailable",
            flag="degraded_cpu",
            latency_ms=23,
            cost=2.2,
        ),
    ],
    instance_counts={"train_a": 3, "train_b": 7, "eval": 5, "gen": 6},
    seed_base=4000,
)


SUITES: list[SuiteDefinition] = [
    CI_BUILD,
    SRE_OUTAGE,
    DEBUG_SEGFAULT,
    PERF_REGRESSION,
]

for _suite in SUITES:
    _suite.scenario_schedules = _mixed_schedules([s.step_id for s in _suite.steps])


def get_suite(suite_id: str) -> Optional[SuiteDefinition]:
    for suite in SUITES:
        if suite.suite_id == suite_id:
            return suite
    return None