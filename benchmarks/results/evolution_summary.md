# APforge evaluation & benchmarking layer - evolution summary

Experiment: **V1 -> experience -> learning -> V2 -> learning -> V3**

Generated: 2026-09-06T02:30:36.690023+00:00  |  Suites: 4  |  Deterministic (fixed seeds)

## Headline metrics (all suites, measured from real executions)

| metric | V1 (naive) | V2 (after learning 1) | V3 (after learning 2) |
|---|---|---|---|
| **held-out eval instances** |  |  |  |
| scenario solve rate | 0.0% | 40.0% | 100.0% |
| step success rate | 0.0% | 83.3% | 100.0% |
| total tool calls | 216 | 96 | 72 |
| unnecessary tool calls | 216 | 36 | 0 |
| failures | 102 | 0 | 0 |
| latency (total ms) | 2941 | 582 | 453 |
| avg latency / call (ms) | 13.62 | 6.06 | 6.29 |
| modeled cost | 327.30 | 96.00 | 72.00 |

| **unseen gen instances** |  |  |  |
| scenario solve rate | 0.0% | 33.3% | 100.0% |
| step success rate | 0.0% | 81.8% | 100.0% |
| total tool calls | 264 | 120 | 88 |
| unnecessary tool calls | 264 | 48 | 0 |
| failures | 126 | 0 | 0 |
| latency (total ms) | 3606 | 726 | 557 |
| avg latency / call (ms) | 13.66 | 6.05 | 6.33 |
| modeled cost | 400.80 | 120.00 | 88.00 |

## Generalization experiment (transfer of learned strategies)

V1/V2/V3 were measured on **unseen instances** (bundles never executed during training). If the learned tool-use strategies transfer, the later stages must solve novel instances without any retraining.

| suite | V1 solve rate | V2 solve rate | V3 solve rate | V1->V3 unnecessary calls |
|---|---|---|---|---|
| ci.build | 0.0% | 33.3% | 100.0% | 66 -> 0 |
| sre.outage | 0.0% | 33.3% | 100.0% | 66 -> 0 |
| debug.segfault | 0.0% | 33.3% | 100.0% | 66 -> 0 |
| perf.regression | 0.0% | 33.3% | 100.0% | 66 -> 0 |

## Narrative

- On the held-out investigation instances, scenario solve rate rose from 0.0% (V1) to 40.0% (V2) to 100.0% (V3).
- Unnecessary tool calls on held-out instances fell from 216 (V1) to 36 (V2) to 0 (V3); total tool calls from 216 to 96 to 72.
- Failures on held-out instances: V1 102, V2 0, V3 0. Measured latency: 2941ms -> 582ms -> 453ms (avg per call 13.62ms -> 6.06ms -> 6.29ms). Modeled cost: 327.30 -> 96.00 -> 72.00.
- On the unseen investigation instances, scenario solve rate rose from 0.0% (V1) to 33.3% (V2) to 100.0% (V3).
- Unnecessary tool calls on unseen instances fell from 264 (V1) to 48 (V2) to 0 (V3); total tool calls from 264 to 120 to 88.
- Failures on unseen instances: V1 126, V2 0, V3 0. Measured latency: 3606ms -> 726ms -> 557ms (avg per call 13.66ms -> 6.05ms -> 6.33ms). Modeled cost: 400.80 -> 120.00 -> 88.00.

## Per-suite stage snapshots (agent version + tool call breakdown)

### ci.build - Flaky CI build investigation

Learning phase_a: 12 turns, 3 successes, 9 failures, 21 strategy updates (agent version 1 -> 13).
Learning phase_b: 28 turns, 27 successes, 1 failure, 29 strategy updates (agent version 13 -> 41).

| stage | agent version (engine) | eval solve rate | gen (unseen) solve rate | eval unnecessary | gen unnecessary |
|---|---|---|---|---|---|
| v1 | 1 | 0.0% | 0.0% | 54 | 66 |
| v2 | 13 | 40.0% | 33.3% | 9 | 12 |
| v3 | 41 | 100.0% | 100.0% | 0 | 0 |

Tool call breakdown on unseen (gen) instances by stage:
- v1: ci.artifacts.fetch=18, ci.branch.log=18, ci.cache.inspect=18, ci.secret.scan=12
- v2: ci.deps.resolve=6, ci.runner.status=18, ci.tests.analyze=6
- v3: ci.config.read=4, ci.deps.resolve=6, ci.runner.status=6, ci.tests.analyze=6

### sre.outage - Production outage triage

Learning phase_a: 12 turns, 3 successes, 9 failures, 21 strategy updates (agent version 1 -> 13).
Learning phase_b: 28 turns, 27 successes, 1 failure, 29 strategy updates (agent version 13 -> 41).

| stage | agent version (engine) | eval solve rate | gen (unseen) solve rate | eval unnecessary | gen unnecessary |
|---|---|---|---|---|---|
| v1 | 1 | 0.0% | 0.0% | 54 | 66 |
| v2 | 13 | 40.0% | 33.3% | 9 | 12 |
| v3 | 41 | 100.0% | 100.0% | 0 | 0 |

Tool call breakdown on unseen (gen) instances by stage:
- v1: config.backup=18, deploy.env=12, metric.host=18, svc.env=18
- v2: config.diff=18, metric.region=6, svc.status=6
- v3: config.diff=6, deploy.history=4, metric.region=6, svc.status=6

### debug.segfault - Native crash debugging

Learning phase_a: 12 turns, 3 successes, 9 failures, 21 strategy updates (agent version 1 -> 13).
Learning phase_b: 28 turns, 27 successes, 1 failure, 29 strategy updates (agent version 13 -> 41).

| stage | agent version (engine) | eval solve rate | gen (unseen) solve rate | eval unnecessary | gen unnecessary |
|---|---|---|---|---|---|
| v1 | 1 | 0.0% | 0.0% | 54 | 66 |
| v2 | 13 | 40.0% | 33.3% | 9 | 12 |
| v3 | 41 | 100.0% | 100.0% | 0 | 0 |

Tool call breakdown on unseen (gen) instances by stage:
- v1: core.symsrv=12, deploy.manifest=18, mem.gc=18, trace.symbols=18
- v2: deploy.binary=18, mem.profile=6, trace.dump=6
- v3: core.files=4, deploy.binary=6, mem.profile=6, trace.dump=6

### perf.regression - Latency regression hunt

Learning phase_a: 12 turns, 3 successes, 9 failures, 21 strategy updates (agent version 1 -> 13).
Learning phase_b: 28 turns, 27 successes, 1 failure, 29 strategy updates (agent version 13 -> 41).

| stage | agent version (engine) | eval solve rate | gen (unseen) solve rate | eval unnecessary | gen unnecessary |
|---|---|---|---|---|---|
| v1 | 1 | 0.0% | 0.0% | 54 | 66 |
| v2 | 13 | 40.0% | 33.3% | 9 | 12 |
| v3 | 41 | 100.0% | 100.0% | 0 | 0 |

Tool call breakdown on unseen (gen) instances by stage:
- v1: conn.handshake=18, latency.avg=18, lock.waits=18, queue.jitter=12
- v2: conn.count=6, latency.p95=6, lock.stats=18
- v3: conn.count=6, latency.p95=6, lock.stats=6, queue.depth=4

## Methodology

- All metrics come from real executions of the harness in this run.
- Latency is the wall-clock ms measured around each real tool call; tools model heavyweight instrumentation with small deterministic delays.
- Cost is a modeled per-instrument charge (units per call) applied to the real recorded call counts.
- Agent versions are bumped by the engine whenever a learning phase changes the strategy; V1/V2/V3 are the measured stages at version 1, after phase A, and after phase B respectively.
- Reproduction: `python benchmarks/run_evolution.py`
