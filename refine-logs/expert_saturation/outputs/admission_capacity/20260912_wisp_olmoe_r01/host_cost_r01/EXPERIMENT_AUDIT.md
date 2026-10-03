# Experiment Audit: `host_cost_r01/attempt01`

**Date:** 2026-09-14  
**Auditor:** Codex GPT-6, reused same-family reviewer  
**Independence / acceptance:** `same-family` / `provisional`  
**Evidence:** native vLLM in-process, one RTX 5090, finite-arrival X runtime with host observation

## Verdict: PASS

No P0 or P1 integrity defect was found. The four retained cells support only `MEASUREMENT_ONLY / OBSERVATIONAL_SOURCE_LOCALIZATION`: the on arm measured real engine/planner/native-kernel host intervals and GC overlap on its own trajectories. They do not identify a pure CPU, Python, GC, scheduler-wait, or GPU root cause, do not yield a subtractable observer cost, and do not support method GO.

## Reconstruction

The retrieved archive is 7,127,208 bytes with SHA256 `dea1b44f5e643ec8ca5c4e977cd56eddb41f38c0af37027c1230c554686270b3`. All 140 members are regular files with unique safe names; the 139 non-manifest payloads match every declared size/SHA256 and the extracted files. All 43 frozen inputs match `input_hashes.json` (SHA256 `97f0f7e719c020568e325666e493c25ea91dcc0c9d08fd57dfb1c3182aafe1e6`). The prepared workload is exactly source indices 16–31 from the declared frozen pool, whose source SHA256 also matches; selection used no result or route (`attempt01/input_provenance.json:2-25`). The rebuilt and remote analyses are semantically equal, with `issues=[]` and no missing cell (`attempt01/analysis.json:2-4`).

| Cell | Req / output | Steps / scheduled tokens | Measured / all layer calls | Engine / planner / kernel spans | GC | Capture s | Process s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0_observer_off` | 16 / 512 | 57 / 2,544 | 912 / 1,024 | 0 / 0 / 0 | 0 | 8.146565 | 62.227846 |
| `1_observer_on` | 16 / 512 | 56 / 2,544 | 896 / 1,008 | 56 / 896 / 896 | 184 | 8.196133 | 59.610601 |
| `2_observer_on` | 16 / 512 | 55 / 2,544 | 880 / 992 | 55 / 880 / 880 | 183 | 8.509791 | 61.816418 |
| `3_observer_off` | 16 / 512 | 57 / 2,544 | 912 / 1,024 | 0 / 0 / 0 | 0 | 8.176339 | 59.715127 |

Totals are 64 measured requests, 2,048 outputs, 10,176 scheduled tokens, 3,600 measured and 4,048 all-phase layer calls, plus four separate one-request/two-output warmups. Every request/output/scheduler/pager join passes, all calls complete, and no preemption or recomputation occurred. Each cell reports 384 expert slots, 4,831,838,208 expert bytes, 1 GiB KV, CPU affinity 0–7, one empty-start private Triton cache, 320/320 completed compile-domain calls with unchanged pager/KV state, and zero measurement-phase compiler or handle-load events. Qualification was disabled, as required for this performance observation (`attempt01/finite_metrics.py:33-90`; `attempt01/analyze_covered.py:29-66`).

The executed command in every cell calls `instrumentation/run_host_cost.py` with the frozen `off/on/on/off` switch (`attempt01/run_attempt.py:44-55`). The wrapper replaces only the measurement capture: off forwards the original call; on explicitly adds CPU diagnostics plus the GC observer, wraps the live planner, runtime kernel call, and `engine.step`, then restores all hooks (`attempt01/instrumentation/run_host_cost.py:75-120,157-191`). Raw evidence agrees: off has both switches false and zero spans/events; on has both true, exact frozen live namespace/source hashes, restored `engine/kernel/planner`, and no restoration error.

For each on cell, every engine ordinal maps one-to-one to a raw `engine_call`; each engine maps one scheduler step and exactly 16 planner plus 16 kernel calls. Every measured pager `call_id` appears exactly once in each layer-span kind with identical layer, rows, context and step, and every child interval is inside its parent engine interval. These are the checks actually executed by the frozen analyzer (`attempt01/analyze_host_cost.py:57-101`), independently reproduced from both raw traces.

| On cell | Engine wall / thread / process CPU ms | Planner wall ms | Kernel-host wall ms | Same-thread GC overlap ms | Wide observation envelope ms |
|---|---:|---:|---:|---:|---:|
| `1_observer_on` | 8156.657 / 8154.735 / 8221.481 | 243.178 | 412.713 | 46.426 | 31.641 |
| `2_observer_on` | 8486.156 / 8484.765 / 8566.168 | 256.909 | 601.275 | 50.228 | 15.357 |

The 329.500 ms engine-wall difference is accompanied by only 3.801 ms more engine-clipped GC overlap. Engine, planner and kernel spans are nested; thread/process/rusage clocks overlap; GC is a clipped interval union; the wide native `cpu_delta` includes before/after observation work. None is added to wall or deducted as savings (`attempt01/analyze_host_cost.py:16-43,97-117`). Both on cells observed `sched_schedstats=0`; retained nonzero schedstat deltas therefore remain unqualified and are not evidence of zero queueing. Thread CPU near wall can include driver CPU work or spinning and does not establish a Python/CPU bottleneck.

The paired capture changes are `+0.608%` and `+4.078%` using the corresponding off cell as denominator (`100*(on/off-1)`). Each on/off pair has only 10/16 equal outputs and a different route; the on repeat itself changes capture by `+3.827%`, with 11/16 equal outputs and a different route. These are complete perturbed trajectories, not a pure observer-overhead estimate. The report preserves this boundary and explicitly withholds significance, a noise floor, quality, SLO, stable benefit, old-run causality, and GO (`REPORT.md:14-34,42-44`; `attempt01/analysis.json:56761-56989`).

## A–F integrity

| Check | Result | Finding |
|---|---|---|
| A. Provenance | PASS | Exact frozen source-order documents/prompts; external/runtime/source hashes and package versions are retained; no GT or result-based selection. |
| B. Denominators/accounting | PASS | Raw units and direct off denominators; capture/process scopes separated; nested clocks and GC are not summed or subtracted. |
| C. Existence/retention | PASS | Four declared cells, logs, raw traces and analysis are present and authenticated; no cell omitted. |
| D. Called paths | PASS | Command, raw switches, live namespace, spans and restoration records prove the claimed on/off paths ran; no reported claim rests on dead validation code. |
| E. Scope | PASS | One fixed 16-document cohort, two fresh engines per switch and ordered off/on/on/off; observer and trajectory effects remain inseparable. |
| F. Classification | PASS | Real request-level runtime measurement without ground truth; not a quality evaluation, network service, production result, or causal method evaluation. |

**Claim ceiling:** The data localize observed host intervals and show observer-sensitive trajectory variation. They do not isolate removable overhead or a unique bottleneck. Same-family reviewer reuse makes this acceptance provisional.
