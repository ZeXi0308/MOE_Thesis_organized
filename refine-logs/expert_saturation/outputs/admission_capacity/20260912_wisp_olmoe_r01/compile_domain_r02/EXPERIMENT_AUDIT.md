# Experiment Audit: `compile_domain_r02/attempt01`

**Date:** 2026-09-14  
**Auditor:** Codex GPT-6, reused same-family reviewer  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`  
**Evidence tier:** native vLLM in-process finite-arrival request measurement on one RTX 5090 with the custom F/X pager  

## Overall verdict: PASS

No P0 or P1 integrity defect was found. The retained r02 artifacts support a `MEASUREMENT_ONLY` result: the declared MoE compile/handle domain completed, the measured requests did not enter an observed Triton compiler or handle-load path, and X reduced observed D2D bytes by about 54%. They do **not** establish a stable complete-request benefit, a noise bound, quality, SLO behavior, or method GO.

## Deterministic reconstruction

### Archive and frozen inputs: PASS

- `attempt01.tar.gz` is 6,318,591 bytes with SHA256 `acf66ad7688296778dfdcdc969d67b190f956b8eb2fe2153caa77b3cf6754cf1`. It contains 134 regular-file members, with no duplicate or unsafe names. The 133 non-manifest payloads match every size and SHA256 in `attempt01/retrieval_manifest.json`; the remaining member is the manifest itself. The extracted `attempt01/` files also match all 133 manifest entries.
- All 40 frozen inputs match `input_hashes.json`; its SHA256 is `de2b8a8e50f8f10333aec4cb57e87ca4e0772c776510503d40035030fe7502ee`, matching the launch-time check at `attempt01/launch.json:5-8`.
- The input source exists locally with the declared SHA256. The prepared requests and token IDs are exactly source indices 16 through 31, with no score, route, or future-output selection (`attempt01/input_provenance.json:2-25`).
- The run froze repository HEAD `c4ae4f5daaea929e1a4358862296d832f1cc67ab`, the external WiSP hashes, and package versions (`attempt01/protocol.json:81-121`). All four execution records match them; all 11 retained runtime sources match `sources.json`.

### Request, scheduler, and pager joins: PASS

| Cell | Mode | Requests / outputs | Scheduler steps / positions | Measured / all layer calls | Capture s | Mean completion s | Process s |
|---|---|---:|---:|---:|---:|---:|---:|
| `0_fullstage` | F | 16 / 512 | 56 / 2,544 | 896 / 1,008 | 8.443981 | 5.667933 | 63.775799 |
| `1_oneshot` | X | 16 / 512 | 57 / 2,544 | 912 / 1,024 | 8.116922 | 5.344573 | 58.836490 |
| `2_oneshot` | X | 16 / 512 | 56 / 2,544 | 896 / 1,008 | 8.360311 | 5.591888 | 61.399337 |
| `3_fullstage` | F | 16 / 512 | 55 / 2,544 | 880 / 992 | 8.353137 | 5.641121 | 60.909512 |

The four cells therefore retain 64 measured requests, 2,048 output tokens, 10,176 scheduled positions, 3,584 measured layer calls, and 4,032 all-phase layer calls. Independent reconstruction verified:

- each request joins to one frozen document and prompt digest;
- all 512 per-cell output events reconstruct the retained 32-token result and token timestamps;
- every request covers computed positions 0 through 158 exactly, with no preemption or recomputation;
- every nonempty scheduler step joins to exactly one call for each of 16 layers, with identical internal request ID, computed position, and prompt length multisets;
- all call IDs are contiguous, every call completed, and H2D/D2D group bytes equal the raw expert-copy lists and pager summaries.

The frozen analyzer performs these identity, request, source, resource, trace, and byte checks at `attempt01/finite_metrics.py:33-90` and the scheduler-to-row join at `attempt01/analysis_source/analyze_shared_pool_execution.py:34-87`. Its retained result has no issues (`attempt01/analysis.json:2-9`), and the per-cell request/output and scheduler totals appear at `attempt01/analysis.json:1333-1334,1479`, `2856-2857,3002`, `4379-4380,4525`, and `5902-5903,6048`.

### Compile, cache, and state coverage: PASS

| Cell | Domain calls | Unique keys | New pipeline: init / setup | Handle-load events | Measurement compiler / load events | Cache entries after | Setup s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0_fullstage` | 320 | 26 | 7 / 22 | 29 | 0 / 0 | 237 | 7.018654 |
| `1_oneshot` | 320 | 22 | 7 / 18 | 25 | 0 / 0 | 204 | 5.460023 |
| `2_oneshot` | 320 | 22 | 7 / 18 | 25 | 0 / 0 | 204 | 5.928610 |
| `3_fullstage` | 320 | 26 | 7 / 22 | 29 | 0 / 0 | 237 | 6.840687 |

- Each cell contains the exact ordered domain `(M=1..160) x (GEMM 1,2)`: 1,280 compile-only helper invocations in total. Every call completed one interception and ended with initialized handles. The lower actual compiler-pipeline count is expected because repeated domain calls reuse in-memory kernels.
- All actual compiler events are classified `cache_hit=false` / `compiler_pipeline` and paired with 108 total `launcher_and_driver_load` events. All occur during initialization or `compile_domain_warmup`; independent counting found **zero measurement-phase `compiler_call` events and zero measurement-phase handle/load events**, which is stronger than merely finding zero measurement misses. The retained phase counts and empty measurement-miss lists are at `attempt01/analysis.json:1517-1527`, `3040-3050`, `4563-4573`, and `6086-6096`.
- Every cell started from its own empty private Triton disk cache and produced a nonempty inventory; driver coverage is true for all cells (`attempt01/results/execution.json:89-107,1113`, `1199-1217,2091`, `2177-2195,3069`, and `3155-3173,4179`). Other backend caches were not declared cold (`attempt01/protocol.json:110-118`).
- The two installed native helper hashes exactly match the frozen sources. Compile setup preserved all 16 layers' slots, mappings, LRU state, last-route fields, `ever_loaded`, six explicit pager statistics, tensor pointers/shapes, records/events, context, and measurement flag. KV request IDs, free/total blocks, and all KV storage pointers/bytes were also exactly equal before and after.

### Resource and wall-time accounting: PASS

All four cells directly report 384 expert slots, 4,831,838,208 unique expert bytes, and 1,073,741,824 KV bytes. F uses private 320 plus stage 64; X uses private 336 plus stage 48. Each cell has two unique expert storages whose sizes sum to the declared expert budget, CPU affinity 0-7, and an empty scheduler request set at measurement start. Representative retained values are at `attempt01/analysis.json:1416-1479`; the same checks are executed for each cell at `attempt01/finite_metrics.py:56-65`.

The four setup intervals total 25.247974 seconds and lie inside their respective subprocess intervals. The independently recomputed subprocess timers total 244.921138 seconds. `process_wall_s` comes directly from the parent subprocess interval (`attempt01/run_attempt.py:110-119`), while capture and request metrics use their declared raw clocks (`attempt01/finite_metrics.py:91-121`). No setup, warmup, compiler, observer, I/O, failure, or shutdown time is subtracted from the process metric; nested CUDA/host spans are not added to wall time (`attempt01/protocol.json:68-79`).

### Pairing and observed drift: PASS

The analyzer uses the frozen F/X/X/F pairing `(F0,X1)` and `(F3,X2)`, then same-arm pairs `(F0,F3)` and `(X1,X2)` (`attempt01/analyze_covered.py:76-95`). Independent recomputation using `100 * (X/F - 1)` reproduced the retained values exactly:

| X minus F | `F0 -> X1` | `F3 -> X2` |
|---|---:|---:|
| Capture wall | -3.873282% | +0.085894% |
| Mean completion latency | -5.705090% | -0.872745% |
| D2D bytes | -54.040452% | -54.623099% |
| Process wall | -7.744802% | +0.804185% |
| Equal per-request outputs | 8/16 | 8/16 |

The raw denominators are direct baseline values, not model-output normalization (`attempt01/finite_metrics.py:110-134`). The comparison values are retained at `attempt01/analysis.json:6100-6205`.

Same-arm capture drift was -1.075846% for F and +2.998548% for X; same-arm mean-completion drift was -0.473054% and +4.627418%, respectively (`attempt01/analysis.json:6221-6319`). The primary capture and process comparisons change sign, and the observed X drift is of similar scale to the favorable first pairing. The two consistent mean-completion decreases and roughly 54% D2D reductions remain descriptive; two engines per mode do not establish significance, a population noise floor, or stable complete-request benefit.

## A-F integrity checks

### A. Ground/reference provenance: PASS

This is a real runtime performance measurement and uses no ground truth or generated reference. The 16-document workload is a source-order slice of the frozen 32-document pool, and its exact prompts are shared across all cells. No output, route, or label selected the slice (`attempt01/input_provenance.json:2-25`).

### B. Normalization and denominators: PASS

TTFT, TPOT, completion, capture, process time, H2D, and D2D are retained in raw units. Percentage changes use the declared F cell as denominator; no metric is divided by the run's own maximum, minimum, or output statistics (`attempt01/finite_metrics.py:110-134`). Capture excludes startup by definition, while process time retains it, and the report keeps those scopes separate.

### C. Result existence and retention: PASS

All four declared cells, raw requests, traces, compilation reports, host observations, logs, and the original remote analysis exist and match the retrieved archive manifest. No result cell is omitted. The failed r01 archive also remains separate.

### D. Actual called paths / dead-code risk: PASS

The executed commands invoke `instrumentation/run_covered.py`, every cell emits 320 domain-call records and host events, and each measurement emits a fully joined F or X pager trace. No reported metric or coverage claim depends only on an uncalled validation branch. This finding is limited to the paths used for the reported claims, not a whole-repository dead-code statement.

### E. Scope: PASS

The report describes one 16-document finite-arrival cohort, two fresh engines per mode, fixed F/X/X/F order, and one GPU. It explicitly withholds significance, quality, SLO, steady-state, pure-D2D causality, and method GO (`REPORT.md:3-5,26-42`; `attempt01/analysis.json:6327`). The language does not exceed the evidence.

### F. Evaluation classification: real runtime measurement, no GT

This is request-level native vLLM in-process measurement with a custom expert-paging runtime. It is not simulation, a quality evaluation, client/network serving, or production evidence.

## Failed r01 separation: PASS

r01 stopped in the first F process because `vars()` was applied to a slotted state. It retained a 42.951954-second failed process, zero compile-domain invocations, no `raw.json`, and therefore no request measurement (`../compile_domain_r01/attempt01/results/0_fullstage/compile_domain.json:1-26`; `../compile_domain_r01/attempt01/results/0_fullstage/status.json:1-4`; `../compile_domain_r01/attempt01/results/execution.json:1-8,64-68`). Its archive SHA256 `b822c24a80f9f97b74a1bcc67f2b4221062d5c88781bc238f568bdeb6584b79b` and failure reason are frozen into r02 (`attempt01/protocol.json:122-131`). No r01 timing enters any r02 comparison.

## Reproduction note

Running the frozen analyzer again produced JSON semantically identical to `attempt01/analysis.json`, with `issues=[]` and no missing cell. `analysis_rebuilt.json` is also semantically identical. The files are not byte-identical because metric dictionaries are emitted from a key-set intersection at `attempt01/finite_metrics.py:129-134`, whose order can vary between Python processes. The original remote analysis remains independently authenticated by the retrieval manifest. This affects byte-level reproducibility only, not keys, values, pairing, or the result claim.

## Claim impact

- **Supported:** all four r02 cells completed and are retained; source, request, scheduler, trace, resource, and byte joins pass.
- **Supported:** the declared M1..160 two-GEMM compile-only domain completed, handles were initialized, and no observed compiler or handle-load event occurred during measurement.
- **Supported as descriptive measurement:** X reduced measured D2D bytes by 54.040% and 54.623% in the two pairings.
- **Unsupported:** X has a stable complete-request or full-process advantage. Capture and process effects changed sign, and same-arm drift is material.
- **Unmeasured:** population noise, confidence intervals, output quality, SLO behavior, steady-state service, Oracle/headroom, and a causal time benefit attributable only to D2D reduction.

The report's final boundary is accurate: explicit coverage removes measurement-time Triton compilation as the explanation for this cohort, while the roughly 54% D2D reduction still does not translate into a stable complete-request net benefit (`REPORT.md:36-44`).
