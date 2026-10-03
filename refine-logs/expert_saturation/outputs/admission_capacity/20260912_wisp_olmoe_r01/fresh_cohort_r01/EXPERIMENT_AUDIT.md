# Experiment Audit Report

**Date:** 2026-09-14  
**Auditor route:** requested `gpt-5.6-sol` / `ultra`; fresh same-family Codex reviewer; provisional  
**Project:** `fresh_cohort_r01`  
**Overall verdict:** **PASS**  
**Integrity status:** `pass`

## P0/P1 findings

| Severity | Finding |
|---|---|
| P0 | None. |
| P1 | None. |

## Checks A-F

| Check | Status | Direct evidence and judgment |
|---|---|---|
| A. Provenance, reference type, outcome selection | PASS | Inputs are pinned Wikitext/tokenizer artifacts with hashes (`input_provenance.json:2-35`). The preparation code deterministically takes eligible articles 65-96 in source order and checks 32 unique document, document-hash, and prefix-hash identities before writing two fixed cohorts (`prepare_inputs.py:43-68`, `prepare_inputs.py:80-100`). The frozen protocol precedes the exclusive, no-retry run; every cell rechecks the 44-file frozen manifest (`run_attempt.py:26-40`, `run_attempt.py:67-75`, `run_attempt.py:89-138`). This supports “no result-based input or cell selection” for the retained iteration. The non-overlap statement is limited to the named prior inputs recorded in `input_provenance.json:36-118`, as the report states. |
| B. Raw metrics and denominators | PASS | Request TTFT, TPOT, maximum ITL, and completion latency are reconstructed directly from retained arrival/completion/token receipt times (`attempt01/analysis_source/analyze_native_pager_transfer.py:13-33`). Capture, process wall, H2D payload, D2D payload, and load envelope are separate fields (`finite_metrics.py:110-126`; `attempt01/analysis_source/analyze_shared_pool_execution.py:23-31`). Pair percentages are `100*(X/F-1)` and cohort summaries first average the two engine metrics per mode, then divide by the F mean (`analyze_fresh_cohort.py:33-43`, `analyze_fresh_cohort.py:109-130`). Independent recomputation from all eight `raw.json` and `pager/calls.jsonl` files matched every reported table value; no model-output maximum/minimum/mean is used to inflate a score. |
| C. Existence, completeness, retention | PASS | `execution.json` is COMPLETE with eight declared cells; every cell returned zero, completed 16 requests/512 outputs, and passed coverage (`attempt01/results/execution.json:2-24`, `attempt01/results/execution.json:1106-1114`, `attempt01/results/execution.json:8355-8363`). The rebuilt analysis has no issue or missing cell (`attempt01/analysis.json:2-4`) and is semantically identical to `analysis_rebuilt.json`. All 221 payload entries in `attempt01/retrieval_manifest.json` were rehashed in this audit with 0 missing/0 mismatched; the retained readback records 222 archive members, 221 payload files, and 44 frozen inputs (`readback_verification.json:2-7`). All four adjacent pairs, both cohort mean ratios, adverse max-ITL changes, and full process costs are retained (`attempt01/analysis.json:12206-12370`, `attempt01/analysis.json:12694-12910`). |
| D. Actual call path / dead code | PASS | The executed command calls `instrumentation/run_covered.py` with the declared cohort and mode (`attempt01/results/execution.json:18-52`). That runner installs the probe and compile setup, then executes the frozen shared-pool runner (`attempt01/instrumentation/run_covered.py:37-80`); the shared runner installs the F/X dispatch and calls the native runner (`attempt01/source/run_shared_pool_pager.py:215-244`); measurement calls `capture_episode` and writes the retained raw record (`attempt01/source/run_native_pager.py:485-527`). The report's metrics flow through `analyze_fresh_cohort.py:71-119`, `analyze_covered.py:21-67`, and `finite_metrics.py:25-134`. Optional kernel-validation functions were not called, but no metric or correctness claim is attributed to them. |
| E. Unit, scope, independence, noise | PASS | The experiment unit is explicitly two document cohorts; engines and requests are not presented as independent treatment replications (`protocol.json:66-66`). The analysis uses two engine-level values per mode within each cohort and forbids pooling 128 requests as independent n (`analyze_fresh_cohort.py:109-143`). The report disclaims significance, noninferiority, stable P99, and a population noise bound (`REPORT.md:30-43`). GPU boundary checks and the advisory lock establish only observed boundary cleanliness, not continuous physical isolation or a noise bound. Scope language does not exceed the evidence. |
| F. Evaluation classification | PASS | Audit taxonomy: **`self_supervised_proxy`**, subtype **real native system measurement with no ground truth by design**. Inputs are real corpus prefixes and outcomes are directly observed native-process timing/copy counters; there is no dataset target, human judgment, simulated outcome, or model-generated reference. This supports only descriptive relative system measurements. `MEASUREMENT_ONLY / CURRENT_X_NET_BENEFIT_NOT_ESTABLISHED` is the correct classification (`REPORT.md:1-5`, `REPORT.md:47-51`). |

## Qualification and treatment-sensitive boundaries

- Compile-domain qualification passed in all eight cells: 320 declared invocations, unchanged pager/KV state, and zero new measurement compiler-pipeline events (`analyze_covered.py:29-54`; `attempt01/analysis.json:1493-1525`, `attempt01/analysis.json:12161-12193`). F compiled 26 keys while X compiled 22, and setup time therefore differs by treatment. Capture excludes setup; full-process wall retains it. The latter is a joint implementation/process cost, not an isolated steady-state effect.
- Kernel/lifecycle qualification is disabled (`attempt01/results/c0_0_fullstage/shared_pool.json:3-4`, `attempt01/results/c0_0_fullstage/shared_pool.json:90-90`). Cross-arm outputs and routes differ and no requests were filtered. Numerical quality or semantic equivalence is therefore unmeasured, exactly as bounded in `REPORT.md:39-47`.
- The warmup uses the first document of its own cohort and retains the policy cache; this is consistent in every cell and explicitly frozen (`protocol.json:53-64`; `attempt01/source/run_native_pager.py:431-438`). The run is not a cold-start or independent-document replication.

## Claim ceiling

Supported: for this current joint X implementation, on one OLMoE model, one RTX 5090, two fixed 16-document cohorts, and two fresh engines per mode/cohort, the observed X/F capture and mean-completion directions are inconsistent; stable complete-request benefit is not established. Unsupported: a population/noise bound, significance, noninferiority, quality equivalence, business SLO, stable tail risk, full long-document prefill, true OOM/offload recovery, EP/multi-GPU behavior, or attribution of the full X/F delta specifically to D2D reduction.

No `.aris` trace was created, per repository/user instruction.
