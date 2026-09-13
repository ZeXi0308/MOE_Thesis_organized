# Experiment Audit Report

**Date:** 2026-09-14 (Asia/Shanghai)  
**Campaign:** `20260914_rotation_runtime_repeat_r01`  
**Measured run:** `execution02_after_finite`  
**Auditor:** GPT-5.6-Sol ultra, fresh read-only same-family reviewer  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`  
**Repository HEAD inspected:** `c4ae4f5daaea929e1a4358862296d832f1cc67ab`  
**Workspace state:** dirty before this review; the audited source and artifacts were treated as immutable and identified by SHA-256.  
**Trace:** no `.aris` file was created, as required by the explicit review scope.

All short artifact references below are relative to `refine-logs/expert_saturation/outputs/admission_capacity/20260914_rotation_runtime_repeat_r01/`.

## Overall Verdict: PASS

The frozen four-cell A/B/B/A repeat, retained request records, action paths, accounting, analysis, same-path diagnostic, initialization-failure record, and final addendum are internally consistent. The reported values match direct recomputation from all four raw files. The final text preserves both throughput directions, describes the repeat as `MEASUREMENT_ONLY`, leaves APC=True `UNRUN`, and does not turn the host-side step-839 observation into a GPU/JIT cause.

- **P0:** 0
- **P1:** 0
- **Integrity status:** `pass`
- **Scientific status:** `OPEN / MEASUREMENT_ONLY`
- **Evidence tier:** `REQUEST_LEVEL / NATIVE_SERVING_INPROCESS_HOST_CAPTURE`
- **Evaluation type:** `self_supervised_proxy` for execution identity and systems performance; there is no external task-quality ground truth.

The one-model, one-GPU, same-cohort scope and same-family review make acceptance provisional. Those declared limits do not contradict the bounded claims and are not integrity findings.

## Independently Verified Result

| Cell | Complete | Wall s | Requests/s | Mean completion s | Max ITL s | Preemptions |
|---|---:|---:|---:|---:|---:|---:|
| block0 native | 32/32 | 24.438142 | 1.309429 | 21.903293 | 4.885155 | 2 natural |
| block0 most_output | 32/32 | 23.478504 | 1.362949 | 21.959875 | 1.064623 | 9 forced + 2 natural |
| block1 most_output | 32/32 | 24.326836 | 1.315420 | 22.803103 | 1.067163 | 9 forced + 2 natural |
| block1 native | 32/32 | 23.927953 | 1.337348 | 21.449650 | 4.710559 | 2 natural |

The raw-derived values agree with `analysis02_after_finite/report.md:7-17` and `RESULTS_AFTER_FINITE_ADDENDUM.md:7-10`:

- block 0 native → most_output: throughput `+4.087%`, mean completion `+0.258%`, maximum ITL `-3.820533 s`; 23/32 requests complete sooner and 9/32 later.
- block 1 native → most_output: throughput `-1.640%`, mean completion `+6.310%`, maximum ITL `-3.643396 s`; 0/32 requests complete sooner and 32/32 later.
- All four cells retain 32 requests × 1,024 output tokens. All within-block and same-role pairs have 32/32 equal output-token sequences. This is execution identity, not task-quality evaluation.

Same-role timing is retained rather than subtracted or used to correct a baseline:

| Same-role block0 → block1 | Wall change | Throughput change | Mean completion change | Max ITL change |
|---|---:|---:|---:|---:|
| native | `-0.510189 s` (`-2.088%`) | `+2.132%` | `-2.071%` | `-0.174596 s` |
| most_output | `+0.848332 s` (`+3.613%`) | `-3.487%` | `+3.840%` | `+0.002540 s` |

The native row is present at `analysis02_after_finite/analysis.json:194076-194097`; the most_output row is present at `analysis02_after_finite/analysis.json:194846-194866`. The compact Markdown renderer leaves its `Native A/A observed drift` table empty but explicitly points to the retained JSON at `analysis02_after_finite/report.md:19-30`; this is a presentation limitation, not missing evidence.

For the two most_output cells, direct wall accounting gives `+0.848332 s = +0.163471 s scheduler +0.555316 s engine excluding scheduler +0.129545 s outside engine calls`, matching `RESULTS_AFTER_FINITE_ADDENDUM.md:16`. Scheduled-work conservation also closes in every cell:

- native: `138725 = 98304 new prefill + 32736 new decode + 7685 recomputed positions`;
- most_output: `174511 = 98304 new prefill + 32736 new decode + 43471 recomputed positions`.

## A. Ground Truth Provenance: PASS

There is no semantic benchmark ground truth in this experiment. The frozen source pins model/tokenizer revisions, dataset revision and shard hash, selection rule, workload hash, request dimensions, and the same-cohort boundary in `execution02_after_finite/gpu_results/cohort2-block0-most_output/config.json:2-45`. The runner verifies workload and prompt-token hashes before execution at `preparation/source/run_probe.py:40-52`.

Output equality is used only as a paired execution-identity check. The final addendum explicitly rejects upgrading it to task quality at `RESULTS_AFTER_FINITE_ADDENDUM.md:12,22`. No model-produced output is presented as external ground truth.

## B. Score Normalization: PASS

The metric implementation retains every arrived request, rejects future execution, checks request/token/timestamp alignment, and declares the denominator as all arrived requests at the actual observation end (`preparation/source/metrics.py:10-29,62-105`). TTFT, TPOT, ITL, completion latency, throughput, and goodput are computed from raw clocks and counts at `preparation/source/metrics.py:106-155`.

No score is divided by the model's own maximum, minimum, mean, or proposed saving. Raw wall time, rate, completion, ITL, preemption, and work totals remain in the result; relative changes are computed only as action versus the predeclared native baseline (`analyze_completion_headroom.py:168-194`).

## C. Result File Existence and Numerical Fidelity: PASS

- The campaign fixes exactly four cells in A/B/B/A order at `preparation/source/campaign.json:12-50`.
- All four successful readbacks have return code 0, `COMPLETE`, 32 completed requests, and the expected 2/11/11/2 preemption counts at `execution02_after_finite/execution.json:2-68`. The execution archive and source identity are retained at `execution02_after_finite/execution.json:70-83`.
- All four raw, metrics, config, engine arguments, decisions, environment, status, warmup, memory, GPU, stdout/stderr, and execution-receipt files exist. The saved metrics match recomputation from raw because the analyzer enforces this before eligibility (`analyze_completion_headroom.py:104-160`).
- Every favorable and unfavorable primary value appears in `analysis02_after_finite/report.md:7-17` and the final addendum at `RESULTS_AFTER_FINITE_ADDENDUM.md:9-24`.

The first `execution/` is not counted as a result. It stopped during the first native engine initialization with return code 1 and `INCOMPLETE` before request measurement (`execution/execution.json:2-15`). No `raw.json` exists under its `gpu_results/`. `INITIALIZATION_FAILURE_ADDENDUM.md:1-3` accurately classifies it as a resource-handoff initialization failure rather than a mechanism result.

## D. Dead Code and Executed Analysis Path: PASS

The result-producing path is live:

- `run_campaign.py` launches one fresh cell process, refuses an existing result directory, records the command and return code, and preserves stdout/stderr (`preparation/source/run_campaign.py:15-56`).
- `run_probe.py` performs common warmups, checks GPU state before measurement, installs the requested policy, captures the actual episode, writes decisions/raw/metrics/status, and refuses a rotation cell with no forced action (`preparation/source/run_probe.py:140-194`).
- The most_output selector uses only current running/waiting/KV/output-count state (`preparation/source/absence_rotation.py:133-205`), and `rotation_native.py:125-211` applies and reconciles the action in the native scheduler path.
- `rotation_runtime_repeat.py:92-116` calls the imported cell validator, within-block comparisons, same-role comparisons, and recovery accounting; it emits `MEASUREMENT_ONLY` only when all four cells and all comparisons qualify.
- Saved metrics are recomputed from raw and checked for exact agreement at `analyze_completion_headroom.py:133-160`; request/work/preemption accounting is invoked on the same path.

No claimed metric is produced only by an uncalled helper, and no offline masking is used as a policy result.

## E. Scope Assessment: PASS

The measured scope is exactly one OLMoE BF16 model, vLLM 0.26.0, one RTX 5090 GPU UUID, one already-observed 32-document cohort, two reverse-order blocks, four fresh engines, 3,072 prompt plus 1,024 output tokens, 50 ms steady arrivals, and one seed. The design says `no new holdout` at `preparation/source/campaign.json:50` and freezes the claim ceiling and retention rules at `preparation/source/DECISIONS.md:3-19`.

The final text stays inside this scope: it reports the sign flip, leaves the problem open, disclaims a full Oracle, noninferiority, quality, business SLO, APC-on, cross-model, continuous-arrival, causal-attribution, and method-GO claim (`RESULTS_AFTER_FINITE_ADDENDUM.md:18-28`). APC=True remains `UNRUN`; it is a proposed next experiment, not part of this result.

## F. Evaluation Type and Evidence Tier: PASS

- **Evaluation type:** `self_supervised_proxy` for paired execution identity and direct systems measurement without semantic ground truth.
- **Evidence tier:** `REQUEST_LEVEL / NATIVE_SERVING_INPROCESS_HOST_CAPTURE`, recorded in raw capture by `preparation/source/native_capture.py:190-204`.
- **Clock and alignment:** one host `perf_counter` origin covers arrivals, scheduler intervals, engine calls, token receipts, and completion; token-level ITL is accepted only when chunks are size one (`preparation/source/native_capture.py:50-100,190-204`).
- **Claim ceiling:** descriptive request-level native-serving behavior for this frozen repeat; no network/client, multi-GPU EP, task-quality, production SLO, or device/kernel attribution.

The step-839 diagnostic is correctly bounded. `most_same_path_after_finite.json:2-39` hashes its raw/decision/config/engine inputs and says timing cannot identify GPU or compilation cost. It finds identical 1,162-step executed schedules, decisions, engine-logical records, and 32 outputs at `most_same_path_after_finite.json:99-177`; the aligned calls are `0.032029/0.032457 s` at `:41-97`. The diagnostic code itself sets the ceiling to observed host execution at `20260913_rotation_strong_baseline_r01/diagnostics/most_output_block_difference.py:59-97`. The final addendum preserves that ceiling at `RESULTS_AFTER_FINITE_ADDENDUM.md:14-16`.

## Required Early Checks

| Check | Status | Evidence |
|---|---|---|
| Future information | PASS | The selector declares and uses present state only (`preparation/source/absence_rotation.py:133-205`); each arm evolves through its own fresh engine (`preparation/source/run_campaign.py:15-56`). |
| Identity/time alignment | PASS | Frozen request, document, prompt-token and arrival checks occur at `analyze_rotation_strong_baseline.py:47-79`; host-clock and one-token-chunk semantics are at `preparation/source/native_capture.py:190-204`; same-path normalization is at `20260913_rotation_strong_baseline_r01/diagnostics/most_output_block_difference.py:77-97`. |
| Timing/accounting | PASS | Nested call/scheduler intervals and exact wall closure are enforced at `analyze_headroom_cost.py:11-42`; successful work is partitioned without overlap at `analyze_headroom_work_cost.py:17-44`; recovery alignment and cost buckets are checked at `analyze_rotation_recovery.py:17-95`. |
| Fair baseline | PASS | Native and most_output share workload, engine arguments, cap and runtime; only completion policy and verified victim order are excluded before matching (`rotation_runtime_repeat.py:100-116`; `analyze_completion_headroom.py:168-190`). A/B/B/A order and same-role drift are retained, and native is never replaced or corrected (`preparation/source/DECISIONS.md:7-15`). |

## Claim Impact

| Claim | Audit impact |
|---|---|
| Four cells completed; 128/128 requests; displayed metrics and deltas match raw | Supported |
| most_output reduced the observed maximum ITL versus native in both blocks | Supported as a descriptive result in this campaign |
| Native-relative throughput repeated in one direction | Refuted by the retained sign flip |
| The old 0.766318 s step-839 call appeared again | Refuted for these two repeats; the aligned calls were about 0.032 s |
| The old stall disappeared for a known reason, or was caused by GPU/JIT/compiler/GC | Unverified and explicitly not claimed |
| Equal generated tokens prove task quality | Unsupported and explicitly not claimed |
| APC=True produced a baseline result | `UNRUN`; no result exists |
| Stable net benefit, full Oracle, production SLO benefit, or method GO | Unsupported and explicitly outside the claim ceiling |

## Action Items

None for integrity. New evidence, including any APC comparison, must remain a separate experiment and cannot be supplied by extending this audit.

**Direct answer:** this bundle passes the targeted integrity review with P0=0 and P1=0. It supports the exact bounded `MEASUREMENT_ONLY` observations, preserves same-role timing variation and the initialization failure, and does not support a stable throughput benefit or a GPU/JIT causal explanation.
