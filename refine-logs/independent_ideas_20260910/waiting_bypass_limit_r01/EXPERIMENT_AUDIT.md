# Waiting-bypass pilot: fresh experiment-integrity audit

**Date:** 2026-09-11  
**Reviewer:** fresh Codex reviewer, same-family, provisional  
**Overall verdict:** **WARN**  
**Integrity status:** **PASS** for the retained runtime evidence and recomputed accounting; **WARN** for claim scope, stale top-level metadata, and any positive/stable performance interpretation.

## Verdict and claim impact

The experiment is a genuine single-GPU vLLM runtime intervention, not a simulation or model-output-derived task evaluation. The bounded policy was called before the native scheduler, changed the mixed-workload first-admission order, reduced actual bypasses from SPT's total/max `6/3` to `3/1`, and preserved independently evolving policy state. All raw timelines and metrics reconcile.

The preregistered positive condition did not pass. `bounded_bypass_once` versus FCFS short-request mean TTFT was **+1.3206%** in forward and **-7.1284%** in reverse, while `bounded_bypass_once` versus SPT made short mean TTFT **+9.6762%** and **+7.7960%** worse. It reduced SPT's long-request completion p95 by **1.7708%** and **2.7790%** and maximum completion by **2.0914%** and **3.1020%**, but it remained worse than FCFS on long p95 in both blocks. Overall request-mean effect versus FCFS changed sign: **+2.4082%** forward and **-0.4602%** reverse. These results fail the frozen requirement that both blocks retain a short-TTFT reduction versus FCFS (`DECISIONS.md:42-46`).

Permissible claim ceiling:

> In one OLMoE/BF16/vLLM-0.26.0/RTX-5090 configuration with one fixed 16-request arrival trace and two reverse-order engine blocks, the one-bypass rule executed as intended and deterministically bounded actual first-admission bypass counts. It descriptively attenuated SPT's long-request tail penalty while giving back short-request TTFT benefit. The pilot did not establish a stable net latency, throughput, SLO, or quality benefit and does not authorize a method GO.

## Checks

### A. Ground-truth and reference provenance — PASS

- This is system-runtime outcome measurement with no task-quality ground truth by design. Host receipt timestamps, scheduler calls, native request state, token IDs, and completions are retained (`runtime/native_capture.py:27-61`, `runtime/native_capture.py:78-125`, `runtime/native_capture.py:148-188`, `runtime/native_capture.py:237-253`). Missing task-quality labels do not make the GPU timings simulated.
- The frozen model/revision, request counts, output length, seed, cap, token budget, and two orderings are recorded before execution (`DECISIONS.md:22-32`; `readback/results/forward/config.json:35-105`; `readback/results/reverse/config.json:35-105`). All ten frozen source/input hashes recomputed correctly.
- Both environments identify vLLM 0.26.0, PyTorch 2.11.0+cu130, CUDA 13.0, one RTX 5090, and no other process before engine initialization (`readback/results/forward/environment.json:2-30`; `readback/results/reverse/environment.json:2-30`). Runtime logs show actual model loading, CUDA/FlashAttention/Triton MoE setup, GPU KV-cache creation, and formal cell execution (`readback/results/forward/stdout.log:1-43`, `readback/results/forward/stdout.log:59-72`; `readback/results/reverse/stdout.log:1-42`, `readback/results/reverse/stdout.log:58-71`).
- Limitation: the 13.8 GB model files are not in this local bundle, so their bytes could not be rehashed by this reviewer. The retained verification record reports all three shards and six support files matching (`model-cache-verification.json:9-70`), and the execution log resolves the pinned revision path.

### B. Metrics, denominators, and accounting — PASS

- No metric is normalized by prediction/output statistics. Relative values use ordinary `(intervention / baseline - 1)` deltas, with raw seconds and per-request arrays retained (`analyze_results.py:146-158`; `analysis/TABLE.md:20-46`).
- TTFT is first host-received token minus planned arrival; TPOT is first-to-last host receipt divided by 127; request latency is completion minus arrival. Failed and unfinished arrivals remain in denominators, while completion/SLO require completed status (`runtime/metrics.py:62-155`).
- The independently recomputed four-bucket identity held for all 384 request executions: arrival-to-submission + submission-to-first-schedule + first-schedule-to-first-token + first-token-to-completion = completion-minus-arrival. The derived artifact states the same boundary and avoids treating phases as pure GPU time (`analysis/REQUEST_ACCOUNTING.md:3-15`, `analysis/REQUEST_ACCOUNTING.md:308-314`).
- All 24 episodes had one-token host chunks, so token-level ITL is resolved. All 12 formal cells completed 16/16 requests; formal SLO pass was 16/16 under the stated loose reference thresholds (`analysis/TABLE.md:20-46`).

### C. Physical result existence and count consistency — PASS

- Recomputed archives match the declared SHA-256, byte count, and 57-member count: forward `a3e23633...45e7` / 2,384,801 bytes and reverse `397257f4...96eb` / 2,403,109 bytes (`ARCHIVE-forward.json:4-6`; `ARCHIVE-reverse.json:4-6`). Both tarballs pass gzip validation, have safe regular-file paths, and byte-match all 107 extracted readback paths.
- Each block contains exactly six formal and six warmup `raw/metrics/checks` triples plus the eight required support files. All 126 JSON files parse. Block statuses record 6+6 complete (`readback/results/forward/status.json:2-5`; `readback/results/reverse/status.json:2-5`).
- Independent recomputation checked 24 raw episodes, 384 request executions, 49,152 output tokens, 7,027 scheduler steps/actions, 72 metric groups, every output-event stream, and the 96-request mixed accounting derivative. It matched `analysis/summary.json` and `analysis/TABLE.md`; no favorable cell was selected or omitted.

### D. Called code and action reachability — PASS

- The engine monkey-patches the live scheduler, applies queue ordering immediately before the native `schedule()`, then records actual positive first admissions from the native result (`runtime/native_capture.py:78-103`). Runtime source hashes are checked against the reviewed native scheduler reference before engine creation (`run_waiting_order.py:99-113`).
- The bounded ordering performs a virtual same-step budget update (`runtime/waiting_order.py:16-29`), while the actual ledger updates only after native positive scheduled tokens and rejects any count above one (`runtime/waiting_order.py:31-53`). Independent reconstruction from raw native scheduled-token insertion order matched every recorded same-step charge.
- Both mixed blocks produced the same first-admission orders: FCFS original order; SPT moves requests 11/13/15 ahead of 10/12/14; bounded alternates 11/10/13/12/15/14. Raw accounting records these orders and counts (`analysis/REQUEST_ACCOUNTING.md:11-15`; `analysis/TABLE.md:63-90`).
- Every arm independently runs its full request/KV/batch/generation timeline to drain; prefix caching is disabled, and the engine must be empty before the next episode (`run_waiting_order.py:37-50`, `run_waiting_order.py:118-143`). All arms carry the same observation hooks; policy-specific sorting/ledger cost remains inside measured request and wall time.
- `runtime/admission_feedback.py` and feedback branches are not reached because this campaign freezes `policy="static"` (`run_waiting_order.py:123-129`; `runtime/native_capture.py:40-42`, `runtime/native_capture.py:68-76`). No result claim relies on that helper.

### E. Scope, repeats, environment, and baseline fairness — WARN

- Fair baselines are FCFS and SPT, plus an all-short no-reorder negative control. Formal order is reversed across two fresh engines, and warmups are excluded as preregistered (`DECISIONS.md:15-32`; `analysis/TABLE.md:48-105`).
- Scope is only one model/revision, one GPU/host, one fixed reused 16-request identity set, one 50 ms arrival regime, two engine blocks, and eight short/eight long requests per mixed cell. Percentiles are therefore small-sample descriptors. There is no Oracle, significance analysis, new workload/seed, HTTP serving, multi-GPU EP, or task-quality evaluation (`DECISIONS.md:45-50`; `readback/results/forward/config.json:67-105`).
- Exact output sequences match only 4-10 of 16 requests across formal arm pairs (`analysis/TABLE.md:149-182`). This is consistent with policy-specific state evolution and `VLLM_BATCH_INVARIANT=0`, but it forbids a quality-preservation claim and prevents attributing every latency delta solely to queue waiting.
- The separately supplied prefill-sharing diagnostic is correctly limited to `STRUCTURAL_OBSERVED_STATE_NO_COUNTERFACTUAL`; it observes three states in each mixed FCFS block but supplies no alternative-policy headroom or latency result (`analysis/prefill_sharing_opportunities.json:2-11`, `analysis/prefill_sharing_opportunities.json:21-89`; `inspect_prefill_sharing.py:50-59`).

### F. Evaluation classification — PASS

- Primary evaluation: `real_runtime_measurement_no_task_quality_gt`, evidence tier `NATIVE_SERVING_INPROCESS_HOST_CAPTURE`, claim tier `REQUEST_LEVEL_EXPLORATORY_INTERVENTION_NO_METHOD_GO`. It is not `simulation_only`, `synthetic_proxy`, or `human_eval`.
- Auxiliary diagnostic: `STRUCTURAL_OBSERVED_STATE_NO_COUNTERFACTUAL`.

## Material findings by priority

1. **P1 / claim-blocking:** Action validity passes, but the preregistered two-block performance condition fails. Stable fairness/latency tradeoff, net benefit, and method-GO claims are unsupported. Descriptive admission-order and bypass-limit claims remain supported.
2. **P2 / provenance presentation:** `README.md:3-10` and `README.md:57-64` still describe the pre-run/download state. `STATUS.json:3` carries an observation timestamp before reverse began, while `STATUS.json:16-17` and `STATUS.json:76-80` retain stale partial-download/runtime-pending fields beside completed fields at `STATUS.json:45-70` and `STATUS.json:89-117`. Treat README as a preparation snapshot and raw/status/process/archive files as execution authority.
3. **P2 / runtime warning:** Both stderr logs retain `SM 12.x requires CUDA >= 12.9` messages (`readback/results/forward/stderr.log:1-2`; `readback/results/reverse/stderr.log:1-2`). CUDA 13.0 is recorded, both processes exit 0, and every cell completes, so this is not evidence of a failed or simulated run.

No P0 integrity issue was found. There is no fake ground truth, self-normalized score, phantom result, future-information policy input, identity drift, hidden failure, archive mismatch, or post-hoc favorable-run selection in the checked scope.

## Research disposition

- **Strongest baselines:** SPT for short TTFT; FCFS for avoiding the long-tail cost. The bounded rule lies between them and has no stable residual net benefit.
- **Oracle/headroom:** UNRUN. The three executable policies are not an Oracle.
- **Failure category:** action semantics alive; stable positive performance formulation not established under the tested regime.
- **Resurrection condition:** only a preregistered new natural request/arrival regime showing a materially different fairness problem, as frozen in `DECISIONS.md:48-50`; threshold/seed scanning does not qualify.
- **Next smallest step:** write the final scientific verdict from these retained results and stop mechanism expansion for the current formulation. No additional GPU experiment is justified by this audit.

## Audit boundary

No GPU experiment was run. No raw or repository file was modified. The audit did not independently rehash absent model-weight bytes, assess task quality, establish statistical generalization, review external prior art, or perform a full vLLM correctness/security audit. No `.aris` trace was created, per the explicit repository/user instruction.

Executor metadata resolution: after retaining this audit, mutable README and STATUS were updated to the completed measurement and failed benefit condition. No raw data, original derived metrics, or reviewer finding was changed.
