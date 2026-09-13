# Experiment Audit Report

**Date:** 2026-09-14 (UTC+08:00)  
**Auditor:** GPT-5.6-Sol ultra, fresh same-family read-only reviewer  
**Review status:** provisional  
**Bundle:** `20260914_ltr_component_probe_r01`  
**Repository HEAD inspected:** `c4ae4f5daaea929e1a4358862296d832f1cc67ab`

## Overall Verdict: PASS

## Integrity Status: pass (same-family provisional)

**P0: 0. P1: 0.** No material integrity defect was found within the stated `MEASUREMENT_ONLY` claim ceiling.

This audit inspected the frozen source/package, execution receipt, all four retained raw attempts, all four decision streams, saved metrics, and the two analysis programs. It performed no GPU, network, or production action and did not use any later gap-localization analysis.

## Checks

### A. Ground Truth Provenance / Future Leakage: PASS

There is no model-quality ground truth in this experiment. The Wikitext material supplies frozen prompt tokens and identities only (`preparation/pkg/inputs_preparation/prepared/long/config.json:17-38`); generated token IDs are retained as model outcomes. Cross-arm token equality is reported only as a diagnostic, while quality is explicitly unmeasured (`RESULTS_ADDENDUM.md:18`, `RESULTS_ADDENDUM.md:24-30`). No model output is relabeled as dataset ground truth.

The action path uses live request state, current owned/history blocks, arrival order, and counters accumulated through prior scheduling decisions (`preparation/pkg/recovery_service_components.py:47-74`, `preparation/pkg/ltr_recompute_native.py:147-203`). Future token receipt times enter only the post-run epoch evaluation (`../../../experiments/admission_capacity/analyze_ltr_component_probe.py:36-83`). Independent replay of all 7,452 decisions across the four attempts reproduced every stored counter state and priority from earlier live/selected history. For every step, the observed pre-action output count equaled the number of token receipts available by that step's start.

### B. Denominators, Nested Timing, and Score Normalization: PASS

Request latency, TTFT, TPOT, ITL, completion throughput, and their populations are computed directly from the common host clock and all arrived requests (`preparation/pkg/metrics.py:10-29`, `preparation/pkg/metrics.py:62-155`). The four completed episodes therefore use 32/32 requests and 32,768/32,768 returned tokens per cell. Relative changes divide an action value by its paired baseline value; no score is divided by the model's own maximum, minimum, or mean (`../../../experiments/admission_capacity/analyze_prefix_cache_baseline.py:206-220`). Raw values remain present.

The component analysis enforces `decision <= wrapped scheduler <= captured scheduler <= episode wall` and does not add nested timers (`../../../experiments/admission_capacity/analyze_ltr_component_probe.py:151-157`). Independent sums reproduced the saved decision, wrapped-scheduler, and captured-scheduler totals exactly. Recovery spans were not summed into wall time.

### C. Result Existence, Numbers, and All-Run Retention: PASS

The execution receipt records all four cells as `COMPLETE`, 32 requests each, with 20/22/22/20 native-recompute preemptions (`execution/execution.json:34-86`; `execution/readback/group-status.json:6-53`). The launch contract refuses an existing result directory and runs exactly `off/on/on/off` (`preparation/pkg/run.sh:7-23`); the wrapper also uses a create-once launch marker (`execute.py:20-38`). All four raw files, decision files, metrics, statuses, logs, warmups, and environment records are retained.

The recorded readback archive SHA-256 is correct, and a one-pass comparison found all 97 regular archive members (1,266,423,099 uncompressed bytes) byte-identical to `execution/readback/`. Both the preparation and readback package copies pass all 16 entries in `SHA256SUMS`. Actual hashes for every raw/decision file match those embedded in `analysis/analysis.json`.

Independent recomputation from the four raw files produced:

| Cell | Wall s | Throughput req/s | Mean completion s | Max ITL s | Preemptions | Repeated positions |
|---|---:|---:|---:|---:|---:|---:|
| block0 off | 27.544777239 | 1.161744737 | 21.458609510 | 4.691627061 | 20 | 69,176 |
| block0 on | 28.160974976 | 1.136324294 | 22.067661471 | 4.928369878 | 22 | 75,742 |
| block1 on | 27.668793637 | 1.156537593 | 21.601234088 | 4.867611827 | 22 | 75,742 |
| block1 off | 28.558399811 | 1.120510960 | 22.332265476 | 4.904622465 | 20 | 69,176 |

These values match `RESULTS_ADDENDUM.md:9-22` and `analysis/analysis.json:846`, `analysis/analysis.json:1790`, `analysis/analysis.json:2734`, and `analysis/analysis.json:3621`.

### D. Metrics and Checks Actually Executed: PASS

The measured pipeline calls `summarize_episode_requests` and writes its output (`preparation/pkg/run_probe.py:185-194`). The analysis independently invokes the frozen metrics implementation, validates saved metrics against raw requests, checks every scheduler/output/memory edge, executes component accounting, and builds quantum epochs (`../../../experiments/admission_capacity/analyze_ltr_component_probe.py:193-225`). `analysis/analysis.json:2` records `MEASUREMENT_ONLY`, and all four cells are eligible. Every metric function relevant to the reported claims is reachable and executed; copied headroom/rotation modules are inactive compatibility baggage and are not claimed as measured mechanisms.

### E. Scope and Baseline Fairness: PASS

All four attempts use identical prompt identities, arrivals, model/revision, BF16 precision, seed, backend, scheduler settings, APC-off cache, 13,960,740,864-byte KV allocation, and 6,656 usable blocks. The only intended policy difference is whether the frozen 200/10 waiting boost participates in ordering; the completion-policy label changes with it (`preparation/pkg/run_probe.py:68-90`, `preparation/pkg/run_probe.py:118-167`). The analyzer verifies the common engine, runtime hashes, input identities, KV qualification, and source inventory before pairing (`../../../experiments/admission_capacity/analyze_ltr_component_probe.py:160-211`, `../../../experiments/admission_capacity/analyze_ltr_component_probe.py:245-274`).

Independent step-by-step checks found, for every action: planned tokens equal native scheduled tokens; planned victims equal returned native preemptions; released blocks equal actual free-block deltas; all selected-history reservations balance; and every held resident's request/KV state is unchanged at the scheduling boundary. The on paths are identical to one another, as are the off paths.

The baseline supports only the claimed component ablation: boost-off is the same custom packing/reservation/recompute backend without priority boost, not native vLLM. The report states this limitation and does not claim superiority over native vLLM, full LTR, or another system (`RESULTS_ADDENDUM.md:24-30`). Two same-input repeats and opposite signs support the descriptive statement that no consistent full-request improvement was observed; they do not establish a variance bound, statistical effect, or general no-go, and none is claimed.

### F. Evaluation Type and Evidence Tier: PASS

**Integrity taxonomy:** `self_supervised_proxy` only in the narrow sense that no quality GT exists by design.  
**System evidence:** `NATIVE_SERVING` / `NATIVE_SERVING_INPROCESS_HOST_CAPTURE` (`preparation/pkg/native_capture.py:191-203`; `RESULTS_ADDENDUM.md:5-7`).  
**Quality evidence:** none.

This is real in-process vLLM execution on one RTX 5090, not a simulation. It remains one model, one deterministic fixed-length cohort, one arrival regime, and two same-input repeats. It does not support production, business-SLO, semantic-quality, full-LTR, Oracle, or method-GO claims.

## Independent Action and Output Findings

- Each boost-on attempt has one effective epoch for request `memory-train-article-0003640`, steps 609-618: 10 consecutive scheduled calls, 3 recompute-only calls, 1 mixed recompute/new-output call, and 6 output-only calls. Seven new tokens return in the epoch; the first arrives 115.693 ms and 108.400 ms after the respective pre-action epoch starts. This verifies actual service, not merely a changed counter.
- Boost-off has the same diagnostic counter epoch but its 10 selected calls are dispersed over steps 647-927 and return three new tokens; it is correctly excluded from claims of applied protection.
- Each cell returns exactly 32,768 tokens. Outputs are 32/32 identical within each policy repeat and 29/32 identical across on/off. The three cross-policy differences begin at output positions 639, 919, and 725 for requests `0000628`, `0002733`, and `0003259`, respectively (`analysis/analysis.json:3727-3888`). No semantic-quality inference is justified or made.
- All four logs report a fused-MoE JIT compilation during inference (`execution/readback/results/*log:49`). That event belongs to the measured engine wall and makes a warmed steady-state interpretation unsafe. The addendum reports raw full-wall results, retains the sign flip, and makes no steady-state or significance claim, so this is a bounded runtime limitation rather than an integrity finding.

## Claim Impact

- **The waiting boost caused a real scheduling change and returned new outputs:** supported for this four-cell component experiment.
- **Throughput, mean completion, and max ITL did not improve consistently across the two blocks:** supported as a descriptive observation only.
- **Current 200/10 component improves or harms full-request performance generally:** unsupported and not claimed.
- **Full LTR, native vLLM superiority, business SLO, production stability, or semantic quality:** unmeasured and explicitly outside scope.

## Action Items

No correction is required for the present `MEASUREMENT_ONLY` addendum. Any later stable-performance claim must eliminate or explicitly model the in-episode JIT/runtime variance and add independent workloads/repeats; any quality claim must add an external quality criterion.
