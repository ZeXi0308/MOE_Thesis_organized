# Experiment Audit Report

**Date:** 2026-09-14
**Auditor:** GPT-5.6-Sol via Codex, fresh same-family reviewer, read-only review, provisional
**Project:** six-cell funding-filter GPU comparison
**Repository snapshot:** `43234de708255cbd2b9da01ad81e2997d06f8f42`
**Reviewed result snapshot:** scoped `REPORT.md:32-66` SHA256 `78f05ad7bef8a394a9fe822c8aca27e943b7fffe873b7dea2ae0e512025c19c0`; full `REPORT.md` snapshot SHA256 `1b8e597bd5118fd5973f59327945f67f68bcb86ef7dffb5b1b1db2f5803fb71d`; `analysis_r02.json` SHA256 `46e09b443b67f86612befdccf57404da4af3ffc14103c824010efc86641e5d02`

## Overall Verdict: WARN

## Integrity Status: warn

The six measured GPU cells, request metrics, action records, KV accounting, and reported within-group comparisons are supported by the retained raw files. I found no P0 or P1 integrity defect. The WARN is a claim-scope limit: this is one reused 32-document cohort, one model/device/runtime, one execution per order position, and no calibrated noise bound. Same-role throughput drift reaches 2.381%, while the filtered-versus-least throughput effect changes sign across blocks (-3.011% and +1.272%). The report's `MEASUREMENT_ONLY` conclusion is therefore supported; a repeatable benefit, task-quality result, general controller result, or production claim is not.

Review scope is limited to the six actual GPU cells, their packaged runtime/analyzers, `analysis_r02.json`, and the measured addendum at `REPORT.md:32-66`. The structural model check and later lifecycle/action-branch sections at `REPORT.md:69-97`, including `model_check_r01.json`, are outside this review and inherit no acceptance from it. The historical preparation text and `analysis_r01.json` remain preparation history, not measured evidence (`REPORT.md:3-17,20-29`; `analysis_r01.json:2-3`). No GPU was used during this audit and no original artifact was modified.

## Checks

### A. Ground Truth Provenance: PASS

The performance targets are directly observed request completion and host-receipt timing, plus scheduler/KV/action records. The 32 prompts come from a pinned WikiText shard with dataset, revision, Arrow hash, tokenizer hashes, and a pre-run reuse/selection rule (`preparation/pkg/inputs_preparation/prepared/heterogeneous/config.json:17-40`). Capture binds request/document/prompt/arrival identities before execution and records cumulative native outputs and completion (`execution/readback/pkg/native_capture.py:24-44,104-177`). The raw capture labels itself `NATIVE_SERVING_INPROCESS_HOST_CAPTURE` and records the limits of host chunk timing (`execution/readback/pkg/native_capture.py:191-204`).

There is no task-answer ground truth. Cross-policy token equality is only an execution-identity diagnostic; the report explicitly refuses to treat it as task quality (`REPORT.md:64`). This is correctly labeled rather than presented as semantic accuracy.

### B. Score Normalization and Full Accounting: PASS

No metric is normalized by a maximum, minimum, or mean of the policy's own outputs. The metric code retains every arrived request, validates timestamp/token alignment, and computes TTFT, TPOT, ITL, throughput, and goodput from the actual observation interval (`execution/readback/pkg/metrics.py:10-29,62-155`). I independently recomputed all six cells from raw request timestamps and output IDs: every saved request metric, population count, percentile, wall duration, and report-table value matched.

The denominator is 32 completed requests per cell, 192 request instances total, with exactly 1,024 returned tokens per request (196,608 token IDs); these are repeated request instances, not 192 independent documents (`REPORT.md:34,64`). The work decomposition also reconciles in every cell: fresh prefill 90,112 positions plus fresh decode 32,736 positions plus recompute equals scheduled work. Decision time is nested within scheduler time, and `scheduler inclusive + engine remainder + outside engine = wall`; it is not added twice (`REPORT.md:62`). The six analysis summaries retain the same work/timing invariants (`analysis_r02.json:33514-34591,68067-69092,102568-103648,137124-138204,171680-172705,206181-207258`).

### C. Result and Number Existence: PASS

All six predeclared arms and their order exist in the frozen campaign (`preparation/pkg/campaign.json:2-45`), including the strongest simple `most_output` baseline and the retain-all rule (`preparation/pkg/campaign.json:62-65`). All six terminal receipts are `COMPLETE`, each with 32 completed requests and native-recompute preemptions; the group returned exit 0 (`execution/execution.json:17-91`). The readback archive hash/size and replacement-GPU origin are recorded (`execution/execution.json:93-102`), and the campaign log retains the six launches in frozen order and group completion (`execution/readback/campaign.log:1-32`).

I independently matched every value in `REPORT.md:40-58` to the corresponding raw and metrics files. The nine predeclared pair results also match `analysis_r02.json`, including all output-equality counts (`analysis_r02.json:207301-209404`). No unfavorable cell was absent or replaced. `analysis_r01.json` still says `UNRUN` and is correctly treated as historical (`analysis_r01.json:2-3`; `REPORT.md:34`).

One nonblocking metadata ambiguity remains: each cell's `safe-cap-qualification.json` says `measurement_status: NOT_YET_RUN` even though the adjacent `status.json` is `COMPLETE` (`execution/readback/results/funding-block0-least_feasible/safe-cap-qualification.json:2-4`; `execution/readback/results/funding-block0-least_feasible/status.json:2-7`). This field describes the pre-measurement qualification snapshot and is not consumed as terminal cell status, but a future reader could misread it.

### D. Dead Code and Actual Action Path: PASS

The critical metric and telemetry code is executed by the probe, not merely defined: the probe installs native capture/memory/rotation instrumentation for the measured episode and writes raw, metrics, and terminal status (`execution/readback/pkg/run_probe.py:183-236`). The memory hook wraps the actual vLLM allocator and `_preempt_request`, records original-call entry/return and before/after pool/request state, and reports the native preemption count (`execution/readback/pkg/memory_telemetry.py:41-107,109-153`). The rotation wrapper makes its decision from the live running/waiting set, current free blocks, and current owned blocks, then invokes the native schedule path and checks reservation and held-KV invariants (`execution/readback/pkg/rotation_native.py:159-220`).

I replayed every recorded proposal from its own cell's retained before-state, maintaining a separate tracker per policy. All 1,054 `most_output` proposals per cell and all 1,282 `least_progress`/`least_feasible` proposals per cell matched; the funding arm used only current owned-block counts and reused no future state. This agrees with the analyzer's present-state replay contract (`refine-logs/expert_saturation/experiments/admission_capacity/analyze_funding_filter_comparison.py:122-164`) and its retained replay receipts (`analysis_r02.json:34570-34585,69071-69087,103624-103643,138180-138199,172684-172700,207237-207253`).

Every recorded preemption called and returned through the original native path. Each victim's computed state reset to zero, its per-group block count became `[0]`, the preemption counter incremented, and the freed-block delta matched the released allocation. At every scheduler boundary, `free + used = 6656`, and request-owned block totals matched pool use. Held requests preserved computed progress and block IDs. Forced/natural counts were respectively 22/4, 20/4, 21/5, 21/5, 20/4, and 22/4 in frozen cell order. No shared future KV, route, request set, or output trace was used to score another policy.

### E. Scope, Independence, and Noise Floor: WARN

The experiment deliberately reuses one 32-document cohort, alternates 2,560/3,072-token prompts, and declares that it is not a fresh holdout or matched dynamic-pressure workload (`preparation/pkg/campaign.json:46-60`; `preparation/pkg/inputs_preparation/prepared/heterogeneous/config.json:32-40`). The two blocks reverse arm order but reuse the same documents and arrivals. They are order checks, not independent workload repeats. The report states this and gives same-role throughput drift of +1.274%, -2.381%, and +1.930% (`REPORT.md:64`). That drift overlaps or exceeds the favorable +1.272% filtered-versus-least block, so no stable effect size or significance claim is available.

Within this six-cell group, the runtime, source hashes, RTX 5090 UUID, and software versions are consistent (`execution/readback/results/funding-block0-least_feasible/environment.json:2-32`). The launcher checks the shared lock, refuses result replacement, records `nvidia-smi`, and aborts if another compute PID exists (`execution/readback/pkg/run.sh:6-25`). The old staging GPU and replacement execution GPU are distinct and explicitly separated (`execution/execution.json:95-101`; `REPORT.md:36`); I found no cross-device timing pair in the measured tables.

The six cells ran sequentially in about seven minutes. Each application warmup completed before measurement, but the logs record warmup-time JIT compilation and use of a default non-device-specific MoE configuration (`execution/readback/results/funding-block0-least_feasible.log:35,46-54`). There is no counterbalanced multi-run estimate, continuous clock/power trace, or thermal noise calibration. These limits do not invalidate the retained descriptive measurements; they cap the conclusion at this execution group.

### F. Evaluation Type and Claim Ceiling: WARN

Classification: **direct request-level native in-process systems measurement without task ground truth**, with **self-supervised output-equality proxy** used only for execution identity. Evidence tier: `REQUEST_LEVEL / NATIVE_SERVING_INPROCESS_HOST_CAPTURE` (`execution/readback/pkg/native_capture.py:191-204`). It is not simulation-only, human evaluation, dataset-answer `real_gt`, HTTP/network serving, multi-GPU EP, or production traffic.

The allowed claim ceiling is: on this replacement RTX 5090, pinned vLLM/runtime, 6,656-block KV pool, reused 32-document episode, and frozen six-cell order, the funding filter changed actual victims/work and did not establish repeatable full-request improvement over least-progress. The strongest simple `most_output` arm remained faster in throughput in both blocks (`REPORT.md:53-60`). Task quality, statistical significance, a fresh-workload effect, cross-device timing, capacity generalization, and an independent method contribution remain unmeasured.

## Independent Recalculation

| Cell | Wall s | req/s | Mean completion s | Max request ITL s | Recomputed positions | Forced / natural preemptions |
|---|---:|---:|---:|---:|---:|---:|
| block0-most_output | 23.871104 | 1.340533 | 21.981054 | 2.112613 | 93,044 | 22 / 4 |
| block0-least_progress | 24.553422 | 1.303281 | 21.046706 | 1.931141 | 83,510 | 20 / 4 |
| block0-least_feasible | 25.315591 | 1.264043 | 21.706353 | 2.039727 | 89,761 | 21 / 5 |
| block1-least_feasible | 24.836317 | 1.288436 | 21.311881 | 1.990427 | 89,761 | 21 / 5 |
| block1-least_progress | 25.152222 | 1.272253 | 21.554351 | 2.008486 | 83,510 | 20 / 4 |
| block1-most_output | 23.570727 | 1.357616 | 21.735523 | 2.152757 | 93,044 | 22 / 4 |

The recomputed within-block deltas and request/output comparisons match `REPORT.md:53-58` exactly. Same-role block0-to-block1 recomputation also matches `analysis_r02.json:208705-209404`.

## Severity and Claim Impact

- **P0 findings:** none.
- **P1 findings:** none.
- **Nonblocking note:** stale `measurement_status: NOT_YET_RUN` inside the pre-measurement safe-cap qualification snapshot; terminal status and analysis use the correct completed artifacts.
- **Supported:** six-cell execution existence; 192 completed request instances; request/wall/work/action metrics; actual native preemption and KV-resource conservation; the bounded finding that the funding filter did not show a repeatable full-request improvement in this group.
- **Needs qualifier:** any description of the funding filter's performance must retain the single-cohort, same-device, reversed-order, uncalibrated-noise boundary and the strongest-simple-baseline comparison.
- **Unsupported:** task-quality improvement, statistical significance, fresh-workload generalization, HTTP/production SLO, multi-GPU behavior, cross-device timing comparison, or independent method contribution.
- **Outside this audit:** structural model validation, lifecycle localization, and post-hoc action-branch diagnostics in `REPORT.md:69-97` and their associated files.

## Action Items

No correction is required for the six-cell measured addendum. Keep its status `MEASUREMENT_ONLY` and keep the post-addendum model/lifecycle/branch material outside this audit's acceptance.

The one smallest discriminating follow-up is the same frozen six-cell protocol on one fresh, document-disjoint 32-request cohort, on the same device/runtime/pool with no threshold changes and a predeclared counterbalanced order. Continue the funding-filter performance hypothesis only if its full-request effect has the same sign beyond the observed same-role drift and remains competitive with `most_output`; otherwise classify this tested formulation as `MEASUREMENT_ONLY / NO_REPEATABLE_FULL_REQUEST_IMPROVEMENT_IN_TESTED_REGIME` rather than a family-wide hard no-go.

**Review independence:** `same-family`
**Acceptance status:** `provisional`
