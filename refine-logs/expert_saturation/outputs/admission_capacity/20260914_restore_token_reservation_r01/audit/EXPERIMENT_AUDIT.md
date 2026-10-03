# Experiment Audit Report

**Date**: 2026-09-14  
**Auditor**: Codex GPT-6, fresh same-family agent, read-only review  
**Project**: `20260914_restore_token_reservation_r01`  
**Review independence**: `same-family`  
**Acceptance status**: `provisional`

Evidence-path convention: paths beginning `experiments/` or `outputs/` are relative to `refine-logs/expert_saturation/`; paths beginning `analysis/`, `RESULTS_ADDENDUM.md`, or `analyze_frozen_cost.py` are relative to the audited target directory.

## Overall Verdict: WARN

## Integrity Status: pass

No P0 or P1 integrity defect was found. The frozen package, runtime receipts, six retained cells, request and clock accounting, executed planner path, obligation lifecycle, result keys, and reported numbers are internally consistent. The evidence supports the report's narrow `NATIVE_SERVING / MEASUREMENT_ONLY / OPEN` conclusion.

The WARN records three P2 claim boundaries: the two order reversals reuse one 32-request cohort, output quality was not evaluated and outputs differ for some requests, and the step-299 cost transfer plus pre-divergence timing residuals are conditional diagnostics rather than an action counterfactual. The result report already states all three boundaries, so these are limits on reuse of the result rather than evidence of fabrication or suppressed outcomes.

| Severity | Count | Result |
|---|---:|---|
| P0 | 0 | No fabricated ground truth, phantom result, corrupt artifact, or invalid execution receipt found. |
| P1 | 0 | No material accounting, action-replay, identity, lifecycle, or claim mismatch found. |
| P2 | 3 | Scope, quality, and causal-transfer boundaries must remain attached to the claims. |

## Checks

### A. Ground Truth Provenance: PASS

This is a systems-performance measurement, not a task-accuracy benchmark. The primary observations come from six actual native-serving request trajectories. The analyzer requires a complete raw run, the pinned engine/runtime, exact request/document/prompt identities, 1,024 returned tokens per request, monotonic request clocks, and raw-to-saved metric equality (`experiments/admission_capacity/analyze_restore_completion_probe.py:141-177`). Output receipts are reconstructed from cumulative engine returns and checked against the stored per-request output tokens and times (`experiments/admission_capacity/analyze_prefix_cache_baseline.py:43-61`).

No external quality ground truth is present or claimed. The addendum says that output equality does not establish quality and lists free-EOS and quality as unmeasured (`outputs/admission_capacity/20260914_restore_token_reservation_r01/RESULTS_ADDENDUM.md:31-37`). The observed equality counts, 27/32 against `guard_all` and 28/32 against `fit_scan`, are disclosed rather than converted into an accuracy score (`outputs/admission_capacity/20260914_restore_token_reservation_r01/RESULTS_ADDENDUM.md:18-25`). There is therefore no fake-GT path.

### B. Score Normalization: PASS

The request denominator is all requests that arrived by the actual observation end, and future/unexecuted requests are rejected (`outputs/admission_capacity/20260914_restore_token_reservation_r01/preparation/pkg/metrics.py:10-29`). TTFT, TPOT, ITL, completion, throughput, and SLO counts are calculated from raw timestamps and completed-request counts; throughput is `completed / observed duration`, with raw population counts retained (`outputs/admission_capacity/20260914_restore_token_reservation_r01/preparation/pkg/metrics.py:62-155`). Matched-pair deltas are direct subtraction and ratios are `action / baseline - 1`, while the raw values and per-request changes remain in the result (`experiments/admission_capacity/analyze_prefix_cache_baseline.py:206-220`).

No metric is divided by the model's own maximum, minimum, or mean. The fixed cost model uses previously frozen coefficients and actual post-cutoff schedules; it is not used to normalize the primary result (`outputs/admission_capacity/20260914_restore_token_reservation_r01/analyze_frozen_cost.py:15-52`).

### C. Result File and Claim Existence: PASS

Preparation records the repository state, 17-file source inventory, package hash, six exact cells, runtime source hashes, absolute reference engine, and preparation-script hash (`outputs/admission_capacity/20260914_restore_token_reservation_r01/preparation/preparation.json:6-30`, `outputs/admission_capacity/20260914_restore_token_reservation_r01/preparation/preparation.json:31-102`). Execution records matching package/runtime receipts, an empty preflight compute-process list, return code zero, readback hash, and all six cells as `COMPLETE`, each with 32 completed requests (`outputs/admission_capacity/20260914_restore_token_reservation_r01/execution/execution.json:2-33`, `outputs/admission_capacity/20260914_restore_token_reservation_r01/execution/execution.json:34-108`). The stale preparation status is explicitly identified as historical and superseded by the six-cell terminal record (`outputs/admission_capacity/20260914_restore_token_reservation_r01/RESULTS_ADDENDUM.md:3-7`).

The analyzer refuses changed metadata or source inventories, checks the parent/helper and reference engine/runtime hashes, validates the 32-request workload, inspects every frozen cell, and emits comparisons only after all cells are eligible (`experiments/admission_capacity/analyze_restore_token_reservation.py:152-204`). The reported cell metrics exist at `analysis/analysis.json:5072-5124`, `analysis/analysis.json:11200-11252`, `analysis/analysis.json:19261-19313`, `analysis/analysis.json:27322-27374`, `analysis/analysis.json:33450-33502`, and `analysis/analysis.json:38572-38624`. The four descriptive comparison records and their actual arm configurations exist at `analysis/analysis.json:38627-38977`, `analysis/analysis.json:38980-39331`, `analysis/analysis.json:39332-39683`, and `analysis/analysis.json:39684-40035`. The shared ledger row reports the same six-cell totals and bounded conclusion (`experiments/admission_capacity/RESULT_LEDGER.md:212`).

Fresh read-only verification reproduced the main `analysis.json` byte-for-byte with SHA-256 `7331f3c9514951ec36a2fd68c193db927fccc0b2214ec04ae2aff4e76c5d8dd4`. Fresh localization and frozen-cost reruns were also byte-identical, with SHA-256 `b8845e3be343e2e990dd3d4eab7fd30426eb64da5c4457782592b9595848320f` and `c8e4ab6ce3497b346b4675b710b2bb812d1b76de2d03ff2f18f051d45dd14f95`. Both 17-file package inventories passed `sha256sum -c`; both gzip/tar archives passed integrity checks.

### D. Executed and Dead Paths: PASS

The frozen runner verifies its package, acquires the shared GPU lock, refuses an existing results directory, checks GPU process occupancy before every cell, and runs the declared `F/G/R/R/G/F` order (`outputs/admission_capacity/20260914_restore_token_reservation_r01/preparation/pkg/run.sh:7-25`). The measured path installs the component on the actual scheduler, captures the episode, finalizes obligations, and writes decisions, obligations, raw data, metrics, and terminal status (`outputs/admission_capacity/20260914_restore_token_reservation_r01/preparation/pkg/run_probe.py:181-219`).

The installed scheduler constructs the planner input from each policy's live state, applies token/victim order to the native scheduler, and fails if native scheduled tokens, victims, or remaining resource reservations differ from the plan (`outputs/admission_capacity/20260914_restore_token_reservation_r01/preparation/pkg/ltr_recompute_native.py:169-229`). The audit analyzer independently replays the exact frozen plan on every recorded pre-state and records the same-pre-state flag-off plan only as a one-step boundary (`experiments/admission_capacity/analyze_restore_token_reservation.py:39-84`). The inherited accounting checks actual actions, preemptions, block conservation, request-state transitions, and nested timing on every scheduler step (`experiments/admission_capacity/analyze_restore_completion_probe.py:60-131`). These functions are called by the six-cell main path (`experiments/admission_capacity/analyze_restore_token_reservation.py:190-201`), and their outputs appear in `analysis.json`; no claimed metric or mechanism depends on a defined-but-unused path.

The focused obligation unit suite passed all six tests in a fresh no-bytecode run. This supplements, but does not replace, the recorded real-run lifecycle receipts.

### E. Scope Assessment: WARN

The scope is six complete cells: three policy arms in forward and reverse order, all using the same 32 old-d6 documents, single RTX 5090, OLMoE BF16, vLLM 0.26.0, one arrival trace, one KV capacity, and length-fixed 3,072/1,024 requests (`outputs/admission_capacity/20260914_restore_token_reservation_r01/RESULTS_ADDENDUM.md:5-7`). Thus 192 request executions represent 32 unique inputs reused across six cells, and the two runs per arm are order reversals on the same cohort rather than independent workload or model replication.

All cells and both orders are retained, including harmed requests and output differences; execution records zero failed cells (`outputs/admission_capacity/20260914_restore_token_reservation_r01/execution/execution.json:34-108`, `outputs/admission_capacity/20260914_restore_token_reservation_r01/RESULTS_ADDENDUM.md:25-31`). The report correctly calls the repeat differences descriptive rather than a noise bound or non-inferiority test (`outputs/admission_capacity/20260914_restore_token_reservation_r01/RESULTS_ADDENDUM.md:31`).

`fit_scan` is the strongest same-group service baseline and `guard_all` is the direct reservation ablation. Native/least/most, a complete LTR ladder, new documents/arrivals, free EOS, multiple models/GPUs, SLOs, and production P99 are not tested in this bundle (`outputs/admission_capacity/20260914_restore_token_reservation_r01/RESULTS_ADDENDUM.md:37-45`). The stated claim ceiling is commensurate with that scope. Any broader method-win, robustness, or production claim would fail this check.

### F. Evaluation Type and Causal Boundary: PASS

The primary evaluation type is `real_runtime_measurement`: independent `NATIVE_SERVING` policy trajectories with complete request outputs and wall-clock/request metrics (`outputs/admission_capacity/20260914_restore_token_reservation_r01/RESULTS_ADDENDUM.md:5-7`). This category is outside the skill's vision-style GT taxonomy; quality remains `unmeasured`. The one-step flag-off planner replay and frozen-cost transfer are `simulation_only`/retrospective diagnostics and are kept separate from realized full-request outcomes (`analysis/analysis.json:41093-41099`, `outputs/admission_capacity/20260914_restore_token_reservation_r01/analyze_frozen_cost.py:55-76`).

Policy-specific state is preserved by six separate executions. The analyzer checks exact request identities and clocks, actual scheduled tokens, victims, block ownership, and resource conservation rather than reusing one policy's future trace (`experiments/admission_capacity/analyze_restore_completion_probe.py:65-128`, `experiments/admission_capacity/analyze_restore_completion_probe.py:166-177`). The flag-off comparison stops at the same recorded pre-state and explicitly supplies no alternate future latency (`experiments/admission_capacity/analyze_restore_token_reservation.py:67-84`).

An obligation starts only for an actually resumed, previously outputting request with more than one pending token. It remains active until an actually observed new output or completion, rejects interruption when enabled, checks retained history blocks, and cannot finalize unresolved (`experiments/admission_capacity/restore_obligation.py:33-56`, `experiments/admission_capacity/restore_obligation.py:64-113`). Independent replay matches each begin/release/start/interruption/reservation event to actual completions and schedules (`experiments/admission_capacity/analyze_restore_completion_probe.py:27-55`).

`pending_tokens == 1` is a visible work-state predicate, not by itself proof of decode output; the preparation contract states that boundary (`outputs/admission_capacity/20260914_restore_token_reservation_r01/REPORT.md:9-13`). The localizer therefore separately counts outputs already returned and new outputs received after a selected call (`analysis/localization/localize.py:22-39`). At the first guard/residual divergence, all 30 eligible ready residents were already outputting and all 30 produced a new output; the action used 994 restore tokens plus those 30 realized one-token services (`analysis/localization/FINDINGS.md:5`). A fresh targeted replay found, in each residual repeat, 1,382 selected occurrences among 1,383 ready-candidate occurrences at withholding events; all selected occurrences were already outputting and all produced a new output, while the one unselected occurrence later became a victim. This validates realized service without redefining every generic `pending1` candidate as decode.

The localization distinguishes ordered-dispatch divergence from token/victim-map divergence (`analysis/localization/localize.py:76-93`). Guard/residual first differ on both at step 406, while fit/residual first differ in order at 406 and in the map at 521 (`analysis/localization/REPORT.md:7-12`). The latter state is not called a common pre-state after the earlier order divergence (`analysis/localization/FINDINGS.md:7`).

The cost transfer starts at the predeclared conditional cutoff 299 and uses the actual future schedule as input (`outputs/admission_capacity/20260914_restore_token_reservation_r01/analyze_frozen_cost.py:10-52`). The addendum reports it separately from complete-request wall metrics and warns that its error can exceed the action difference and that it is neither an online predictor nor an Oracle (`outputs/admission_capacity/20260914_restore_token_reservation_r01/RESULTS_ADDENDUM.md:31-35`). Likewise, measured elapsed time already differed before the first ordered action; the localization reports that residual and does not attribute all whole-run timing to step 406 (`analysis/localization/FINDINGS.md:5-7`).

## Findings

### P2-01 — Same-cohort scope and repeat dependence

The two order reversals help expose run-order drift but do not provide independent workload, seed, model, or runtime replication. Treat the four pairwise percentages as descriptive effects on this cohort. Do not attach statistical robustness or broad capacity claims. Evidence: `RESULTS_ADDENDUM.md:5-7`, `RESULTS_ADDENDUM.md:31`, `RESULTS_ADDENDUM.md:37-45`.

### P2-02 — Output quality is unmeasured

Some paired requests have different output sequences. The bundle preserves their hashes and equality flags but has no reference answers, semantic judge, or free-EOS quality measurement. No quality-neutrality claim is supported. Evidence: `RESULTS_ADDENDUM.md:18-25`, `RESULTS_ADDENDUM.md:31-37`, `analysis/analysis.json:38633-38922`.

### P2-03 — Conditional timing and replay are not full causal estimates

The primary whole-request metrics are realized full episodes, but the step-299 cost transfer is conditioned on actual future schedules, and elapsed time differs before the first ordered action. These diagnostics can localize and check consistency; they cannot estimate an action's counterfactual request benefit or Oracle headroom. Evidence: `analyze_frozen_cost.py:10-52`, `analyze_frozen_cost.py:70-76`, `analysis/localization/FINDINGS.md:5-7`, `RESULTS_ADDENDUM.md:33-35`, `RESULTS_ADDENDUM.md:41`.

## Claim Impact

- **Supported, bounded**: On this old 32-request cohort, the residual policy executed the intended token reservation, delivered the selected one-token services, and retained every first-output obligation to actual new output.
- **Supported, descriptive**: Relative to `guard_all`, residual improved throughput and mean completion in both order blocks while increasing the global maximum ITL.
- **Supported, negative boundary**: Relative to same-group `fit_scan`, residual lost throughput and mean completion in both order blocks; the current implementation did not win the pause/service-volume tradeoff.
- **Unsupported and not claimed**: superiority to native/least/most, a full-LTR or production SLO result, an action Oracle, quality neutrality, robustness across cohorts/models/GPUs, novelty, or method GO.

## Action Items

- Keep `MEASUREMENT_ONLY / OPEN`, the same-cohort qualifier, and the full list of unmeasured domains on every reuse of these numbers.
- Keep conditional cost-transfer and one-step flag-off results separate from realized full-request deltas.
- If a broader mechanism claim is pursued, run the already specified fresh-cohort strong-baseline experiment and add a quality protocol before making quality-neutrality claims.

## Fresh Verification and Unavailable Checks

Fresh verification wrote only under `/private/tmp/token-reservation-audit.v24Tam` and did not alter source, frozen packages, raw results, or prior derived artifacts. It included byte-identical reruns of the main analyzer, localization, and cost transfer; package/archive hash checks; six obligation unit tests; all-cell request/horizon checks; and a targeted pending-one realized-service replay.

No new GPU execution or network/remote-host query was performed. Current remote environment state, GPU tensor/KV equality, free-EOS semantic quality, statistical independence, and performance on a new cohort/model/GPU therefore remain unavailable. The recorded source/archive/runtime receipts were verified locally. No `.aris` trace was created, as required for this review.
