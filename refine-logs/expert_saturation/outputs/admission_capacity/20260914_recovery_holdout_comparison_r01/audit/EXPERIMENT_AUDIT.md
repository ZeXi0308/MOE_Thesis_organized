# Experiment Audit Report

**Date:** 2026-09-14  
**Generated:** 2026-09-14T15:03:42Z  
**Auditor:** Codex GPT-6 fresh same-family agent; read-only review; provisional  
**Project:** `20260914_recovery_holdout_comparison_r01`  
**Repository HEAD recorded by the experiment:** `de64dae5ea4fa3c92ec605e9847e817d8e68f3ac` (shared dirty worktree)  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`

## Overall Verdict: PASS

## Integrity Status: pass

P0: **0**. P1: **0**. P2: **1**. The eight-cell native-serving result is real, complete, numerically reproducible from retained raw files, and stated within its measured scope. The P2 concerns provenance durability of a secondary historical cost diagnostic, not the primary measurements or claim.

## Scope and method

This audit covered U and the listed direct analyzer, input, d6-reference, cost-model, execution, raw-result, diagnostic, and plot dependencies. It did not audit later funding/staged-store experiments or changing repository ledgers. It used no GPU, network, or `.aris`, and changed no source, raw, frozen, analysis, or report input.

The requested `analyze_absence_rotation.py` name resolves in the actual call chain to the imported `analyze_apc_rotation.py` and the runtime policy module `absence_rotation.py`; see `analyze_recovery_holdout_comparison.py:13-15,59-76,122-125`.

## Issues

### P2-1 — Historical cost-model training provenance is not self-authenticating

`20260914_recovery_progress_model_r01/cost_baseline.json:13-18` records two absolute training paths, call count, and RMSE, but no training-raw hashes or calibration-script hash. The transfer adapter pins the frozen model and source (`20260914_restore_token_reservation_r01/analyze_frozen_cost.py:8-12,55-60`), so it detects model drift but not later mutation of those historical training files.

The audit recorded both training raw hashes and independently reconstructed all three coefficients, 3021 calls, and RMSE exactly. Impact is limited because the U report correctly labels this as observed-schedule, retrospective diagnosis and rejects it as an action predictor, counterfactual, Oracle, or physical kernel cost (`RESULTS_ADDENDUM.md:80-89`; `analysis/cost_transfer/analysis.json:1702`). Future cost diagnostics should retain the calibration script and training raw hashes beside the frozen model.

## Checks

### A. Ground Truth Provenance: PASS

The workload is tied to a pinned WikiText train-shard Arrow hash, model/tokenizer revisions, document IDs, row intervals, document hashes, and prompt-token hashes (`execution/readback/results/cohort3-block0-most_output/config.json:17-39`). All 34 resolved holdout-preparation files/assets matched their receipts. Performance truth comes from retained native runtime request/token timestamps and scheduler events, not generated labels. Model outputs are used only for sequence equality; quality is explicitly unevaluated (`RESULTS_ADDENDUM.md:54,91-97`).

### B. Score Normalization: PASS

Rates use all completed requests or output tokens divided by the raw observation window; latency distributions retain raw seconds (`execution/readback/pkg/metrics.py:10-59,62-156`). Pairwise relative changes use the baseline cell's raw value and retain raw deltas/per-request values (`analyze_prefix_cache_baseline.py:206-220`). No metric is normalized by a model-output maximum, minimum, or mean.

### C. Result File Existence and Numeric Agreement: PASS

Both execution receipts are terminal COMPLETE and retain all eight 32-request cells (`execution/execution.json:34-130`; `execution/readback/group-status.json:1-97`). The analysis requires all eight eligible before comparisons (`analyze_recovery_holdout_comparison.py:219-228`), and reports all eight COMPLETE (`analysis/REPORT.md:3-16`). Independent raw recomputation matched all eight wall/rate/latency distributions, exclusive work-accounting totals, 16 comparisons, output-equality counts, and harmed/improved request counts to absolute tolerance `1e-12`.

### D. Live Metric Call Paths: PASS

`run_probe.py:13-20,210-228` imports the packaged metric/capture functions, executes the native episode, and writes raw plus metrics. The main analyzer reloads the packaged metrics and independently inspects each cell (`analyze_recovery_holdout_comparison.py:206-219`). No claimed result depends on an uncalled metric implementation.

### E. Scope Assessment: PASS

The claim language matches one 32-document, same-source-distribution cohort, fixed 3072/1024 lengths, steady arrivals, one OLMoE/vLLM/RTX-5090 configuration, and two correlated reverse-order repeats. It explicitly excludes independent-population, quality, business-SLO, Oracle, significance, multi-model/GPU, and method-GO claims (`analysis/analysis.json:68741-68749`; `RESULTS_ADDENDUM.md:91-101`).

### F. Evaluation Type: PASS

System evidence is direct `NATIVE_SERVING` measurement. For the skill's GT taxonomy, semantic evaluation is `self_supervised_proxy` only in the narrow sense that no quality GT exists by design; quality was not evaluated. It is not a synthetic-reference accuracy evaluation or a simulation-only result.

## Targeted integrity checks

- **Causal cutoff / future leakage:** PASS. Online actions use each engine's current/past state; scheduler-start availability is checked at `analyze_recovery_holdout_comparison.py:79-119`. The one future-schedule calculation is isolated and labeled post-hoc (`analysis/cost_transfer/analysis.json:1702`).
- **Independent policy state:** PASS. Eight fresh output directories/engines run serially under one lock (`execution/readback/pkg/run.sh:7-27`); cross-arm scheduling paths differ, while same-arm repeated paths and outputs match (`RESULTS_ADDENDUM.md:71-78`).
- **Request/token/time alignment:** PASS. All 256 requests complete with exactly 262144 outputs; stream interval is one token (`engine_args.json:17`), and the analyzer checks request identity, scheduler steps, token availability, and completion clocks.
- **Accounting non-overlap:** PASS. Prefill/recompute/decode positions close exclusively; decision and wrapped-scheduler time remain nested inside inclusive scheduler and wall time (`RESULTS_ADDENDUM.md:58-69`).
- **Baseline/config/resource equality:** PASS. The four arms share engine fields and exact runtime sources (`preparation/preparation_checks.json:5-52`); actual KV is 13,960,740,864 bytes and 6656 usable 16-token blocks (`engine_args.json:15`; `safe-cap-qualification.json:35-54`).
- **Retention and output semantics:** PASS. Both reverse-order blocks, unfavorable requests, repeats, and first output differences remain (`RESULTS_ADDENDUM.md:37-56,71-78`). Equality is not presented as quality.
- **Lifecycle status:** PASS. Preparation and frozen REPORT retain their historical UNRUN state (`preparation/preparation.json:1-7`; `REPORT.md:3,33`); terminal execution and addendum are separately authoritative (`RESULTS_ADDENDUM.md:3`; `execution/execution.json:1-33`).
- **Isolation:** PASS. Group preflights show no GPU process (`execution/execution.json:8-29`); per-cell environment records an empty pre-init process list and the exact GPU/runtime/source hashes (`environment.json:2-32`). Raw measurement snapshots contained exactly the cell's own process.
- **Historical victim diagnostic:** PASS within `CPU_OBSERVED_PRE_ACTION_IDENTITY_ONLY`; its report forbids trajectory/latency attribution (`preparation_checks/victim_cost_identity/REPORT.md:3-20`).
- **Plot provenance:** PASS. The script asserts eight eligible cells and iterates every cell in both panels (`analysis/plot_tradeoffs.py:12-38,53-64`). A fresh render produced a byte-identical PNG; SVG differences were limited to generated timestamp/element IDs. Visual inspection found all eight points and no hidden cell.

## Deterministic verification

The main analyzer was rerun in `/private/tmp/recovery-holdout-integrity-20260914.PhPkJE/analysis` with metadata SHA `28f7a20c1eaecd30540aafcc9634e377b68b74423c97b2e2ca4b71e10f883940`. Its `analysis.json` (`7d1ac0e5...1d59`) and `REPORT.md` (`29c63a2a...61e`) were byte-identical to U. The old victim-cost diagnostic reran byte-identically under its recorded Python 3.9 runtime. No GPU campaign was rerun.

## Claim impact

- **Supported:** In both blocks, `most_output` beats `fit_scan` and `guard_residual` on completed-request throughput and global maximum request ITL; all costs and harmed requests remain in the denominator (`RESULTS_ADDENDUM.md:9-18,37-54`).
- **Supported:** `most_output` has slower mean completion than native and harms 27/28 requests on completion plus 25/25 on per-request maximum ITL across the two blocks (`RESULTS_ADDENDUM.md:41-54`).
- **Supported with scope qualifier:** The document-disjoint holdout excludes the prior 128 identities, but remains the same pinned source shard and fixed workload family (`RESULTS_ADDENDUM.md:5-7,91`).
- **Qualified diagnostic only:** The frozen cost transfer and old d6 victim identity support accounting/interpretation boundaries, not action prediction, causal performance attribution, or an Oracle (`RESULTS_ADDENDUM.md:80-89,95`).
- **Unsupported and explicitly excluded:** quality, business SLO, statistical significance, independent-population generalization, multi-model/GPU behavior, MoE specificity, novelty, global optimality, or method GO (`RESULTS_ADDENDUM.md:91-99`).

Core and raw SHA-256 values are recorded in `EXPERIMENT_AUDIT.json`. This same-family audit is provisional and does not upgrade the experiment's `MEASUREMENT_ONLY` claim ceiling.
