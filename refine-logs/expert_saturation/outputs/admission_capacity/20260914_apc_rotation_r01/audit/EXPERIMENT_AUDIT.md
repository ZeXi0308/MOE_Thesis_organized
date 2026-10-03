# Experiment Audit Report

**Date**: 2026-09-14  
**Generated at**: 2026-09-13T17:55:35Z  
**Auditor**: fresh Codex agent, same-family, read-only review  
**Review status**: provisional  
**Project**: APC-on native versus frozen `most_output` rotation  
**Repository HEAD**: `c4ae4f5daaea929e1a4358862296d832f1cc67ab` (dirty worktree; target artifacts were hashed and reviewed in place)

## Overall Verdict: WARN

## Integrity Status: pass with bounded-scope warnings

**P0 findings: 0. P1 findings: 0.**

The four retained native-runtime cells and the reported descriptive tradeoff are internally supported. Independent recomputation from all four `raw.json` files reproduced request identity, completion, output receipts, timing, executed-position accounting, preemption counts, forced-release deltas, policy replay, and the published wall/throughput/mean-completion/max-ITL values. The WARN is about evidence scope: the cohort was previously observed, the two repeats reuse that same cohort, the hold branch was never activated, and no quality, Oracle, APC-on baseline ladder, or generalization evidence was produced.

No `.aris` trace was created, as explicitly required for this bounded audit. No source, raw result, analysis, or author report was modified.

## Four correctness checks

### 1. Identity and alignment: PASS

- The frozen order is `native / most / most / native`; all cells use `cohort2`, cap 32, APC on, and independent engines (`preparation/source/campaign.json:11-53`).
- The execution record shows four serial cells on the same GPU UUID, empty foreign-process snapshots at each cell boundary, exit code 0, 32 completed requests per cell, and final `COMPLETE` (`execution/execution.json:1-100`).
- Independent raw replay verified 32 unique request IDs per cell, the frozen document/prompt hash and arrival for every request, internal-to-source mappings, 1024 output tokens, and monotonic arrival/admission/receipt/completion ordering. Every output receipt reconstructed the retained request output and token timeline exactly.
- The readback archive SHA-256 is `91d4936121aa6c5e34ae02b1a020a380e0d033e24c332eda1e396c83611df009`; all 86 regular members (1,214,217,785 uncompressed bytes) matched the extracted files byte for byte. The staged source archive SHA-256 is `eb40e068a70cc185587159dfaf1b13839dc49be7c46f0df0c1896ba46d0adf3d` (`execution/execution.json:89-100`).

### 2. Causal cutoff and future leakage: PASS for the executed policy

- `AbsenceRotation.decide` uses only the current step, current running/waiting state, current free blocks, and event history recorded by prior actual preempt/resume events (`preparation/source/absence_rotation.py:111-205`). `most_output` is the number of output tokens already produced at the decision point, not future completion information (`preparation/source/absence_rotation.py:51-70`, `preparation/source/absence_rotation.py:191-197`).
- The adapter forms every decision before calling the patched native scheduler, then records actual preempt/resume outcomes afterward (`preparation/source/rotation_native.py:174-264`). It does not assign model token state directly; the forced action invokes native `_preempt_request` (`preparation/source/rotation_native.py:137-158`).
- Independent step-by-step replay reproduced every proposal in both `most_apc` runs from each step's `before` snapshot plus prior actual events. Each run had 1,029 checked proposals, 9 funded forced preemptions, and identical normalized decision/execution fingerprints (`analysis/analysis.json:1276-1283`, `analysis/analysis.json:1990-1997`).
- This replay validates the chosen policy's online inputs and execution. It is not an Oracle for unchosen victims or no-swap counterfactuals. The author report states that boundary correctly (`RESULTS_ADDENDUM.md:16`, `RESULTS_ADDENDUM.md:26`).

### 3. Accounting and metric correctness: PASS

- Host timestamps and token receipts are retained by `capture_episode`; failed calls are not promoted to executed work, and successful calls update the execution high-water mark only after `engine.step()` returns (`preparation/source/native_capture.py:64-100`, `preparation/source/native_capture.py:133-204`).
- Independent interval-union accounting closed in all four cells:

  | Cell | Scheduled | Repeated executed | Fresh prefill | Fresh decode |
  |---|---:|---:|---:|---:|
  | block0 native | 137,717 | 6,677 | 98,304 | 32,736 |
  | block0 most | 167,312 | 36,272 | 98,304 | 32,736 |
  | block1 most | 167,312 | 36,272 | 98,304 | 32,736 |
  | block1 native | 137,717 | 6,677 | 98,304 | 32,736 |

  In every cell, `scheduled = repeated + fresh prefill + fresh decode`. APC cache jumps are reported separately from actually repeated execution; the action adds 29,595 repeated positions relative to native in both blocks (`analysis/analysis.json:520-562`, `analysis/analysis.json:1097-1271`, `analysis/analysis.json:1811-1985`, `analysis/analysis.json:2525-2567`).
- The request metrics use all arrived requests and direct timestamps, not a model-derived score or a favorable subset (`preparation/source/metrics.py:62-155`). All 32 requests completed in every cell, so throughput is `32 / full observation wall` and mean completion is the mean of `completion - arrival` over the same 32 requests.
- Independent raw recomputation exactly reproduced the four published rows and paired deltas (`RESULTS_ADDENDUM.md:7-14`). In particular, `most` reduced max-ITL by 3.713069 s and 3.630799 s, while mean completion worsened by 0.554193 s and 1.020063 s. Throughput changed by +1.692445% and -0.334092%, so the three objectives do not jointly improve.
- The mutually exclusive wall buckets close: `wall = scheduler inclusive + engine excluding scheduler + outside engine`; decision time is a subset of scheduler time and was not deducted (`RESULTS_ADDENDUM.md:20`). Qualification and telemetry overhead remain in observed wall time.
- The two `most` paths have identical discrete execution, decision, and output hashes while wall differs by 0.332850 s. The report correctly preserves both repeats and does not treat them as an independent noise bound (`RESULTS_ADDENDUM.md:20`).

### 4. Action-specific state, free blocks, and APC qualification: PASS in the qualified domain; WARN for unexercised hold branch

- Each measurement starts after a drained prefix-cache reset with 7,671/7,671 free usable blocks, zero live/negative references, and zero hashed blocks (`execution/group_readback/results/cohort2-block0-most_apc/prefix-cache-reset.json:1-28`).
- The runtime is one `UnitaryKVCacheCoordinator` / `FullAttentionSpec`, no sliding/chunk attention, one 16-token block granularity, no speculative/lookahead/context parallelism, and the seven pinned vLLM source hashes are retained (`execution/group_readback/results/cohort2-block0-most_apc/safe-cap-qualification.json:1-54`, `execution/group_readback/results/cohort2-block0-most_apc/environment.json:12-29`). All four cells carried the same source inventory and hashes.
- The adapter rejects partial hits, pending CoW, shared live blocks, duplicate ownership, negative refcounts, and any mismatch between active physical references and request-owned blocks (`preparation/source/rotation_native.py:30-52`). Before and after every forced exchange, that live assertion executed successfully.
- Independent replay verified 9 forced native preemptions in each `most` run. Their actual free-block deltas were exactly `[245, 246, 248, 250, 250, 250, 250, 250, 249]`; every delta equaled the candidate, expected exclusive-owner, and native observed release fields, and each victim block table became empty. The protected target received scheduled work every recovery step with free blocks still covering its remaining conservative full-history need (`preparation/source/rotation_native.py:133-172`, `preparation/source/rotation_native.py:217-258`).
- The analyzer and this audit can verify executed fail-closed receipts and aggregate/free-delta conservation, but cannot reconstruct every live block ID/refcount offline. This limitation is stated in both the analyzer scope and report (`analysis/REPORT.md:5`, `RESULTS_ADDENDUM.md:22`).
- `held_request_steps` is zero in both `most` runs (`analysis/analysis.json:1276-1283`, `analysis/analysis.json:1990-1997`). The hold code was evaluated but no request crossed its hold condition, so this experiment does not validate a state-preserving held transition. The report does not claim otherwise (`RESULTS_ADDENDUM.md:22`).

## A-F integrity checks

### A. Ground-truth provenance: PASS

Performance outcomes come from retained native-runtime request/token timestamps, scheduler receipts, block-pool telemetry, and native preemption events. There is no synthetic reference or model-output-derived performance ground truth. Output equality is used only for identity/result consistency; semantic quality and natural-EOS behavior remain unmeasured (`RESULTS_ADDENDUM.md:14`, `RESULTS_ADDENDUM.md:22`).

### B. Score normalization: PASS

No metric is normalized by the action arm's own maximum, minimum, or mean. Raw seconds, request/s, request-level changes, and executed token-position counts are retained. Relative changes use the paired native baseline denominator and accompany raw values (`experiments/admission_capacity/analyze_apc_rotation.py:167-172`, `RESULTS_ADDENDUM.md:7-14`).

### C. Result existence and number matching: PASS

All four raw files, terminal status, reset, environment, decisions, metrics, and analysis files exist. Raw SHA-256 values independently matched:

- block0 native: `357939e1dc75a894ed86289319938c6104ae9f9cc9565febe009afbdcdb202ae`
- block0 most: `397cfd04ffe3a56002cc3e6ce8262d69e2d27d0c1d25e92df31624ef6f801fd6`
- block1 most: `60042e157a5326ab97141ccaf1817938cbc4f5ae09ef2c5d96c653ad9ec0a961`
- block1 native: `d8ddf171644c083611b92b8552fbb49d295cbaabf46e384943f9d0463f3f3403`

The analysis status is `MEASUREMENT_ONLY`, every cell is eligible and complete, and the addendum's numerical rows, per-request slower/worse counts, repeat fingerprints, and scope statements match the raw-derived analysis (`analysis/REPORT.md:1-12`, `RESULTS_ADDENDUM.md:3-30`).

### D. Dead-code / claimed-output path: PASS with one coverage warning

The request summarizer, distribution function, raw work accounting, policy replay, compare path, and final report path all produced retained outputs (`preparation/source/metrics.py:10-29`, `preparation/source/metrics.py:40-59`, `experiments/admission_capacity/analyze_apc_rotation.py:47-109`, `experiments/admission_capacity/analyze_apc_rotation.py:196-224`). No claimed metric was found to come from an uncalled function. The conditional hold transition was not exercised, as noted above, and therefore carries no empirical safety/effectiveness claim.

### E. Scope assessment: WARN

Actual scope is one OLMoE checkpoint, one fixed 32-request cohort and arrival regime, one GPU/runtime/configuration, and two correlated same-input executions per policy. `cohort2` was already observed and is not a new holdout. APC-on headroom and nearest-system baselines, unchosen-action Oracle, shared-prefix/partial-hit/CoW regimes, heterogeneous lengths, a second model, external HTTP service behavior, business SLOs, and task quality were not tested (`RESULTS_ADDENDUM.md:22-26`).

The addendum stays within this scope: it reports a reproducible measurement tradeoff, leaves the research question `OPEN`, and denies method GO and family-wide NO-GO (`RESULTS_ADDENDUM.md:3`). Therefore this is a scope warning, not a claim-integrity failure.

### F. Evaluation type: native runtime measurement; quality UNRUN

- Performance/resource evaluation: direct `NATIVE_SERVING` measurement from real in-process execution.
- Policy-input validation: causal replay of the executed policy only.
- Output identity: deterministic consistency diagnostic.
- Semantic/task quality: `UNRUN`.
- Counterfactual Oracle and statistical/generalization evaluation: `UNRUN`.

## Non-blocking metadata note

`remaining_blocks_at_full_reservation: 247` is the remainder after the derived safe cap 29 (`7671 - 29*256`), despite its ambiguous name. The requested cap32 full-history margin is `7671 - 32*256 = -521`; the analyzer records that separately and the addendum corrects the interpretation (`execution/group_readback/results/cohort2-block0-most_apc/safe-cap-qualification.json:43-54`, `RESULTS_ADDENDUM.md:30`). No reported comparison consumed the ambiguous field.

## Claim impact

- Four APC-on native-runtime cells completed with the declared frozen inputs and resources: **supported**.
- Frozen `most_output` actually changed native execution and shortened the maximum observed ITL in both same-cohort pairs: **supported as a descriptive measurement**.
- The action preserved all three primary objectives: **unsupported**; mean completion worsened twice and throughput changed sign.
- Exact output equality proves task quality: **unsupported**; quality was not evaluated.
- Stepwise replay provides an action Oracle: **unsupported**; it only checks the selected online path.
- The current fixed rule is a method GO, a general solution, or a family NO-GO: **unsupported**.

## Required claim ceiling

`NATIVE_SERVING / MEASUREMENT_ONLY`: in this single fixed APC-on cohort and resource regime, the frozen rotation rule reduces the longest observed pause while increasing repeated execution and mean request completion time. Throughput is not directionally stable across the two same-input executions. The result identifies a beneficiary/victim tradeoff and does not establish quality, SLO success, generalization, Oracle headroom, or method GO.

