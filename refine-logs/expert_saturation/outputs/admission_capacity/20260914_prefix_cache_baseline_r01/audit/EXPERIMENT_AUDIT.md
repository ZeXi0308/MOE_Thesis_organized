# Experiment Audit Report

**Date:** 2026-09-14  
**Auditor:** fresh Codex sub-agent, read-only semantic review  
**Review class:** same-family, provisional  
**Repository HEAD:** `c4ae4f5daaea929e1a4358862296d832f1cc67ab` (dirty; unrelated worktree changes were preserved)  
**Project:** `20260914_prefix_cache_baseline_r01`

## Overall Verdict: WARN

## Integrity Status: warn

No P0 or P1 integrity defect was found. The two finite claims in `RESULTS_ADDENDUM.md` are supported: in this exact cohort, APC-on reduced strictly repeated scheduled positions by 1,008 in both pairs, while the two victims' first recovery service steps remained 1030 and 1026. The warning is for one unused resource-metadata label inconsistency described below, not for a mismatch in the reported result.

The scientific ceiling remains `NATIVE_SERVING / MEASUREMENT_ONLY`. This audit does not support significance, a stable full-request speedup, task-quality equivalence, an Oracle, a method GO, or a general APC NO-GO.

## P0 / P1 Findings

- **P0:** none.
- **P1:** none.
- **Bounded note below P1:** `safe_static.py` writes `remaining_blocks_at_full_reservation = usable - safe * per_request`, where `safe=29`, rather than using the configured cap 32 (`preparation/source/safe_static.py:58-62`). The retained result consequently says 247 at `gpu_results/cohort2-block0-apc_off/safe-cap-qualification.json:46-54`. That value is the remainder at safe cap 29: `7671 - 29*256 = 247`; cap-32 full-reservation headroom is `7671 - 32*256 = -521`. The same file correctly records `full_cap_reserved_blocks=8192` and `full_reservation_sufficient=false`. No current addendum or analyzer decision consumes the mislabeled 247 field, so this does not change the present claim. Future analysis must not interpret it as cap-32 headroom.

## Audit Scope

The review read the three requested preparation/execution/analysis programs; the frozen `run_probe.py`, `native_capture.py`, `memory_telemetry.py`, `metrics.py`, actual `safe_static.py`, launchers and campaign; the preparation archive; execution attempts and ledger; all four returned archives, `raw.json`, `metrics.json`, environment, cache reset, cache-accounting and resource files; generated analysis; historical preparation report; and the result addendum. The requested top-level `PROTOCOL.md` did not exist. Its preparation-era content is retained in `REPORT.md` and frozen `DECISIONS.md`; `RESULTS_ADDENDUM.md:28` explicitly leaves that historical `PREPARED_UNRUN` record unchanged.

The auditor did not run GPU work, alter author files, inspect current remote state, fetch upstream vLLM, or generate `.aris`. The returned environment reports vLLM 0.26.0 and retains hashes for seven runtime source files (`gpu_results/cohort2-block0-apc_on/environment.json:2-27`), but this audit only binds the measurements to those retained runtime hashes; it does not independently certify that remote source tree against an upstream tag.

## Four Required Correctness Checks

### 1. Future leakage and policy-specific state: PASS

There is no learned selector, future-known Oracle, offline action choice, or trace-reuse counterfactual in this comparison. The frozen campaign declares four fresh engines in fixed `off/on/on/off` order (`execution/frozen/campaign.json:11-54`), and each cell launches a fresh process with only the APC flag changed (`execution/frozen/run_campaign.py:15-49`). Each arm advances its own request, KV, scheduler and completion state. Analysis pairs completed arms after execution; it does not use one arm's future state to construct another (`analyze_prefix_cache_baseline.py:206-220`).

### 2. Identity and time alignment: PASS

The frozen loader verifies the workload hash, 32 aligned request/prompt/arrival entries, 3,072-token prompts and 1,024 requested outputs (`execution/frozen/run_probe.py:37-49`). The analyzer checks source ID, document ID, prompt-token digest, arrival, internal mapping, completion length and shared-clock ordering for every request (`analyze_prefix_cache_baseline.py:172-183`). It also conserves every engine-call/scheduler-step/receipt relation and rejects multi-token receipt ambiguity (`analyze_prefix_cache_baseline.py:43-106`).

Independent projections of all four raw files had the same measurement-identity digest and the same complete output-token digest. Each file contains 32 completed requests, 32,768 one-token receipts, 1,348 completed engine calls and no failed engine call. The four returned cells and archives are recorded separately with return code 0 and terminal `COMPLETE` (`execution/execution.json:4-72`).

### 3. Accounting, cold reset and resources: WARN

The request metric denominator is every request arrived by the actual observation end; throughput is completed requests divided by `observation_end - min(arrival)` (`execution/frozen/metrics.py:10-29`, `execution/frozen/metrics.py:62-152`). All four cells have 32 planned/arrived/completed, zero failed/unfinished requests and are eligible for complete-episode comparison (`gpu_results/cohort2-block0-apc_off/metrics.json:2-11`, `gpu_results/cohort2-block0-apc_off/metrics.json:33260-33264`; the same checks passed for the other three files).

The exclusive executed-position accounting is conserved. APC-off has `138725 = 7685 repeated + 98304 fresh prefill + 32736 fresh decode`; APC-on has `137717 = 6677 + 98304 + 32736` (`analysis/analysis.json:249-257`, `analysis/analysis.json:794-802`, `analysis/analysis.json:1346-1354`, `analysis/analysis.json:1898-1906`). The 1,008-token `computed_adjustment` in each APC-on run is a skipped cached position range, not itself an executed-token count (`analysis/analysis.json:829-840`, `analysis/analysis.json:1381-1392`). Separately, unioning successful executed intervals establishes 1,008 fewer actually repeated positions. Failed allocation attempts are not counted as execution.

Recovery gaps are request-local and overlap. For block 0 off, request 0017069 spans 16.9255-21.5889 s while request 0016821 spans 19.4270-21.4731 s (`analysis/analysis.json:258-281`); summing them would double-count wall time. The addendum correctly retains wall as the complete observation interval and states that token positions are not GPU time (`RESULTS_ADDENDUM.md:14-20`).

Every process used the same 16,089,350,144-byte KV allocation, 7,671 usable 16-token blocks, model storage and one GPU UUID. The APC-on reset receipt shows a drained engine, 656 hashed free blocks before reset, a successful reset, and zero hashes/refcounts with all 7,671 blocks free afterward (`gpu_results/cohort2-block0-apc_on/prefix-cache-reset.json:2-27`); the other three reset receipts satisfy the same postcondition. The preflight retained exact model-shard hashes and an empty compute-process list (`execution/attempts/1789319340098121000/preflight.json:2-23`). The WARN is solely the non-claim metadata label noted above.

### 4. Baseline fairness and repeat retention: PASS

APC-off is the direct native baseline. Normalized `config.json` and `engine_args.json` are byte-identical across all four cells after removing `enable_prefix_caching`; the analyzer repeats this check and requires identical software and GPU UUID (`analyze_prefix_cache_baseline.py:206-220`). The fixed order counterbalances APC once in each direction (`execution/frozen/campaign.json:11-53`). All three warmup input projections are identical across cells, every warmup completed, and the measured output sequences are identical across all four executions.

The two same-role runs are real fresh-process executions, retained as separate archives and ledger entries, rather than duplicated rows or values computed from one trace (`execution/execution.json:4-72`). They remain correlated same-input repeats. Their wall changes are about 0.79% for APC-off and 0.34% for APC-on, while the two off-to-on wall changes are about -0.50% and -0.04% (`analysis/analysis.json:2463-2473`, `analysis/analysis.json:2738-2748`, `analysis/analysis.json:3013-3023`, `analysis/analysis.json:3288-3298`). `RESULTS_ADDENDUM.md:14` correctly makes no significance, non-inferiority, robustness or stable-net-benefit claim.

## A-F Integrity Checklist

### A. Ground Truth Provenance: PASS

This is a native systems measurement with no task ground truth. Prompts come from the frozen WikiText cohort with pinned dataset, tokenizer and prompt hashes (`execution/frozen/cohorts/cohort2/inputs_preparation/prepared/long/config.json:17-40`). Generated tokens are used only for identity/equality and completion accounting; the addendum explicitly excludes task quality (`RESULTS_ADDENDUM.md:20`). There is no model-generated reference presented as real ground truth.

### B. Score Normalization: PASS

TTFT, TPOT, ITL, completion rate and throughput are computed directly from retained timestamps and counts (`execution/frozen/metrics.py:106-152`). Pair comparisons use raw subtraction or `action/baseline - 1` (`analyze_prefix_cache_baseline.py:206-220`). No metric is divided by the model's own maximum, minimum or mean prediction.

### C. Result File Existence and Claim Match: PASS

All four raw/result directories, returned archives, terminal statuses and claimed keys exist. Archive hashes in the execution ledger match the retained archives, and each archived `raw.json` hashes identically to its extracted copy. The preparation archive and executed frozen archive both hash to `6266a0dff1411cbf5403beb8ccab18918325ed912b2054e68985e286302290fd`, matching the staged verification (`execution/attempts/1789319340098121000/staged-verification.json:1-4`) and execution ledger (`execution/execution.json:74-87`).

The addendum's four wall/throughput/mean-completion/max-ITL/repeated-position rows match `analysis.json` (`RESULTS_ADDENDUM.md:7-16`; `analysis/analysis.json:249-257`, `analysis/analysis.json:516-537`, `analysis/analysis.json:794-840`, `analysis/analysis.json:1068-1089`, `analysis/analysis.json:1346-1392`, `analysis/analysis.json:1620-1641`, `analysis/analysis.json:1898-1906`, `analysis/analysis.json:2165-2186`). A clean CPU re-run of the analyzer reproduced the result; the only JSON differences were absolute versus relative path strings.

### D. Dead Code Detection: PASS

`summarize_episode_requests` calls `summarize_requests`, which calls the distribution helper (`execution/frozen/metrics.py:10-29`, `execution/frozen/metrics.py:40-59`, `execution/frozen/metrics.py:62-156`), and `run_probe.py` writes those outputs to every cell's `metrics.json` (`execution/frozen/run_probe.py:199-215`). The analyzer calls the retained metric path and the strict work-accounting path (`analyze_prefix_cache_baseline.py:182-200`, `analyze_prefix_cache_baseline.py:233-246`). No claimed metric comes from an uncalled function.

### E. Scope Assessment: PASS

Actual scope is one RTX 5090, one OLMoE revision, one 32-request cohort, one fixed-KV pressure point, one fixed arrival trace and two correlated repeats per role. `RESULTS_ADDENDUM.md:20-24` states these exclusions and does not call the evaluation comprehensive, robust or significant. The scope is sufficient only for the exact finite measurement claim.

### F. Evaluation Type: self_supervised_proxy

Closest checklist class: `self_supervised_proxy` because no external task GT is required for this native runtime measurement. More precisely, it is `native_system_measurement_no_task_gt`, with complete request outputs and schedules retained. Evidence tier is `NATIVE_SERVING` in-process, not external HTTP or production service (`RESULTS_ADDENDUM.md:5`, `RESULTS_ADDENDUM.md:20`).

## Hash-Bound Inputs

| Artifact | SHA256 |
|---|---|
| `analyze_prefix_cache_baseline.py` | `590ee20ee3648bb097abf82f29b23ac1e249ca588216bba4b0b13d713d18a473` |
| `prepare_prefix_cache_baseline.py` | `eaf81e8b3c5f64edd52a2e17a3de6a7c70aa294e575a81bb5763ec3ee182b1ad` |
| `run_frozen_kv_remote.py` | `5bed0643468a8de17122d1ff3855790db7a6755626dd494f9fbc608ecdc4d83f` |
| `preparation/execution.tar.gz` | `6266a0dff1411cbf5403beb8ccab18918325ed912b2054e68985e286302290fd` |
| `execution/execution.json` | `43366d88d2e68626687eed3de92ed917c41721a61b251a8416a6129ec4a5a524` |
| `analysis/analysis.json` | `a1b58aed882e12941217435ab7e815567abd3b7c5b663d3bcf337344e3908206` |
| `RESULTS_ADDENDUM.md` | `f994ef22340fd357aed652cd5ff9d0bb3c28f069a173ee2ddae8660e2f5bdb5a` |
| block0 APC-off `raw.json` | `8aebd8a233d497a577b31c5fdea407dc80378580f0cf953656574a451ec78b90` |
| block0 APC-on `raw.json` | `cfe9ce89ecf486289c3b3972f13d06d674941a2b75583b234be32724d8692868` |
| block1 APC-on `raw.json` | `061ccaf6acb1c825806c17c9b221bc8f5e050cb2ecc144725465fb7bb8169ae3` |
| block1 APC-off `raw.json` | `b9ce36e1aec77855bdda8baf4f253f62f48cb5bcebee7d6db3265b8605996a11` |

The machine-readable report contains the remaining metrics, environment, reset and cache-ledger hashes.

## Claim Impact

- **APC-on reduced actually repeated positions from 7,685 to 6,677 in this exact cohort:** supported.
- **APC-on did not change the two first recovery service steps and did not remove the longest exposed pause:** supported.
- **APC-on produced a stable or significant full-request speedup:** unsupported and not claimed.
- **Output-token equality proves task-quality equivalence:** unsupported and not claimed.
- **APC, rotation, or an APC-aware method is generally GO/NO-GO:** unsupported and not claimed.

## Scientific Handoff

- **Verdict:** `MEASUREMENT_ONLY`; integrity `WARN` only for unused metadata naming.
- **Evidence type:** `NATIVE_SERVING`, in-process host capture.
- **What was measured:** four fresh native APC off/on/on/off executions, complete request timing, schedule/KV/preemption traces and retained outputs.
- **What was not measured:** significance, independent workload population, external service behavior, quality, second model/runtime, Oracle or APC-aware rotation.
- **Strongest baseline:** native APC off versus native APC on under the same frozen engine/workload settings.
- **Oracle/headroom:** unmeasured.
- **Claim ceiling:** APC reused 1,008 post-preemption positions and reduced repeated execution in this single cohort, without changing recovery start steps.
- **Failure category:** the current longest recovery wait remains exposed; neither run failure nor general APC failure.
- **Resurrection condition:** a different prefix-sharing/cache-survival/capacity regime requires a new real execution.
- **One next smallest experiment:** if authorized later, the already stated APC-on native versus APC-on most-output four-cell comparison, with the corrected resource-field semantics and no threshold scan.
