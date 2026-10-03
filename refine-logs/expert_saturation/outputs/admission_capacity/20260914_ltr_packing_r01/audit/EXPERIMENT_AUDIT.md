# Experiment Audit Report

**Date:** 2026-09-14  
**Auditor:** Codex GPT-6, fresh same-family agent, read-only review  
**Project:** `20260914_ltr_packing_r01`  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`  
**Overall verdict:** `PASS`  
**Integrity status:** `pass`

## Severity

- P0: **0**
- P1: **0**
- No fake ground truth, self-normalized score, phantom result, action/receipt mismatch, denominator error, future-trace counterfactual, or scope inflation was found within the declared component-measurement ceiling.
- This is one bounded integrity review. It does not certify full LTR, a quality evaluation, an Oracle, statistical significance, or production SLO behavior.

## Review contract

- Review-time repository HEAD: `de64dae5ea4fa3c92ec605e9847e817d8e68f3ac`; the worktree already contained unrelated/concurrent dirty files. The executed package is assessed through its sealed hashes rather than the current checkout.
- Evidence read: the supplied bundle reports, preparation metadata and package, execution receipt and complete readback, all four result cells, packing/localization analyses, the three supplied analysis programs, and the JIT semantics addendum.
- Research question audited: whether the reported four-cell native execution honestly supports the bounded conclusion that rank-prefix did not remove unfinished-recovery churn or consistently improve full-request outcomes in this fixed old-d6 regime.
- Weakest integrity link audited: policy-specific past-only state -> recorded plan -> native scheduled tokens/victims -> block/resource receipts -> request/output timing.
- Allowed claim ceiling: `NATIVE_SERVING` in-process, custom FCFS/current-history-reservation/native-RECOMPUTE component comparison; `MEASUREMENT_ONLY`.
- Stop condition: no P0/P1 after one direct evidence pass. No GPU rerun, network access, source/raw mutation, threshold scan, or `.aris` trace was performed.

## A. Ground Truth Provenance: PASS

This experiment does not use a generated answer as quality ground truth. WikiText supplies frozen prompts and identities, while the evaluated outcomes are native scheduler actions, KV ownership receipts, host receipt times, and generated token prefixes. The input source, revision, Arrow hash, tokenizer hashes, deterministic selection rule, and non-holdout scope are recorded in each cell config (`execution/readback/results/block0-d6-packing-fit_scan/config.json:17-35`). `run_probe.py` re-hashes every prompt before engine construction (`preparation/pkg/run_probe.py:40-52`), and `native_capture.py` records cumulative outputs and fails on prefix mutation or incorrect fixed completion length (`preparation/pkg/native_capture.py:150-177`).

There is no quality reference or quality score. The results explicitly state that only 25/32 cross-arm outputs match and that quality equivalence cannot be claimed (`RESULTS_ADDENDUM.md:24`). This is therefore not a fake-GT case.

## B. Score Normalization: PASS

Request metrics are computed directly from arrival, token receipt, and completion timestamps. TTFT, TPOT, ITL, request latency, completion throughput, and SLO-reference attainment use fixed formulas and fixed 5 s / 0.2 s thresholds (`preparation/pkg/metrics.py:62-155`). Pairwise percentages are action/baseline ratios, not division by a maximum, minimum, or mean of the model's own predictions (`refine-logs/expert_saturation/experiments/admission_capacity/analyze_prefix_cache_baseline.py:206-220`; `refine-logs/expert_saturation/experiments/admission_capacity/analyze_ltr_packing_probe.py:196-205`).

The stored `slo_attainment=1.0` is a raw fixed-threshold diagnostic. It is not used as a business-SLO claim; the report excludes business SLO-goodput (`RESULTS_ADDENDUM.md:24`). No suspicious score normalization was found.

## C. Result Existence and Exact Numbers: PASS

The declared campaign contains exactly four cells in reversed pair order (`preparation/pkg/campaign.json:2-30`). The retained campaign log has one START and END for each cell and a terminal campaign marker (`execution/readback/campaign.log:17-25`). The execution receipt records all four as `COMPLETE`, 32 requests each, with preemption counts 22/25/25/22 and no error (`execution/execution.json:34-87`). Each cell's `status.json` independently agrees; for example, the first fit cell is complete with 32 requests and 22 native-RECOMPUTE preemptions (`execution/readback/results/block0-d6-packing-fit_scan/status.json:2-7`).

Independent recomputation from the four raw files reproduced every reported headline value:

| Cell | Wall s | Requests/s | Mean completion s | Max request ITL s | Calls | Preemptions | Recomputed positions |
|---|---:|---:|---:|---:|---:|---:|---:|
| block0 fit-scan | 27.670858657 | 1.156451283 | 21.618075060 | 4.834701002 | 1862 | 22 | 75742 |
| block0 rank-prefix | 28.196696216 | 1.134884731 | 22.277188004 | 6.732402464 | 1872 | 25 | 74982 |
| block1 rank-prefix | 27.708315525 | 1.154887960 | 21.775504319 | 6.576920833 | 1872 | 25 | 74982 |
| block1 fit-scan | 28.234744502 | 1.133355395 | 22.144992097 | 4.933291839 | 1862 | 22 | 75742 |

These values match `RESULTS_ADDENDUM.md:9-16` and the derived cells, including the first fit cell's work totals and wall (`analysis/analysis.json:261-269`, `analysis/analysis.json:4603-4607`). All 128 planned requests completed exactly 1024 outputs, for 131,072 retained outputs. Per-request TTFT, TPOT, all 1023 ITLs, and completion latency were independently recalculated for all 128 requests and matched every stored `metrics.json` row.

The two matched comparisons also reproduce the reported sign flip: rank-prefix throughput is -1.864891% in block0 and +1.899895% in block1; mean completion is +3.048897% and -1.668494%; max ITL is worse by 1.897701 s and 1.643629 s (`analysis/analysis.json:20171-20205`, `analysis/analysis.json:20503-20537`). The fit repeat wall changes by +0.563886 s and the prefix repeat by -0.488381 s, while same-policy schedule paths are identical (`analysis/analysis.json:20835-20848`, `analysis/analysis.json:21167-21180`).

### Archive and attempt identity

- Input archive SHA256: `1fea4a612af77a0062163cec33566765c2df25ade4021407756ec43b1734ef73`.
- Readback archive SHA256: `2825c8181d893af9482adbc6b5f07e027dea465f3631fb6036b690d75179cacc`, matching `execution/execution.json:31-33` and `RESULTS_ADDENDUM.md:40`.
- All 97 regular members of `execution/readback.tar.gz` byte-match the extracted `execution/readback/` tree.
- All 16 preparation-package manifest entries match both `preparation/pkg/` and returned `execution/readback/pkg/`; preparation metadata records the same archive and expected runtime source hashes (`preparation/preparation.json:7-29`).
- The driver hash is `155a5bec8585434e5da0a625f3c6a047981f4727986db8414bbe796ca4fb147d`, matching `execution/execution.json:7`.
- Within this bundle, no extra result cell or retry is present. `run.sh` refuses an existing results directory and executes the four cells serially (`preparation/pkg/run.sh:7-24`); `execute.py` uses exclusive launch/result creation and never auto-relaunches an interrupted remote group (`execute.py:20-38`, `execute.py:80-108`). This does not claim knowledge of attempts outside the declared bundle/remote directory.

## D. Dead Code and Called Metrics: PASS

Every metric helper in `metrics.py` is on the executed call graph: `run_probe.py` calls `summarize_episode_requests` (`preparation/pkg/run_probe.py:190-195`), which calls `summarize_requests`, `_finite`, and `_distribution` (`preparation/pkg/metrics.py:10-29`, `preparation/pkg/metrics.py:32-59`, `preparation/pkg/metrics.py:62-155`). The packing analyzer calls raw work accounting, component accounting, quantum accounting, residency accounting, and the request metrics (`refine-logs/expert_saturation/experiments/admission_capacity/analyze_ltr_packing_probe.py:116-140`). Their outputs appear in `analysis.json`.

The reused package contains inactive headroom/rotation code and inherited `headroom_observer` / `rotation_config` metadata. The frozen packing path installs only `install_component` (`preparation/pkg/run_probe.py:169-179`); no reported packing metric is sourced from those inactive branches. This is not a dead claimed metric.

## E. Scope and Baselines: PASS

The evidence is one model, one RTX 5090, one old 32-document cohort, one steady 50 ms arrival regime, two policies, and two reversed-order same-input repeats. The report calls the repeats correlated rather than a noise bound, retains both unfavorable signs, and explicitly excludes a fresh holdout, second cohort/regime, full LTR predictor, CPU SWAP, Oracle, quality, significance, production P99, and business SLO (`RESULTS_ADDENDUM.md:3-5`, `RESULTS_ADDENDUM.md:24-30`).

The matched strongest baseline for this question is the same custom backend with fit-scan. Rank-prefix changes only the infeasible-candidate stop rule; common LTR200/10, FCFS ranking, current-history reservation, native recompute, model/runtime, workload, and resource budget are held fixed (`preparation/pkg/ltr_recompute_native.py:44-81`, `RESULTS_ADDENDUM.md:26`). Cross-group native/least/most/headroom results are not used as matched wall comparisons. Scope language does not exceed the evidence.

## F. Evaluation Type: PASS

- Skill classification: `self_supervised_proxy` because no external quality GT is required or used.
- More precise subtype: `real_native_system_measurement_no_quality_gt`.
- Evidence layer: `NATIVE_SERVING_INPROCESS_HOST_CAPTURE` from synchronous vLLM 0.26.0 execution, not simulation (`preparation/pkg/native_capture.py:192-204`; `execution/readback/results/block0-d6-packing-fit_scan/environment.json:2-9`).
- Scientific status: `MEASUREMENT_ONLY`.

## Targeted Integrity Checks

### Runtime, package, and input identity: PASS

All four `engine_args.json` files are byte-identical. The model/revision, BF16 dtype, seed, 4096 max length, 32 sequences, 1024 batched tokens, APC off, FCFS, full-ISL reservation, and exact 13,960,740,864-byte KV pool are recorded at `execution/readback/results/block0-d6-packing-fit_scan/engine_args.json:2-19`. The runtime records vLLM 0.26.0 and seven vLLM source hashes (`execution/readback/results/block0-d6-packing-fit_scan/environment.json:2-31`); all four cells carry the same seven hashes. The two pinned hashes also match both preflight receipts (`execution/execution.json:8-29`).

All four request cohorts independently match the same 32 request/document IDs, arrival times, 3072-token prompt hashes, and internal/external mapping. The frozen workload hash matches its config. There were zero identity mismatches.

### Policy-specific state and past-only boost: PASS

Each policy is a separate engine process and retains its own raw requests, scheduler steps, memory trace, output events, and decisions. The LTR counter reads only live IDs and past selection/non-selection, resets after 200 idle scheduler calls, and spends at most 10 selected calls (`preparation/pkg/recovery_service_components.py:35-77`). The planner uses current priority, arrival, residence, block ownership, current history, and pending tokens (`preparation/pkg/ltr_recompute_native.py:26-81`).

An independent replay of the counter state over all 7,468 scheduler calls produced zero priority/state mismatches. For every call, the pre-action output count also equaled the number of already received outputs at that scheduler start. Post-run token times are used only to classify outcomes, not to drive an action; the analysis states this boundary (`analysis/REPORT.md:5`). No future trace is reused as an Oracle or counterfactual.

### Planned tokens, victims, and resource receipts: PASS with declared replay limit

For all 7,468 scheduler calls, independently checked:

- recorded `tokens == actual_scheduled == native scheduler-step allocations`;
- recorded victims equal native preemption hooks and returned preempted IDs;
- every victim's actual owned-block release equals the pool free-block delta;
- `free_after_reservation = free_before + released - current-history need`;
- remaining selected-history need is no larger than free blocks after scheduling;
- held residents preserve their recorded scheduler state;
- decision time <= wrapped component scheduler time <= captured scheduler time.

There were zero mismatches. The runtime itself fails if the native action differs from the component plan or if selected histories lose their reservation (`preparation/pkg/ltr_recompute_native.py:184-218`), and the analysis rechecks the same relationships from raw memory receipts (`refine-logs/expert_saturation/experiments/admission_capacity/analyze_ltr_component_probe.py:86-157`).

The exact planner cannot be reconstructed from prestate because the runner did not save resolved `long_prefill_token_threshold`. All four cells are correctly marked `LIMITED_ACTUAL_PLAN_AND_RESOURCE_RECEIPTS`, `replayed_steps=0` (`analysis/analysis.json:4598-4601`, `analysis/analysis.json:9880-9883`). This limits planner replay; it does not invalidate the recorded plan/native-action/resource equality or the current claim ceiling.

### Denominator and nested wall accounting: PASS

The full measured episode identity is:

```text
wall = scheduler inclusive + engine excluding scheduler + outside engine
```

Independent recomputation gives:

| Cell | Wall | Scheduler inclusive | Decision nested | Engine excluding scheduler | Outside engine |
|---|---:|---:|---:|---:|---:|
| block0 fit | 27.670859 | 1.096043 | 0.248181 | 26.112081 | 0.462735 |
| block0 prefix | 28.196696 | 1.209376 | 0.261837 | 26.422978 | 0.564343 |
| block1 prefix | 27.708316 | 1.083290 | 0.252802 | 26.186458 | 0.438568 |
| block1 fit | 28.234745 | 1.120247 | 0.257707 | 26.574881 | 0.539616 |

Decision and wrapped scheduler costs are nested and never added to wall. This matches the stored component totals (`analysis/analysis.json:586-597`, `analysis/analysis.json:5281-5292`) and the report's accounting statement (`RESULTS_ADDENDUM.md:22`). Recovery spans overlap and are not summed as wall.

### Preemption-bounded residency and quantum boundary: PASS

Residencies close at the next actual preemption of the same request; completion-only endings are separate. In each fit run, 16 resumed residencies are re-preempted and the 0/1/2-new-output counts are 2/8/3. In each prefix run, 18 are re-preempted and the counts are 6/0/0 (`analysis/analysis.json:4577-4596`, `analysis/analysis.json:9858-9878`). The six prefix zero-output segments end after actual recompute positions 994, 995, 996, 997, 998, and 1998; this was independently derived from the raw selection intervals.

Each fit run has one observed boost epoch; each prefix run has eight. Every epoch has 10 selected calls: 3 pure recompute calls, 1 mixed recompute/decode call, and 6 decode-only calls, returning 7 new outputs. Every epoch records `exhausted_without_new_output=0` (`analysis/analysis.json:633-674`, `analysis/analysis.json:5330-5609`). The first-output comparison uses the successful enclosing engine-call return, so an output returned by the exhausting call counts as service (`refine-logs/expert_saturation/experiments/admission_capacity/analyze_ltr_component_probe.py:36-83`). This is a post-run boundary classification, not a pre-action predictor.

### Matched pairs, repeats, and first divergence: PASS

Both fit repeats have identical schedules and 32/32 identical full outputs; both prefix repeats do as well. Each fit/prefix pair has 25/32 identical full outputs. At scheduler step 406 in both blocks, the first action divergence begins from equal recorded counter/block/pool/running state and an equal 11,295-output returned prefix; fit-scan schedules the common 30 one-token decodes plus a 994-token recovery, while rank-prefix schedules only the common 30 decodes (`analysis/localization/REPORT.md:7-16`; `analysis/localization/analysis.json:8`, `analysis/localization/analysis.json:1004-1005`).

The recorded prestate is not a full engine or KV-byte checkpoint, and the analysis says so (`analysis/localization/REPORT.md:5`). Timing already differs before the first action in opposite directions, so later wall/completion differences are not wholly attributed to packing (`analysis/localization/REPORT.md:21`).

### Oracle and JIT claim boundary: PASS

No fixed-trace action Oracle is claimed. The report says the full-request Oracle was not run and local zero-output recompute cannot be summed into an achievable request benefit (`RESULTS_ADDENDUM.md:28`). The first-output retention mechanism is labeled an untested next hypothesis (`analysis/localization/REPORT.md:23`).

All four logs contain one generic `warning_once` fused-MoE JIT warning after monitor activation. The current results correctly do not assign it to the measured episode, infer machine-code recompilation, count occurrences, or explain wall drift (`RESULTS_ADDENDUM.md:24`). The supplied correction explains why the warning cannot support those claims (`../20260914_ltr_component_probe_r01/JIT_LOG_SEMANTICS_ADDENDUM.md:5-18`, `../20260914_ltr_component_probe_r01/JIT_LOG_SEMANTICS_ADDENDUM.md:40-50`).

## Claim Impact

- Four-cell exact request/wall/throughput/completion/max-ITL/preemption/recompute values: **supported**.
- Rank-prefix leaves more zero-output re-preemption segments and has worse maximum request ITL in both reversed pairs: **supported within this fixed old-d6 component regime**.
- Rank-prefix provides a stable throughput or average-completion improvement: **unsupported and correctly not claimed**.
- Cross-arm generation quality equivalence: **unsupported and correctly not claimed**.
- Full LTR, CPU-SWAP reproduction, action Oracle, statistical significance, business SLO, production P99, or problem-family NO-GO: **unsupported and correctly excluded**.
- First-new-output retention feasibility or net benefit: **UNRUN / hypothesis only**.

## Action Items

No correction is required for the current bounded `MEASUREMENT_ONLY` claims.

If a later claim requires exact planner reconstruction, record the resolved scheduler configuration, especially `long_prefill_token_threshold`, before execution. If a later claim requires stable performance or quality, use a fresh cohort/regime with controlled independent repeats and an explicit quality protocol. Preserve the JIT semantics addendum as the correction authority for all warning-based interpretations.

## Hashes of Primary Audited Inputs

| Path | SHA256 |
|---|---|
| `preparation/execution.tar.gz` | `1fea4a612af77a0062163cec33566765c2df25ade4021407756ec43b1734ef73` |
| `execution/readback.tar.gz` | `2825c8181d893af9482adbc6b5f07e027dea465f3631fb6036b690d75179cacc` |
| `RESULTS_ADDENDUM.md` | `4fe4693ea080219145dfc6a8b97bbcdcebf26d27a6a6afea6b5411aa113d7635` |
| `analysis/analysis.json` | `8019c1ed019296ee801c1c14deb87917019405f347e6a6b89e1170cab6050e06` |
| `analysis/localization/analysis.json` | `28365675d76a6fdd4cff8b48f2f636d1bab4254dc0cc4398aa63ca301eab0d43` |
| `analysis/localization/localize.py` | `a2a03fab7d9c2b55db3772d977f6e131cbde909fbb491a0e4bd2fe1425362099` |
| `refine-logs/expert_saturation/experiments/admission_capacity/analyze_ltr_packing_probe.py` | `7a88ba39a17e536f87b60788023b3042d5b9366853d32c20fd214fd5b8bbfb38` |
| `refine-logs/expert_saturation/experiments/admission_capacity/analyze_ltr_component_probe.py` | `342738deed09c6fb585137a81ed5cd3186abaa54445d0443dbbed3669ccbd1a6` |
| `refine-logs/expert_saturation/experiments/admission_capacity/analyze_prefix_cache_baseline.py` | `590ee20ee3648bb097abf82f29b23ac1e249ca588216bba4b0b13d713d18a473` |
| `../20260914_ltr_component_probe_r01/JIT_LOG_SEMANTICS_ADDENDUM.md` | `6764bad8d290142fe3ef7a06dca0e35db26706012952243b2e61dd4eb7d50a2c` |

Raw and decision hashes for all four cells are recorded in `EXPERIMENT_AUDIT.json`.
