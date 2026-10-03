# StableBatch Observable Selector Experiment Audit

**Date:** 2026-08-10  
**Mode:** Research experiment / rapid exploration  
**Auditor:** GPT-5.6-Sol ultra (fresh same-family agent, read-only, provisional)  
**Project:** StableBatch MaxGate-v1 observable selector pilot

## Overall Verdict: PASS_WITH_LIMITATIONS

**Integrity status:** PASS  
**Blocking findings:** none (P0: 0, P1: 0)  
**Decision:** the experiment can continue; no integrity repair is required.

The frozen descriptive verdict is reproduced exactly: within this single-RTX-5090, 16-document, 240-cell, offline proxy run, MaxGate-v1 did not outperform the predeclared balanced-shuffle selector (`A_O=-3`, `A_S=3`). This supports `WEAKENS_MAXGATE_V1_NOT_BETTER_THAN_SHUFFLE` for this pilot only.

## Decisive Checks

### A. Reference provenance: PASS

- Inputs come from hash-bound fixed token windows (`run_single_contribution_pilot.py:312`).
- The all-M1 reference is explicitly a self-supervised model-internal counterfactual, not external ground truth (`configs/observable_selector_pilot_v1.json:87`).
- Rewards are direct downstream route-set distance changes, `D_U-D_policy` (`run_observable_selector_pilot.py:822`).
- The result labels itself `self_supervised_proxy_offline_same_cell_single_action_value_replay` (`summary.json:12`).

### B. Metric and normalization integrity: PASS

- Raw reward totals are accumulated directly; rates use the fixed 240-cell denominator (`run_observable_selector_pilot.py:897`).
- Raw totals and positive/tie/negative counts are retained (`summary.json:25`, `summary.json:116`).
- No model-output maximum or other self-normalizing denominator is used.

### C. Artifact and result integrity: PASS

- Acceptance is correctly marked smoke-only and scientifically ineligible (`../observable_selector_acceptance_20260810_run02/RUN_STATUS.json:1`).
- The formal request binds the exact acceptance artifacts, runner, base runner, config, and frozen lock (`run_request.json:2`, `run_request.json:28`).
- Both manifests match all current file sizes and SHA-256 hashes; remote and local copies match for all 27 compared paths.
- Formal status is `COMPLETE`, and the manifest binds all result ledgers and the atomic status (`RUN_STATUS.json:1`, `MANIFEST.json:3`).
- Independent raw-ledger recomputation found zero mismatches (`../../audits/observable_selector_20260810_run01/INDEPENDENT_RECOMPUTE.json:11`).

### D. Selection and control integrity: PASS

- Observable scanning records gate weights and expert IDs without decoding M1/M64 outcomes (`run_observable_selector_pilot.py:225`).
- MaxGate receives only an allowlisted observable view; forbidden outcome fields fail closed (`run_observable_selector_pilot.py:290`).
- Assignments and `POLICY_SELECTION_LOCK.json` are written before result extraction (`run_observable_selector_pilot.py:1296`).
- The balanced-shuffle rank, arm order, and side-call order regenerate exactly from frozen seeds; every rank occurs 30 times.
- The frozen verdict recomputes from raw rows with zero mismatches: 35 opportunity cells across 8 documents; O counts `13/209/18`; S counts `10/220/10`; `A_O=-3`; `A_S=3`.
- Targeted tests pass 15/15 (10 observable-selector tests plus 5 base-runner tests).

### E. Scope: WARN (P2 only)

- Scope is one OLMoE revision, one RTX 5090/software stack, 16 document windows, 240 victim-layer cells, one formal run, and one predeclared balanced-shuffle realization.
- The three repeats per arm establish bitwise artifact stability; they are not independent scientific seeds.
- This supports the frozen descriptive pilot verdict, not a population-level statistical statement.

### F. Evaluation type: PASS

Primary type: `causal_model-internal_measurement`.  
Reference type: `self_supervised_proxy`.

It is real GPU/model execution, but it is not serving, expert-parallel, multi-GPU, prevalence, external-correctness, or generalization evidence.

## P0/P1 Findings

None. No issue was found that would change the current result, break the selector/shuffle control, introduce outcome leakage, or make the frozen descriptive conclusion uninterpretable.

## P2 Notes

1. Keep the single-GPU/local-MoE/offline-proxy boundary on downstream claims.
2. Do not relabel the all-M1 counterfactual as external ground truth.
3. Fresh document-level replication is needed only if the claim is later elevated from this descriptive pilot to a general statistical claim; it is not required to accept the current result.

## Claim Impact

- `WEAKENS_MAXGATE_V1_NOT_BETTER_THAN_SHUFFLE`: **supported for the frozen pilot**.
- “MaxGate-v1 improves over the matched shuffled selector”: **contradicted in this run**.
- “MaxGate is statistically inferior in general”: **unsupported**.
- Serving, dynamic batching, EP, multi-GPU, prevalence, or external-correctness claims: **unsupported**.

## Review Assurance

- Reviewer model: `gpt-5.6-sol`
- Reasoning: `ultra`
- Review independence: `same-family`
- Acceptance status: `provisional`
- Trace: `.aris/traces/experiment-audit/2026-08-10_run06/`
