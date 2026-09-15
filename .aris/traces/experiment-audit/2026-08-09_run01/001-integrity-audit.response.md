# Experiment Integrity Audit — FrontierCredit Full-DAG Pilot

**Auditor:** fresh same-family reviewer  
**Acceptance:** provisional  
**Audit date:** 2026-08-09

## Verification performed

- Read both evaluation scripts and all listed artifacts.
- Parsed and validated all 13,386 lines of `pilot_results.json`.
- Re-ran the pilot; all five generated artifacts were byte-identical.
- Verified all four manifest file hashes and the runner hash.
- Recomputed all per-cell metrics, captures, medians, deadlines, node conservation, and action timings.
- Ran the test suite: **9/9 PASS**.
- Independently enumerated the unrestricted eligible-queue action space; the stored Oracle optimum matches in all eight current cells.

## A. Ground Truth Provenance

**Status: PASS**

**Evidence**

- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:3-10`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:858-875`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:888-917`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:920-994`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:840-855`
- `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/RUN_STATUS.json:2-4`

**Details**

The evaluation uses hand-constructed deterministic request DAGs and a hard-coded synthetic service curve. Its only reference is an exact planning Oracle over the simulator. Schema-only `layer_ready_us` placeholders are explicitly not consumed as counterfactual timing.

There is no dataset GT, model-generated GT, human evaluation, real-model trace, or physical measurement. The artifacts consistently label the evaluation `simulation_only` and `scientific_result_eligible=false`. No model-derived reference is disguised as ground truth.

**Evaluation type: `simulation_only`**

## B. Score Normalization

**Status: FAIL**

**Evidence**

- Capture denominator and zero handling: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:858-866`
- Per-cell outcome-based baseline selection: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:1007-1029`
- Median over those independently selected baselines: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:1043-1067`
- Four eligible-cell capture vectors:
  - `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/pilot_results.json:4047-4063`
  - `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/pilot_results.json:5554-5570`
  - `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/pilot_results.json:7514-7530`
  - `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/pilot_results.json:9449-9465`
- Headline decision: `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/pilot_results.json:13345-13360`

**Details**

The basic capture formula is sound:

`(immediate_flow - candidate_flow) / (immediate_flow - oracle_flow)`

Raw flow/tardiness/miss values are retained, zero headroom returns `null`, and scores are not clipped. Negative captures such as `-1.125` appear in the result.

The failure is the headline aggregation. For every cell, the evaluator uses that same cell's observed flow to select whichever of four simple policies performed best, then takes the median of those cell-wise maxima. This is an outcome-selected Oracle portfolio, not one "strongest simple policy."

Independent recomputation over the four eligible cells gives:

| Baseline | Captures | Median |
|---|---|---:|
| immediate | 0, 0, 0, 0 | 0.0000 |
| edf | 0, 0, 0, 1 | 0.0000 |
| max_rows | 1, 1, -1.125, 0.625 | 0.8125 |
| queue_local_credit | 1, 1, -1.125, 0.625 | 0.8125 |
| Cell-wise post-hoc best | 1, 1, 0, 1 | **1.0000** |

No fixed simple baseline reaches the frozen `0.90` threshold. The reported `1.0` crosses it only because baseline identity is selected separately using each evaluation cell's outcome. Therefore `SIMPLE_BASELINE_SUFFICIENT` is not supported as a single-baseline claim.

## C. Result File Existence and Consistency

**Status: WARN**

**Evidence**

- Status: `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/RUN_STATUS.json:1-6`
- Manifest: `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/MANIFEST.json:1-11`
- Narrative values: `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/pilot_summary.md:3-25`
- Result status and decision: `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/pilot_results.json:13344-13385`
- Imported, unbound dependency: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:22-37`
- Manifest-writing scope: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:1240-1251`

**Details**

All requested files exist. Every narrative value matches the JSON. Status, verdict, evaluation type, and scientific eligibility agree across files. All declared hashes match. A fresh run reproduced every generated file byte-for-byte.

The warning is provenance completeness: the manifest hashes the runner but not imported `core.py`, even though `Contribution`, `ServiceCatalog`, validation, and service interpolation come from it. It also does not bind the test script, Python/runtime version, or repository commit. Current reproducibility is verified, but the manifest alone is not a complete executable provenance capsule.

## D. Dead Code Detection

**Status: PASS**

**Evidence**

- Policy construction: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:997-1004`
- Policy, sham, and Oracle execution: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:1090-1127`
- Metric and capture consumption: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:589-609`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:1007-1040`

**Details**

Every implemented policy, the identity sham, exact Oracle, raw metric function, and capture function is executed and represented in the results.

Minor hygiene findings do not invalidate the result:

- `POLICY_NAMES` at lines 42-49 is unused and therefore does not enforce policy-set completeness.
- The four-row service-curve point at lines 871-874 is unreachable with only two requests per cell; recorded flushes use only one or two rows.

## E. Scope Assessment

**Status: WARN**

**Evidence**

- Frozen grid construction: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:920-994`
- One execution per generated cell: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:1090-1104`
- Protocol dimensions: `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/pilot_protocol.json:1-22`
- Only four eligible cells: `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/pilot_results.json:13345-13350`
- Evidence ceiling: `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/pilot_summary.md:23-25`

**Details**

Actual scope:

- 8 deterministic configurations.
- 2 requests per cell.
- 2 decode steps × 2 layers × top-2 = 16 nodes per cell.
- One run per configuration.
- No random seeds, repeats, uncertainty intervals, real traces, models, or hardware.
- Only 4/8 cells have nonzero Oracle headroom.

The narrative correctly calls this a development-only simulation and explicitly rejects broader claims. The warning concerns the verdict language: "strongest simple policy" obscures a per-cell post-hoc portfolio and median aggregation, including one eligible cell where the best simple capture is `0.0`.

The protocol file also omits the decision thresholds and selector semantics; those appear only in the result. There is no independent evidence that this aggregation was frozen before inspecting outcomes.

## F. Causal and Oracle Integrity

**Status: FAIL**

**Evidence**

Successor readiness:

- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:402-455`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/test_frontiercredit_full_dag_pilot.py:93-116`

Online view:

- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:655-727`

Identity-sham construction:

- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:612-624`
- Staggered future arrival: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:928-938`
- Declared visibility contract: `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_223803/pilot_protocol.json:9-11`

Shared transition:

- Policy path: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:730-785`
- Oracle path: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:788-855`
- Whole-ready flush: `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:557-582`

Oracle pruning and tests:

- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:489-503`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/test_frontiercredit_full_dag_pilot.py:118-157`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/test_frontiercredit_full_dag_pilot.py:174-202`

**Details**

- **Successor readiness: PASS.** A successor is revealed only after all current top-k siblings complete plus the combine delay. Different actions therefore change downstream readiness.
- **Actual FrontierCredit future visibility: PASS.** The normal view iterates only revealed nodes and does not expose unrevealed route suffixes.
- **Identity sham: FAIL.** `_observation_map` constructs its swap from every request ID in the full episode. In a staggered cell, at `t=0` only `r0` has arrived, yet slot 1 of `r0` is labeled with future request ID `r1`. A direct probe produced:

  `now=0.0, arrived=('r0',), sham observed groups=[('r0',0,0), ('r1',0,0)]`

  Thus an online sham view exposes an unrevealed future request identity, contradicting `future arrivals/routes hidden`. The existing sham test uses an aligned-arrival episode, so it cannot catch this.
- **Identity-only physical mutation: PASS.** Sham changes the observation grouping but not queues, nodes, route assignments, or state transitions.
- **Shared transition: PASS.** Policies and Oracle call the same settle, whole-ready flush, and bounded-hold functions.
- **Current Oracle numeric exactness: PASS, implementation caveat.** Independent exhaustive enumeration without `_decision_queues` replica pruning produced the same optimum for all eight cells. However, the comment that lowest-replica canonicalization cannot delete a physical schedule is false generally: it removes schedules that launch a higher replica now while deliberately leaving a lower replica idle. Existing tests prove only a tiny expectation and dominance in one cell, not equivalence over the declared action space.

Because the sham violates a declared online-causality invariant, F is a failure even though it does not change the current simple-baseline verdict.

## Overall Verdict

**Overall verdict: FAIL**  
**Integrity status: fail**  
**Evaluation type: simulation_only**  
**Reason code: `OUTCOME_SELECTED_BASELINE_AND_SHAM_FUTURE_ID_LEAK`**

This is a same-family provisional review. Deterministic file consistency is strong, but the evaluation logic does not justify its baseline-sufficiency headline, and one declared causal-visibility invariant is false.

## Claim Impacts

| Claim | Impact |
|---|---|
| **C1: full-DAG simulator result exists and is internally consistent** | **Needs qualifier.** Existence, hashes, raw metrics, DAG replay, and byte-for-byte reproduction are supported. Clean protocol-level consistency is not accepted because the sham violates declared online visibility and the manifest omits an executable dependency. |
| **C2: strongest simple baseline is sufficient in the frozen 8-cell simulation** | **Unsupported as written.** The `1.0` score belongs to a post-hoc cell-wise best-of-four envelope. The best fixed simple baseline has median capture `0.8125`, below `0.90`. A narrower claim—"the cell-wise simple-policy Oracle envelope reaches median capture 1.0"—is supported. |
| **C3: any real-model, GPU, serving, EP/NCCL/RDMA, natural-workload, production, or general claim** | **Unsupported.** The artifact explicitly marks itself simulation-only and scientifically ineligible. |

## Required Actions

1. Replace the per-cell post-hoc baseline selector with either one baseline frozen before evaluation, or an explicit causal selector that chooses among simple policies using only revealed online state. Report every fixed policy's median, aggregate raw flow, and worst-cell behavior.
2. Rename any retained envelope metric to `cellwise_best_simple_oracle_envelope`; do not call it a single policy or baseline.
3. Fix the identity sham so no label is derived from an unarrived request. Use causally assigned opaque sham IDs and add a staggered-arrival test asserting that the view contains no future identity.
4. Either remove lowest-replica pruning or explicitly narrow the action-space contract and prove equivalence. Add an all-eight-cell unrestricted-enumeration regression test.
5. Add the decision thresholds, selector definition, `launch_cost_us`, `core.py` hash, test-script hash, runtime version, and repository commit to the frozen protocol/manifest.
6. Emit a replayable Oracle ledger with action timestamps, queues, node IDs, and service times.
7. Preserve this artifact unchanged and rerun into a new directory after fixes.
