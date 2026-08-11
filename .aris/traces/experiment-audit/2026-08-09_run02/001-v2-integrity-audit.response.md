## A. Ground Truth Provenance

Status: PASS

Evidence:

- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:3-9`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:982-988`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:991-1107`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:893-980`
- `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_230235/pilot_protocol.json:16-32`
- `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_230235/pilot_results.json:15027-15116`

Details:

- There is no dataset GT, human GT, real model trace, or measured GPU service target.
- Requests, routes, deadlines, replicas, and the `10/14/20 µs` service curve are deterministic synthetic fixtures.
- The Oracle is an exact optimum only inside the declared simulator and is labeled `future_known_upper_bound`, `simulation_only`, and `scientific_result_eligible=false`.
- `layer_ready_us` values are explicitly schema-only placeholders and are not used as counterfactual timing.
- Classification: `simulation_only`. No disguised real GT or model-derived pseudo-GT was found.

## B. Score / Decision Integrity

Status: PASS

Evidence:

- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:45-59`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:971-980`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:1120-1279`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:1325-1369`
- Per-cell summaries: `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_230235/pilot_results.json:1397-1436`, `2832-2871`, `4501-4540`, `6170-6209`, `8391-8430`, `10587-10626`, `12788-12827`, `14984-15023`
- Final decision: `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_230235/pilot_results.json:15028-15072`

Independent recomputation produced:

| Cell | Immediate | EDF | MaxRows | QueueLocal | Frontier | Sham | Oracle | Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned/aligned/loose | 120 | 120 | 120 | 120 | 120 | 120 | 120 | 0 |
| aligned/aligned/tight | 120 | 120 | 120 | 120 | 120 | 120 | 120 | 0 |
| aligned/staggered/loose | 149 | 149 | 123 | 123 | 123 | 123 | 123 | 26 |
| aligned/staggered/tight | 149 | 149 | 123 | 123 | 123 | 123 | 123 | 26 |
| crossed/aligned/loose | 168 | 168 | 186 | 186 | 158 | 186 | 152 | 16 |
| crossed/aligned/tight | 168 | 152 | 158 | 158 | 158 | 180 | 152 | 16 |
| crossed/staggered/loose | 149 | 149 | 155 | 155 | 155 | 183 | 149 | 0 |
| crossed/staggered/tight | 149 | 149 | 155 | 155 | 155 | 177 | 149 | 0 |

For the four nonzero-headroom cells, capture vectors are:

- Immediate: `[0, 0, 0, 0]`
- EDF: `[0, 0, 0, 1]`
- MaxRows: `[1, 1, -1.125, 0.625]`
- QueueLocal: `[1, 1, -1.125, 0.625]`
- Frontier: `[1, 1, 0.625, 0.625]`
- Sham: `[1, 1, -1.125, -0.75]`

Recomputed aggregate statistics:

- Immediate: median `0`, worst `0`, aggregate flow `634`
- EDF: median `0`, worst `0`, aggregate flow `618`
- MaxRows: median `0.8125`, worst `-1.125`, aggregate flow `590`
- QueueLocal: median `0.8125`, worst `-1.125`, aggregate flow `590`
- Frontier aggregate flow: `562`
- Sham aggregate flow: `612`
- Oracle aggregate flow: `550`

Other recomputed decision fields:

- Fixed QueueLocal median capture: `0.8125`
- Frontier-minus-reference vector: `[0, 0, 1.75, 0]`; median `0`
- Identity-gap vector: `[0, 0, 1.75, 1.375]`; median `0.6875`
- Frontier miss deltas: `[0, 0, -1, 0]`
- Applicable eligible cells: `4`
- Cellwise envelope captures: `[1, 1, 0, 1]`; median `1.0`

The envelope is computed only for diagnostics and is absent from the positive-decision expression. Zero headroom becomes `null`, not `1.0`; negative capture remains unclamped.

`queue_local_credit` is fixed across all cells and is explicitly disclosed as selected after v1 inspection and not preregistered. The output is correctly bounded as `AUDIT_CORRECTED_DESCRIPTIVE_REANALYSIS`. It does not manufacture support: median Frontier increment is `0 < 0.1`, yielding `FRONTIER_SIGNAL_NOT_SUPPORTED`.

Non-invalidating wording hardening: the latent `SIMPLE_BASELINE_SUFFICIENT` branch at runner lines `1248-1252` should be renamed to include `ON_FROZEN_SIMULATION_CELLS` if ever emitted.

## C. Result / Provenance Consistency

Status: PASS

Evidence:

- `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_230235/RUN_STATUS.json:1-7`
- `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_230235/MANIFEST.json:1-22`
- `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_230235/pilot_summary.md:1-39`
- `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_230235/pilot_protocol.json:1-42`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:1373-1517`

Details:

- Fresh tests: `13/13 PASS`.
- Fresh CLI replay under the manifest-recorded CPython `3.14.6` generated all five artifacts byte-for-byte identically in `/tmp/frontiercredit_v2_audit_replay_230235`.
- Output-directory independence is therefore confirmed; no output path appears in the serialized artifacts.
- All four manifest artifact hashes and all three source hashes match current bytes.
- Current HEAD matches `8fe396078ca365afb9ea5d06d8b88c9c01e7a825`.
- Runtime fields match the current executable, implementation, version, and platform.
- The old commit does not falsely claim source reachability: the manifest explicitly says the source hashes bind uncommitted/untracked code.
- `RUN_STATUS`, protocol, results, summary prose, decision numbers, and claim ceiling are mutually consistent.
- A cap-1 run raised `UNSOLVED_EXACT_STATE_LIMIT` and created no output directory.

## D. Dead Code / Scope

Status: WARN

Evidence:

- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:25-40`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:1033-1117`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:1282-1369`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/core.py:653-1199`
- `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_230235/pilot_results.json:15074-15116`

Details:

- Actual scope: eight deterministic factorial cells, two requests/cell, two steps, two layers, top-2, two pinned replicas.
- There is one deterministic trajectory per policy/cell and no stochastic seed or replicate axis.
- Seven records per cell—five online policies, one sham, one Oracle—give 56 stored replays.
- Every claimed policy, metric, sham, and Oracle path is executed and appears in results.
- Much of shared `core.py` after line 653 is not used by this pilot. This is shared-module dead scope, not phantom evidence.
- `deadline_miss_delta_must_be_nonpositive` is serialized but the implementation directly hardcodes `<= 0`; changing that configuration value would have no effect.
- `formal_gate2_unchanged=true` is metadata, not a derived verification of the separate formal Gate-2 implementation.
- Scope language is honest: `simulation_only`, `scientific_result_eligible=false`, and the claim ceiling expressly excludes prevalence, real models, GPU serving, EP/NCCL/RDMA, and production SLOs.

## E. Causal / Sham Integrity

Status: PASS

Evidence:

- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:397-480`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:616-665`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:668-840`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/test_frontiercredit_full_dag_pilot.py:99-213`
- Sham counters in the eight summary blocks cited under B.

Details:

- A successor layer/step is revealed only after all predecessor top-k siblings complete plus `combine_us`; changing actions therefore changes successor readiness.
- Policies receive only `DecisionView`; it is constructed from revealed active nodes and currently ready/running/completed state. No future suffix is exposed.
- In staggered cells at `t=0`, only `r0` is present; changing `r1` identity and route leaves the sham view and decision unchanged.
- Sham rotation uses only currently revealed request IDs at the identical step/layer and is a no-op with fewer than two visible groups.
- Sham changes only the observed sibling grouping. Physical queues, target replicas, routes, service timing, and episode nodes remain unchanged.
- Independently recomputed decision/member counters match all eight artifacts: `8/16`, `8/16`, `8/16`, `8/16`, `19/38`, `18/36`, `18/36`, `17/34`.
- All four eligible cells are sham-applicable, exceeding the threshold of two.
- No subtler future-identity or future-route leak was found.

## F. Oracle / Action-Space Integrity

Status: PASS

Evidence:

- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:92-99`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:483-507`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:531-586`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:771-968`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py:991-1030`
- `/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/test_frontiercredit_full_dag_pilot.py:215-291`
- Oracle ledger blocks: `/Users/leandrozhao/Desktop/毕设论文资料/artifacts/frontiercredit_pilot/20260809_230235/pilot_results.json:366-515`, `1801-1950`, `3340-3495`, `5009-5164`, `6678-6939`, `8899-9160`, `11095-11356`, `13296-13557`

Details:

- Policies and Oracle use the same full `_eligible_queues` set across every idle executor; the previous replica-order pruning is absent.
- Each action flushes the entire currently ready queue or performs one legal bounded hold.
- Oracle enumerates every queue action plus hold, memoizes the complete settled state, and fails closed at the state cap.
- Independent exhaustive enumeration reconstructed eligibility directly from ready state and executor occupancy without calling `_decision_queues`: 232,732 settled states, 87,036 decision states, 14,792 multi-idle states, maximum four simultaneous queue choices.
- All eight stored Oracle records matched flow, tardiness, misses, request completion, launches, service, action tokens, and state count.
- All detailed ledgers independently replayed exactly and conserved every node.

Hardening gaps do not alter the current result:

- The repository action-space test checks only initial states and compares `_decision_queues` directly with `_eligible_queues`; add an all-reachable-state independent regression.
- The built-in ledger assertion checks flow, tardiness, and misses, but should also assert request-completion mapping, launches, and total service.
- "On any idle executor" means any eligible queue pinned to an idle target executor, not arbitrary queue-to-executor reassignment. Rewording would prevent scope misreading.

## Final disposition

- Overall verdict: WARN
- Integrity status: `warn`
- Evaluation type: `simulation_only`
- Reason code: `simulation_only_posthoc_descriptive_scope`
- Acceptance status: `provisional_same_family`

Claim impacts:

- C1 — simulator/artifact existence and internal consistency: SUPPORTED at the deterministic artifact level.
- C2 — corrected descriptive `FRONTIER_SIGNAL_NOT_SUPPORTED` on the frozen eight cells: SUPPORTED, narrowly. It means the present FrontierCredit formulation did not clear its descriptive continuation rule in these cells.
- C3 — unseen cells, real models, GPU/serving, EP/NCCL/RDMA, production, or general mechanism claims: UNSUPPORTED / UNVERIFIED.

Actions:

- Required before any confirmatory/general claim: preregister comparator, thresholds, and unseen workloads, then run a genuinely untouched confirmation set.
- Required before C3: obtain real route/service evidence and the corresponding GPU/serving or multi-GPU EP measurements.
- Required for non-provisional acceptance: independent/cross-family review.
- Optional hardening: add exhaustive reachable-state regression, extend the Oracle replay assertion, clarify pinned-executor wording, make the deadline-threshold flag operational or remove it, and archive a test command/exit receipt.
- No workspace artifact requires correction to retain the narrow C1/C2 conclusion.

Prior memory was used only to preserve the older "formal gates remain open" boundary; every current-result finding above was verified live.
