# Independent Audit: oracle_action_sweep_20260810_run01

## Bounded verdict

**PASS — 0 P0, 0 P1.**

This is an independent deterministic recomputation from the sealed `cell_results.jsonl`. It did not call or import the runner's `classify_results`, did not rerun the model or GPU experiment, and did not modify the original run or its `MANIFEST.json`.

## Bound sources

| Artifact | SHA-256 |
|---|---|
| `cell_results.jsonl` | `2ab83c452ec41b0e949bb13c0bdf4f599e9d3e0d201e86f800c6dbbd72981654` |
| `summary.json` | `fde35cfb609e07ae2d9439a0227fd30e97807e05df4456fe7c300e4f12f6bd80` |
| `MANIFEST.json` | `921ddee7a3b5147db76b147ba14e049241c62feabb27e16acedb59ec4cc491b2` |

The MANIFEST listed 12 run artifacts. Every listed file's size and SHA-256 matched. `RUN_STATUS.json` was `COMPLETE`, `scientific_result_eligible=true`, and its verdict agreed with `summary.json`.

## Cell and action closure

- 240 rows, 240 unique cells, contiguous indices `0..239`.
- Exactly 8 candidate actions per cell, ranks `0..7`: 1,920 action results total.
- For all 1,920 actions, `reward = unprotected_distance_vs_R - distance_vs_R`; mismatch count: 0.
- For all 240 cells, the recomputed best reward matched, and ties selected the lowest top-k rank; mismatch count: 0.
- Abstaining behavior matched `protect best_rank iff best_reward > 0`; mismatch count: 0.
- There were 33 positive cells and 33 selected-positive confirmation records. All 33 were `PASS`, matched the selected rank and changed-route record, and no non-positive cell carried a confirmation.
- MaxGate used rank 0 in every cell and independently summed to **-3**.
- Frozen shuffle independently summed to **3**; every rank was selected exactly 30 times.

## Decisive metric recomputation

| Metric | Independent formula | Value |
|---|---|---:|
| No intervention | `sum_cell(0)` | 0 |
| MaxGate-v1 | `sum_cell(reward(rank=0))` | -3 |
| Frozen shuffle | `sum_cell(reward(source_shuffled_rank))` | 3 |
| Forced oracle | `sum_cell(max_rank reward)` | 37 |
| Abstaining oracle | `sum_cell(max(0, max_rank reward))` | 37 |
| Positive cells / victims | `count(best>0)` / distinct victim count | 33 / 8 |
| Full-restoration selected cells | selected positive action with distance 0 | 31 |
| Unprotected distance | `sum_cell(unprotected_distance_vs_R)` | 43 |
| Remaining distance | `43 - 37` | 6 |
| Recovery fraction | `37 / 43` | 0.8604651162790697 |
| Uniform-rank random, all cells | `sum(all 1,920 rewards) / 8` | `9/4 = 2.25` |
| Budget-matched conditional random | sum of per-positive-cell rank means | `39/2 = 19.5` |
| Budget-matched global random | `33 * sum(all rewards) / (240 * 8)` | `99/320 = 0.309375` |
| Oracle advantage over conditional random | `37 - 39/2` | `35/2 = 17.5` |
| Oracle advantage over global random | `37 - 99/320` | `11741/320 = 36.690625` |

The positive best-rank counts matched `{0:13, 1:9, 2:4, 3:6, 4:0, 5:0, 6:0, 7:1}`. All 16 per-victim aggregates and all eight budget-curve entries (`1, 4, 8, 16, 32, 60, 120, 240`) matched `summary.json` exactly.

## STRONG gate

| Gate | Threshold | Recomputed | Result |
|---|---:|---:|---|
| Advantage over budget-matched conditional random | >= 8.0 | 17.5 | PASS |
| Advantage over budget-matched global random | >= 8.0 | 36.690625 | PASS |
| Abstaining-oracle recovery fraction | >= 0.25 | 0.8604651162790697 | PASS |
| Distinct victims with positive oracle reward | >= 4 | 8 | PASS |

All four gates pass. The independently derived bounded verdict is `STRONG_ORACLE_ACTION_VALUE_SIGNAL`.

## Findings and boundary

- P0: none.
- P1: none.

This PASS establishes deterministic artifact-level closure for this exact sealed run. It does not independently replay the confirmation actions or model execution, and it does not establish online-selector value, serving behavior, expert-parallel behavior, prevalence, or generalization beyond this run.
