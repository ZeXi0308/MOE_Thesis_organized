# Protection-yield triplet: request and peer follow-up

All three cells completed 128/128 requests. Q1 / yield / plain Q10 produced 1419.121 / 1413.993 / 1404.383 output tokens/s; mean flow was 37.704 / 38.153 / 38.041s. Maximum generation gap was 2.038 / 3.656 / 2.313s. Yield met the rate and mean-flow budgets against Q1 but failed the predeclared lower-max-gap criterion. Gap p95 was 1.772 / 2.857 / 1.924s.

The yield arm recorded 25 early releases. Each had an `ASYNC_LOAD_ADMITTED` native receipt, then a new output and final completion for the admitted head. Median admission-to-new-output was 46.5 ms. All 25 original protected targets also completed; their next output arrived a median 15.2 ms after release. Ten of the 25 releases were followed by at least one actual preemption of that original target (20 later preemptions in total); 14 admitted heads were later preempted (31 events). These are overlapping event follow-ups, not additive action costs.

## Six largest yield-arm token gaps

| Request suffix | Gap (s) | Interval (s) | Preemption → admission (s) | Admission → output (ms) | Role in yield actions |
|---|---:|---:|---:|---:|---|
| 0018772 | 3.656 | 74.83–78.49 | 3.607 | 48.3 | head at 81.02s |
| 0018510 | 3.461 | 74.81–78.27 | 3.411 | 48.3 | none |
| 0018206 | 3.445 | 74.18–77.63 | 3.391 | 53.1 | none |
| 0019990 | 3.441 | 74.40–77.84 | 3.384 | 55.5 | head at 80.61s |
| 0020531 | 3.437 | 74.63–78.06 | 3.381 | 55.2 | head at 73.71s, original_target at 81.24s |
| 0018626 | 3.160 | 73.97–77.13 | 3.110 | 48.8 | head at 80.49s, head at 81.24s |

The six gaps overlap within 73.97–78.49s. Their observed preemption-to-admission intervals account for 98.36–98.66% of each token gap. Five preemptions match ordinary capacity-rotation victim commits; the sixth request was naturally preempted after its ordinary ten-output protection release. Within the union window the log has 13 protection starts, 12 extensions, 12 `OUTPUT_GOAL_REACHED` releases, and zero yield actions. Some of these requests took part in yield actions before or after their longest gap, so this does not identify an equal-state counterfactual.

## Interpretation

The early-release gate can admit a paused head and produce a new output on this observed trajectory. It did not improve the full-cohort tail: the worst delays were concentrated in later recovery after actual preemption, during a window without yield actions. The visible preemption-to-admission interval is not wholly provable as queue waiting from these sparse logs. This one seen-input triplet does not support a performance claim or a new parameter choice.

Source: `A_PROTECTION_YIELD_TRIPLET_RESULT_R01_20261001.json` SHA-256 `907afbec34e81b92e3ff934654efe319f60869a93c20ed5111a8f1b1d0180cbf`; detailed rows and original hashes: `A_PROTECTION_YIELD_TRIPLET_TARGET_PEER_R01_20261001.json`.
