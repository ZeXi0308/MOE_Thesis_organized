# Cohort3 recovery lifecycle accounting

## Verdict

**The exact expensive short-recovery gap does not survive the strongest baselines on this holdout.** Native and `most_output` both have zero residencies that are re-preempted after only 1–2 new output tokens. Fit-scan still has 11 such residencies per cell and `guard_residual` has 5, so this formulation cannot be presented as residual headroom after native/most.

This does not make `most_output` free of repeated recovery. It has 18 `served_then_discarded` residencies and 64,835 recompute positions per cell; its closed service intervals are at least 4 outputs rather than 1–2. Native has only 6 recovery residencies, all of which serve to completion in this accounting. These observations do not establish a performance winner.

## Per-cell ledger

Both block executions have identical lifecycle counts for each arm.

| Arm | Zero-output then re-preempted: residencies / positions | 1–2-output then re-preempted: residencies / outputs / positions | Total recompute positions | Served then discarded: residencies / positions | Served to completion: residencies / positions |
|---|---:|---:|---:|---:|---:|
| `native` | 0 / 0 | 0 / 0 / 0 | 21,426 | 0 / 0 | 6 / 21,426 |
| `most_output` | 0 / 0 | 0 / 0 / 0 | 163,764 | 18 / 64,835 | 26 / 98,929 |
| `fit_scan` | 2 / 5,973 | 11 / 14 / 37,851 | 75,742 | 14 / 48,329 | 6 / 21,440 |
| `guard_residual` | 0 / 0 | 5 / 5 / 17,505 | 89,448 | 18 / 64,024 | 7 / 25,424 |

The first, fourth, and fifth outcome columns are mutually exclusive and conserve total recomputation. The 1–2-output column is a subset of `served_then_discarded` and must not be added again. No arm has a `not_resumed`, censored-no-output, or censored-after-output residency.

## Scope

- Input campaign: `20260914_recovery_holdout_comparison_r01`, eight completed native-serving cells: four arms × two block-order repeats.
- Workload: one shared cohort3 of 32 unique requests/documents/prompts per cell; 256 completed request executions and 262,144 output tokens in total. The repeats are not independent workload samples.
- Holdout separation: cohort3 has zero request-ID, document-ID, and prompt-token-hash overlap with the old lifecycle cohort checked here.
- Runtime boundary: APC off, native recompute, 6,656 usable GPU blocks, no host KV offload. Output time is host receipt immediately after synchronous `LLMEngine.step()` return, not client/network receipt.
- Evidence type: `REANALYSIS_OF_NATIVE_SERVING`, `MEASUREMENT_ONLY`. The analysis reads each policy's actual raw trajectory; it does not replay one policy over another policy's future state.

## Claim boundary

Measured: actual recovery positions, new outputs before the next same-request preemption, state invalidation, next-residency prefix reexecution, and terminal completion buckets.

Not measured by this lifecycle report: isolated GPU recompute time, recoverable milliseconds, quality, full-request latency/throughput tradeoff, method benefit, Oracle headroom, or novelty. In particular, `most_output` eliminating 1–2-token intervals cannot be converted into a performance win, and its larger total recompute count cannot be converted into a latency loss without the separate matched full-request analysis.

Strongest baseline result: native and `most_output` cover the exact 1–2-output short-recovery existence claim on cohort3. What remains is a different question: whether the longer service granted by `most_output` is worth its additional total recovery work and its effect on other requests.

Failure category for the proposed residual: `FORMULATION_NOT_RESIDUAL_AFTER_STRONG_BASELINE` on this one fixed-shape cohort3. This does not falsify recovery-service tradeoffs in other arrival, EOS, context-length, model, or memory regimes.

The smallest next step is to join this ledger, without adding its overlapping diagnostics, to the already-produced matched full-request tradeoff for these same eight cells. No new GPU run is required for that check.
