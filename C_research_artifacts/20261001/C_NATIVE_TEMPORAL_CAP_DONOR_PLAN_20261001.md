# One native cap-donor comparator pilot

## Question

The frozen fresh-cohort retirement test is complete and supports both primary comparisons against full-bound FIFO. Its native tradeoffs remain. Before expanding the candidate, test a precise temporal-reuse comparator: can one fully reserved decoder promise its known-cap release to a FIFO head, with actual quota transfer, and what complete-request cost results?

The [CPU rule](C_TEMPORAL_CAP_DONOR_NOTE_20261001.md) and current global retirement envelope have incomparable acceptance sets. The global rule exploits staggered resident retirement but charges new requests in full. The donor rule can account for the new receiver's bounded growth. This is a partial CacheOPT-inspired mechanism ablation, **not a complete CacheOPT reproduction or a resolved closest-method comparison**.

## Frozen policy and one measurement

Run one `native_temporal_donor_1` cell on the existing new GPU and shared lock. Use the exact already tested `20261001_c_retirement_fresh_inputs_v1` bytes, all128 requests, naturalEOS/max1024, fixed32 sequence slots/1024-token batch/4096usable KV blocks, nooffload, pinned runtime, samewarmups, seed and20/4target. This cohort is now viewed; this is a development comparator pilot. No new arrival draw, input filter or target selection.

At an all-pure-decode boundary, admit the ordinary full-bound FIFO prefix. For its blocked head, let `F` be remaining uncommitted logical quota, `P` its prompt length, `Q` its full upper bound. Choose the first eligible existing decoder with horizon `H=max_tokens-output_tokens` and full quota `Qd` satisfying `H+1 <= 16F-P` and `Qd >= Q-F`. Allocate initial logical quotaF, record one promise, then stop new admission until transfer/cancellation and the next all-decode boundary. Native code remains the physical allocator.

All prior decoders remain a protected prefix and must schedule/produce one token per call through actual completion. On the donor's genuine native free, transfer `Q-F` to the receiver before any later admission. If the receiver finishes first, release only its committed quota and cancel the promise. Preserve both same-call free orders. Only one promise may be outstanding, and any former partial receiver remains ineligible as a donor; this is deliberately stricter than merely forbidding concurrent chains. These restrictions and the absence of CacheOPT's predictor, spatial reuse, SLO allocation and preemption logic must appear with results.

## Outcomes and decisions

First distinguish real lifecycle from performance: all128 complete, allquota released, every promise transferred or cancelled, zero preemptions/progress/capacity violations. Record actual promises/transfers/cancellations, initial and finalquota, full-bound and physical peaks, and decision/bookkeeping costs. Zero promises means no observed temporal action in this cohort, not a failed implementation; do not change a guard or input to force an action.

Report the fixed20/4 goodput, all20frontier points, TTFT/hostgap/arrival-to-completion costs, outputs and finishreasons, and all per-request regressions against **both** completed FIFO references and **both** completed retirement references from the freshblock. These are temporally separated descriptive pilot comparisons, not matched repeat or equal-work causality. Do not select only the favorable reference.

If it has useful native actions or materially challenges the candidate, freeze a compact matched comparison next. If it is weaker, retain the completed comparator and narrow its conclusion; do not tune guard/seed/inputs to obtain a win. A progress, capacity or lifecycle error invalidates this execution; preserve it and fix only the specific implementation error before a separately versioned run.

## Execution

Use the existing `/root/autodl-tmp/moe-research-gpu.lock`, inode2304:29005388732, GPU `GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36`. Nonblocking try-lock; busy means deferred with no child. Existing V3 helper provides900s timeout, inherited lockFD, owned-child cleanup and post-exit GPU drain. Output root `c-native-temporal-cap-donor-pilot-v1` must be new. Release GPU before analysis; no second job or retry loop. Freeze the adapter/cell/wrapper source hashes before upload. This short test uses the user's existing shared machine budget and does not publish anything.
