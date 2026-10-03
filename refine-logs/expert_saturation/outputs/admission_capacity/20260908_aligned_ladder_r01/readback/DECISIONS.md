# Capture-aligned admission ladder — frozen before its GPU execution

Frozen 2026-09-08, before any GPU time is spent. Nothing below may be changed
after seeing results; if it must change, this file is superseded by a new dated
decision file and both are retained.

## Why this experiment

The reconstruction in `../analysis/` establishes, from the sealed 44-episode
native captures, that pure-decode step time is a staircase over the engine's
CUDA-graph capture sizes: flat to within 2.1–4.0% inside a bucket, jumping
12.4–30.1% across one. Capture edges `[1,2,4,8,16,24,32]` are confirmed twice —
recovered from the timing data, and printed by the engine itself
(`cudagraph_capture_sizes`, 7 decode FULL graphs).

The already-measured static caps confirm the load-bearing consequence without
new execution: cap12 and cap16 both sit in bucket 16, and their pure-step cost
ratio is 0.9949 (steady) / 0.9764 (bursty) with TPOT ratio 1.0049 / 0.9943 —
statistically indistinguishable cost — while cap16's goodput is +3.44% / +142.58%.
cap12 has the worst mean padding waste of every cap in both regimes (0.194 / 0.165)
and is the only measured cap not on a capture point.

The frozen ITL feedback controller that was ruled `CONDITIONAL_NO_GO` on
2026-09-06 used the ladder `{8,12,16,32}` and drove decode width to 9–15,
producing mean padding waste 0.301 — the highest of all 44 episodes. That is a
concrete, previously uncharged reason for its residual, and it is a property of
the ladder's coordinates, not of the feedback rule's logic.

## The single change under test

Run the **same** feedback rule, capture code, engine configuration, texts,
arrivals and SLO as the 2026-09-06 policy probe. Change exactly one thing:

    ladder {8,12,16,32}  ->  {8,16,24,32}

Both ladders contain 8, 16 and 32. They differ only in the middle rung: 12 sits
strictly inside bucket 16; 24 is a capture point. No code change is required —
`run_native_capacity.py` derives `feedback_caps` from `--caps`.

This is a coordinate change, not a new mechanism. No new parameter, no predictor,
no expert signal, no threshold search, no added overhead.

## Command

```bash
python run_native_capacity.py --prepared-dir prepared \
  --output-dir results/forward --caps 8,16,24,32 --engine-max-seqs 32 \
  --arrival-scales 1 --repeats 1 --include-feedback \
  --ttft-slo-s 0.20 --tpot-slo-s 0.009 --warmup-condition-rounds 1
```

Reverse engine adds `--reverse-conditions`. Two fresh engines, 12 episodes each,
24 total. Plan generation already verified on CPU: arms are
static8/16/24/32 + shadow32 + feedback32, each in steady and bursty.

## Frozen predictions

Stated before execution. Ordered by how much each would change the research
judgement.

| # | Prediction | Falsified if |
|---|---|---|
| F1 | static24 pure-step median lands in `[9.30, 9.51]` ms, i.e. on the 24-bucket plateau already observed at widths 17–24 | static24 step cost falls outside that interval, which breaks the staircase reading |
| F2 | feedback32 on the aligned ladder shows lower mean padding waste than the 2026-09-06 feedback32 (0.301 steady) | aligned-ladder waste is equal or higher, meaning the ladder was not what pinned width inside buckets |
| F3 | aligned feedback32 does not worsen median request TPOT relative to the 2026-09-06 feedback32 | TPOT degrades, meaning bucket alignment costs more than it saves |
| F4 | aligned feedback32 still fails to beat the best measured static point in its own engine | it wins, which would upgrade this from a cost-accounting result to a method signal |

F4 is deliberately the null. The honest expectation from the reconstruction is
that alignment removes a real but *small* cost term in steady state — the median
interference share is only 0.048–0.074 and removing it entirely still leaves
cap32 steady TPOT at 9.32 ms against a 9 ms SLO. Alignment is therefore expected
to be necessary but not sufficient. A positive F4 would be a surprise, and would
require a controlled repeat before being reported as anything.

## What this experiment cannot establish

- It is single model, single GPU, 32 repeated texts, fixed 128/128 tokens, no EOS.
- It does not test expert signals (U/C). Those remain `UNRUN`.
- Padding waste does **not** monotonically predict goodput: steady cap32 has
  waste 0.161 yet the best goodput, because concurrency gain outweighs padding
  loss. Alignment is a newly charged cost term, not a new objective function.
- bursty has essentially no coalescing action space (arrivals already land in
  groups of 8, i.e. exactly on a capture point), so any bursty effect here comes
  from the ladder's rungs, not from grouping.
- Nothing here licenses a claim about vLLM's default capture list being wrong,
  about other engines, or about production serving.

## Retention

All 24 episodes retained regardless of outcome. Canonical selection rule declared
now: the forward engine is canonical for reporting, the reverse engine is the
controlled repeat; neither may be dropped or swapped after seeing metrics. Raw
cells are never edited; corrections go to an `ADDENDUM.md`.
