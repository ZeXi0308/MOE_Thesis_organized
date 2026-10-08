# Frozen fresh-cohort comparison

## Question and decision

The recovered development bound/retirement/retirement/bound block completed all 512 requests. Both pairs improved the fixed 20/4 goodput (+14.2%/+11.7%) and mean flow (−5.9%/−4.8%), with no capacity or progress violation. The next question is whether this unchanged policy retains a benefit against full-bound FIFO on articles and arrival times that were not used to choose it. Native no-offload is measured alongside it to expose any remaining cost relative to the default scheduler.

One six-cell block, in this fixed order: **native 1, bound FIFO 1, retirement 1, retirement 2, bound FIFO 2, native 2**. Each cell serves all 128 external requests through drain. Compare retirement 1 with bound 1/native 1 and retirement 2 with bound 2/native 2; report both directions and both within-arm repeats. This order brackets the candidate with the references and measures all three on the same replacement GPU. Earlier GPU measurements are contextual only.

Retain support for the primary fresh-cohort hypothesis only if both paired retirement-versus-bound differences in 20/4 goodput are positive and all completion/capacity/progress conditions hold. Report native comparisons and costs even if unfavorable. A sign reversal weakens the claim and is retained; do not change the arrival draw, thresholds, seed, inputs or policy to rescue this test. A correctness violation invalidates that execution and stops the block; preserve the failure. Two repeats describe this block, not a population-level confidence interval or sufficient paper validation.

## Input and resource freeze

Use the input rule frozen before construction in `C_NATIVE_RETIREMENT_FRESH_INPUT_PLAN_20261001.md`: eligible article ranks 832–959, one Poisson draw with rate 5 requests/s and seed 20261001. The resulting span is 22.682329 s, prompts 414–3066 tokens. This observed pressure is retained without resampling. The old 128 prompts reproduced exactly with the local tokenizer. Input directory `20261001_c_retirement_fresh_inputs_v1`:

- config.json: `634e7715618879775daf312fc98d77c1b25f4cd8e97007c101604ded317738a3`
- workload.json: `9576395c9540c71be86dd182df03ab6d39cf1c961a6a24dc851d40991cf0a985`
- INPUT_STATS.json: `797ccd408a60ee01bed2ac370f51f7145fa4f9ea3bcaecc5a1a4535afd034ac1`

The raw input config records the actual maximum prompt 3066; the measurement qualification uses the predeclared conservative ceiling 3072. Per-request policies use actual prompt lengths. This does not reduce the measurement's 32 sequence slots. Model/revision, greedy seed 20260905, EOS enabled/min_tokens 0/max_tokens 1024, 4096 usable KV blocks, batch 1024, no offload, pinned runtime, 25 CPU threads and existing warmups are unchanged. The retirement and bound adapters retain their frozen SHA-256 values `261b8f2697042f75130203019e8722c424cca3e71d82ac309c78f29ec1bc8eb6` and `b6a601d1edc05804bee70d7f6ecdb2f00e1dd017b2280163df2db88250fb50ee`.

New user-supplied endpoint: port 22937; GPU `GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36`. Use the existing `/root/autodl-tmp/moe-research-gpu.lock`, inode `2304:29005388732`, by nonblocking flock. Never recreate it. The user-authorized 100-hour machine budget is shared with A/B; this block is expected to take about 12–15 minutes, with a 900-second child timeout. Lock covers initialization, warmups, measurement, drain and process exit. No second job is queued. A busy lock defers this block to a natural work boundary.

## Outcomes and claim scope

Keep the prior primary target TTFT ≤20 s and maximum distinct host-return gap ≤4 s, goodput over the entire episode. Report all 20 fixed frontier points, full output work and termination, per-request TTFT/flow/gap improvements and regressions, throughput, preemptions, capacity peaks, actual incremental admissions and decision cost. Continuous deadline curves remain descriptive. Chunk interiors remain unobserved; changed outputs preclude an equal-work causal speedup claim.

This is a fresh-cohort test relative to recorded C policy selection. It does not resolve historical missing source receipts, whole-Parquet/Arrow equivalence, semantic output quality, cap-heavy generation, nearest-method implementation or generalization beyond this model/resource domain. Positive results alone do not mark the paper ready for review.
