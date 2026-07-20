# Routing Drift Attribution Report

model: `allenai/OLMoE-1B-7B-0924`
samples: `32`
seq_len: `128`
dtype: `bfloat16`
dataset: `wikitext2`

## Method

For each sample and strategy, three forward passes are compared:

1. **Full forward** (BF16, no approximation) — caches per-layer routing decisions.
2. **Approx (locked routing)** — applies approximation but forces the router to
   reuse the full model's expert selection. Isolates pure numerical error.
3. **Approx (free routing)** — applies approximation and lets the router
   re-select experts from perturbed hidden states. Includes numerical error
   + routing drift.

**drift contribution** = KL_free - KL_locked
**drift fraction** = drift_contribution / KL_free

## Summary

| strategy | mean KL (free) | mean KL (locked) | drift contribution | drift fraction | numerical fraction |
|---|---|---|---|---|---|
| rank8_int4 | 0.361361 | 0.179343 | 0.182018 | 0.4814 | 0.5186 |
| rank1_int4 | 20.989216 | 16.364636 | 4.624580 | 0.2211 | 0.7789 |
| uniform_int4 | 27.152189 | 22.671950 | 4.480239 | 0.1477 | 0.8523 |

## Interpretation

- **rank8_int4**: drift fraction 48.14% → moderate routing drift — single-layer delta profiles may underestimate cascading loss; validate additivity with a sanity check
- **rank1_int4**: drift fraction 22.11% → numerical error dominates — the linear additive delta model is well justified; single-layer profiles are reliable
- **uniform_int4**: drift fraction 14.77% → numerical error dominates — the linear additive delta model is well justified; single-layer profiles are reliable
