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
| uniform_fp8 | 0.292581 | 0.116730 | 0.175851 | 0.5936 | 0.4064 |
| rank8_fp8 | 0.130455 | 0.028825 | 0.101629 | 0.7264 | 0.2736 |
| rank1_fp8 | 0.257345 | 0.078630 | 0.178716 | 0.6705 | 0.3295 |
| rank8_int4 | 0.361361 | 0.179343 | 0.182018 | 0.4814 | 0.5186 |

## Interpretation

- **uniform_fp8**: drift fraction 59.36% → moderate routing drift — single-layer delta profiles may underestimate cascading loss; validate additivity with a sanity check
- **rank8_fp8**: drift fraction 72.64% → routing drift dominates — the linear additive delta model needs cascading correction or routing alignment (EAQuant-style)
- **rank1_fp8**: drift fraction 67.05% → routing drift dominates — the linear additive delta model needs cascading correction or routing alignment (EAQuant-style)
- **rank8_int4**: drift fraction 48.14% → moderate routing drift — single-layer delta profiles may underestimate cascading loss; validate additivity with a sanity check
