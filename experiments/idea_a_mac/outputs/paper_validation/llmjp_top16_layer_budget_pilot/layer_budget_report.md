# Layer-Wise Fixed Tail-Budget Experiment

## Boundary

This is a Mac fake-quant quality experiment.  Calibration profiles are used
only to rank layers; their KL values are not added to predict end-to-end KL.
Every frozen allocation is evaluated end-to-end on a disjoint held-out slice.

## Setup

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- calibration: `wikitext2:validation` offset `0`, n=`4`
- test: `wikitext2:validation` offset `128`, n=`8`
- top-k: `16`; base tail count: `8`; total INT4 layer-rank slots: `128`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 48.909251 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed_tail8 | 0.625000 | 51.565605 | 2.656354 | 0.027236 | 0.023422 | 0.032975 |
| kl_profile_7_9 | 0.625000 | 51.858026 | 2.948775 | 0.024767 | 0.021271 | 0.029451 |
| p95_profile_7_9 | 0.625000 | 51.858026 | 2.948775 | 0.024767 | 0.021271 | 0.029451 |
| gate_mass_profile_7_9 | 0.625000 | 51.412299 | 2.503048 | 0.023224 | 0.019927 | 0.027802 |
| anti_kl_profile_7_9 | 0.625000 | 51.242781 | 2.333530 | 0.026969 | 0.022903 | 0.032589 |
| kl_profile_6_8_10 | 0.625000 | 50.931257 | 2.022006 | 0.021198 | 0.018507 | 0.025091 |

## Paired bootstrap versus fixed tail

Positive `reference_minus_candidate_kl` means the candidate is better.

| reference | candidate | reference_minus_candidate_kl | ci_low | ci_high | probability_candidate_better |
|---|---|---|---|---|---|
| fixed_tail8 | anti_kl_profile_7_9 | 0.000267 | -0.000747 | 0.001104 | 0.752000 |
| fixed_tail8 | gate_mass_profile_7_9 | 0.004013 | 0.003184 | 0.005006 | 1.000000 |
| fixed_tail8 | kl_profile_6_8_10 | 0.006039 | 0.004472 | 0.007854 | 1.000000 |
| fixed_tail8 | kl_profile_7_9 | 0.002469 | 0.001669 | 0.003407 | 1.000000 |
| fixed_tail8 | p95_profile_7_9 | 0.002469 | 0.001622 | 0.003522 | 1.000000 |

## Allocation interpretation

- `kl_profile_*`: protect layers with larger single-layer incremental KL and spend more tail ranks on less-sensitive layers.
- `p95_profile_*`: the same idea using the P95 per-sample KL risk score.
- `gate_mass_profile_*`: protect layers whose fixed tail carries more calibration gate mass.
- `anti_*`: equal-payload negative control that deliberately spends more INT4 ranks on sensitive layers.

These are regular per-layer fixed layouts.  A positive result still needs a
real two-lane kernel; a negative result means layer-wise rank budgeting does
not close the quality gap to token-wise gate selection.
