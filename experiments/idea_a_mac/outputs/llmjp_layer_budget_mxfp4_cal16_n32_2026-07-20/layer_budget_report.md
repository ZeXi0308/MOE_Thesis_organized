# Layer-Wise Fixed Tail-Budget Experiment

## Boundary

This is a Mac fake-quant quality experiment.  Calibration profiles are used
only to rank layers; their KL values are not added to predict end-to-end KL.
Every frozen allocation is evaluated end-to-end on a disjoint held-out slice.

## Setup

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- calibration: `wikitext2_docs:validation` offset `0`, n=`16`
- test: `wikitext2_docs:validation` offset `20`, n=`32`
- top-k: `16`; base tail count: `8`; total low-bit layer-rank slots: `128`
- tail precision: `mxfp4`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 13.927147 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed_tail8 | 0.625000 | 13.935694 | 0.008548 | 0.007750 | 0.007327 | 0.008271 |
| kl_profile_7_9 | 0.625000 | 13.990248 | 0.063101 | 0.009179 | 0.008746 | 0.009718 |
| p95_profile_7_9 | 0.625000 | 13.990248 | 0.063101 | 0.009179 | 0.008746 | 0.009718 |
| gate_mass_profile_7_9 | 0.625000 | 13.982005 | 0.054858 | 0.009207 | 0.008803 | 0.009717 |
| anti_kl_profile_7_9 | 0.625000 | 13.968123 | 0.040976 | 0.007996 | 0.007608 | 0.008450 |
| kl_profile_6_8_10 | 0.625000 | 13.960593 | 0.033446 | 0.008999 | 0.008690 | 0.009375 |

## Paired bootstrap versus fixed tail

Positive `reference_minus_candidate_kl` means the candidate is better.

| reference | candidate | reference_minus_candidate_kl | ci_low | ci_high | probability_candidate_better |
|---|---|---|---|---|---|
| fixed_tail8 | anti_kl_profile_7_9 | -0.000245 | -0.000463 | -0.000020 | 0.010000 |
| fixed_tail8 | gate_mass_profile_7_9 | -0.001457 | -0.001602 | -0.001296 | 0.000000 |
| fixed_tail8 | kl_profile_6_8_10 | -0.001249 | -0.001466 | -0.001018 | 0.000000 |
| fixed_tail8 | kl_profile_7_9 | -0.001429 | -0.001652 | -0.001224 | 0.000000 |
| fixed_tail8 | p95_profile_7_9 | -0.001429 | -0.001641 | -0.001213 | 0.000000 |

## Allocation interpretation

- `kl_profile_*`: protect layers with larger single-layer incremental KL and spend more tail ranks on less-sensitive layers.
- `p95_profile_*`: the same idea using the P95 per-sample KL risk score.
- `gate_mass_profile_*`: protect layers whose fixed tail carries more calibration gate mass.
- `anti_*`: equal-payload negative control that deliberately spends more INT4 ranks on sensitive layers.

These are regular per-layer fixed layouts.  A positive result still needs a
real two-lane kernel; a negative result means layer-wise rank budgeting does
not close the quality gap to token-wise gate selection.
