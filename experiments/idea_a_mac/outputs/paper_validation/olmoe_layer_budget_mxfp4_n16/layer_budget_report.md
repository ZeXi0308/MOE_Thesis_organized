# Layer-Wise Fixed Tail-Budget Experiment

## Boundary

This is a Mac fake-quant quality experiment.  Calibration profiles are used
only to rank layers; their KL values are not added to predict end-to-end KL.
Every frozen allocation is evaluated end-to-end on a disjoint held-out slice.

## Setup

- model: `allenai/OLMoE-1B-7B-0924`
- calibration: `wikitext2:validation` offset `0`, n=`8`
- test: `wikitext2:validation` offset `128`, n=`16`
- top-k: `8`; base tail count: `4`; total low-bit layer-rank slots: `64`
- tail precision: `mxfp4`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 19.834418 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed_tail4 | 0.625000 | 19.757937 | -0.076481 | 0.007141 | 0.005252 | 0.009728 |
| kl_profile_3_5 | 0.625000 | 19.757576 | -0.076842 | 0.007529 | 0.005788 | 0.009861 |
| p95_profile_3_5 | 0.625000 | 19.757576 | -0.076842 | 0.007529 | 0.005788 | 0.009861 |
| gate_mass_profile_3_5 | 0.625000 | 19.752641 | -0.081777 | 0.006985 | 0.005161 | 0.009186 |
| anti_kl_profile_3_5 | 0.625000 | 19.833705 | -0.000713 | 0.007055 | 0.005251 | 0.009314 |
| kl_profile_2_4_6 | 0.625000 | 19.790264 | -0.044154 | 0.007103 | 0.006094 | 0.008400 |

## Paired bootstrap versus fixed tail

Positive `reference_minus_candidate_kl` means the candidate is better.

| reference | candidate | reference_minus_candidate_kl | ci_low | ci_high | probability_candidate_better |
|---|---|---|---|---|---|
| fixed_tail4 | anti_kl_profile_3_5 | 0.000086 | -0.000556 | 0.000775 | 0.630000 |
| fixed_tail4 | gate_mass_profile_3_5 | 0.000156 | -0.000555 | 0.000866 | 0.642000 |
| fixed_tail4 | kl_profile_2_4_6 | 0.000038 | -0.000941 | 0.001432 | 0.480000 |
| fixed_tail4 | kl_profile_3_5 | -0.000388 | -0.001068 | 0.000329 | 0.136000 |
| fixed_tail4 | p95_profile_3_5 | -0.000388 | -0.001075 | 0.000355 | 0.138000 |

## Allocation interpretation

- `kl_profile_*`: protect layers with larger single-layer incremental KL and spend more tail ranks on less-sensitive layers.
- `p95_profile_*`: the same idea using the P95 per-sample KL risk score.
- `gate_mass_profile_*`: protect layers whose fixed tail carries more calibration gate mass.
- `anti_*`: equal-payload negative control that deliberately spends more INT4 ranks on sensitive layers.

These are regular per-layer fixed layouts.  A positive result still needs a
real two-lane kernel; a negative result means layer-wise rank budgeting does
not close the quality gap to token-wise gate selection.
