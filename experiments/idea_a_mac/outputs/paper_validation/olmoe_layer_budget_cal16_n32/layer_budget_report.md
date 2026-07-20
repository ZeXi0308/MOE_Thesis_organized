# Layer-Wise Fixed Tail-Budget Experiment

## Boundary

This is a Mac fake-quant quality experiment.  Calibration profiles are used
only to rank layers; their KL values are not added to predict end-to-end KL.
Every frozen allocation is evaluated end-to-end on a disjoint held-out slice.

## Setup

- model: `allenai/OLMoE-1B-7B-0924`
- calibration: `wikitext2:validation` offset `0`, n=`16`
- test: `wikitext2:validation` offset `128`, n=`32`
- top-k: `8`; base tail count: `4`; total INT4 layer-rank slots: `64`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 18.791504 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed_tail4 | 0.625000 | 19.238104 | 0.446600 | 0.030320 | 0.026038 | 0.036061 |
| kl_profile_3_5 | 0.625000 | 18.986270 | 0.194766 | 0.023680 | 0.020362 | 0.028300 |
| p95_profile_3_5 | 0.625000 | 19.105864 | 0.314360 | 0.024110 | 0.020800 | 0.029415 |
| gate_mass_profile_3_5 | 0.625000 | 19.158958 | 0.367454 | 0.025531 | 0.021700 | 0.030998 |
| anti_kl_profile_3_5 | 0.625000 | 19.442517 | 0.651013 | 0.040911 | 0.035812 | 0.047523 |
| kl_profile_2_4_6 | 0.625000 | 18.998177 | 0.206673 | 0.019569 | 0.017592 | 0.022053 |

## Paired bootstrap versus fixed tail

Positive `reference_minus_candidate_kl` means the candidate is better.

| reference | candidate | reference_minus_candidate_kl | ci_low | ci_high | probability_candidate_better |
|---|---|---|---|---|---|
| fixed_tail4 | anti_kl_profile_3_5 | -0.010590 | -0.013338 | -0.008182 | 0.000000 |
| fixed_tail4 | gate_mass_profile_3_5 | 0.004790 | 0.003035 | 0.006689 | 1.000000 |
| fixed_tail4 | kl_profile_2_4_6 | 0.010751 | 0.007342 | 0.015882 | 1.000000 |
| fixed_tail4 | kl_profile_3_5 | 0.006641 | 0.005053 | 0.008494 | 1.000000 |
| fixed_tail4 | p95_profile_3_5 | 0.006211 | 0.004511 | 0.008276 | 1.000000 |

## Allocation interpretation

- `kl_profile_*`: protect layers with larger single-layer incremental KL and spend more tail ranks on less-sensitive layers.
- `p95_profile_*`: the same idea using the P95 per-sample KL risk score.
- `gate_mass_profile_*`: protect layers whose fixed tail carries more calibration gate mass.
- `anti_*`: equal-payload negative control that deliberately spends more INT4 ranks on sensitive layers.

These are regular per-layer fixed layouts.  A positive result still needs a
real two-lane kernel; a negative result means layer-wise rank budgeting does
not close the quality gap to token-wise gate selection.
