# Layer-Wise Fixed Tail-Budget Experiment

## Boundary

This is a Mac fake-quant quality experiment.  Calibration profiles are used
only to rank layers; their KL values are not added to predict end-to-end KL.
Every frozen allocation is evaluated end-to-end on a disjoint held-out slice.

## Setup

- model: `allenai/OLMoE-1B-7B-0924`
- calibration: `wikitext2:validation` offset `0`, n=`16`
- test: `wikitext2:validation` offset `128`, n=`32`
- top-k: `8`; base tail count: `4`; total low-bit layer-rank slots: `64`
- tail precision: `nvfp4`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 18.791504 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed_tail4 | 0.625000 | 18.669896 | -0.121608 | 0.005713 | 0.004721 | 0.007021 |
| kl_profile_3_5 | 0.625000 | 18.678355 | -0.113149 | 0.005751 | 0.004746 | 0.007162 |
| p95_profile_3_5 | 0.625000 | 18.678355 | -0.113149 | 0.005751 | 0.004746 | 0.007162 |
| gate_mass_profile_3_5 | 0.625000 | 18.730241 | -0.061263 | 0.005581 | 0.004705 | 0.006604 |
| anti_kl_profile_3_5 | 0.625000 | 18.730241 | -0.061263 | 0.005581 | 0.004705 | 0.006604 |
| kl_profile_2_4_6 | 0.625000 | 18.745112 | -0.046393 | 0.006335 | 0.005401 | 0.007490 |

## Paired bootstrap versus fixed tail

Positive `reference_minus_candidate_kl` means the candidate is better.

| reference | candidate | reference_minus_candidate_kl | ci_low | ci_high | probability_candidate_better |
|---|---|---|---|---|---|
| fixed_tail4 | anti_kl_profile_3_5 | 0.000132 | -0.000462 | 0.000741 | 0.669000 |
| fixed_tail4 | gate_mass_profile_3_5 | 0.000132 | -0.000477 | 0.000714 | 0.661000 |
| fixed_tail4 | kl_profile_2_4_6 | -0.000622 | -0.001051 | -0.000174 | 0.003000 |
| fixed_tail4 | kl_profile_3_5 | -0.000038 | -0.000411 | 0.000345 | 0.408000 |
| fixed_tail4 | p95_profile_3_5 | -0.000038 | -0.000386 | 0.000377 | 0.412000 |

## Allocation interpretation

- `kl_profile_*`: protect layers with larger single-layer incremental KL and spend more tail ranks on less-sensitive layers.
- `p95_profile_*`: the same idea using the P95 per-sample KL risk score.
- `gate_mass_profile_*`: protect layers whose fixed tail carries more calibration gate mass.
- `anti_*`: equal-payload negative control that deliberately spends more INT4 ranks on sensitive layers.

These are regular per-layer fixed layouts.  A positive result still needs a
real two-lane kernel; a negative result means layer-wise rank budgeting does
not close the quality gap to token-wise gate selection.
