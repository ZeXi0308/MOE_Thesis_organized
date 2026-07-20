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
- tail precision: `mxfp4`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 18.791504 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed_tail4 | 0.625000 | 18.734759 | -0.056745 | 0.006840 | 0.005646 | 0.008491 |
| kl_profile_3_5 | 0.625000 | 18.731078 | -0.060426 | 0.007747 | 0.006380 | 0.009630 |
| p95_profile_3_5 | 0.625000 | 18.746821 | -0.044683 | 0.007489 | 0.006301 | 0.009069 |
| gate_mass_profile_3_5 | 0.625000 | 18.733454 | -0.058050 | 0.006917 | 0.005781 | 0.008285 |
| anti_kl_profile_3_5 | 0.625000 | 18.733454 | -0.058050 | 0.006917 | 0.005781 | 0.008285 |
| kl_profile_2_4_6 | 0.625000 | 18.812152 | 0.020647 | 0.007244 | 0.006557 | 0.008079 |

## Paired bootstrap versus fixed tail

Positive `reference_minus_candidate_kl` means the candidate is better.

| reference | candidate | reference_minus_candidate_kl | ci_low | ci_high | probability_candidate_better |
|---|---|---|---|---|---|
| fixed_tail4 | anti_kl_profile_3_5 | -0.000078 | -0.000560 | 0.000396 | 0.378000 |
| fixed_tail4 | gate_mass_profile_3_5 | -0.000078 | -0.000566 | 0.000399 | 0.338000 |
| fixed_tail4 | kl_profile_2_4_6 | -0.000404 | -0.001135 | 0.000567 | 0.174000 |
| fixed_tail4 | kl_profile_3_5 | -0.000907 | -0.001488 | -0.000353 | 0.000000 |
| fixed_tail4 | p95_profile_3_5 | -0.000649 | -0.001272 | -0.000065 | 0.015000 |

## Allocation interpretation

- `kl_profile_*`: protect layers with larger single-layer incremental KL and spend more tail ranks on less-sensitive layers.
- `p95_profile_*`: the same idea using the P95 per-sample KL risk score.
- `gate_mass_profile_*`: protect layers whose fixed tail carries more calibration gate mass.
- `anti_*`: equal-payload negative control that deliberately spends more INT4 ranks on sensitive layers.

These are regular per-layer fixed layouts.  A positive result still needs a
real two-lane kernel; a negative result means layer-wise rank budgeting does
not close the quality gap to token-wise gate selection.
