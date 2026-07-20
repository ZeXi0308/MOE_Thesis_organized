# Layer-Wise Fixed Tail-Budget Experiment

## Boundary

This is a Mac fake-quant quality experiment.  Calibration profiles are used
only to rank layers; their KL values are not added to predict end-to-end KL.
Every frozen allocation is evaluated end-to-end on a disjoint held-out slice.

## Setup

- model: `jamesdborin/tiny-mixtral`
- calibration: `builtin:validation` offset `0`, n=`2`
- test: `builtin:validation` offset `4`, n=`2`
- top-k: `2`; base tail count: `1`; total low-bit layer-rank slots: `2`
- tail precision: `mxfp4`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 38001.037446 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed_tail1 | 0.625000 | 38139.063966 | 138.026521 | 0.002981 | 0.002030 | 0.003933 |
| kl_profile_0_2 | 0.625000 | 38436.684187 | 435.646741 | 0.001758 | 0.000647 | 0.002869 |
| p95_profile_0_2 | 0.625000 | 38436.684187 | 435.646741 | 0.001758 | 0.000647 | 0.002869 |
| gate_mass_profile_0_2 | 0.625000 | 37302.502215 | -698.535230 | 0.004963 | 0.004689 | 0.005238 |
| anti_kl_profile_0_2 | 0.625000 | 37302.502215 | -698.535230 | 0.004963 | 0.004689 | 0.005238 |

## Paired bootstrap versus fixed tail

Positive `reference_minus_candidate_kl` means the candidate is better.

| reference | candidate | reference_minus_candidate_kl | ci_low | ci_high | probability_candidate_better |
|---|---|---|---|---|---|
| fixed_tail1 | anti_kl_profile_0_2 | -0.001982 | -0.002658 | -0.001306 | 0.000000 |
| fixed_tail1 | gate_mass_profile_0_2 | -0.001982 | -0.002658 | -0.001306 | 0.000000 |
| fixed_tail1 | kl_profile_0_2 | 0.001224 | 0.001064 | 0.001384 | 1.000000 |
| fixed_tail1 | p95_profile_0_2 | 0.001224 | 0.001064 | 0.001384 | 1.000000 |

## Allocation interpretation

- `kl_profile_*`: protect layers with larger single-layer incremental KL and spend more tail ranks on less-sensitive layers.
- `p95_profile_*`: the same idea using the P95 per-sample KL risk score.
- `gate_mass_profile_*`: protect layers whose fixed tail carries more calibration gate mass.
- `anti_*`: equal-payload negative control that deliberately spends more INT4 ranks on sensitive layers.

These are regular per-layer fixed layouts.  A positive result still needs a
real two-lane kernel; a negative result means layer-wise rank budgeting does
not close the quality gap to token-wise gate selection.
