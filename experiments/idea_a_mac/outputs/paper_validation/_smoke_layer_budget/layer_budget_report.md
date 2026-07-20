# Layer-Wise Fixed Tail-Budget Experiment

## Boundary

This is a Mac fake-quant quality experiment.  Calibration profiles are used
only to rank layers; their KL values are not added to predict end-to-end KL.
Every frozen allocation is evaluated end-to-end on a disjoint held-out slice.

## Setup

- model: `jamesdborin/tiny-mixtral`
- calibration: `builtin:validation` offset `0`, n=`2`
- test: `builtin:validation` offset `4`, n=`2`
- top-k: `2`; base tail count: `1`; total INT4 layer-rank slots: `2`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 38001.037446 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed_tail1 | 0.625000 | 38386.508611 | 385.471165 | 0.003111 | 0.003025 | 0.003196 |
| kl_profile_3_5 | 0.625000 | 38521.589518 | 520.552073 | 0.002040 | 0.000918 | 0.003162 |
| p95_profile_3_5 | 0.625000 | 38521.589518 | 520.552073 | 0.002040 | 0.000918 | 0.003162 |
| gate_mass_profile_3_5 | 0.625000 | 37682.242947 | -318.794499 | 0.008702 | 0.007728 | 0.009676 |
| anti_kl_profile_3_5 | 0.625000 | 37682.242947 | -318.794499 | 0.008702 | 0.007728 | 0.009676 |

## Allocation interpretation

- `kl_profile_*`: protect layers with larger single-layer incremental KL and spend more tail ranks on less-sensitive layers.
- `p95_profile_*`: the same idea using the P95 per-sample KL risk score.
- `gate_mass_profile_*`: protect layers whose fixed tail carries more calibration gate mass.
- `anti_*`: equal-payload negative control that deliberately spends more INT4 ranks on sensitive layers.

These are regular per-layer fixed layouts.  A positive result still needs a
real two-lane kernel; a negative result means layer-wise rank budgeting does
not close the quality gap to token-wise gate selection.
