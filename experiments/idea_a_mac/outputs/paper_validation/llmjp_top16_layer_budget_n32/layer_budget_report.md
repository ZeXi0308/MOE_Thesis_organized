# Layer-Wise Fixed Tail-Budget Experiment

## Boundary

This is a Mac fake-quant quality experiment.  Calibration profiles are used
only to rank layers; their KL values are not added to predict end-to-end KL.
Every frozen allocation is evaluated end-to-end on a disjoint held-out slice.

## Setup

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- calibration: `wikitext2:validation` offset `0`, n=`8`
- test: `wikitext2:validation` offset `128`, n=`32`
- top-k: `16`; base tail count: `8`; total INT4 layer-rank slots: `128`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 25.425516 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed_tail8 | 0.625000 | 25.734461 | 0.308945 | 0.017388 | 0.014998 | 0.020826 |
| kl_profile_7_9 | 0.625000 | 25.642497 | 0.216981 | 0.016073 | 0.014048 | 0.019267 |
| p95_profile_7_9 | 0.625000 | 25.699378 | 0.273862 | 0.015326 | 0.013355 | 0.018247 |
| gate_mass_profile_7_9 | 0.625000 | 25.648624 | 0.223108 | 0.015336 | 0.013398 | 0.018279 |
| anti_kl_profile_7_9 | 0.625000 | 25.778145 | 0.352629 | 0.017024 | 0.014711 | 0.020429 |
| kl_profile_6_8_10 | 0.625000 | 25.606155 | 0.180638 | 0.013968 | 0.012236 | 0.016369 |

## Paired bootstrap versus fixed tail

Positive `reference_minus_candidate_kl` means the candidate is better.

| reference | candidate | reference_minus_candidate_kl | ci_low | ci_high | probability_candidate_better |
|---|---|---|---|---|---|
| fixed_tail8 | anti_kl_profile_7_9 | 0.000364 | 0.000010 | 0.000747 | 0.975000 |
| fixed_tail8 | gate_mass_profile_7_9 | 0.002052 | 0.001499 | 0.002793 | 1.000000 |
| fixed_tail8 | kl_profile_6_8_10 | 0.003420 | 0.002549 | 0.004522 | 1.000000 |
| fixed_tail8 | kl_profile_7_9 | 0.001315 | 0.000838 | 0.001947 | 1.000000 |
| fixed_tail8 | p95_profile_7_9 | 0.002062 | 0.001522 | 0.002769 | 1.000000 |

## Allocation interpretation

- `kl_profile_*`: protect layers with larger single-layer incremental KL and spend more tail ranks on less-sensitive layers.
- `p95_profile_*`: the same idea using the P95 per-sample KL risk score.
- `gate_mass_profile_*`: protect layers whose fixed tail carries more calibration gate mass.
- `anti_*`: equal-payload negative control that deliberately spends more INT4 ranks on sensitive layers.

These are regular per-layer fixed layouts.  A positive result still needs a
real two-lane kernel; a negative result means layer-wise rank budgeting does
not close the quality gap to token-wise gate selection.
