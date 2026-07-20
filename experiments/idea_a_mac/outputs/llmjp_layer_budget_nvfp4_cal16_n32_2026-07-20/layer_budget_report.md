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
- tail precision: `nvfp4`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 13.927147 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed_tail8 | 0.625000 | 13.947031 | 0.019884 | 0.007243 | 0.006890 | 0.007674 |
| kl_profile_7_9 | 0.625000 | 13.984370 | 0.057223 | 0.007361 | 0.006988 | 0.007840 |
| p95_profile_7_9 | 0.625000 | 13.953573 | 0.026426 | 0.007308 | 0.006961 | 0.007756 |
| gate_mass_profile_7_9 | 0.625000 | 13.953181 | 0.026034 | 0.007414 | 0.007089 | 0.007832 |
| anti_kl_profile_7_9 | 0.625000 | 13.932730 | 0.005583 | 0.009226 | 0.008885 | 0.009646 |
| kl_profile_6_8_10 | 0.625000 | 13.944907 | 0.017760 | 0.007278 | 0.006860 | 0.007858 |

## Paired bootstrap versus fixed tail

Positive `reference_minus_candidate_kl` means the candidate is better.

| reference | candidate | reference_minus_candidate_kl | ci_low | ci_high | probability_candidate_better |
|---|---|---|---|---|---|
| fixed_tail8 | anti_kl_profile_7_9 | -0.001983 | -0.002131 | -0.001840 | 0.000000 |
| fixed_tail8 | gate_mass_profile_7_9 | -0.000171 | -0.000324 | -0.000020 | 0.013000 |
| fixed_tail8 | kl_profile_6_8_10 | -0.000035 | -0.000275 | 0.000174 | 0.404000 |
| fixed_tail8 | kl_profile_7_9 | -0.000118 | -0.000244 | -0.000001 | 0.023000 |
| fixed_tail8 | p95_profile_7_9 | -0.000064 | -0.000175 | 0.000053 | 0.167000 |

## Allocation interpretation

- `kl_profile_*`: protect layers with larger single-layer incremental KL and spend more tail ranks on less-sensitive layers.
- `p95_profile_*`: the same idea using the P95 per-sample KL risk score.
- `gate_mass_profile_*`: protect layers whose fixed tail carries more calibration gate mass.
- `anti_*`: equal-payload negative control that deliberately spends more INT4 ranks on sensitive layers.

These are regular per-layer fixed layouts.  A positive result still needs a
real two-lane kernel; a negative result means layer-wise rank budgeting does
not close the quality gap to token-wise gate selection.
