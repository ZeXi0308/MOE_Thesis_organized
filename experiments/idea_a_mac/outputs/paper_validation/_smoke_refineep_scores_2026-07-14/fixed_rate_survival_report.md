# Fixed-Rate Survival Analysis

- inference unit: document/request cluster
- aligned documents per strategy: 1
- paired bootstrap replicates: 100
- multiple-comparison control: Holm adjustment over requested paired KL tests
- boundary: fake-quant quality analysis; no native codec, EP network, or latency result

## Paired comparisons

| candidate | reference | candidate_kl | reference_kl | kl_delta | kl_delta_ci_low | kl_delta_ci_high | relative_kl_change | ppl_delta | ppl_delta_ci_low | ppl_delta_ci_high | bootstrap_sign_p | holm_adjusted_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block_gate8_mxfp4 | rank_tail4_mxfp4 | 0.00855873 | 0.00914042 | -0.000581691 | -0.000581691 | -0.000581691 | -0.0636395 | 1.39195 | 1.39195 | 1.39195 | 0 | 0 |
| block_reserr8_residual_mxfp4 | block_gate8_f436_mxfp4 | 0.00938601 | 0.00783196 | 0.00155405 | 0.00155405 | 0.00155405 | 0.198425 | -0.596153 | -0.596153 | -0.596153 | 0 | 0 |

## Gap recovery

Recovery is only defined in bootstrap replicates where the target KL is below the
baseline KL. A low `valid_bootstrap_fraction` means the denominator is unstable
and the recovery ratio must not be used as evidence.

| type | candidate | baseline | target | recovery | recovery_ci_low | recovery_ci_high | valid_bootstrap_fraction |
|---|---|---|---|---|---|---|---|
| rank_to_gate | block_gate8_mxfp4 | rank_tail4_mxfp4 | gate_threshold_mxfp4 | 0.182423 | 0.182423 | 0.182423 | 1 |
| baseline_to_oracle | block_reserr8_residual_mxfp4 | block_gate8_residual_mxfp4 | block_resbenefit8_residual_mxfp4 | nan | nan | nan | 0 |
