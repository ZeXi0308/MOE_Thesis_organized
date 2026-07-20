# Fixed-Rate Survival Analysis

- inference unit: document/request cluster
- aligned documents per strategy: 12
- paired bootstrap replicates: 10000
- multiple-comparison control: Holm adjustment over requested paired KL tests
- boundary: fake-quant quality analysis; no native codec, EP network, or latency result

## Paired comparisons

| candidate | reference | candidate_kl | reference_kl | kl_delta | kl_delta_ci_low | kl_delta_ci_high | relative_kl_change | ppl_delta | ppl_delta_ci_low | ppl_delta_ci_high | bootstrap_sign_p | inference_valid_n_ge_5 | holm_adjusted_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixed_contribution | mixed_rank | 0.00830924 | 0.0102864 | -0.00197715 | -0.00275024 | -0.00124387 | -0.19221 | -0.0865721 | -0.140353 | -0.0188743 | 0.00019998 | True | 0.00119988 |
| mixed_contribution | mixed_gate_mass | 0.00830924 | 0.010705 | -0.00239572 | -0.00332482 | -0.00150057 | -0.223795 | -0.0454953 | -0.0883222 | -0.00951559 | 0.00019998 | True | 0.00119988 |
| mixed_contribution | mixed_pair_contribution | 0.00830924 | 0.00861885 | -0.000309604 | -0.00107711 | 0.0004588 | -0.0359218 | -0.0116328 | -0.0543196 | 0.0230275 | 0.427957 | True | 0.855914 |
| mixed_contribution | global_contribution | 0.00830924 | 0.0067929 | 0.00151634 | 0.00101249 | 0.00199466 | 0.223224 | -0.0417864 | -0.0929055 | 0.00954417 | 0.00019998 | True | 0.00119988 |
| mixed_contribution | mixed_qerr | 0.00830924 | 0.00823482 | 7.44224e-05 | -0.000623307 | 0.000695829 | 0.00903753 | -0.0217453 | -0.0659016 | 0.0188341 | 0.80372 | True | 0.855914 |
| mixed_contribution | mixed_random | 0.00830924 | 0.052695 | -0.0443858 | -0.0479632 | -0.0408349 | -0.842314 | -0.70418 | -0.906796 | -0.505911 | 0.00019998 | True | 0.00119988 |

## Gap recovery

Recovery is only defined in bootstrap replicates where the target KL is below the
baseline KL. A low `valid_bootstrap_fraction` means the denominator is unstable
and the recovery ratio must not be used as evidence.

(none)
