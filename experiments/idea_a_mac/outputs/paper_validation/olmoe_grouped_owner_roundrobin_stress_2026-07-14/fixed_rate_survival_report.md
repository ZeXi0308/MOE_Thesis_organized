# Fixed-Rate Survival Analysis

- inference unit: document/request cluster
- aligned documents per strategy: 12
- paired bootstrap replicates: 10000
- multiple-comparison control: Holm adjustment over requested paired KL tests
- boundary: fake-quant quality analysis; no native codec, EP network, or latency result

## Paired comparisons

| candidate | reference | candidate_kl | reference_kl | kl_delta | kl_delta_ci_low | kl_delta_ci_high | relative_kl_change | ppl_delta | ppl_delta_ci_low | ppl_delta_ci_high | bootstrap_sign_p | inference_valid_n_ge_5 | holm_adjusted_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixed_contribution | mixed_rank | 0.00540806 | 0.00662106 | -0.00121299 | -0.00171555 | -0.000763655 | -0.183203 | -0.0120869 | -0.0587935 | 0.0354906 | 0.00019998 | True | 0.00119988 |
| mixed_contribution | mixed_gate_mass | 0.00540806 | 0.0063949 | -0.000986834 | -0.00122599 | -0.000748785 | -0.154316 | -0.0146445 | -0.0565228 | 0.0301436 | 0.00019998 | True | 0.00119988 |
| mixed_contribution | mixed_pair_contribution | 0.00540806 | 0.00535794 | 5.01173e-05 | -9.10573e-05 | 0.000203892 | 0.00935383 | 0.0037386 | -0.0194361 | 0.0306588 | 0.527147 | True | 1 |
| mixed_contribution | global_contribution | 0.00540806 | 0.00482964 | 0.000578424 | 0.000177954 | 0.000937426 | 0.119765 | -0.0176203 | -0.051037 | 0.022134 | 0.00619938 | True | 0.0185981 |
| mixed_contribution | mixed_qerr | 0.00540806 | 0.00540375 | 4.31505e-06 | -0.000386173 | 0.000439945 | 0.000798528 | -0.0116261 | -0.032671 | 0.0154067 | 0.974703 | True | 1 |
| mixed_contribution | mixed_random | 0.00540806 | 0.0257683 | -0.0203602 | -0.0258591 | -0.0150318 | -0.790127 | -0.106452 | -0.17707 | -0.0373013 | 0.00019998 | True | 0.00119988 |

## Gap recovery

Recovery is only defined in bootstrap replicates where the target KL is below the
baseline KL. A low `valid_bootstrap_fraction` means the denominator is unstable
and the recovery ratio must not be used as evidence.

(none)
