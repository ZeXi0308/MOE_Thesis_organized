# Fixed-Rate Survival Analysis

- inference unit: document/request cluster
- aligned documents per strategy: 60
- paired bootstrap replicates: 10000
- multiple-comparison control: Holm adjustment over requested paired KL tests
- boundary: fake-quant quality analysis; no native codec, EP network, or latency result

## Paired comparisons

| candidate | reference | candidate_kl | reference_kl | kl_delta | kl_delta_ci_low | kl_delta_ci_high | relative_kl_change | ppl_delta | ppl_delta_ci_low | ppl_delta_ci_high | bootstrap_sign_p | inference_valid_n_ge_5 | holm_adjusted_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| peerblock_contrib64_mxfp4 | rank_tail8_mxfp4 | 0.00569434 | 0.00643081 | -0.000736474 | -0.000895542 | -0.000584952 | -0.114523 | -0.0162655 | -0.0421926 | 0.00930218 | 0.00019998 | True | 0.00139986 |
| peerblock_contrib64_mxfp4 | gate_threshold_mxfp4 | 0.00569434 | 0.00605497 | -0.00036063 | -0.00051091 | -0.000215571 | -0.0595594 | 0.0188127 | -0.00443038 | 0.042429 | 0.00019998 | True | 0.00139986 |
| peerblock_qerr64_mxfp4 | rank_tail8_mxfp4 | 0.00578773 | 0.00643081 | -0.000643081 | -0.000802723 | -0.000484654 | -0.1 | -0.0227052 | -0.0503525 | 0.00492859 | 0.00019998 | True | 0.00139986 |
| peerblock_qerr64_mxfp4 | gate_threshold_mxfp4 | 0.00578773 | 0.00605497 | -0.000267237 | -0.000437964 | -9.61858e-05 | -0.0441352 | 0.012373 | -0.0150931 | 0.0402555 | 0.00219978 | True | 0.00439956 |
| peerblock_qerr64_mxfp4 | peerblock_contrib64_mxfp4 | 0.00578773 | 0.00569434 | 9.33931e-05 | -9.12535e-05 | 0.000279518 | 0.016401 | -0.00643969 | -0.027798 | 0.0151169 | 0.321568 | True | 0.321568 |
| peerblock_random64_mxfp4 | peerblock_contrib64_mxfp4 | 0.054462 | 0.00569434 | 0.0487677 | 0.0469157 | 0.050774 | 8.56424 | 0.741961 | 0.647854 | 0.847258 | 0.00019998 | True | 0.00139986 |
| peerblock_reversegate64_mxfp4 | peerblock_contrib64_mxfp4 | 0.0634098 | 0.00569434 | 0.0577155 | 0.0554369 | 0.0601564 | 10.1356 | 0.851654 | 0.743297 | 0.972357 | 0.00019998 | True | 0.00139986 |

## Gap recovery

Recovery is only defined in bootstrap replicates where the target KL is below the
baseline KL. A low `valid_bootstrap_fraction` means the denominator is unstable
and the recovery ratio must not be used as evidence.

| type | candidate | baseline | target | recovery | recovery_ci_low | recovery_ci_high | valid_bootstrap_fraction |
|---|---|---|---|---|---|---|---|
| rank_to_gate | peerblock_contrib64_mxfp4 | rank_tail8_mxfp4 | gate_threshold_mxfp4 | 1.95952 | 1.5082 | 2.67655 | 1 |
| rank_to_gate | peerblock_qerr64_mxfp4 | rank_tail8_mxfp4 | gate_threshold_mxfp4 | 1.71103 | 1.22259 | 2.46013 | 1 |
