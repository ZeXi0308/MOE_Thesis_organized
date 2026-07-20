# Fixed-Rate Survival Analysis

- inference unit: document/request cluster
- aligned documents per strategy: 16
- paired bootstrap replicates: 10000
- multiple-comparison control: Holm adjustment over requested paired KL tests
- boundary: fake-quant quality analysis; no native codec, EP network, or latency result

## Paired comparisons

| candidate | reference | candidate_kl | reference_kl | kl_delta | kl_delta_ci_low | kl_delta_ci_high | relative_kl_change | ppl_delta | ppl_delta_ci_low | ppl_delta_ci_high | bootstrap_sign_p | inference_valid_n_ge_5 | holm_adjusted_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| peerblock_contrib64_mxfp4 | rank_tail4_mxfp4 | 0.00554059 | 0.00615575 | -0.000615161 | -0.000959392 | -0.000197511 | -0.0999326 | -0.0191679 | -0.0481473 | 0.0102353 | 0.00679932 | True | 0.0271973 |
| peerblock_contrib64_mxfp4 | gate_threshold_mxfp4 | 0.00554059 | 0.00585629 | -0.000315701 | -0.000570802 | -6.44599e-05 | -0.053908 | -0.0264661 | -0.0661698 | 0.00629779 | 0.0139986 | True | 0.0419958 |
| peerblock_qerr64_mxfp4 | rank_tail4_mxfp4 | 0.00546512 | 0.00615575 | -0.000690629 | -0.000945371 | -0.000431606 | -0.112192 | 0.00367503 | -0.0283158 | 0.034628 | 0.00019998 | True | 0.00139986 |
| peerblock_qerr64_mxfp4 | gate_threshold_mxfp4 | 0.00546512 | 0.00585629 | -0.000391169 | -0.000764451 | -6.45705e-05 | -0.0667946 | -0.00362316 | -0.0438985 | 0.0288141 | 0.0183982 | True | 0.0419958 |
| peerblock_qerr64_mxfp4 | peerblock_contrib64_mxfp4 | 0.00546512 | 0.00554059 | -7.54681e-05 | -0.000409886 | 0.000227938 | -0.0136209 | 0.0228429 | -0.00220999 | 0.0514138 | 0.665933 | True | 0.665933 |
| peerblock_random64_mxfp4 | peerblock_contrib64_mxfp4 | 0.0214625 | 0.00554059 | 0.0159219 | 0.0136155 | 0.0183204 | 2.87369 | 0.14077 | 0.0894538 | 0.201869 | 0.00019998 | True | 0.00139986 |
| peerblock_reversegate64_mxfp4 | peerblock_contrib64_mxfp4 | 0.0323995 | 0.00554059 | 0.0268589 | 0.0226744 | 0.0316731 | 4.84766 | 0.201357 | 0.128656 | 0.278198 | 0.00019998 | True | 0.00139986 |

## Gap recovery

Recovery is only defined in bootstrap replicates where the target KL is below the
baseline KL. A low `valid_bootstrap_fraction` means the denominator is unstable
and the recovery ratio must not be used as evidence.

| type | candidate | baseline | target | recovery | recovery_ci_low | recovery_ci_high | valid_bootstrap_fraction |
|---|---|---|---|---|---|---|---|
| rank_to_gate | peerblock_contrib64_mxfp4 | rank_tail4_mxfp4 | gate_threshold_mxfp4 | 2.05424 | 1.1514 | 13.3772 | 0.9311 |
| rank_to_gate | peerblock_qerr64_mxfp4 | rank_tail4_mxfp4 | gate_threshold_mxfp4 | 2.30625 | 1.10057 | 19.9227 | 0.9311 |
