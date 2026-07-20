# Fixed-Rate Survival Analysis

- inference unit: document/request cluster
- aligned documents per strategy: 16
- paired bootstrap replicates: 10000
- multiple-comparison control: Holm adjustment over requested paired KL tests
- boundary: fake-quant quality analysis; no native codec, EP network, or latency result

## Paired comparisons

| candidate | reference | candidate_kl | reference_kl | kl_delta | kl_delta_ci_low | kl_delta_ci_high | relative_kl_change | ppl_delta | ppl_delta_ci_low | ppl_delta_ci_high | bootstrap_sign_p | inference_valid_n_ge_5 | holm_adjusted_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| peerblock_gate64_mxfp4 | rank_tail4_mxfp4 | 0.00530369 | 0.00556718 | -0.000263497 | -0.000637021 | 0.000104135 | -0.0473304 | -0.0216217 | -0.0624895 | 0.0164554 | 0.156584 | True | 0.626337 |
| peerblock_gate64_mxfp4 | gate_threshold_mxfp4 | 0.00530369 | 0.00483144 | 0.000472248 | 0.000192249 | 0.000779079 | 0.0977449 | -0.0215882 | -0.0459408 | 0.00254381 | 0.00039996 | True | 0.00239976 |
| peerblock_contrib64_mxfp4 | rank_tail4_mxfp4 | 0.0048206 | 0.00556718 | -0.000746585 | -0.00107191 | -0.000450349 | -0.134105 | 0.0324068 | -0.0087231 | 0.0708423 | 0.00019998 | True | 0.0019998 |
| peerblock_contrib64_mxfp4 | gate_threshold_mxfp4 | 0.0048206 | 0.00483144 | -1.084e-05 | -0.000248252 | 0.000229214 | -0.00224363 | 0.0324403 | 0.00565009 | 0.0580773 | 0.946705 | True | 0.946705 |
| peerblock_qerr64_mxfp4 | rank_tail4_mxfp4 | 0.00465759 | 0.00556718 | -0.000909596 | -0.00124255 | -0.000603036 | -0.163385 | 0.0138374 | -0.0359201 | 0.0615151 | 0.00019998 | True | 0.0019998 |
| peerblock_qerr64_mxfp4 | gate_threshold_mxfp4 | 0.00465759 | 0.00483144 | -0.000173851 | -0.000446695 | 9.92928e-05 | -0.0359833 | 0.0138708 | -0.0242764 | 0.054374 | 0.208379 | True | 0.626337 |
| peerblock_qerr64_mxfp4 | peerblock_contrib64_mxfp4 | 0.00465759 | 0.0048206 | -0.000163011 | -0.000342925 | 1.7667e-05 | -0.0338155 | -0.0185695 | -0.0577101 | 0.0187112 | 0.0763924 | True | 0.381962 |
| peerblock_qbenefit64_mxfp4 | peerblock_qerr64_mxfp4 | 0.00473644 | 0.00465759 | 7.88556e-05 | -4.37881e-05 | 0.000208445 | 0.0169306 | -0.0277772 | -0.0639346 | 0.0048873 | 0.214379 | True | 0.626337 |
| peerblock_random64_mxfp4 | peerblock_qerr64_mxfp4 | 0.0182587 | 0.00465759 | 0.0136011 | 0.0118622 | 0.0156096 | 2.92021 | 0.104636 | 0.0511811 | 0.155725 | 0.00019998 | True | 0.0019998 |
| peerblock_reversegate64_mxfp4 | peerblock_qerr64_mxfp4 | 0.0294649 | 0.00465759 | 0.0248073 | 0.02124 | 0.0289813 | 5.32621 | 0.226447 | 0.135181 | 0.31663 | 0.00019998 | True | 0.0019998 |

## Gap recovery

Recovery is only defined in bootstrap replicates where the target KL is below the
baseline KL. A low `valid_bootstrap_fraction` means the denominator is unstable
and the recovery ratio must not be used as evidence.

| type | candidate | baseline | target | recovery | recovery_ci_low | recovery_ci_high | valid_bootstrap_fraction |
|---|---|---|---|---|---|---|---|
| rank_to_gate | peerblock_gate64_mxfp4 | rank_tail4_mxfp4 | gate_threshold_mxfp4 | 0.358136 | -0.162619 | 0.743601 | 1 |
| rank_to_gate | peerblock_contrib64_mxfp4 | rank_tail4_mxfp4 | gate_threshold_mxfp4 | 1.01473 | 0.681535 | 1.35779 | 1 |
| rank_to_gate | peerblock_qerr64_mxfp4 | rank_tail4_mxfp4 | gate_threshold_mxfp4 | 1.23629 | 0.870204 | 1.68522 | 1 |
