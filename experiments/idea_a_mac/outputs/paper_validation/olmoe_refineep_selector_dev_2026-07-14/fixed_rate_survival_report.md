# Fixed-Rate Survival Analysis

- inference unit: document/request cluster
- aligned documents per strategy: 16
- paired bootstrap replicates: 10000
- multiple-comparison control: Holm adjustment over requested paired KL tests
- boundary: fake-quant quality analysis; no native codec, EP network, or latency result

## Paired comparisons

| candidate | reference | candidate_kl | reference_kl | kl_delta | kl_delta_ci_low | kl_delta_ci_high | relative_kl_change | ppl_delta | ppl_delta_ci_low | ppl_delta_ci_high | bootstrap_sign_p | inference_valid_n_ge_5 | holm_adjusted_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block_gate8_mxfp4 | rank_tail4_mxfp4 | 0.00505781 | 0.00556718 | -0.000509378 | -0.000831208 | -0.000243094 | -0.0914965 | 0.0196969 | -0.0226543 | 0.0605068 | 0.00019998 | True | 0.00219978 |
| block_gate8_mxfp4 | gate_threshold_mxfp4 | 0.00505781 | 0.00483144 | 0.000226367 | 3.29825e-05 | 0.000414176 | 0.046853 | 0.0197304 | -0.000289929 | 0.0411638 | 0.0253975 | True | 0.10159 |
| block_contrib8_mxfp4 | rank_tail4_mxfp4 | 0.00456999 | 0.00556718 | -0.000997197 | -0.00125486 | -0.000754563 | -0.17912 | 0.00337346 | -0.0386068 | 0.045512 | 0.00019998 | True | 0.00219978 |
| block_contrib8_mxfp4 | gate_threshold_mxfp4 | 0.00456999 | 0.00483144 | -0.000261451 | -0.000477971 | -5.25982e-05 | -0.0541146 | 0.00340689 | -0.027784 | 0.0339866 | 0.0117988 | True | 0.0589941 |
| block_qerr8_mxfp4 | gate_threshold_mxfp4 | 0.0046257 | 0.00483144 | -0.000205739 | -0.000420545 | -6.22818e-06 | -0.0425834 | 0.0298197 | 0.0061318 | 0.0610081 | 0.0447955 | True | 0.134387 |
| block_contrib8_mxfp4 | block_qerr8_mxfp4 | 0.00456999 | 0.0046257 | -5.57126e-05 | -0.000182551 | 6.82406e-05 | -0.0120441 | -0.0264128 | -0.0560659 | -0.00246504 | 0.391561 | True | 0.783122 |
| block_reserr16_residual_mxfp4 | block_qerr16_f436_mxfp4 | 0.00430504 | 0.00409093 | 0.000214108 | 8.5816e-05 | 0.000345366 | 0.0523373 | 0.0151742 | -0.0112152 | 0.0375827 | 0.0009999 | True | 0.0069993 |
| block_contrib16_residual_mxfp4 | block_contrib16_f436_mxfp4 | 0.00427158 | 0.00436617 | -9.45922e-05 | -0.000367007 | 0.000218775 | -0.0216648 | 0.000480554 | -0.0302118 | 0.0244635 | 0.523948 | True | 0.783122 |
| block_qerr16_f436_mxfp4 | gate_threshold_matchedwire_mxfp4 | 0.00409093 | 0.00451173 | -0.000420797 | -0.000711527 | -0.000140865 | -0.0932674 | 0.00909531 | -0.0282488 | 0.0482004 | 0.0029997 | True | 0.0179982 |
| block_random8_mxfp4 | block_contrib8_mxfp4 | 0.0182111 | 0.00456999 | 0.0136411 | 0.0116812 | 0.0157944 | 2.98494 | 0.113172 | 0.045533 | 0.172556 | 0.00019998 | True | 0.00219978 |
| block_reversegate8_mxfp4 | block_contrib8_mxfp4 | 0.0274178 | 0.00456999 | 0.0228478 | 0.019654 | 0.0265655 | 4.99953 | 0.179441 | 0.108995 | 0.255276 | 0.00019998 | True | 0.00219978 |

## Gap recovery

Recovery is only defined in bootstrap replicates where the target KL is below the
baseline KL. A low `valid_bootstrap_fraction` means the denominator is unstable
and the recovery ratio must not be used as evidence.

| type | candidate | baseline | target | recovery | recovery_ci_low | recovery_ci_high | valid_bootstrap_fraction |
|---|---|---|---|---|---|---|---|
| rank_to_gate | block_gate8_mxfp4 | rank_tail4_mxfp4 | gate_threshold_mxfp4 | 0.692329 | 0.388664 | 0.953739 | 1 |
| rank_to_gate | block_contrib8_mxfp4 | rank_tail4_mxfp4 | gate_threshold_mxfp4 | 1.35536 | 1.06537 | 1.77498 | 1 |
| rank_to_gate | block_qerr8_mxfp4 | rank_tail4_mxfp4 | gate_threshold_mxfp4 | 1.27963 | 1.00692 | 1.7204 | 1 |
| baseline_to_oracle | block_reserr16_residual_mxfp4 | block_gate16_residual_mxfp4 | block_resbenefit16_residual_mxfp4 | 0.99423 | 0.426247 | 1.8596 | 0.9919 |
