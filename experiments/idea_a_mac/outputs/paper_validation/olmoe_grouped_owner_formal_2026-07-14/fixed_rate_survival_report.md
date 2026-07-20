# Fixed-Rate Survival Analysis

- inference unit: document/request cluster
- aligned documents per strategy: 12
- paired bootstrap replicates: 10000
- multiple-comparison control: Holm adjustment over requested paired KL tests
- boundary: fake-quant quality analysis; no native codec, EP network, or latency result

## Paired comparisons

| candidate | reference | candidate_kl | reference_kl | kl_delta | kl_delta_ci_low | kl_delta_ci_high | relative_kl_change | ppl_delta | ppl_delta_ci_low | ppl_delta_ci_high | bootstrap_sign_p | inference_valid_n_ge_5 | holm_adjusted_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixed_contribution | mixed_rank | 0.00542953 | 0.00652258 | -0.00109306 | -0.00140602 | -0.000738085 | -0.16758 | 0.00132146 | -0.0254669 | 0.0339868 | 0.00019998 | True | 0.00119988 |
| mixed_contribution | mixed_gate_mass | 0.00542953 | 0.00626147 | -0.000831943 | -0.00117943 | -0.000446056 | -0.132867 | 0.00997235 | -0.0383496 | 0.0488899 | 0.00019998 | True | 0.00119988 |
| mixed_contribution | mixed_pair_contribution | 0.00542953 | 0.00556318 | -0.000133651 | -0.00043305 | 0.000127629 | -0.0240242 | 0.00787472 | -0.0247889 | 0.0425842 | 0.353965 | True | 0.707929 |
| mixed_contribution | global_contribution | 0.00542953 | 0.00521146 | 0.00021807 | 2.00897e-05 | 0.000407751 | 0.0418443 | 0.0309156 | -0.00271866 | 0.0720293 | 0.0309969 | True | 0.0929907 |
| mixed_contribution | mixed_qerr | 0.00542953 | 0.00540873 | 2.07964e-05 | -0.000293682 | 0.000317913 | 0.00384496 | 0.017786 | -0.00999578 | 0.050421 | 0.886911 | True | 0.886911 |
| mixed_contribution | mixed_random | 0.00542953 | 0.0153985 | -0.00996896 | -0.0130597 | -0.00741266 | -0.647399 | -0.0246901 | -0.0816164 | 0.0236886 | 0.00019998 | True | 0.00119988 |

## Gap recovery

Recovery is only defined in bootstrap replicates where the target KL is below the
baseline KL. A low `valid_bootstrap_fraction` means the denominator is unstable
and the recovery ratio must not be used as evidence.

(none)
