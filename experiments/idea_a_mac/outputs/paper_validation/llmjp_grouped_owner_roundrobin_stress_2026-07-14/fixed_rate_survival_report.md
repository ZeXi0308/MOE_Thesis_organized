# Fixed-Rate Survival Analysis

- inference unit: document/request cluster
- aligned documents per strategy: 12
- paired bootstrap replicates: 10000
- multiple-comparison control: Holm adjustment over requested paired KL tests
- boundary: fake-quant quality analysis; no native codec, EP network, or latency result

## Paired comparisons

| candidate | reference | candidate_kl | reference_kl | kl_delta | kl_delta_ci_low | kl_delta_ci_high | relative_kl_change | ppl_delta | ppl_delta_ci_low | ppl_delta_ci_high | bootstrap_sign_p | inference_valid_n_ge_5 | holm_adjusted_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixed_contribution | mixed_rank | 0.00710826 | 0.00815891 | -0.00105065 | -0.00154218 | -0.000554448 | -0.128773 | 0.022397 | -0.0234353 | 0.0751922 | 0.00039996 | True | 0.00159984 |
| mixed_contribution | mixed_gate_mass | 0.00710826 | 0.00800704 | -0.000898778 | -0.00129492 | -0.000512591 | -0.112249 | 0.0551831 | 0.0201583 | 0.0947089 | 0.00019998 | True | 0.00119988 |
| mixed_contribution | mixed_pair_contribution | 0.00710826 | 0.00731503 | -0.000206771 | -0.000435949 | 2.2074e-05 | -0.0282666 | -0.00220535 | -0.0358055 | 0.0246438 | 0.0805919 | True | 0.161184 |
| mixed_contribution | global_contribution | 0.00710826 | 0.00669744 | 0.000410815 | 9.60525e-05 | 0.000741335 | 0.061339 | 0.0504161 | 0.00522 | 0.102929 | 0.00879912 | True | 0.0263974 |
| mixed_contribution | mixed_qerr | 0.00710826 | 0.00700137 | 0.000106886 | -0.000217394 | 0.000411163 | 0.0152664 | 0.00910068 | -0.0251895 | 0.0484026 | 0.50495 | True | 0.50495 |
| mixed_contribution | mixed_random | 0.00710826 | 0.0360612 | -0.0289529 | -0.0302905 | -0.0277387 | -0.802883 | -0.145233 | -0.276802 | -0.00741899 | 0.00019998 | True | 0.00119988 |

## Gap recovery

Recovery is only defined in bootstrap replicates where the target KL is below the
baseline KL. A low `valid_bootstrap_fraction` means the denominator is unstable
and the recovery ratio must not be used as evidence.

(none)
