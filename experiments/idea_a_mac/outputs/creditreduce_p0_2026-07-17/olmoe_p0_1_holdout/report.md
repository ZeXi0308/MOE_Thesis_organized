# CreditReduce Mac P0 Result

> Numerical/full-model evidence only; no GPU, network, latency, or actual-wire claim.

## Configuration

- model: `allenai/OLMoE-1B-7B-0924`
- phase/status: `p0_1_holdout` / `COMPLETE`
- topology: EP8, ranks/domain=4, `contiguous`
- samples/seq_len: 64 / 256

## Opportunity

- p_eligible: {'point': 0.9901342569483482, 'lcb95': 0.9895344725026374, 'ucb95': 0.9907088006705548}
- rho_credit: {'point': 0.7466049226593316, 'lcb95': 0.7461201646204494, 'ucb95': 0.7470801549970382}

## Endpoint quality

```text
           endpoint  corpus_ppl  mean_token_kl_vs_late  top1_disagreement_rate  elapsed_seconds_diagnostic_only  delta_nll_mean  delta_nll_lcb95_one_sided  delta_nll_ucb95_one_sided  delta_nll_ci_low_two_sided  delta_nll_ci_high_two_sided  nll_margin quality_status  logical_payload_bytes  minimal_bitmap_bytes  scale_bytes  accounted_bytes  saving_vs_late_payload
          late_bf16    7.778348               0.000000                0.000000                       356.015925        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005    NONINFERIOR           4291653632.0                   0.0          0.0     4291653632.0                0.000000
pretrained_original    7.780326               0.001567                0.015074                       317.891652        0.000254                  -0.000584                   0.001104                   -0.000740                     0.001271       0.005    NONINFERIOR                    NaN                   NaN          NaN              NaN                     NaN
   stock_early_bf16    7.778649               0.001531                0.015564                       337.061818        0.000039                  -0.001071                   0.001117                   -0.001288                     0.001327       0.005    NONINFERIOR           1071513600.0                   0.0          0.0     1071513600.0                0.750326
   clean_early_bf16    7.781344               0.001560                0.015380                       356.583639        0.000385                  -0.000674                   0.001447                   -0.000878                     0.001639       0.005    NONINFERIOR           1071513600.0                   0.0          0.0     1071513600.0                0.750326
 uniform_early_fp32    7.778095               0.000003                0.000061                       294.253375       -0.000032                  -0.000087                   0.000000                   -0.000092                     0.000000       0.005    NONINFERIOR           2143027200.0                   0.0          0.0     2143027200.0                0.500652
            pd_full    7.778095               0.000003                0.000061                       296.639368       -0.000032                  -0.000087                   0.000000                   -0.000092                     0.000000       0.005    NONINFERIOR           2118938624.0                   0.0          0.0     2118938624.0                0.506265
  uniform_early_fp8    7.776262               0.002748                0.021446                       299.404636       -0.000268                  -0.001331                   0.000764                   -0.001546                     0.000963       0.005    NONINFERIOR            535756800.0                   0.0    1046400.0      536803200.0                0.874919
```

## Hard decision

```json
{
  "overall": "FAIL",
  "gates": {
    "opportunity_eligible": "PASS",
    "opportunity_credit": "PASS",
    "early_bf16_must_fail": "FAIL",
    "pd_full_noninferior": "PASS",
    "pd_full_equals_uniform_early_fp32": "PASS",
    "legacy_patch_exact": "PASS",
    "late_repeat_deterministic": "PASS",
    "pd_full_payload_cap": "PASS",
    "uniform_fp8_not_dominant": "FAIL"
  },
  "fp8_dominates": true,
  "phase": "p0_1_holdout",
  "evidence_boundary": "full-model numerical quality and logical remote hidden-vector payload only"
}
```
