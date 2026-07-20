# CreditReduce Mac P0 Result

> Numerical/full-model evidence only; no GPU, network, latency, or actual-wire claim.

## Configuration

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- phase/status: `p0_1_holdout` / `COMPLETE`
- topology: EP8, ranks/domain=4, `contiguous`
- samples/seq_len: 64 / 256

## Opportunity

- p_eligible: {'point': 1.0, 'lcb95': 1.0, 'ucb95': 1.0}
- rho_credit: {'point': 0.8749734106736948, 'lcb95': 0.8749185240028913, 'ucb95': 0.875027352545348}

## Endpoint quality

```text
           endpoint  corpus_ppl  mean_token_kl_vs_late  top1_disagreement_rate  elapsed_seconds_diagnostic_only  delta_nll_mean  delta_nll_lcb95_one_sided  delta_nll_ucb95_one_sided  delta_nll_ci_low_two_sided  delta_nll_ci_high_two_sided  nll_margin quality_status  logical_payload_bytes  minimal_bitmap_bytes  scale_bytes  accounted_bytes  saving_vs_late_payload
          late_bf16   13.061050               0.000000                0.000000                       181.985199        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005    NONINFERIOR           2147026944.0                   0.0          0.0     2147026944.0                0.000000
pretrained_original   13.070421               0.001516                0.019792                       119.721754        0.000717                  -0.000088                   0.001509                   -0.000237                     0.001666       0.005    NONINFERIOR                    NaN                   NaN          NaN              NaN                     NaN
   stock_early_bf16   13.058319               0.001483                0.020404                       211.854786       -0.000209                  -0.000912                   0.000515                   -0.001035                     0.000642       0.005    NONINFERIOR            268435456.0                   0.0          0.0      268435456.0                0.874973
   clean_early_bf16   13.065569               0.001561                0.019792                       208.054656        0.000346                  -0.000429                   0.001115                   -0.000578                     0.001265       0.005    NONINFERIOR            268435456.0                   0.0          0.0      268435456.0                0.874973
 uniform_early_fp32   13.061050               0.000000                0.000000                       196.388062        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005    NONINFERIOR            536870912.0                   0.0          0.0      536870912.0                0.749947
            pd_full   13.061050               0.000000                0.000000                       207.628359        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005    NONINFERIOR            536870912.0                   0.0          0.0      536870912.0                0.749947
  uniform_early_fp8   13.084105               0.003390                0.026777                       212.695706        0.001764                   0.000554                   0.002971                    0.000318                     0.003207       0.005    NONINFERIOR            134217728.0                   0.0    1048576.0      135266304.0                0.936998
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
