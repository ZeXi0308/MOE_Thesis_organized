# CreditReduce Mac P0 Result

> Numerical/full-model evidence only; no GPU, network, latency, or actual-wire claim.

## Configuration

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- phase/status: `dev` / `PARTIAL`
- topology: EP8, ranks/domain=4, `contiguous`
- samples/seq_len: 1 / 32

## Opportunity

- p_eligible: {'point': 1.0, 'lcb95': 1.0, 'ucb95': 1.0}
- rho_credit: {'point': 0.8747553816046967, 'lcb95': 0.8747553816046967, 'ucb95': 0.8747553816046967}

## Endpoint quality

```text
           endpoint  corpus_ppl  mean_token_kl_vs_late  top1_disagreement_rate  elapsed_seconds_diagnostic_only  delta_nll_mean  delta_nll_lcb95_one_sided  delta_nll_ucb95_one_sided  delta_nll_ci_low_two_sided  delta_nll_ci_high_two_sided  nll_margin quality_status  logical_payload_bytes  minimal_bitmap_bytes  scale_bytes  accounted_bytes  saving_vs_late_payload
          late_bf16   11.631989               0.000000                0.000000                         1.979005        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005    NONINFERIOR              4186112.0                   0.0          0.0        4186112.0                0.000000
pretrained_original   11.322242               0.002650                0.000000                         0.549053       -0.026990                  -0.026990                  -0.026990                   -0.026990                    -0.026990       0.005   INCONCLUSIVE                    NaN                   NaN          NaN              NaN                     NaN
   stock_early_bf16   11.351723               0.002265                0.032258                         1.152878       -0.024389                  -0.024389                  -0.024389                   -0.024389                    -0.024389       0.005   INCONCLUSIVE               524288.0                   0.0          0.0         524288.0                0.874755
   clean_early_bf16   11.462180               0.003517                0.000000                         0.905608       -0.014706                  -0.014706                  -0.014706                   -0.014706                    -0.014706       0.005   INCONCLUSIVE               524288.0                   0.0          0.0         524288.0                0.874755
 uniform_early_fp32   11.631989               0.000000                0.000000                         0.707773        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005   INCONCLUSIVE              1048576.0                   0.0          0.0        1048576.0                0.749511
            pd_full   11.631989               0.000000                0.000000                         0.681358        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005   INCONCLUSIVE              1048576.0                   0.0          0.0        1048576.0                0.749511
  uniform_early_fp8   11.350712               0.008804                0.000000                         0.785820       -0.024479                  -0.024479                  -0.024479                   -0.024479                    -0.024479       0.005   INCONCLUSIVE               262144.0                   0.0       2048.0         264192.0                0.936888
```

## Hard decision

```json
{
  "overall": "NOT_TESTED",
  "reason": "only the complete sealed p0_1_holdout can decide P0-1",
  "phase": "dev",
  "evidence_boundary": "full-model numerical quality and logical remote hidden-vector payload only"
}
```
