# CreditReduce Mac P0 Result

> Numerical/full-model evidence only; no GPU, network, latency, or actual-wire claim.

## Configuration

- model: `allenai/OLMoE-1B-7B-0924`
- phase/status: `dev` / `PARTIAL`
- topology: EP8, ranks/domain=4, `contiguous`
- samples/seq_len: 1 / 32

## Opportunity

- p_eligible: {'point': 0.993503248375812, 'lcb95': 0.993503248375812, 'ucb95': 0.993503248375812}
- rho_credit: {'point': 0.744127936031984, 'lcb95': 0.744127936031984, 'ucb95': 0.744127936031984}

## Endpoint quality

```text
           endpoint  corpus_ppl  mean_token_kl_vs_late  top1_disagreement_rate  elapsed_seconds_diagnostic_only  delta_nll_mean  delta_nll_lcb95_one_sided  delta_nll_ucb95_one_sided  delta_nll_ci_low_two_sided  delta_nll_ci_high_two_sided  nll_margin quality_status  logical_payload_bytes  minimal_bitmap_bytes  scale_bytes  accounted_bytes  saving_vs_late_payload
          late_bf16   10.649511               0.000000                     0.0                         9.374665        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005    NONINFERIOR              8196096.0                   0.0          0.0        8196096.0                0.000000
pretrained_original   10.772986               0.000649                     0.0                         0.742003        0.011528                   0.011528                   0.011528                    0.011528                     0.011528       0.005   INCONCLUSIVE                    NaN                   NaN          NaN              NaN                     NaN
   stock_early_bf16   10.719860               0.000604                     0.0                         1.405276        0.006584                   0.006584                   0.006584                    0.006584                     0.006584       0.005   INCONCLUSIVE              2097152.0                   0.0          0.0        2097152.0                0.744128
   clean_early_bf16   10.710870               0.000615                     0.0                         1.097039        0.005745                   0.005745                   0.005745                    0.005745                     0.005745       0.005   INCONCLUSIVE              2097152.0                   0.0          0.0        2097152.0                0.744128
 uniform_early_fp32   10.649511               0.000000                     0.0                         1.100976        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005   INCONCLUSIVE              4194304.0                   0.0          0.0        4194304.0                0.488256
            pd_full   10.649511               0.000000                     0.0                         1.356296        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005   INCONCLUSIVE              4141056.0                   0.0          0.0        4141056.0                0.494753
  uniform_early_fp8   10.821708               0.001273                     0.0                         1.343882        0.016040                   0.016040                   0.016040                    0.016040                     0.016040       0.005   INCONCLUSIVE              1048576.0                   0.0       2048.0        1050624.0                0.871814
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
