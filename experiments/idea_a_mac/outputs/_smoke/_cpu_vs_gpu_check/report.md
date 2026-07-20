# CreditReduce Mac P0 Result

> Numerical/full-model evidence only; no GPU, network, latency, or actual-wire claim.

## Configuration

- model: `allenai/OLMoE-1B-7B-0924`
- phase/status: `dev` / `PARTIAL`
- topology: EP8, ranks/domain=4, `contiguous`
- samples/seq_len: 2 / 256

## Opportunity

- p_eligible: {'point': 0.9885200974421438, 'lcb95': 0.9874893201513487, 'ucb95': 0.9895466148049107}
- rho_credit: {'point': 0.746406820950061, 'lcb95': 0.7452093250335652, 'ucb95': 0.7475993679348487}

## Endpoint quality

```text
           endpoint  corpus_ppl  mean_token_kl_vs_late  top1_disagreement_rate  elapsed_seconds_diagnostic_only  delta_nll_mean  delta_nll_lcb95_one_sided  delta_nll_ucb95_one_sided  delta_nll_ci_low_two_sided  delta_nll_ci_high_two_sided  nll_margin quality_status  logical_payload_bytes  minimal_bitmap_bytes  scale_bytes  accounted_bytes  saving_vs_late_payload
          late_bf16    9.555953               0.000000                0.000000                        16.521146        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005    NONINFERIOR            134512640.0                   0.0          0.0      134512640.0                0.000000
pretrained_original    9.530248               0.002170                0.023529                         6.037771       -0.002694                  -0.004437                  -0.000950                   -0.004437                    -0.000950       0.005    NONINFERIOR                    NaN                   NaN          NaN              NaN                     NaN
   stock_early_bf16    9.519059               0.001859                0.017647                         7.611693       -0.003868                  -0.007048                  -0.000688                   -0.007048                    -0.000688       0.005    NONINFERIOR             33480704.0                   0.0          0.0       33480704.0                0.751096
   clean_early_bf16    9.536971               0.001384                0.013725                         7.654839       -0.001988                  -0.002816                  -0.001161                   -0.002816                    -0.001161       0.005    NONINFERIOR             33480704.0                   0.0          0.0       33480704.0                0.751096
 uniform_early_fp32    9.555953               0.000000                0.000000                         7.668559        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005    NONINFERIOR             66961408.0                   0.0          0.0       66961408.0                0.502192
            pd_full    9.555953               0.000000                0.000000                         7.864444        0.000000                   0.000000                   0.000000                    0.000000                     0.000000       0.005    NONINFERIOR             66138112.0                   0.0          0.0       66138112.0                0.508313
  uniform_early_fp8    9.562087               0.002243                0.011765                         7.924143        0.000642                  -0.003254                   0.004538                   -0.003254                     0.004538       0.005    NONINFERIOR             16740352.0                   0.0      32696.0       16773048.0                0.875305
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
