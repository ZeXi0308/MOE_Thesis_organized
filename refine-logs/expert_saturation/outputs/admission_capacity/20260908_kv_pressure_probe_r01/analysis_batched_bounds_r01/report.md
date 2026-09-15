# Batched idle-expert bounds (arithmetic only)

measured expert weights 12.00 GiB, KV region 15.00 GiB, top-k 8/64

The `uniform null` column is a null model, **not** this router's behaviour.
`disjoint floor` and `concentrated ceiling` bracket every possible routing.

| batch width | uniform null idle | idle experts | idle GiB | vs per-token claim | floor GiB | ceiling GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8750 | 56.0 | 10.50 | 100.0% | 10.50 | 10.50 |
| 2 | 0.7656 | 49.0 | 9.19 | 87.5% | 9.00 | 10.50 |
| 4 | 0.5862 | 37.5 | 7.03 | 67.0% | 6.00 | 10.50 |
| 8 | 0.3436 | 22.0 | 4.12 | 39.3% | 0.00 | 10.50 |
| 12 | 0.2014 | 12.9 | 2.42 | 23.0% | 0.00 | 10.50 |
| 16 | 0.1181 | 7.6 | 1.42 | 13.5% | 0.00 | 10.50 |
| 24 | 0.0406 | 2.6 | 0.49 | 4.6% | 0.00 | 10.50 |
| 32 | 0.0139 | 0.9 | 0.17 | 1.6% | 0.00 | 10.50 |

## What this corrects

- `20260908_kv_pressure_probe_r01` reported idle expert 10.50 GiB using the per-token ratio.
- That value is an upper bound valid only at batch width 1.
- At the batch widths this repository actually measured (16-32) the uniform null
  predicts 1.42-0.17 GiB,
  i.e. 1.6%-13.5% of the reported figure.

## What still has to be measured

Real routers concentrate load, so the true idle fraction lies between the floor
and ceiling columns and cannot be derived here. The decisive quantity is the
per-step, per-layer **union** of experts selected across all tokens in the batch,
which requires a route trace from the live engine.
