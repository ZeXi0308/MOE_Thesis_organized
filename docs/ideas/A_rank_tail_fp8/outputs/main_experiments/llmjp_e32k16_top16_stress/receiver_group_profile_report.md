# Receiver-Group Rank Profile Report

model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
samples: `128`
dataset: `wikitext2`
seq_len: `128`
dtype: `bfloat16`
receiver_groups: `4`
receiver_mapping: `contiguous`

## C1 rank long-tail

- rank-16 median share across layers: `0.020489`
- rank1/rank16 median ratio across layers: `9.389673`
- verdict: **强成立**

## Receiver-group heterogeneity

| receiver_group | rank-16 median share across layers | selected token count |
|---:|---:|---:|
| 0 | 0.020907 | 52724 |
| 1 | 0.020782 | 53956 |
| 2 | 0.020305 | 54385 |
| 3 | 0.020660 | 54343 |

Summary:

- Median max/min receiver-group spread for rank-16 median share: `1.061x`
- Median max/mean receiver traffic imbalance across layers for rank-16: `1.076x`

Interpretation:

- `layer x rank` explains the global importance trend.
- `receiver_group` explains where that traffic and residual sensitivity lands.
- A static LUT shaped as `layer x receiver_group x rank -> precision` is therefore more deployable than one global rank-only rule when receiver-side congestion matters.
