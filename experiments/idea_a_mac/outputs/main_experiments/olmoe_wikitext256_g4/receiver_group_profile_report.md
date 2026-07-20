# Receiver-Group Rank Profile Report

model: `allenai/OLMoE-1B-7B-0924`
samples: `256`
dataset: `wikitext2`
seq_len: `128`
dtype: `bfloat16`
receiver_groups: `4`
receiver_mapping: `contiguous`

## C1 rank long-tail

- rank-8 median share across layers: `0.049137`
- rank1/rank8 median ratio across layers: `5.434607`
- verdict: **强成立**

## Receiver-group heterogeneity

| receiver_group | rank-8 median share across layers | selected token count |
|---:|---:|---:|
| 0 | 0.048698 | 95768 |
| 1 | 0.048170 | 95818 |
| 2 | 0.049921 | 96548 |
| 3 | 0.048868 | 99626 |

Summary:

- Median max/min receiver-group spread for rank-8 median share: `1.150x`
- Median max/mean receiver traffic imbalance across layers for rank-8: `1.200x`

Interpretation:

- `layer x rank` explains the global importance trend.
- `receiver_group` explains where that traffic and residual sensitivity lands.
- A static LUT shaped as `layer x receiver_group x rank -> precision` is therefore more deployable than one global rank-only rule when receiver-side congestion matters.
