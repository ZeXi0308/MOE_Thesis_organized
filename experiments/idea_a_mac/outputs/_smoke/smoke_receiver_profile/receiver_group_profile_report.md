# Receiver-Group Rank Profile Report

model: `jamesdborin/tiny-mixtral`
samples: `2`
dataset: `builtin`
seq_len: `32`
dtype: `float32`
receiver_groups: `2`
receiver_mapping: `contiguous`

## C1 rank long-tail

- rank-2 median share across layers: `0.388353`
- rank1/rank2 median ratio across layers: `1.649153`
- verdict: **不成立或证据不足**

## Receiver-group heterogeneity

| receiver_group | rank-2 median share across layers | selected token count |
|---:|---:|---:|
| 0 | 0.366195 | 34 |
| 1 | 0.388438 | 32 |

Summary:

- Median max/min receiver-group spread for rank-2 median share: `1.055x`
- Median max/mean receiver traffic imbalance across layers for rank-2: `1.364x`

Interpretation:

- `layer x rank` explains the global importance trend.
- `receiver_group` explains where that traffic and residual sensitivity lands.
- A static LUT shaped as `layer x receiver_group x rank -> precision` is therefore more deployable than one global rank-only rule when receiver-side congestion matters.
