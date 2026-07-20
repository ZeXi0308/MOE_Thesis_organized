# Receiver-Group Rank Profile Report

model: `NickyNicky/Mixtral-TinyMistral-8x248M-Instruct_oasst2_chatML_Intel_orca_dpo_pairs_DPO_V1`
samples: `128`
dataset: `wikitext2`
seq_len: `128`
dtype: `bfloat16`
receiver_groups: `4`
receiver_mapping: `contiguous`

## C1 rank long-tail

- rank-2 median share across layers: `0.000135`
- rank1/rank2 median ratio across layers: `14656.703125`
- verdict: **强成立**

## Receiver-group heterogeneity

| receiver_group | rank-2 median share across layers | selected token count |
|---:|---:|---:|
| 0 | 0.000125 | 43332 |
| 1 | 0.000136 | 40657 |
| 2 | 0.000147 | 43408 |
| 3 | 0.000118 | 38479 |

Summary:

- Median max/min receiver-group spread for rank-2 median share: `2.390x`
- Median max/mean receiver traffic imbalance across layers for rank-2: `1.189x`

Interpretation:

- `layer x rank` explains the global importance trend.
- `receiver_group` explains where that traffic and residual sensitivity lands.
- A static LUT shaped as `layer x receiver_group x rank -> precision` is therefore more deployable than one global rank-only rule when receiver-side congestion matters.
