# C1 Long-Tail Report

model: `allenai/OLMoE-1B-7B-0924`
samples: `4`
seq_len: `64`
top_k: `8`
dtype: `bfloat16`

- rank-k median share across layers: `0.055791`
- rank1/rankk median ratio across layers: `4.304844`
- C1 verdict: **强成立**

Interpretation:

- 强成立：多数层 rank-k contribution 很小，可以继续保留 drop / aggressive quantization。
- 弱成立：可以做差分量化，但 drop 要谨慎。
- 不成立或证据不足：不要主打 top-k internal long-tail，先收缩为 rank 是 deployable importance proxy。
