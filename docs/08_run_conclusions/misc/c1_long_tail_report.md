# C1 Long-Tail Report

model: `jamesdborin/tiny-mixtral`
samples: `8`
seq_len: `96`
top_k: `2`

- rank-k median share across layers: `0.425999`
- rank1/rankk median ratio across layers: `1.330066`
- C1 verdict: **不成立或证据不足**

Interpretation:

- 强成立：多数层 rank-k contribution 很小，可以继续保留 drop / aggressive quantization。
- 弱成立：可以做差分量化，但 drop 要谨慎。
- 不成立或证据不足：不要主打 top-k internal long-tail，先收缩为 rank 是 deployable importance proxy。
