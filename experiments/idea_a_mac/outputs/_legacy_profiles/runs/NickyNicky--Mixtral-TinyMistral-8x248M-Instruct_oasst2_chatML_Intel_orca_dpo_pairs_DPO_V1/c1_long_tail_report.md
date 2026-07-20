# C1 Long-Tail Report

model: `NickyNicky/Mixtral-TinyMistral-8x248M-Instruct_oasst2_chatML_Intel_orca_dpo_pairs_DPO_V1`
samples: `8`
seq_len: `96`
top_k: `2`

- rank-k median share across layers: `0.000152`
- rank1/rankk median ratio across layers: `10221.473877`
- C1 verdict: **强成立**

Interpretation:

- 强成立：多数层 rank-k contribution 很小，可以继续保留 drop / aggressive quantization。
- 弱成立：可以做差分量化，但 drop 要谨慎。
- 不成立或证据不足：不要主打 top-k internal long-tail，先收缩为 rank 是 deployable importance proxy。
