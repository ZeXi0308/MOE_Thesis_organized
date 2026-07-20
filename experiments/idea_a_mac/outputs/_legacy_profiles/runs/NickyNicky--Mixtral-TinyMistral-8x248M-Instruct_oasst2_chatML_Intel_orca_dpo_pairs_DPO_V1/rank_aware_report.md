# Rank-Aware Approximation Report

model: `NickyNicky/Mixtral-TinyMistral-8x248M-Instruct_oasst2_chatML_Intel_orca_dpo_pairs_DPO_V1`
samples: `8`
seq_len: `96`

Rank-k INT4 is better than Rank-1 INT4 on KL.

See `approx_results.csv` for byte saving, KL, PPL delta, and local relative MSE.
