# Rank-Aware Approximation Report

model: `allenai/OLMoE-1B-7B-0924`
samples: `4`
seq_len: `64`
dtype: `bfloat16`

Rank-k INT4 is better than Rank-1 INT4 on KL.

See `approx_results.csv` for byte saving, KL, PPL delta, and local relative MSE.
