# Rank-Aware Approximation Report

model: `jamesdborin/tiny-mixtral`
samples: `8`
seq_len: `96`

Rank-k INT4 is better than Rank-1 INT4 on KL.

See `approx_results.csv` for byte saving, KL, PPL delta, and local relative MSE.
