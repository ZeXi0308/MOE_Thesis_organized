# Idea A Rank-LUT Foundational Claim -- Real GPU Rigorous Re-Verification (olmoe)

- documents: 128 (fresh offset=600), top_k=8, seq_len=128

## Policy summary (byte saving, mean KL, 95% CI)
           policy  byte_saving  mean_kl  kl_ci_low  kl_ci_high
             full      0.00000 0.000000   0.000000    0.000000
       rank1_int4      0.09375 0.208147   0.198083    0.219494
       rankk_int4      0.09375 0.004326   0.004062    0.004617
      uniform_fp8      0.50000 0.003845   0.003632    0.004072
fp8top6_rest_int4      0.56250 0.009574   0.009113    0.010103
fp8top4_rest_int4      0.62500 0.023341   0.022498    0.024243
fp8top3_rest_int4      0.65625 0.036035   0.034729    0.037489
fp8top2_rest_int4      0.68750 0.051709   0.049958    0.053607
     uniform_int4      0.75000 0.258886   0.249373    0.269695

## Claim 1: matched-byte-budget tail-vs-head asymmetry (the 'smoking gun')
- rank1_int4 (head) mean KL = 0.208147
- rankk_int4 (tail) mean KL = 0.004326
- head - tail diff: mean=0.203821, 95% CI=[0.193556, 0.214369]
- head/tail ratio = 48.11x  (GO threshold: CI_low > 0 and ratio > 5.0x)
- VERDICT: GO

## Claim 2: FP8-first tail-INT4 Pareto frontier
- monotone non-decreasing KL as saving increases: True
- uniform_int4 KL CI low = 0.249373
- worst fp8top*_rest_int4 KL CI high = 0.053607
- safety margin (uniform_int4 / worst fp8top*) = 4.651908733179959
- VERDICT: NO-GO