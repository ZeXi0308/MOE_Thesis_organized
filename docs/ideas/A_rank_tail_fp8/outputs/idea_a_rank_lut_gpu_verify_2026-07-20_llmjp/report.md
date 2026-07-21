# Idea A Rank-LUT Foundational Claim -- Real GPU Rigorous Re-Verification (llmjp)

- documents: 128 (fresh offset=600), top_k=16, seq_len=128

## Policy summary (byte saving, mean KL, 95% CI)
            policy  byte_saving  mean_kl  kl_ci_low  kl_ci_high
              full     0.000000 0.000000   0.000000    0.000000
        rank1_int4     0.046875 0.201978   0.194765    0.209602
        rankk_int4     0.046875 0.001721   0.001633    0.001809
       uniform_fp8     0.500000 0.005853   0.005678    0.006075
fp8top12_rest_int4     0.562500 0.006387   0.006198    0.006585
 fp8top8_rest_int4     0.625000 0.008614   0.008286    0.009020
 fp8top6_rest_int4     0.656250 0.011696   0.011373    0.012080
 fp8top4_rest_int4     0.687500 0.033945   0.033493    0.034475
      uniform_int4     0.750000 0.207476   0.199354    0.216642

## Claim 1: matched-byte-budget tail-vs-head asymmetry (the 'smoking gun')
- rank1_int4 (head) mean KL = 0.201978
- rankk_int4 (tail) mean KL = 0.001721
- head - tail diff: mean=0.200257, 95% CI=[0.193177, 0.207602]
- head/tail ratio = 117.35x  (GO threshold: CI_low > 0 and ratio > 5.0x)
- VERDICT: GO

## Claim 2: FP8-first tail-INT4 Pareto frontier
- monotone non-decreasing KL as saving increases: True
- uniform_int4 KL CI low = 0.199354
- worst fp8top*_rest_int4 KL CI high = 0.034475
- safety margin (uniform_int4 / worst fp8top*) = 5.782520951552427
- VERDICT: GO