Native FP8 long-decode occupancy and descriptive BF16 comparison

From research_K_expert_skip project root, with matplotlib available:

  python figures/f1_long_decode_r07/build_long_decode.py \
    --run runs/olmoe_fp8_matrix_r09b \
    --model OLMoE-1B-7B-Instruct --experts 64 --top-k 8 \
    --weight-format 'FP8 W8A8' \
    --output-dir figures/f1_long_decode_fp8_r09b

The existing builder was reused unchanged. bench/status.json is COMPLETE,
63/63 cells. source.json records input hashes; the used raw FP8 sources also
match the completed run SHA256.json (comparison_sources.json).

F1_long_decode.pdf/png/svg: measured FP8 trajectory and uniform iid reference.
occupancy_by_batch.csv: seven pooled numerator/denominator rows.
occupancy_by_layer.csv: 112 layer/batch rows; not independent repetitions.
F1_format_comparison.pdf/png/svg: BF16 r07 and FP8 r09b descriptive overlay.
comparison_bf16_fp8.csv: exact inputs and FP8-minus-BF16 percentage-point differences.
comparison_sources.json: comparison input identity and interpretation limits.

Comparison joins the two occupancy_by_batch.csv files by batch and checks
E/k, prefix/output/decode lengths, layer count, and routed-slot denominator.
The difference is 100*(FP8_singleton_slots - BF16_singleton_slots)/routed_slots.
No mean across format-specific runs or confidence interval is computed.

Both are native-generated fixed-length FULL graph trajectories, prefix128,
2048 outputs/request, 2047 real decode steps, 16 routed layers, E64/k8.
The counted replay is joined only after its generated-token identity with
that format's performance pass is certified. All decode rows are unpadded.
The first output is a prefill prediction and excluded from these counters.
FP8 is online W8A8; nonquantized model dtype and auto KV remain BF16.

FP8 singleton percent for B=1/2/4/8/16/32/64:
100.00000000, 83.14606742, 55.01171684, 29.74002542, 12.05689729, 3.31878588, 0.60527673
FP8 singleton numerators: 262016, 435712, 576558, 623389, 505456, 278264, 101499
Slot denominators per format: 262016, 524032, 1048064, 2096128, 4192256, 8384512, 16769024
FP8 minus BF16 (percentage points): 0.00000000, -2.89524304, -7.62768304, -3.66933699, -1.06479662, -0.12880893, -0.04402164

In this pair of runs, FP8 has lower singleton fractions for B>=2. Both
formats have larger B32/B64 singleton fractions than the uniform iid
reference. These observations do not identify a causal quantization effect:
quantization can change generated content and routes, and there is only one
execution per format/batch. Complete per-expert marginals are absent, so
no fitted skew curve separates marginal skew from within-batch correlation.
EOS is ignored; occupancy is not natural task quality, physical DRAM,
policy byte savings, or latency. The BF16/FP8 overlay does not fix content
or routing. No GPU was used for these derived artifacts.
