Native long-decode occupancy from r07

Run from the research_K_expert_skip project root with a CPU Python environment
containing matplotlib:

  python figures/f1_long_decode_r07/build_long_decode.py \
    --run runs/olmoe_bf16_matrix_r07 \
    --model OLMoE-1B-7B-Instruct --experts 64 --top-k 8 \
    --output-dir figures/f1_long_decode_r07

The builder also accepts the bench directory or its measurements.jsonl.
For a later completed FP8, DeepSeek, or Qwen run, change --run, --model,
--experts, --top-k, and --output-dir. A nonstandard format can be named with
--weight-format. E/k are checked against every recorded routed layer. The
default batch population is 1/2/4/8/16/32/64; --batches can specify a completed
subset explicitly. The input bench must be COMPLETE. Raw data are not changed.

Files:
  F1_long_decode.pdf/png/svg: measured occupancy and uniform iid reference.
  occupancy_by_batch.csv: seven raw numerator/denominator aggregates.
  occupancy_by_layer.csv: 112 layer/batch rows, not independent repetitions.
  source.json: the small source/hash manifest and parsing boundaries.
  interpretation.txt: scope, trajectory, and future format-comparison limits.

r07 results, B increasing:
  singleton percent: 100, 86.041310, 62.639400, 33.409362,
                     13.121694, 3.447595, 0.649298.
  singleton counts: 262016, 450884, 656501, 700303, 550095, 289064, 108881.
  slot denominators: 262016 * B.
Each point covers 16 routed layers * 2047 real decode steps = 32752 layer
events, with prefix 128 and 2048 fixed outputs per request. The first output
is a prefill prediction and is not in these decode counters. All recorded
decode runtime rows have actual B == padded B and FULL graph mode.

The existing plot_results parsing checks that the joined measurement's
logical counters equal the raw counted replay snapshot and that the run
record certifies exact generated-token identity with performance. The builder
does not independently rerun generation. It additionally checks layer sums,
decode populations, and E/k to prevent misleading aggregation/labels.

r05's short BF16 capture used prefix 32 and eight decode steps; its seven-B
singleton percentages were 100, 91.40625, 50.56152, 31.37207, 11.72485,
3.21960, and 0.75836. The long capture observes later native-generated states
and uses a different prefix. Differences are descriptive, not a controlled
effect of observation duration. Both show shrinking singleton opportunity;
the uniform model understates the long-run B32/B64 ratios (1.59312/0.02221%
versus measured 3.44759/0.64930%). Complete per-expert marginals are absent,
so this does not identify skew versus correlation and no fitted curve is made.

A later FP8 capture with the same prefix 128 / output 2048 external protocol
can be compared within that workload definition. Quantization may change
generated content and routes; such a comparison does not hold them fixed.
No confidence interval treats layers or tokens as independent source runs.
No physical DRAM, latency, or natural-completion quality claim follows from
these occupancy ratios. No GPU was used for this derived artifact.
