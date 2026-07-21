# LUT Evaluation Report

model: `allenai/OLMoE-1B-7B-0924`
samples: `32`
seq_len: `128`
dtype: `bfloat16`
epsilon: `0.1`
baseline PPL: `15.0088`

## Results

| method | actual KL | predicted KL | additivity ratio | PPL delta | byte saving | bottleneck saving |
|---|---|---|---|---|---|---|
| milp | 9.414166 | 0.100000 | 94.14 | 1.441762 | 0.5580 | 0.5408 |
| rank_only | 9.441152 | 0.099902 | 94.50 | 1.853826 | 0.5566 | 0.5451 |
| greedy | 6.622170 | 0.078749 | 84.09 | 1.083744 | 0.5308 | 0.5302 |
| all_bf16 | -0.000000 | 0.000000 | 0.00 | 0.000000 | 0.0000 | 0.0000 |
| uniform_int4 | 27.152189 | 0.223908 | 121.27 | 6.678207 | 0.7500 | 0.7500 |

## Additivity Check

The additivity ratio (actual KL / predicted KL) measures how well the linear
additive delta model predicts the end-to-end accuracy loss.

- ratio ≈ 1.0: additivity holds well
- ratio > 1.5: cascading effects (routing drift) cause underestimation
- ratio < 0.8: overestimation (delta profiles are conservative)
