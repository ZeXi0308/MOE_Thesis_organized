# Idea A Main Experiment Report

## 1. 实验目标

这组实验把前置实验扩展成更接近论文主实验的版本，覆盖四个问题：

1. 更大样本下，top-k 内部 rank 长尾是否稳定？
2. 换一个 MoE 模型后，rank-aware 结论是否仍成立？
3. `receiver_group` 维度是否有必要进入 `Rank-LUT[layer, receiver_group, rank]`？
4. 近似策略是否能形成 accuracy / traffic / simulated latency tradeoff？

结论：**Idea A 的主线有足够实验支撑，但应写成 rank-aware mixed precision combine，而不是 drop-first。**

## 2. 实验配置

| model | profile data | approx data | receiver groups | routing |
|---|---:|---:|---:|---|
| `allenai/OLMoE-1B-7B-0924` | WikiText-2 validation 256 samples, seq_len 128 | 32 samples, seq_len 128 | 4 | top-8 |
| `NickyNicky/Mixtral-TinyMistral-8x248M...` | WikiText-2 validation 128 samples, seq_len 128 | 64 samples, seq_len 128 | 4 | top-2 |

本地设备是 Mac CPU-only，因此 serving latency 是基于真实 routing / receiver traffic 统计的模拟值，不是多 GPU 集群实测。

## 3. C1：rank 长尾稳定成立

| model | tail rank | tail-rank median share | rank1/tail-rank median ratio | verdict |
|---|---:|---:|---:|---|
| OLMoE top-8 | rank-8 | 0.049137 | 5.434607 | 强成立 |
| Mixtral-TinyMistral top-2 | rank-2 | 0.000135 | 14656.703125 | 强成立 |

解释：

- OLMoE 是最贴近 Idea A 的证据，因为它是真 top-8 MoE。
- Mixtral-TinyMistral 是 supporting evidence，说明 rank-aware 现象不是 OLMoE 偶然现象。
- 两个模型共同支持：`rank` 可以作为低成本的重要性 proxy。

## 4. C2：同等 byte saving 下，压低 rank 远优于压 rank1

### OLMoE top-8

所有 `rankN_int4` 都只压一个 rank，byte saving 相同，都是 `0.09375`。

| strategy | byte saving | KL vs full | local relative MSE | PPL delta |
|---|---:|---:|---:|---:|
| uniform_int4 | 0.75000 | 27.1522 | 0.085818 | 6.6782 |
| rank1_int4 | 0.09375 | 20.9892 | 0.029478 | 4.4878 |
| rank8_int4 | 0.09375 | 0.3614 | 0.001274 | -0.0291 |

关键结论：

- `rank8_int4` 相比 `rank1_int4`，KL 低约 `58.1x`。
- `rank8_int4` 相比 `rank1_int4`，local relative MSE 低约 `23.1x`。
- `uniform_int4` 虽然 byte saving 更高，但 KL 明显失控，不能作为默认主策略。

### Mixtral-TinyMistral top-2

`rank1_int4` 和 `rank2_int4` 的 byte saving 相同，都是 `0.375`。

| strategy | byte saving | KL vs full | local relative MSE | PPL delta |
|---|---:|---:|---:|---:|
| uniform_int4 | 0.75000 | 139.1945 | 0.202308 | 89.8437 |
| rank1_int4 | 0.37500 | 118.7166 | 0.189019 | 56.9790 |
| rank2_int4 | 0.37500 | 2.8718 | 0.001349 | 2.8840 |

关键结论：

- `rank2_int4` 相比 `rank1_int4`，KL 低约 `41.3x`。
- `rank2_int4` 相比 `rank1_int4`，local relative MSE 低约 `140.1x`。
- 第二个模型支持同一个方向：不是 OLMoE 偶然现象。

## 5. Receiver-group 维度

### Profile heterogeneity

| model | receiver-group median-share max/min | traffic max/mean | interpretation |
|---|---:|---:|---|
| OLMoE rank-8 | 1.150x | 1.200x | group 差异中等，主要用于拥塞/部署控制 |
| Mixtral rank-2 | 2.390x | 1.189x | group 差异更明显，可支撑 group-aware LUT |

### Group-specific approximation

OLMoE：只压某个 receiver group 的 rank-8。

| strategy | byte saving | KL vs full | local relative MSE |
|---|---:|---:|---:|
| group0_rank8_int4 | 0.02371 | 0.2142 | 0.000444 |
| group1_rank8_int4 | 0.02282 | 0.1776 | 0.000242 |
| group2_rank8_int4 | 0.02379 | 0.1701 | 0.000329 |
| group3_rank8_int4 | 0.02333 | 0.1651 | 0.000249 |

Mixtral-TinyMistral：只压某个 receiver group 的 rank-2。

| strategy | byte saving | KL vs full | local relative MSE |
|---|---:|---:|---:|
| group0_rank2_int4 | 0.09735 | 0.4138 | 0.000366 |
| group1_rank2_int4 | 0.09286 | 1.0421 | 0.000368 |
| group2_rank2_int4 | 0.09965 | 1.1114 | 0.000329 |
| group3_rank2_int4 | 0.08536 | 0.5387 | 0.000251 |

解释：

- `rank` 是最强信号；`receiver_group` 是次级但实际有用的部署维度。
- OLMoE 的 group 差异不是极端大，不能夸成“receiver group 决定一切”。
- Mixtral-TinyMistral 的 group-specific KL 差异更明显，支持 `layer x receiver_group x rank` 比全局 rank-only LUT 更灵活。
- 合理写法：`receiver_group` 用来表达接收端拥塞和 placement 差异，而不是替代 rank importance。

## 6. Serving-side simulation

Simulation 使用真实 routing 统计和 receiver traffic，假设 100 Gbps link。绝对 latency 只作模拟，主要看相对 saving。

### OLMoE top-8

| strategy | byte saving | bottleneck-byte saving | simulated latency saving | KL vs full |
|---|---:|---:|---:|---:|
| uniform_int4 | 0.7500 | 0.7500 | 0.7500 | 27.1522 |
| rank1_int4 | 0.0938 | 0.0822 | 0.0822 | 20.9892 |
| rank8_int4 | 0.0938 | 0.0945 | 0.0945 | 0.3614 |

### Mixtral-TinyMistral top-2

| strategy | byte saving | bottleneck-byte saving | simulated latency saving | KL vs full |
|---|---:|---:|---:|---:|
| uniform_int4 | 0.7500 | 0.7500 | 0.7500 | 139.1945 |
| rank1_int4 | 0.3750 | 0.4000 | 0.4000 | 118.7166 |
| rank2_int4 | 0.3750 | 0.3500 | 0.3500 | 2.8718 |

解释：

- Uniform INT4 的 traffic saving 最大，但 accuracy 代价不可接受。
- Rank-aware INT4 的 traffic saving 较小，但 accuracy 代价低得多，是可部署的 Pareto 点。
- Same-byte comparison 下，低 rank 近似显著优于 rank1 近似。

## 7. 论文主张建议

建议把 Idea A 的论文主线写成：

**Profile-Guided Receiver-Aware Rank-LUT Mixed-Precision Combine for MoE Serving**

核心 claim：

1. Profile 阶段统计 `layer x receiver_group x rank` 的 contribution 和 traffic。
2. Serving 阶段用静态 LUT 选择 expert output precision。
3. `rank` 提供低成本 importance proxy。
4. `receiver_group` 表达接收端 placement / congestion 差异。
5. 目标是在可控 KL / PPL 代价下减少 combine traffic 和 receiver bottleneck latency。

不要这样写：

- 不要把 drop 当默认主策略。
- 不要说 receiver group 差异在所有模型上都极端大。
- 不要把本地 simulation 写成真实多 GPU latency measurement。

## 8. 产物路径

OLMoE 主实验：

- `experiments/idea_a_mac/outputs/main_experiments/olmoe_wikitext256_g4/receiver_group_profile_report.md`
- `experiments/idea_a_mac/outputs/main_experiments/olmoe_wikitext256_g4/approx_results.csv`
- `experiments/idea_a_mac/outputs/main_experiments/olmoe_wikitext256_g4/serving_simulation.csv`
- `experiments/idea_a_mac/outputs/main_experiments/olmoe_wikitext256_g4/figures/rank_sweep_kl.png`
- `experiments/idea_a_mac/outputs/main_experiments/olmoe_wikitext256_g4/figures/serving_accuracy_tradeoff_kl.png`

Mixtral-TinyMistral supporting experiment:

- `experiments/idea_a_mac/outputs/main_experiments/mixtral_tinymistral_wikitext128_g4/receiver_group_profile_report.md`
- `experiments/idea_a_mac/outputs/main_experiments/mixtral_tinymistral_wikitext128_g4/approx_results.csv`
- `experiments/idea_a_mac/outputs/main_experiments/mixtral_tinymistral_wikitext128_g4/serving_simulation.csv`
- `experiments/idea_a_mac/outputs/main_experiments/mixtral_tinymistral_wikitext128_g4/figures/rank_sweep_kl.png`
- `experiments/idea_a_mac/outputs/main_experiments/mixtral_tinymistral_wikitext128_g4/figures/serving_accuracy_tradeoff_kl.png`
