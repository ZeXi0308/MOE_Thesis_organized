# Top-16 Stress Test Report

## 1. 实验目的

本实验用于回答一个更直接的问题：

> 当 MoE routing 的 top-k 扩大到真实 top-16 时，rank-aware mixed precision 是否仍然成立？

该实验不是替代 OLMoE top-8 主证据，而是作为 larger-top-k stress test，验证 `rank` 作为 importance proxy 在更大 top-k 下是否仍有明显信号。

## 2. 实验配置

- 模型：`llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- 架构：MixtralConfig
- layers：16
- hidden size：512
- experts：32
- top-k：16
- dtype：bfloat16
- profile 数据：WikiText-2 validation，128 samples，seq_len 128
- approximation 数据：WikiText-2 validation，64 samples，seq_len 128
- receiver groups：4

该模型是研究 checkpoint，不是主流线上 serving 模型。因此本实验应写成 **top-16 pressure / stress test**，不要写成唯一主证据。

## 3. C1：top-16 rank 长尾

Profile 结果：

- rank-16 median share across layers：`0.020489`
- rank1/rank16 median ratio across layers：`9.389673`
- C1 verdict：**强成立**

解释：

在真实 top-16 routing 下，rank-16 的 median contribution share 约为 `2.05%`，rank1 contribution 约为 rank16 的 `9.39x`。这说明当 top-k 扩大到 16 时，低 rank tail 仍然明显存在。

## 4. Approximation 结果

| strategy | byte saving | KL vs full | local relative MSE | PPL delta |
|---|---:|---:|---:|---:|
| full | 0.000000 | 0.0000 | 0.00000000 | 0.0000 |
| uniform_int4 | 0.750000 | 20.6570 | 0.04243705 | 5.6714 |
| rank1_int4 | 0.046875 | 20.4631 | 0.03618669 | 6.1548 |
| rank8_int4 | 0.046875 | 0.3507 | 0.00003117 | 0.0033 |
| rank16_int4 | 0.046875 | 0.1898 | 0.00000653 | 0.0057 |

关键对比：

- `rank16_int4` 和 `rank1_int4` 的 byte saving 相同，都是 `0.046875`。
- `rank16_int4` 的 KL 为 `0.1898`，`rank1_int4` 的 KL 为 `20.4631`，rank16 低约 `107.8x`。
- `rank16_int4` 的 local relative MSE 为 `0.00000653`，`rank1_int4` 为 `0.03618669`，rank16 低约 `5543x`。
- `rank8_int4` 已经很稳，`rank16_int4` 更稳，符合 rank 越靠后 contribution 越低的预期。

## 5. Receiver-group 观察

Profile 中 rank-16 的 receiver group 差异：

| receiver_group | rank-16 median share | selected token count |
|---:|---:|---:|
| 0 | 0.020907 | 52724 |
| 1 | 0.020782 | 53956 |
| 2 | 0.020305 | 54385 |
| 3 | 0.020660 | 54343 |

Summary:

- rank-16 median-share max/min spread：`1.061x`
- rank-16 traffic max/mean：`1.076x`

解释：

这个 top-16 checkpoint 上 receiver group 差异较弱，说明它主要支撑 **rank 长尾**，而不是强 receiver-group sensitivity。receiver_group 仍可作为部署控制维度，但不要用该模型夸大 group 差异。

## 6. Serving Simulation

假设 100 Gbps link，基于真实 routing traffic 做 simulation：

| strategy | byte saving | bottleneck-byte saving | simulated latency saving | KL vs full |
|---|---:|---:|---:|---:|
| uniform_int4 | 0.7500 | 0.7500 | 0.7500 | 20.6570 |
| rank1_int4 | 0.0469 | 0.0515 | 0.0515 | 20.4631 |
| rank8_int4 | 0.0469 | 0.0463 | 0.0463 | 0.3507 |
| rank16_int4 | 0.0469 | 0.0424 | 0.0424 | 0.1898 |

解释：

- Uniform INT4 traffic saving 最大，但 KL 代价大。
- `rank16_int4` 只节省约 `4.69%` bytes，但精度代价极低。
- 在相同 single-rank byte saving 下，rank16 明显优于 rank1。

## 7. 对 Idea A 的影响

该 top-16 stress test 强化了核心 claim：

> 在更大的 top-k routing 下，rank tail 仍然存在，并且 lowest-rank INT4 近似明显优于 rank1 INT4。

建议论文中这样定位：

- OLMoE top-8：主证据，贴近真实 top-8 serving。
- Mixtral-TinyMistral top-2：跨模型 supporting evidence。
- LLM-jp E32-k16 top-16：larger-top-k stress test。

不要这样写：

- 不要把该模型写成真实线上主流 serving 模型。
- 不要用它证明 receiver group 差异很强。
- 不要用它替代 OLMoE 主证据。

## 8. 产物

- `receiver_group_profile_report.md`
- `rank_share_by_layer.csv`
- `receiver_rank_share.csv`
- `receiver_group_variability.csv`
- `approx_results.csv`
- `serving_simulation.csv`
- `figures/rank_sweep_kl.png`
- `figures/rank_sweep_local_mse.png`
- `figures/serving_accuracy_tradeoff_kl.png`
