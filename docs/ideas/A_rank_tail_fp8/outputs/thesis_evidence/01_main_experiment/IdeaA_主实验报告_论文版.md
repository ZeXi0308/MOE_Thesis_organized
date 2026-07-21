# Idea A 主实验报告：Receiver-Aware Rank-LUT Mixed-Precision Combine for MoE Serving

## 摘要

本报告围绕 Idea A：**Profile-Guided Receiver-Aware Rank-LUT Mixed-Precision Combine for MoE Serving**，完成了一组本地可复现实验，用于验证该方向是否具备继续发展为毕业论文主线的实验基础。

实验核心问题是：在 MoE 推理的 combine 阶段，是否可以利用 router 产生的 top-k rank 作为重要性代理，对低贡献 expert output 使用更低精度，从而减少通信字节与 receiver 侧瓶颈，同时保持较小精度损失。

当前实验结果支持该方向：

- 在 OLMoE top-8 模型上，rank-8 的 median contribution share 仅为 `4.91%`，rank1/rank8 contribution ratio 为 `5.43x`。
- 在同样 byte saving `9.375%` 下，`rank8_int4` 的 KL 为 `0.3614`，而 `rank1_int4` 的 KL 为 `20.9892`，低约 `58.1x`。
- 在第二个 MoE 模型 Mixtral-TinyMistral 上，同样观察到 rank-aware 近似显著优于 rank1 近似：`rank2_int4` KL `2.8718`，`rank1_int4` KL `118.7166`。
- receiver_group 维度存在可观测差异，尤其在 Mixtral-TinyMistral 上更明显，支持将 LUT 设计为 `layer x receiver_group x rank -> precision`。
- serving simulation 表明，rank-aware INT4 能在较低 accuracy 损失下形成可解释的 traffic / latency tradeoff；uniform INT4 虽然省流量更多，但 accuracy 代价过高，不适合作为默认策略。

因此，本报告建议将 Idea A 的主线收敛为：

> **Profile-Guided Receiver-Aware Rank-LUT Mixed-Precision Combine for MoE Serving**

而不是 drop-first 或 uniform quantization 方案。

---

## 1. 研究问题

MoE 模型在推理时，每个 token 通常只路由到 top-k 个 experts。分布式 MoE serving 中，expert output 需要被传回原 token 所在 receiver 进行 weighted combine：

```text
router top-k -> selected experts -> expert outputs -> receiver combine
```

该 combine 阶段可能产生大量跨设备通信。传统做法通常对所有 selected expert output 使用相同精度，例如 BF16 或统一 INT4。但这种 uniform 方案没有利用 router rank 内部的贡献差异。

本实验验证三个问题：

1. **C1：top-k 内部是否存在 rank 长尾？**  
   即 lowest-rank expert output 的 contribution 是否显著小于 rank1。

2. **C2：rank-aware mixed precision 是否比错误 rank 或 uniform quantization 更稳？**  
   即在相同 byte saving 下，近似 low-rank expert output 是否明显优于近似 rank1。

3. **C3：receiver_group 是否有必要进入 LUT？**  
   即不同 receiver group 的 traffic、contribution 和近似误差是否存在差异，足以支持 `Rank-LUT[layer, receiver_group, rank]`。

---

## 2. 方法设计

### 2.1 Contribution 定义

对于每个 token、每层 MoE、每个 selected expert rank，记录：

```text
contribution = gate_weight * ||expert_output||
share = contribution / sum(contribution over top-k)
```

其中：

- `gate_weight` 来自 router top-k softmax。
- `expert_output` 是 expert MLP 输出。
- `rank` 表示 router 选出的 top-k 内部排序，rank1 是最高 gate，rank-k 是最低 gate。

该指标用于判断 top-k 内部是否存在可利用的 rank 长尾。

### 2.2 近似策略

本实验主要比较以下策略：

| 策略 | 含义 |
|---|---|
| `full` | 所有 selected expert output 保持 BF16 |
| `uniform_int4` | 所有 selected expert output 都做 INT4 fake quantization |
| `rank1_int4` | 只将 rank1 expert output 做 INT4 |
| `rankk_int4` / `rank8_int4` / `rank2_int4` | 只将最低 rank expert output 做 INT4 |
| `groupX_rankk_int4` | 只将某个 receiver group 上的最低 rank expert output 做 INT4 |

其中 `rank1_int4` 是重要反例：它与 `rankk_int4` byte saving 相同，但压缩的是最重要的 rank。

### 2.3 Receiver Group 建模

由于本机没有多 GPU 环境，实验采用静态 expert placement 模拟 receiver group：

```text
expert_id -> receiver_group
```

默认使用 contiguous mapping，将 experts 按 id 连续划分到 4 个 receiver groups。该设置不声称等价于真实集群拓扑，但可以模拟 receiver 侧 traffic / placement 差异，并用于验证 `receiver_group` 是否值得进入 LUT 维度。

### 2.4 Serving Simulation

serving simulation 使用真实 routing 统计得到每层、每个 receiver group 的 bytes：

```text
strategy_bytes = selected_count * hidden_size * bytes_per_element
```

再以 receiver group 的最大流量作为该层瓶颈：

```text
layer_bottleneck_bytes = max(bytes_per_receiver_group)
```

假设链路带宽为 100 Gbps，估算相对 latency saving。注意：

> 这里是基于真实 routing traffic 的 simulation，不是多 GPU 实测 latency。

---

## 3. 实验设置

| 模型 | MoE 配置 | Profile 数据 | Approx 数据 | Receiver groups |
|---|---|---:|---:|---:|
| `allenai/OLMoE-1B-7B-0924` | 16 layers, 64 experts, top-8 | WikiText-2 validation 256 samples, seq_len 128 | 32 samples, seq_len 128 | 4 |
| `NickyNicky/Mixtral-TinyMistral-8x248M...` | 12 layers, 8 experts, top-2 | WikiText-2 validation 128 samples, seq_len 128 | 64 samples, seq_len 128 | 4 |

运行环境：

- Mac 本地 CPU-only。
- dtype：`bfloat16`。
- 数据集：WikiText-2 validation。
- 指标：
  - rank contribution share
  - next-token logit KL vs full
  - local combine relative MSE
  - perplexity delta
  - total communication bytes
  - receiver bottleneck bytes
  - simulated latency saving

---

## 4. 实验结果

### 4.1 C1：rank 长尾稳定成立

| 模型 | tail rank | tail-rank median share | rank1/tail-rank median ratio | 结论 |
|---|---:|---:|---:|---|
| OLMoE top-8 | rank-8 | `0.049137` | `5.434607` | 强成立 |
| Mixtral-TinyMistral top-2 | rank-2 | `0.000135` | `14656.703125` | 强成立 |

OLMoE 是最关键的证据，因为它是真 top-8 MoE。结果表明 rank-8 的 contribution median 约为 `4.91%`，低于 10% 强成立阈值。

Mixtral-TinyMistral 的 rank 长尾更极端，虽然它是 top-2 MoE，但可作为跨模型 supporting evidence，说明 rank-aware 现象不是 OLMoE 偶然结果。

结论：

> Router rank 可以作为低成本 importance proxy。低 rank expert output 更适合近似压缩。

### 4.2 C2：同等 byte saving 下，压低 rank 远优于压 rank1

#### OLMoE top-8

所有 `rankN_int4` 都只压一个 rank，byte saving 相同，均为 `0.09375`。

| strategy | byte saving | KL vs full | local relative MSE | PPL delta |
|---|---:|---:|---:|---:|
| `uniform_int4` | `0.75000` | `27.1522` | `0.085818` | `6.6782` |
| `rank1_int4` | `0.09375` | `20.9892` | `0.029478` | `4.4878` |
| `rank8_int4` | `0.09375` | `0.3614` | `0.001274` | `-0.0291` |

关键对比：

- `rank8_int4` 相比 `rank1_int4`，KL 低约 `58.1x`。
- `rank8_int4` 相比 `rank1_int4`，local relative MSE 低约 `23.1x`。
- `uniform_int4` 虽然 byte saving 高，但 KL 明显失控。

#### Mixtral-TinyMistral top-2

`rank1_int4` 和 `rank2_int4` 的 byte saving 相同，均为 `0.375`。

| strategy | byte saving | KL vs full | local relative MSE | PPL delta |
|---|---:|---:|---:|---:|
| `uniform_int4` | `0.75000` | `139.1945` | `0.202308` | `89.8437` |
| `rank1_int4` | `0.37500` | `118.7166` | `0.189019` | `56.9790` |
| `rank2_int4` | `0.37500` | `2.8718` | `0.001349` | `2.8840` |

关键对比：

- `rank2_int4` 相比 `rank1_int4`，KL 低约 `41.3x`。
- `rank2_int4` 相比 `rank1_int4`，local relative MSE 低约 `140.1x`。

结论：

> 在相同 byte saving 下，近似 lowest-rank expert output 明显优于近似 rank1。Idea A 的 rank-aware mixed precision 主张成立。

### 4.3 Receiver Group 维度

#### Receiver group heterogeneity

| 模型 | tail rank | receiver-group median-share max/min | traffic max/mean | 解释 |
|---|---:|---:|---:|---|
| OLMoE | rank-8 | `1.150x` | `1.200x` | group 差异中等，主要用于拥塞/部署控制 |
| Mixtral-TinyMistral | rank-2 | `2.390x` | `1.189x` | group 差异更明显，可支撑 group-aware LUT |

OLMoE 上 receiver_group 差异存在，但不是极端大。因此不能把 receiver_group 夸成唯一主因。更准确的说法是：

> `rank` 是主要 importance signal；`receiver_group` 是部署侧 traffic / placement / congestion 控制维度。

#### Group-specific approximation

OLMoE：只压某个 receiver group 的 rank-8。

| strategy | byte saving | KL vs full | local relative MSE |
|---|---:|---:|---:|
| `group0_rank8_int4` | `0.02371` | `0.2142` | `0.000444` |
| `group1_rank8_int4` | `0.02282` | `0.1776` | `0.000242` |
| `group2_rank8_int4` | `0.02379` | `0.1701` | `0.000329` |
| `group3_rank8_int4` | `0.02333` | `0.1651` | `0.000249` |

Mixtral-TinyMistral：只压某个 receiver group 的 rank-2。

| strategy | byte saving | KL vs full | local relative MSE |
|---|---:|---:|---:|
| `group0_rank2_int4` | `0.09735` | `0.4138` | `0.000366` |
| `group1_rank2_int4` | `0.09286` | `1.0421` | `0.000368` |
| `group2_rank2_int4` | `0.09965` | `1.1114` | `0.000329` |
| `group3_rank2_int4` | `0.08536` | `0.5387` | `0.000251` |

结论：

> receiver_group 维度有必要进入 LUT，但它应作为 rank-aware 策略的补充维度，而不是替代 rank。

建议 LUT 形式：

```text
Rank-LUT[layer, receiver_group, rank] -> precision
```

### 4.4 Serving-side Simulation

#### OLMoE top-8

| strategy | byte saving | bottleneck-byte saving | simulated latency saving | KL vs full |
|---|---:|---:|---:|---:|
| `uniform_int4` | `0.7500` | `0.7500` | `0.7500` | `27.1522` |
| `rank1_int4` | `0.0938` | `0.0822` | `0.0822` | `20.9892` |
| `rank8_int4` | `0.0938` | `0.0945` | `0.0945` | `0.3614` |

#### Mixtral-TinyMistral top-2

| strategy | byte saving | bottleneck-byte saving | simulated latency saving | KL vs full |
|---|---:|---:|---:|---:|
| `uniform_int4` | `0.7500` | `0.7500` | `0.7500` | `139.1945` |
| `rank1_int4` | `0.3750` | `0.4000` | `0.4000` | `118.7166` |
| `rank2_int4` | `0.3750` | `0.3500` | `0.3500` | `2.8718` |

解释：

- Uniform INT4 的 traffic saving 最大，但 accuracy 代价不可接受。
- Rank-aware INT4 的 traffic saving 较小，但 KL / MSE 明显更可控。
- Same-byte comparison 下，低 rank 近似比 rank1 近似可靠得多。

结论：

> Rank-aware mixed precision 是更合理的 accuracy-traffic Pareto 点。

---

### 4.5 Top-16 Stress Test

为了验证当 top-k 进一步增大时，rank-aware 假设是否仍然成立，补充运行了一个真实 top-16 MoE checkpoint：

```text
llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M
```

该模型配置为：

- 16 layers
- hidden size 512
- 32 experts
- `num_experts_per_tok = 16`
- profile：WikiText-2 validation 128 samples
- approximation：WikiText-2 validation 64 samples

Profile 结果：

| model | top-k | tail rank | tail-rank median share | rank1/tail-rank median ratio | 结论 |
|---|---:|---:|---:|---:|---|
| LLM-jp E32-k16 | 16 | rank-16 | `0.020489` | `9.389673` | 强成立 |

Approximation 结果：

| strategy | byte saving | KL vs full | local relative MSE | PPL delta |
|---|---:|---:|---:|---:|
| `uniform_int4` | `0.750000` | `20.6570` | `0.04243705` | `5.6714` |
| `rank1_int4` | `0.046875` | `20.4631` | `0.03618669` | `6.1548` |
| `rank8_int4` | `0.046875` | `0.3507` | `0.00003117` | `0.0033` |
| `rank16_int4` | `0.046875` | `0.1898` | `0.00000653` | `0.0057` |

关键结论：

- `rank16_int4` 与 `rank1_int4` byte saving 相同，均为 `0.046875`。
- `rank16_int4` 的 KL 比 `rank1_int4` 低约 `107.8x`。
- `rank16_int4` 的 local relative MSE 比 `rank1_int4` 低约 `5543x`。
- `rank8_int4` 已经很稳，`rank16_int4` 更稳，说明更大 top-k 下 rank tail 仍然明显。

该模型是研究 checkpoint，不是主流线上 serving 模型。因此它应作为 **top-16 stress test**，用于增强 larger-top-k 说服力；不要把它替代 OLMoE top-8 主证据。

---

## 5. 图表与产物

OLMoE 主实验目录：

```text
experiments/idea_a_mac/outputs/main_experiments/olmoe_wikitext256_g4/
```

关键文件：

- `receiver_group_profile_report.md`
- `rank_share_by_layer.csv`
- `receiver_rank_share.csv`
- `receiver_group_variability.csv`
- `approx_results.csv`
- `serving_simulation.csv`
- `figures/rank_sweep_kl.png`
- `figures/rank_sweep_local_mse.png`
- `figures/serving_accuracy_tradeoff_kl.png`
- `figures/serving_latency.png`

Mixtral-TinyMistral supporting experiment 目录：

```text
experiments/idea_a_mac/outputs/main_experiments/mixtral_tinymistral_wikitext128_g4/
```

关键文件：

- `receiver_group_profile_report.md`
- `rank_share_by_layer.csv`
- `receiver_rank_share.csv`
- `receiver_group_variability.csv`
- `approx_results.csv`
- `serving_simulation.csv`
- `figures/rank_sweep_kl.png`
- `figures/rank_sweep_local_mse.png`
- `figures/serving_accuracy_tradeoff_kl.png`
- `figures/serving_latency.png`

Top-16 stress test 目录：

```text
experiments/idea_a_mac/outputs/main_experiments/llmjp_e32k16_top16_stress/
```

关键文件：

- `top16_stress_test_report.md`
- `receiver_group_profile_report.md`
- `rank_share_by_layer.csv`
- `approx_results.csv`
- `serving_simulation.csv`
- `figures/rank_sweep_kl.png`
- `figures/rank_sweep_local_mse.png`
- `figures/serving_accuracy_tradeoff_kl.png`

---

## 6. 对论文主线的影响

### 6.1 推荐题目

推荐使用：

> **Profile-Guided Receiver-Aware Rank-LUT Mixed-Precision Combine for MoE Serving**

中文可以写成：

> **面向 MoE Serving 的 Profile-Guided Receiver-Aware Rank-LUT 混合精度 Combine 优化**

### 6.2 核心贡献写法

可以写成三点：

1. **Rank-aware contribution profiling**  
   通过 profile 统计 `gate_weight * ||expert_output||`，发现 top-k 内部存在稳定 rank 长尾。

2. **Receiver-aware Rank-LUT policy**  
   构造 `Rank-LUT[layer, receiver_group, rank] -> precision`，在 serving 阶段使用静态查表方式决定 expert output 精度。

3. **Accuracy-traffic tradeoff**  
   在保持较小 KL / MSE 代价的前提下，减少 expert output communication bytes 和 receiver bottleneck traffic。

### 6.3 不建议的写法

不要写成：

```text
我们证明 receiver_group 是最主要影响因素。
```

更稳的写法：

```text
Rank 是主要 importance proxy；receiver_group 则用于表达 receiver-side placement 与 congestion 差异，使 LUT 更贴近 serving 部署。
```

不要写成：

```text
本实验证明多 GPU serving latency 显著下降。
```

更稳的写法：

```text
基于真实 routing traffic 的 serving simulation 显示，该策略能够减少 receiver bottleneck bytes，并形成更好的 accuracy-traffic tradeoff。真实多 GPU latency 需要后续在分布式环境中验证。
```

不要把 drop 当默认主策略。当前结果支持的主策略是：

```text
BF16 / INT4 mixed precision
```

drop 只适合作为 aggressive ablation。

---

## 7. 局限性

当前实验仍有以下限制：

1. **没有真实多 GPU 通信实测**  
   serving latency 是基于 routing traffic 的模拟值。

2. **receiver_group 是静态 placement 模拟**  
   当前使用 expert id 到 receiver group 的静态映射，尚未接入真实集群 placement。

3. **approximation 样本数低于 profile 样本数**  
   OLMoE profile 使用 256 条，但 approximation sweep 使用 32 条，主要受 Mac CPU-only 计算限制。

4. **尚未实现自动 LUT 搜索**  
   当前实验验证了 rank / receiver_group 维度的有效性，但还没有实现完整 optimizer，例如在 KL budget 下搜索最优 precision assignment。

---

## 8. 下一步建议

如果继续推进为最终毕业论文，建议补三类实验：

1. **更完整的数据规模**
   - OLMoE approximation sweep 从 32 条扩展到 128 或 256 条。
   - 增加更长 seq_len，例如 256。

2. **自动 LUT 生成**
   - 输入：profile 统计、receiver traffic、KL/MSE 预算。
   - 输出：`Rank-LUT[layer, receiver_group, rank] -> precision`。
   - 对比：global rank-only LUT、receiver-aware LUT、oracle。

3. **真实 serving 环境验证**
   - 多 GPU 或多进程模拟 all-to-all。
   - 记录真实 communication time、receiver bottleneck、end-to-end decode latency。

---

## 9. 总结

当前实验已经足以支持 Idea A 作为毕业论文主线继续推进。最重要的证据是：

- top-k 内部存在稳定 rank 长尾；
- 同等 byte saving 下，近似 low-rank expert output 远优于近似 rank1；
- 跨两个 MoE 模型结论一致；
- top-16 stress test 进一步说明 larger top-k 下 rank tail 仍然明显；
- receiver_group 维度有实际价值，但应作为部署和拥塞维度谨慎表述；
- rank-aware mixed precision 比 uniform INT4 更适合作为默认策略。

最终建议：

> 将论文主线聚焦于 `Rank-LUT[layer, receiver_group, rank] -> precision` 的 mixed-precision combine，而不是 drop 或 uniform quantization。
