# Graceful EP / QTree-EP 生死判决（2026-07-13）

> **已失效，禁止继续引用。** 严格复核发现本版本存在 baseline accumulation-order 不等价、单文档伪重复、test reuse 和 wire 语义混算。修正版见 `../graceful_qtree_corrected_v3_2026-07-13/严格复核后的生死实验结论.md`。

## 一句话结论

不能把当前 Idea A 原样升级成“FP8/FP4 two-lane → Graceful EP → QTree-EP”这一整条主线。

- **Criticality-Aware Graceful EP 作为核心机制：NO-GO。** 即使只让前 4 个 MoE 层、25% token 的后 4 个 rank 未按时到达，相对正常 FP8-head/MXFP4-tail 已增加 `0.007971` KL，95% paired bootstrap CI 为 `[0.006568, 0.009449]`，但逻辑 wire saving 只额外增加约 `1.64` 个百分点（相对 uniform FP8 从 `23.64%` 到 `25.29%`）。
- **原始 two-lane QTree：NO-GO。** 它把每节点一个 partial 拆成 head/tail 两个 partial，EP=8 时反而比节点级 uniform FP8 多 `45.12%` 逻辑字节；EP=16 时多 `15.93%`。它没有打赢最基本的 hierarchical FP8 baseline。
- **Topology-conditioned critical-single partial：窄条件 GO。** 每节点始终只发一个 partial；只在该 partial 全由 tail contribution 构成时使用 MXFP4，否则使用 FP8。EP=8 时相对节点级 FP8 再省 `2.22%`，额外 KL 为 `0.000054`，CI 跨 0；EP=16 时再省 `11.18%`，额外 KL 为 `0.000968`，CI `[0.000544, 0.001408]`。这说明“拓扑分组后出现 tail-only partial 的概率随 EP 形态变化”是值得继续验证的新洞察，但还不足以成为已成立的论文主线。

因此，当前最稳妥的处理是：**保留现有 FP8-first rank-segmented FP4 combine；Graceful 降为极端拥塞下的可选消融；砍掉 two-lane QTree；仅把 critical-single partial 作为下一轮 topology-aware 候选。**

## 1. 实验回答了什么

### 1.1 设置

- 模型：`allenai/OLMoE-1B-7B-0924`，16 个 MoE 层，top-8；
- 测试集：WikiText held-out validation offset 128，16 条样本，seq_len 128，共 1,218 个有效 next-token；
- 校准集：与测试不重叠的 offset 0，8 条样本；
- 数值：FP8 proxy 与硬件对齐的 MXFP4 fake quant；
- 质量：从输入到最终 logits 的端到端 token KL 与 corpus PPL，而非单层 local MSE；
- 统计：按样本 paired bootstrap，1,000 次；补救策略相对 raw 的比较额外使用 10,000 次 bootstrap；
- 通信：逻辑 combine wire-byte proxy，包含 scale/metadata；不是实测网络字节、kernel latency、TPOT 或 P99。

完整原始表见 `survival_summary.csv`、`paired_comparisons.csv` 和 `sample_metrics.csv`。

## 2. Graceful EP：为什么没有通过

正常 two-lane 的基准是：

| 策略 | mean token KL | 相对 uniform FP8 wire saving |
|---|---:|---:|
| uniform FP8 per expert | 0.004678 | 0.00% |
| FP8-head / MXFP4-tail | 0.007141 | 23.64% |

在此基础上模拟 tail contribution 因 deadline 未到达：

| 拥塞方式 | mean token KL | 相对正常 two-lane 的额外 KL | 95% CI | 相对 uniform FP8 saving |
|---|---:|---:|---:|---:|
| 前 4 层，25% token，后 4 rank miss | 0.015112 | 0.007971 | [0.006568, 0.009449] | 25.29% |
| 前 4 层，50% token，后 4 rank miss | 0.024162 | 0.017021 | [0.013182, 0.021894] | 26.94% |
| 前 8 层，50% token，后 4 rank miss | 0.039643 | 0.032502 | [0.025792, 0.039973] | 30.23% |
| 后 4 层，100% token，后 4 rank miss | 0.095852 | 0.088712 | [0.077944, 0.098267] | 30.23% |

关键洞察有三点：

1. **Graceful 的安全区非常窄。** 最温和配置只比正常 two-lane 多省约 `1.64` 个百分点，却让 KL 从 `0.007141` 增至 `0.015112`。
2. **越接近输出的层越不能简单降级。** 同样 drop 后 4 rank，最后 4 层的 KL 远高于最前 4 层，说明系统控制不能只看 rank 或瞬时队列，必须带 layer/trajectory quality budget。
3. **按剩余 gate mass 做 renormalization consistently 更差。** 例如前 4 层、25% token 时 KL 从 `0.015112` 升至 `0.025045`。不能把“保持权重和为 1”当成无损补救。

### 2.1 校准 alpha 也没有救回主线

我们在 disjoint calibration prompts 上，为 `(layer, miss-tail-count)` 拟合一个无额外 wire-byte 的标量补偿：

| 配置 | calibrated - raw KL | 95% CI | 判断 |
|---|---:|---:|---|
| 前 4 层，25% token，后 4 rank | +0.000519 | [-0.000358, 0.001451] | 无改善 |
| 前 4 层，50% token，后 4 rank | -0.001738 | [-0.004054, -0.000158] | 小幅改善，但总额外 KL 仍为 0.015283 |
| 前 8 层，50% token，后 4 rank | -0.000171 | [-0.001272, 0.001161] | 无显著改善 |
| 前 4 层，全部 token，后 2 rank | +0.000882 | [-0.001579, 0.004804] | 无改善 |
| 后 4 层，全部 token，后 4 rank | -0.010069 | [-0.014404, -0.005482] | 有改善，但 KL 仍高达 0.085784 |

因此补偿器只能缓解部分误差，不能改变 Graceful 的质量/字节斜率。若真实系统没有非常大的 tail-latency 收益，它不值得成为核心贡献。

## 3. QTree：真正的强基线是什么

hierarchical aggregation 本身不是创新。正确基线是“节点内精确累加，每节点只发送一个 FP8 partial”，而不是逐 expert FP8。

| EP 配置 | 策略 | mean token KL | 相对节点 FP8 的逻辑字节变化 |
|---|---|---:|---:|
| EP=8, 4 GPU/node | node uniform FP8 | 0.004365 | baseline |
| EP=8, 4 GPU/node | two-lane partial | 0.007402 | **+45.12%** |
| EP=16, 4 GPU/node | node uniform FP8 | 0.004410 | baseline |
| EP=16, 4 GPU/node | two-lane partial | 0.007271 | **+15.93%** |

two-lane 复制了 partial vector 与 metadata；它虽然保留 rank criticality，却破坏了拓扑聚合的最大收益。再次对已经量化的 two-lane partial 量化还会额外增加 `0.003189` KL，CI `[0.002460, 0.003997]`。原始 QTree 设计应停止。

## 4. 唯一值得继续的种子：critical-single partial

新策略不再发 head/tail 两条 lane，而是：

1. 在 NVLink domain / node 内先形成一个 partial；
2. 若该 partial 包含任一 head-rank contribution，使用 FP8；
3. 只有 partial 完全由 tail contribution 构成时才使用 MXFP4；
4. 每个 node/token 始终最多一个 partial，不破坏 hierarchical combine 的向量数优势。

| EP 配置 | critical-single KL | 相对 node FP8 的额外 KL | 95% CI | 相对 node FP8 再省字节 |
|---|---:|---:|---:|---:|
| EP=8, 4 GPU/node | 0.004420 | 0.000054 | [-0.000349, 0.000464] | 2.22% |
| EP=16, 4 GPU/node | 0.005378 | 0.000968 | [0.000544, 0.001408] | 11.18% |

uniform MXFP4 node partial 虽能再省约 47.3%，但 EP=8/16 的额外 KL 分别为 `0.058290/0.050584`，明显不可用。critical-single 的价值不是“FP4 partial 总是好”，而是用 contribution composition 识别少数安全的 topology partial。

这里出现了一个比原始 QTree 更具体的研究问题：

> 当 EP degree、每节点 GPU 数和路由稀疏性改变时，tail-only topology partial 的出现概率如何变化？能否在不增加 partial 数量的前提下，利用其关键性组成选择 wire format？

EP=8 到 EP=16 的增量 saving 从 `2.22%` 上升到 `11.18%`，是一个有洞察的趋势，但目前只有一个模型、一个 placement 和两个 EP 点，不能据此宣称普遍规律。

## 5. 对论文主线的硬建议

### 现在不要写成

> contribution criticality → transport service levels → graceful degradation → QTree two-lane → TPOT/P99。

这条链上两个核心机制已经没有通过离线生死实验；而 TPOT/P99 尚无真实硬件证据。

### 当前可守住的版本

> FP8-first rank-segmented FP4 combine，并探索不增加 partial 数量的 topology-conditioned precision placement。

其中：

- 已有 R-layout 仍是核心方法；
- critical-single 是候选第二机制，不是已证实贡献；
- Graceful 只保留为 rare emergency fallback / negative result ablation；
- two-lane QTree 和 mass renorm 直接停止投入。

## 6. critical-single 的下一轮真正生死门槛

只建议再投入一轮短验证，并设置明确停止条件：

1. 在 OLMoE、LLM-jp 和至少一个不同 top-k / expert-count 模型上扫描 `EP={4,8,16,32}`、`GPU/node={4,8}`；
2. 使用真实或至少 trace-derived expert placement，不只用 modular placement；
3. 报告 tail-only node partial 比例、增量 wire saving、KL CI，验证 `EP/topology → composition → saving` 因果链；
4. 以 hierarchical FP8 为唯一主 baseline，并加入 `sum-then-quantize`、uniform MXFP4、per-expert R-layout；
5. 保留门槛：多数有效配置相对 hierarchical FP8 再省至少 `8%` 慢链路字节，额外 KL 不超过 `0.001`，且不增加 partial vector 数；
6. 通过后才进入两节点 kernel，测 RDMA bytes、pack/dequant 开销、TPOT/P99；未通过则回到纯 R-layout Idea A，不再扩展 QTree 故事。

## 7. 证据边界

本轮能证明：端到端质量趋势、不同补救的相对优劣，以及逻辑数据形态是否可能打赢强基线。

本轮不能证明：

- RDMA priority / QoS 是否能隔离 head 与 tail；
- node partial 是否真的减少跨节点字节或 operator latency；
- kernel pack/dequant/accumulation 开销；
- TPOT、P99、goodput 或多租户 SLO；
- 该规律可泛化到其他模型、路由和拓扑。

所以最终判决是：**整条升级路线不能直接成立；但实验从失败中筛出了一个更窄、更符合强 baseline 的 topology-aware precision placement 种子。**
