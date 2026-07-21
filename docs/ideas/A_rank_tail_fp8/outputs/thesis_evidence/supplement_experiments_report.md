# Idea A 补充实验报告：Layer Sensitivity + Routing Drift Attribution

> 日期：2026-06-24
> 模型：`allenai/OLMoE-1B-7B-0924`（16 layers, 64 experts, top-8）
> 数据集：WikiText-2 validation
> 样本数：32（approximation），seq_len 128
> dtype：bfloat16
> receiver groups：4，contiguous mapping
> 平台：Mac M5 Pro / 48GB unified memory，CPU-only

---

## 0. 实验目的

在主实验（`IdeaA_主实验报告_论文版.md`）已验证 C1 长尾和 rank-aware 近似优势的基础上，补充两个关键实验，补全 proposal 中尚未验证的 claim：

1. **Layer Sensitivity（实验 C / Phase 6）**：验证 LUT 的 `layer` 维度是否有直接证据支撑。若层间差异不大，LUT 退化为 `(receiver_group, rank) -> precision`；若差异显著，layer 留在 LUT key 中。

2. **Routing Drift Attribution（C2 归因）**：将端到端精度损失拆解为 routing drift 和纯数值误差两部分，回答 drift 在 combine 近似损失中的占比，验证 C2 claim 并确定 MILP accuracy 约束的标定方式。

---

## 1. Layer Sensitivity

### 1.1 方法

固定策略为 `rank8_int4`（主实验推荐主策略），每次只对一个 MoE layer 启用近似，其余层保持 BF16。对每层独立测量端到端 KL / PPL delta / local relative MSE vs full baseline。

通过标准（与 `IdeaA_前置实验设计_MacM5Pro.md` §5 一致）：
- **KL ratio (max/min) >= 3x** → layer 维度成立
- **< 1.5x** → LUT 退化为 `(receiver_group, rank) -> precision`

### 1.2 结果

| metric | value |
|---|---|
| num layers | 16 |
| max KL | 0.161946（layer 0） |
| min KL | 0.028234（layer 13） |
| **KL ratio (max/min)** | **5.74x** |
| max local MSE | 0.00249264（layer 15） |
| min local MSE | 0.00000046（layer 2） |
| MSE ratio (max/min) | 5364.54x |

完整逐层数据（`layer_sensitivity.csv`）：

| layer | KL | PPL delta | local MSE |
|---|---|---|---|
| 0 | 0.161946 | -0.048779 | 0.00065847 |
| 1 | 0.111872 | -0.015562 | 0.00000920 |
| 2 | 0.108914 | +0.004839 | 0.00000046 |
| 3 | 0.101960 | -0.033648 | 0.00002806 |
| 4 | 0.082244 | -0.028513 | 0.00004110 |
| 5 | 0.078681 | +0.003573 | 0.00056848 |
| 6 | 0.070917 | +0.008860 | 0.00043397 |
| 7 | 0.065395 | +0.011927 | 0.00078813 |
| 8 | 0.053701 | -0.015874 | 0.00026743 |
| 9 | 0.049557 | -0.001561 | 0.00021499 |
| 10 | 0.040249 | -0.007301 | 0.00025135 |
| 11 | 0.041865 | -0.018852 | 0.00040284 |
| 12 | 0.036394 | -0.015250 | 0.00065681 |
| 13 | 0.028234 | +0.000820 | 0.00046120 |
| 14 | 0.028415 | -0.008496 | 0.00086249 |
| 15 | 0.126962 | +0.014074 | 0.00249264 |

### 1.3 结论

> **Layer sensitivity is significant**（KL ratio 5.74x >= 3x）。
> `layer` 维度应保留在 LUT key 中：`Rank-LUT[layer, receiver_group, rank] -> precision`。

层间敏感度分布呈现清晰模式：

- **最敏感层（top 5）**：layer 0 (KL=0.162), layer 15 (KL=0.127), layer 1 (KL=0.112), layer 2 (KL=0.109), layer 3 (KL=0.102)
- **最不敏感层（bottom 5）**：layer 13 (KL=0.028), layer 14 (KL=0.028), layer 12 (KL=0.036), layer 10 (KL=0.040), layer 11 (KL=0.042)

模式解释：
- **早期层（0-3）最敏感**：处理原始 token embedding，近似扰动对下游影响大。
- **最后一层（15）也高度敏感**：直接决定 output logits，local MSE 最高（0.00249）。
- **中间偏后层（10-14）最不敏感**：hidden state 已经过多层变换，对单层近似有缓冲。

对 MILP 的影响：
- 高敏感层（0, 1, 2, 3, 15）固定为 BF16，不进入优化变量。
- 低敏感层（10-14）可更激进降精度，是 MILP 优化的主要受益层。
- 变量规模从 $O(L \times |\text{groups}| \times k \times |\mathcal{P}|)$ 缩减为 $O(L_{\text{low}} \times |\text{groups}| \times k \times |\mathcal{P}|)$，其中 $L_{\text{low}} \approx 11$。

---

## 2. Routing Drift Attribution

### 2.1 方法

对每个样本和策略，执行三遍 forward：

1. **Full forward**（BF16，无近似）——缓存每层 routing decisions `(selected_experts, routing_weights)`。
2. **Approx + locked routing**——套用近似策略，但强制 router 使用 full forward 的 expert selection。隔离纯数值误差。
3. **Approx + free routing**——套用近似策略，router 基于扰动后的 hidden state 重新选 expert。包含数值误差 + routing drift。

分解定义：

```text
drift_contribution = KL_free - KL_locked
drift_fraction     = drift_contribution / KL_free
numerical_fraction = 1 - drift_fraction
```

策略组：`rank8_int4`（主策略）、`rank1_int4`（反例）、`uniform_int4`（激进策略）。

### 2.2 结果

| strategy | mean KL (free) | mean KL (locked) | drift contribution | drift fraction | numerical fraction |
|---|---:|---:|---:|---:|---:|
| `rank8_int4` | 0.3614 | 0.1793 | 0.1820 | **48.14%** | 51.86% |
| `rank1_int4` | 20.9892 | 16.3646 | 4.6246 | 22.11% | 77.89% |
| `uniform_int4` | 27.1522 | 22.6720 | 4.4802 | 14.77% | 85.23% |

逐样本数据见 `drift_per_sample.csv`（32 samples × 3 strategies = 96 rows）。

### 2.3 关键发现

#### 发现 1：主策略 rank8_int4 的 drift fraction 为 48.14%——moderate routing drift

主策略 `rank8_int4` 的端到端 KL 极低（0.3614），但其中约一半（48.14%）来自 routing drift，另一半来自纯数值误差。

这意味着：
- **routing drift 是真实存在的贡献者**，不能忽略。
- 但绝对值很小（drift_contribution = 0.182），不会导致精度失控。
- 对 MILP accuracy 约束的影响：**单层 δ profile 可能低估 cascading loss**，需要做 additivity sanity check。建议 δ 从 end-to-end（free routing）实验标定，而非从 locked routing 实验。

#### 发现 2：rank1_int4 和 uniform_int4 的 drift fraction 较低，但绝对 drift 更高

| strategy | absolute drift | drift fraction | total KL |
|---|---:|---:|---:|
| rank8_int4 | 0.182 | 48.14% | 0.361 |
| rank1_int4 | 4.625 | 22.11% | 20.989 |
| uniform_int4 | 4.480 | 14.77% | 27.152 |

`rank1_int4` 和 `uniform_int4` 的绝对 drift（4.6 和 4.5）远高于 `rank8_int4`（0.18），但因为总 KL 也极高（21 和 27），所以 fraction 反而更低。

解释：高精度损失策略下，数值误差是主因，routing drift 是次要增量。而低精度损失策略（rank8_int4）下，数值误差已经很小，routing drift 的相对占比反而凸显。

#### 发现 3：C2 claim 获得支持

proposal 中 C2 的措辞是"端到端精度退化的相当一部分来自 routing drift"。在主策略 `rank8_int4` 上，drift fraction = 48.14%，属于"相当一部分"。C2 claim 成立。

但需注意：
- C2 不声称 drift 是主因（48% < 50%），而是"相当一部分"。
- 对于高损失策略（rank1_int4, uniform_int4），drift 不是主因，但这两者本来就不作为主策略。
- proposal 中的风险回退方案——"routing drift 占主导：把 $\delta_{l,R,p}$ 收紧成 per-layer cap，考虑叠加 EAQuant 式 routing alignment"——当前不需要触发，因为 drift 不是主导（< 50%）。

### 2.4 对 MILP accuracy 约束的影响

| drift fraction | 对 δ 模型的影响 | 当前状态 | 对策 |
|---|---|---|---|
| > 60%（drift 主导） | 线性可加性不成立 | 未出现 | — |
| 30%–60%（moderate） | 可加性有偏差，需校验 | **rank8_int4 = 48.14%** | δ 从 end-to-end 标定 + additivity sanity check |
| < 30%（numerical 主导） | 可加性成立 | rank1_int4, uniform_int4 | 单层 δ profile 可直接用 |

建议：
1. **δ 标定方式**：从 end-to-end（free routing）实验测量 $\delta_{l,R,p}$，而非从 locked routing 实验。这样 drift 效应自然包含在 δ 中。
2. **Additivity sanity check**：离线对比"全层全 rank 启用 $p$ 的预测损失 $\sum w \cdot \delta$"vs"实测损失"，验证层间可加性偏差。若偏差 > 20%，考虑引入 cascading correction。
3. **MILP 约束形式不变**：线性可加约束 $\sum \delta \cdot x \cdot \text{freq} / \sum \text{freq} \le \epsilon$ 仍然可用，但 ε 需要留出 ~50% 的 drift margin。

---

## 3. 综合结论：对论文主线的影响

### 3.1 LUT 三个维度全部获得直接证据

| 维度 | 证据来源 | 判定 |
|---|---|---|
| `rank` | 主实验 C1/C2（rank8 vs rank1, 58.1x KL） | 强成立 |
| `receiver_group` | 主实验 §4.3（group spread 1.15x–2.39x） | 成立（部署维度） |
| `layer` | **本次实验 §1**（KL ratio 5.74x） | **强成立** |

LUT 最终形式确认为：

```text
Rank-LUT[layer, receiver_group, rank] -> precision
```

三个维度都有实验支撑，不需要退化。

### 3.2 C2 claim 获得支持

| claim | 证据 | 状态 |
|---|---|---|
| C1：top-k 内部 rank 长尾 | 主实验（3 模型强成立） | ✅ 已验证 |
| C2：相当一部分精度损失来自 routing drift | **本次实验 §2**（rank8_int4 drift fraction 48.14%） | ✅ 已验证 |

### 3.3 高敏感层已识别

MILP 的变量规模可缩减：高敏感层（0, 1, 2, 3, 15）固定 BF16，不进入优化。

低敏感层（10, 11, 12, 13, 14）可优先分配低精度档位。

### 3.4 δ 标定策略已确定

由于主策略 drift fraction 为 moderate（48%），δ 应从 end-to-end free routing 实验标定，并做 additivity sanity check。

---

## 4. 产物清单

### Layer Sensitivity

```
experiments/idea_a_mac/outputs/layer_sensitivity/olmoe_wikitext32_g4/
  layer_sensitivity.csv          # 逐层 KL / PPL / MSE
  layer_sensitivity_report.md    # 完整报告
  layer_sensitivity.partial.csv  # 增量保存（容错）
```

### Drift Attribution

```
experiments/idea_a_mac/outputs/drift_attribution/olmoe_wikitext32_g4/
  drift_per_sample.csv           # 逐样本 × 策略（96 rows）
  drift_summary.csv              # 按策略聚合
  drift_attribution_report.md    # 完整报告
  drift_per_sample.partial.csv   # 增量保存（容错）
```

### 实验日志

```
experiments/idea_a_mac/outputs/layer_sensitivity_log.txt
experiments/idea_a_mac/outputs/drift_attribution_log.txt
```

---

## 5. 下一步

两个补充实验已完成，Idea A 的全部 claim（C1/C2）和 LUT 三个维度（layer/receiver_group/rank）均有直接实验证据。建议按以下顺序推进：

1. **自动 LUT 生成**：输入 profile 统计 + receiver traffic + KL/MSE 预算，输出 `Rank-LUT[layer, receiver_group, rank] -> precision`。对比 global rank-only LUT、receiver-aware LUT、oracle。
2. **δ 标定 + additivity check**：从 end-to-end 实验标定 $\delta_{l,R,p}$，验证层间可加性。
3. **迁移到 DeepSeek-V2-Lite / Qwen-MoE**：验证跨模型泛化性。
4. **真实多 GPU serving 验证**：记录真实 communication time、receiver bottleneck、end-to-end decode latency。
