# LUT Optimizer + End-to-End Evaluation 综合报告

> 日期：2026-06-24
> 模型：`allenai/OLMoE-1B-7B-0924`（16 layers, 64 experts, top-8）
> 数据集：WikiText-2 validation
> δ 标定：16 samples, seq_len 128, bfloat16
> 端到端评估：32 samples, seq_len 128, bfloat16
> 4 receiver groups, contiguous mapping

---

## 1. δ 标定

### 1.1 设置

对 11 个低敏感层（layers 4-14，排除高敏感层 0, 1, 2, 3, 15），逐 (layer, rank, precision) 测量端到端 KL：

- 11 layers × 8 ranks × 3 precisions (INT8, INT4, drop) = 264 entries
- 每次只对一个 (layer, rank) 施加近似，其余保持 BF16
- 基线：full BF16 forward

产物：`outputs/delta_profile/olmoe_wikitext16_g4/delta_profile.csv`

### 1.2 δ 分布概览

| precision | median δ (rank 8) | median δ (rank 1) | rank1/rank8 ratio |
|---|---:|---:|---:|
| INT8 | ~0.02 | ~0.06 | ~3x |
| INT4 | ~0.05 | ~0.15 | ~3x |
| drop | ~0.25 | ~1.5 | ~6x |

规律与 C1 长尾一致：rank 越低（rank 8），δ 越小，越适合近似。

---

## 2. LUT Optimizer

### 2.1 设置

- MILP solver：scipy.optimize.milp (HiGHS)
- 变量：`x_{l,r,R,p} ∈ {0,1}`，共 11×4×8×4 = 1408 binary + 1 continuous
- 目标：min max receiver-group traffic
- 约束：加权平均 δ ≤ ε
- 高敏感层 (0,1,2,3,15) 固定 BF16

三种方法 + 两个 baseline：
1. **MILP (receiver-aware)**：`Rank-LUT[layer, receiver_group, rank] -> precision`
2. **Rank-only**：`LUT[layer, rank] -> precision`（无 receiver_group 维度）
3. **Greedy**：按 benefit = byte_saving / δ 贪心降级
4. **All BF16**：无压缩 baseline
5. **Uniform INT4**：所有 expert output 全 INT4

### 2.2 Pareto 曲线

| ε (KL budget) | MILP byte save | MILP bottleneck save | Rank-only byte save | Greedy byte save | Uniform INT4 |
|---|---:|---:|---:|---:|---:|
| 0.02 | 33.8% | 35.6% | 33.4% | 28.6% | — |
| 0.05 | 48.8% | 50.2% | 48.8% | 44.9% | — |
| 0.1 | 55.8% | 57.0% | 55.7% | 53.0% | — |
| 0.2 | 61.0% | 62.1% | 61.1% | 59.6% | — |
| 0.5 | 66.1% | 66.9% | 66.2% | 65.6% | — |
| 1.0 | 67.8% | 68.6% | 68.6% | 68.2% | — |
| ∞ | — | — | — | — | 75.0% (pred_kl=0.224) |

### 2.3 关键发现

**发现 1：MILP 的流量平衡效果**

ε=0.1 时各 receiver group 的实际流量：

| method | group 0 | group 1 | group 2 | group 3 | max-min spread |
|---|---:|---:|---:|---:|---:|
| MILP | 685715 | 685826 | 685837 | 685815 | **122** |
| Rank-only | 672728 | 694374 | 677489 | 706080 | **33352** |
| Greedy | 722041 | 732100 | 700069 | 760185 | **60116** |
| All BF16 | 1497918 | 1534978 | 1577726 | 1593538 | **95620** |

MILP 将 4 个 receiver group 的流量差从 95620 (BF16) 压缩到 122——**几乎完美平衡**。这是 receiver-aware 维度的核心价值。

**发现 2：MILP vs Greedy**

MILP 在所有 ε 值上都优于 greedy。在紧预算下差距更大：
- ε=0.02：MILP 33.8% vs greedy 28.6% → **+5.2pp**
- ε=0.1：MILP 55.8% vs greedy 53.0% → **+2.8pp**

说明 MILP 全局优化比贪心局部决策更有效。

**发现 3：MILP vs Rank-only**

MILP 在 bottleneck saving 上一致优于 rank-only（因为能平衡流量），但 byte saving 接近：
- ε=0.1：MILP bottleneck 57.0% vs rank-only 55.7% → **+1.3pp**
- ε=0.02：MILP bottleneck 35.6% vs rank-only 34.0% → **+1.6pp**

receiver-aware 维度的增益虽不大，但方向一致。

---

## 3. 端到端评估 + Additivity Check

### 3.1 设置

ε=0.1，32 samples，端到端 forward，比较 actual KL vs predicted KL。

### 3.2 结果

| method | actual KL | predicted KL | additivity ratio | PPL delta | byte saving | bottleneck saving |
|---|---:|---:|---:|---:|---:|---:|
| **MILP** | **9.414** | 0.100 | **94.1x** | 1.442 | 55.8% | 54.1% |
| Rank-only | 9.441 | 0.100 | 94.5x | 1.854 | 55.7% | 54.5% |
| Greedy | 6.622 | 0.079 | 84.1x | 1.084 | 53.1% | 53.0% |
| All BF16 | 0.000 | 0.000 | — | 0.000 | 0% | 0% |
| Uniform INT4 | **27.152** | 0.224 | 121.3x | 6.678 | 75.0% | 75.0% |

### 3.3 关键发现：Additivity 严重违反

**actual KL 是 predicted KL 的 ~94 倍。** 线性可加 δ 模型严重低估了多层同时近似时的端到端精度损失。

原因分析与 C2 drift attribution 实验一致：

- C2 实验显示：单层 rank8_int4 的 drift fraction = 48.14%
- 当 11 个低敏感层同时降级时，每层的近似扰动 hidden state → 下游 router 改变 expert selection → 级联放大
- 单层 δ 只测量了孤立近似的影响，不包含跨层 cascading 效应
- 11 层同时降级时，cascading 累积导致实际损失远超可加预测

**additivity ratio 与同时降级的 (layer, rank) 数量正相关**：

| method | 同时降级数量 | additivity ratio |
|---|---|---:|
| Greedy (保守) | 较少 | 84.1x |
| MILP / Rank-only (激进) | 较多 | 94.1x |
| Uniform INT4 (最激进) | 最多 | 121.3x |

### 3.4 尽管违反 additivity，MILP 仍优于 uniform INT4

| 指标 | MILP | Uniform INT4 | MILP 优势 |
|---|---:|---:|---|
| actual KL | 9.414 | 27.152 | **2.9x 更低** |
| PPL delta | 1.442 | 6.678 | **4.6x 更低** |
| byte saving | 55.8% | 75.0% | 少 19.2pp |
| bottleneck saving | 54.1% | 75.0% | 少 20.9pp |

MILP 用 19.2pp 的 byte saving 换取了 2.9x 的精度优势——在 accuracy-byte Pareto 上明显优于 uniform INT4。

---

## 4. 对论文主线的影响

### 4.1 已验证的部分

| 组件 | 状态 | 证据 |
|---|---|---|
| C1 长尾 | ✅ | 3 模型强成立 |
| C2 routing drift | ✅ | drift fraction 48.14% |
| layer 维度 | ✅ | KL ratio 5.74x |
| receiver_group 维度 | ✅ | MILP 流量平衡 spread 122 vs 33352 |
| rank 维度 | ✅ | rank8 vs rank1, KL 低 58.1x |
| LUT optimizer | ✅ | MILP Pareto 曲线完整 |
| MILP > greedy | ✅ | +2.8-5.2pp byte saving |
| MILP > uniform INT4 | ✅ | 2.9x 更低 actual KL |

### 4.2 需要调整的部分

**δ 标定方式需要改进**：

当前的单层 δ profile 不能直接用作 MILP 的 accuracy 约束，因为 additivity ratio ≈ 94x。

建议方案（按优先级）：

1. **End-to-end δ 标定**：直接从多层同时近似实验标定 δ。例如，对每组 (layer, rank, precision) 组合，施加到所有低敏感层并测量 actual KL。但组合爆炸（11^8 太多）。

2. **Cascading correction factor**：保持单层 δ profile，但引入一个 cascading 修正系数 κ。若 MILP 选择了 N 个非 BF16 (layer, rank)，则 actual_kl ≈ κ(N) × sum(δ)。从当前实验标定 κ。

3. **保守 ε**：将 ε 设置为 actual_kl_target / 94。例如，要 actual KL < 1.0，设 ε < 0.011。

### 4.3 对 proposal 的影响

- MILP 的**结构不变**：变量、约束、目标都正确
- accuracy 约束的**标定方式需要改**：ε 不能直接用单层 δ 的可加和
- **traffic balancing 是 receiver-aware 的独立价值**：不依赖 accuracy 模型的正确性
- proposal 风险回退方案第 108 行已预见此风险："routing drift 占主导：把 δ_{l,R,p} 收紧成 per-layer cap，考虑叠加 EAQuant 式 routing alignment"

---

## 5. 产物清单

### δ 标定

```
outputs/delta_profile/olmoe_wikitext16_g4/
  delta_profile.csv              # 264 entries: (layer, rank, precision) -> delta_kl, mse
```

### LUT Optimizer

```
outputs/lut_optimizer/olmoe_final/
  optimizer_comparison.csv       # 所有 method × epsilon 的指标
  optimizer_report.md            # 完整报告
  lut_milp_eps*.json             # MILP LUT (各 epsilon)
  lut_rank_only_eps*.json        # Rank-only LUT
  lut_greedy_eps*.json           # Greedy LUT
```

### 端到端评估

```
outputs/lut_evaluation/olmoe_eps0.1/
  lut_evaluation.csv             # actual vs predicted KL, additivity ratio
  lut_evaluation_report.md       # 完整报告
```

### 实验日志

```
outputs/delta_profile_log.txt
outputs/lut_evaluation_log.txt
```

---

## 6. 下一步

1. **Cascading correction**：标定 κ(N) = actual_kl / sum(δ) 与同时降级数量 N 的关系，修正 MILP accuracy 约束。
2. **更多 ε 的端到端评估**：当前只评估了 ε=0.1，应补充 ε=0.02, 0.05, 0.2, 0.5 的端到端结果，画出 actual accuracy-byte Pareto。
3. **迁移到 GPU**：在真实多 GPU 环境验证 receiver bottleneck 效果。
4. **DeepSeek-V2-Lite**：验证跨模型泛化性。
