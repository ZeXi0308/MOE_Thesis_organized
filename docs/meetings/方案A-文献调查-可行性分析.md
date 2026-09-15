# 方案 A 文献调查与可行性分析

> Profile-Guided Gate-Bucket Partial Combine for MoE Inference  
> 调查时间：2025年6月

---

## 一、调查目标

验证两个"生死假设"的文献支撑情况，确认选题是否有 prior art 撞车。

- **假设 1（长尾）**：Combine contribution $g_{t,e} \cdot \|o_{t,e}\|$ 呈长尾分布，尾部可近似。
- **假设 2（routing drift 主因）**：端到端精度退化的主因是 routing drift（误差通过改变下游 gate 的 routing 决策被非线性放大），而非单层线性误差累加。

---

## 二、长尾分布——有无先例？

### 2.1 直接证据：无人画过

搜索关键词 `"combine contribution distribution"`、`"$g_{t,e} \cdot \|o_{t,e}\|$"`、`"gate weight output norm long tail MoE"`，**零命中**。

所有现有工作分析的是 **expert 粒度的永久重要性**（哪些 expert 可以整体删除），而非 **per-token combine 粒度**的贡献分布。

### 2.2 间接旁证（强烈支持长尾假设）

| 论文 | 核心发现 | 对本假设的支撑 |
|------|---------|---------------|
| **MoDES** (2025) | 多模态 MoE 推理中可跳过 **88% 的 expert**，仍保留 97% 性能 | extreme concentration 的强旁证 |
| **Not All Experts are Equal** (ACL 2024) | 某些 expert 在全部 token 上几乎无贡献，可安全永久删除 | expert 重要性极度不均 |
| MoE 负载均衡是公认训练难题 | 路由天然集中到少数 expert，训练时需加 auxiliary loss 强行打散 | gate score 分布本身即 skewed |

### 2.3 结论

**长尾大概率成立。** 大 top-k（如 top-8）下尾部 expert 贡献可能微乎其微。但 combine contribution distribution plot 仍是一张没人画过的图，是 Phase 0 必须出的关键产出。

---

## 三、Routing Drift——是否被证实？

### 3.1 现有文献直接证实 routing instability 真实存在

| 论文 | 发现 | 证据等级 |
|------|------|---------|
| **EAQuant** (2025/2026, ArXiv) | 量化 MoE 三大挑战之一是 **"Routing Instability"**：量化噪声改变路由器决策，导致 token-expert 错配。提出 "Routing Consistency Alignment" 强制对齐路由分布 | ★★★ 直接证实 |
| **Router Choice Matters** (ICLR 2026 under review) | 核心论点："stabilizing router rankings during calibration is key to accurate low-bit MoE inference" | ★★★ 直接证实 |
| RL + MoE 训练论文 | 动态路由在 RL 训练中引起 **"routing drift"**，导致重要性采样方差高甚至灾难退化 | ★★ 间接证实 |

### 3.2 但这些论文的策略和你的不同

| | EAQuant / Router Choice Matters | 你的方案 A |
|---|---|---|
| 误差来源 | 量化权重/激活 | 丢弃/低精度传尾部 expert 输出 |
| 对 routing drift 的态度 | "消灭它" | "接受它会发生，只在不敏感层启用" |
| 是否做 error decomposition | 否 | **是——你做端到端误差分解（线性 vs routing drift）** |

**上述论文替你证明了 routing drift 是真实存在的退化通道——这反而强化了你做 characterization 的动机。** 审稿人不能说"这问题不存在"。

### 3.3 结论

Routing drift 在 MoE 社区已被认定为独立且重要的退化机制。你的创新在于：**首次将它作为 combine 近似场景下的误差分解 characterization 来做，而非直接当 bug 修复。**

---

## 四、层敏感度差异——低敏感层是否存在？

### 4.1 关键文献：LExI (2025)

**LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference**

核心发现：
- 不同层对扰动敏感度差异巨大
- 早期层和特定功能层高度敏感，减少 expert 数会显著降精度
- 中间层和后期层可大幅削减 active experts，几乎无损
- 在 Qwen1.5-MoE 上，同吞吐量下自适应方案精度比固定 top-k 高 10%

**LExI 自己的推广声明：**
> "The sensitivity profiling methodology extends beyond expert allocation, serving as a foundation for diverse optimization problems such as layer-specific mixed-precision quantization."

这句话直接给你留了位置——你做的是 layer-specific combine approximation based on sensitivity profiling。

### 4.2 结论

**低敏感层大概率存在，已有过硬文献支撑。** 你需要做的是把 LExI 的 layer sensitivity profiling 框架映射到 combine 近似场景上，并加入 routing drift 的误差分解维度。

---

## 五、Combine 近似——是否有 prior art？

### 5.1 搜索详情

| 搜索词 | 命中 | 
|--------|-----|
| `"partial combine" MoE` | 零 |
| `"combine approximation" MoE` | 零 |
| `"gate-based expert output dropping" MoE` | 零 |
| `"approximate weighted sum" MoE communication inference` | 零 |
| `"combine phase" MoE "precision reduction" OR "quantization"` | 零 |

### 5.2 现有相关工作及其与本方案的差异

| 方向 | 代表论文 | 做了什么 | 与本方案的区别 |
|------|---------|---------|---------------|
| **Dispatch 优化** | DeepEP, RailS, ExpertFlow | 优化 token→expert 发送阶段的通信/放置 | 全在 dispatch 阶段，不动 combine |
| **Expert 剪枝** | Not All Experts are Equal (ACL 2024) | 永久删除贡献小的 expert | 模型级永久剪枝，非 per-token 动态 |
| **Expert 跳过** | MoDES (2025), DynaMoE (2026) | 推理时决定跳过哪些 expert 的**计算** | 省的是计算，非通信。两者正交、可互补 |
| **全模型量化** | EAQuant, Router Choice Matters | 将所有权重/激活量化为低精度 | 全局量化，非选择性 combine 降精度 |
| **层自适应 expert 数** | LExI (2025) | 调整每层激活几个 expert | 调的是 dispatch 端 expert 数，非 combine 端精度 |
| **MoE 推理综述** | A Survey on Inference Optimization for MoE (2024) | 全面综述 | 模型层讲了剪枝/量化/蒸馏，系统层讲了分布式/调度，**无 combine 近似条目** |

### 5.3 结论

**Combine 阶段近似是一个真实的文献空白。** 现有所有工作要么在 dispatch 端（送前），要么在 expert 端（算前），要么在全模型量化（一刀切）。没有人做"所有 expert 照算、但 combine 时按 gate 权重选择性降精度/丢弃"这件事。

---

## 六、综合风险评估

| 风险 | 初始评估 | 更新评估 | 依据 |
|------|---------|---------|------|
| 长尾不够长，可近似空间太小 | 🔴 致命 | 🟢 **低** | 旁证极强（MoDES 跳 88%），但需亲手画图 |
| routing drift 太大，几乎没有低敏感层 | 🟡 中 | 🟢 **低** | LExI 已证明层敏感度差异巨大 |
| 已有 prior art 撞车 | 🟡 中 | 🟢 **极低** | 零命中，combine 近似是明确空白 |

---

## 七、关键 Related Work 清单

以下论文需要在论文的 Related Work 中处理：

### 需对比/区分
1. **EAQuant** (2025) — MoE 量化的 routing instability，证明 routing drift 存在
2. **Router Choice Matters** (ICLR 2026) — 路由排序稳定性决定量化质量
3. **Not All Experts are Equal** (ACL 2024) — expert 重要性不均
4. **MoDES** (2025) — expert 跳过，省计算非通信
5. **LExI** (2025) — 层敏感度 profiling
6. **DynaMoE** (2026) — 动态 expert 激活数
7. **A Survey on MoE Inference Optimization** (2024) — 综述，确认 combine 空白

### 需关注的最接近竞争者
- **MoDES + EAQuant 的交叉方向**：如果有人在 dispatch 跳过 + routing consistency 基础上，自然延伸到 combine 近似，时间窗口有限
- **LExI 的扩展**：作者明确写了 sensitivity profiling 可推广到"layer-specific mixed-precision quantization"——如果他们把 precision 从 compute 扩展到 communication，就会撞车

---

## 八、时间窗口判断

三股力量正在向 combine 方向汇聚：
- MoDES 证明了"不激活某些 expert"几乎无损
- EAQuant 证明了 routing stability 是关键
- LExI 证明了层敏感度 profiling 是有效方法论

**但还没有人把它们组合成 "combine 近似 + 长尾 characterization + routing drift decomposition" 这一枪。** 时间窗口可能不到一个会议周期，建议尽快推进 Phase 0 实验出图。
