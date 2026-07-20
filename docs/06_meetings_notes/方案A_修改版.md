# 方案 A（修改版）：Receiver-Port 拥塞下的 Gate-Aware Partial Combine

> Profile-Guided, Receiver-Aware Partial Combine for MoE Inference

本文档基于导师 (Qiaolun) 在 first_meeting 中的 7 条反馈，对原方案 A 做了系统性修改。**原版的"灵魂"保留**——combine 阶段是 MoE 通信里唯一可以做 gate-aware 差分精度的位置；**改动集中在三处**：拥塞模型简化、优化变量规范化、claim 顺序重排（先做基础精度实验、再做 drift 拆解）。

---

## 1. 一句话定位

现有 MoE 通信优化几乎都集中在 dispatch 和 expert placement 上；combine 阶段被忽视的不是"传输本身"，而是它独有的一个自由度——gate 权重作为重要性信号，免费可得、可差分化、且在 dispatch 阶段结构上根本做不到。本工作首先实测 **drop/quantize 在 combine 上的端到端精度代价**与 **top-k 内部 contribution 长尾**是否成立，再在此 evidence 之上构建一个 **receiver-port 拥塞导向**的离线 ILP 求解器与 **静态查表** runtime。

---

## 2. 核心结构论点（保留）

在 MoE 里，combine 是唯一可以做 gate-aware 差分精度传输的位置。

- **Dispatch 阶段**：每个 token 被复制 $k$ 份发给 top-$k$ expert，所有副本是同一个 hidden state；要按 gate 给不同副本不同精度，就得做 $k$ 次不同精度的量化——要么浪费、要么破坏 collective 的规整性。
- **Combine 阶段**：每个 `(token, expert)` 对应一个独立 output $o_{t,e}$，且 $g_{t,e}$ 已知；按 gate 区分精度天然规整。

combine 的数学形式：$y_t=\sum_{e\in S(t)} g_{t,e}\cdot o_{t,e}$，线性可结合。
- $y_t$：token $t$ 经 combine 后的 hidden state；$S(t)$：被 routing 选中的 expert 集合
- $g_{t,e}$：expert $e$ 对 token $t$ 的 gate 权重（$\sum_{e\in S(t)}g_{t,e}=1$）；$o_{t,e}$：expert $e$ 的 d 维输出

> 这个论点在原版里就站得住，导师也没反对。

---

## 3. Claims（按导师建议重排优先级）

### C0｜端到端精度损失本身有多大？（**新增，最高优先级**）

> 导师反馈第 2 条："你要先证明：drop / quantize 到底带来多大精度损失。"

**先回答最基础的问题**：在 combine 阶段对 expert output 做 FP8 / INT4 量化、或者直接 drop top-k 中尾部的若干 expert，模型 perplexity / 下游 benchmark 分数到底掉多少？这是任何后续故事的起点——**如果这个数字本身就大到不可接受，整个方案就废了；如果小到忽略不计，则不需要 gate-aware 这套**。

实验设计见 §6。

### C1｜top-k 内部 contribution 长尾（**降级为待验证假设，需先实验**）

> 导师反馈第 3 条：现有 evidence 主要是 expert-level 流量不均，不一定能直接证明 token 内部 top-k contribution 长尾。建议参考 AICB。

**预设**：在某 token 的 top-k 内部，按 $g_{t,e}\cdot \|o_{t,e}\|$ 排序后呈现长尾分布。

**现状**：MoDES (2025)、Not All Experts are Equal (ACL 2024) 证明的是**全 expert 池**层面的不均；**top-k 内部**的分布**没有直接 evidence**。

**验证手段**：
- 用 AICB 生成 / 模拟 MoE 通信流量，观察 expert-level 的流量差异是否在加 routing bias 后放大；
- 在真实模型上记录每个 token 的 $\{g_{t,e}\cdot \|o_{t,e}\|\}_{e\in S(t)}$，画累积分布。

**触发条件**：如果 C1 不成立（即 top-k 内部分布接近均匀），方案需要回退到"按 expert 全局降精度"——这种情况下卖点变弱，但仍然可以做 receiver-port 拥塞导向的均一精度策略。

### C2｜routing drift 是端到端损失的主因（**降级为辅助实验**）

> 导师反馈第 2 条：先量整体损失，再拆 drift。

**实验**：两遍近似前向。
- 第一遍：完整近似（量化/丢弃），让 gate **自由**重新选 expert；
- 第二遍：完整近似，但 gate 选择**锁死**为原模型 routing。
- 两者的端到端精度差 = routing drift 单独贡献；剩余部分 = 纯数值误差。

EAQuant (2025, arxiv 2506.13329) 把"Routing Fragility Under Quantization Noise"列为 MoE 量化三大挑战之一，是间接旁证。**这个拆解只有在 C0 显示"损失非平凡"时才有意义**——所以放在 C0 之后。

---

## 4. 拥塞模型（**重大修改：从 link-level 退到 receiver-port**）

> 导师反馈第 1 条：先简化成 receiver 端口拥塞，参考 INFOCOM 那类工作的简化假设——网络内部资源近似无限，只考虑 receiver 端口是否成为热点。

### 简化假设

- **网络 fabric 内部不建模**（不区分 spine / leaf / pod，假设 bisection 带宽充足）；
- **唯一拥塞点是 receiver 端口**：某个 GPU/server 接收太多 expert output 时，入口排队成为瓶颈；
- **拥塞 → 排队 → TBT**：receiver $r$ 在 step $s$ 的入口流量 $D_r^{(s)}$ 决定该卡的 combine 阶段 wall-clock，进而影响 TBT。

### 拥塞函数

把每个 receiver 看作一个 M/D/1 或简单的线性排队模型：

$$\text{queue\_delay}_r(s) = \frac{D_r^{(s)}}{\mu_r - D_r^{(s)}/\Delta t}$$

或者最简单的版本——直接 $\max_r D_r^{(s)}$ 作为关键路径代理：

$$\text{TBT}^{(s)} \approx \alpha + \beta \cdot \max_r D_r^{(s)}$$

>  $\mu_r$：receiver $r$ 的入口带宽（GB/s）；$D_r^{(s)}$：step $s$ 流入 $r$ 的总字节；$\Delta t$：step 时长。

**这就直接回应了导师反馈第 7 条**：TBT 不再是孤立的 SLO 约束，而是 $\max_r D_r^{(s)}$ 的单调函数——**减少 receiver $r$ 的入口流量 $\Leftrightarrow$ 缩短 TBT**，目标和收益不再脱节。

---

## 5. 三维决策（修改后的 WHERE / WHICH-LAYER / HOW）

### WHERE｜按 receiver-group 而不是 link 分桶（**修改**）

把所有 receiver GPU 按入口流量预测分成几个组（"热 receiver" / "中 receiver" / "冷 receiver"），决策粒度上移到 **receiver-group**：

- 热 receiver 组：入口流量预测最高 → 进来的 expert output 按 gate 桶降精度；
- 冷 receiver 组：入口流量充裕 → 全精度。

**这比原版的"按链路"简单得多，也比原版的"按 sender 拓扑层级"更对应导师建议**。

### WHICH-LAYER｜按敏感度筛（**保留**）

离线跑端到端实验，对每一层单独施加近似、测 perplexity，得到一张层敏感度热力图。低敏感的层启用近似，高敏感的层全精度。

> 这一项导师没动，原版方案保留。

### HOW｜静态查表（**保留思路、简化 key**）

> 导师反馈第 4 条：LUT key 设计要简化。

LUT key 简化为 **`(layer_id, receiver_group, gate_bucket)`**——三维、低基数、可枚举。
- `layer_id`：32 层（DeepSeek-V2-Lite 量级）
- `receiver_group`：3–4 个分组
- `gate_bucket`：把 $g_{t,e}$ 离散到 4–8 个桶

key 总数 $\approx 32 \times 4 \times 8 = 1024$，runtime 查表 O(1)。

值域：$\{\text{BF16, FP8, INT4, drop}\}$。drop 路径配 gate 重归一化 $y_t\approx\frac{1}{\sum_{e\in\text{kept}}g_{t,e}}\sum_{e\in\text{kept}}g_{t,e}o_{t,e}$ 抵消偏差。

---

## 6. 离线优化问题（**重写：0/1 ILP + 经验 accuracy 约束**）

> 导师反馈第 5 条：精度选择要写成 0/1 indicator + 求和=1 约束。
> 导师反馈第 6 条：accuracy 约束不能是抽象 ε，要从离线 profile 表里拿。

### 决策变量（0/1 indicator 形式）

对每个三元组 $(l, R, B)$（layer, receiver-group, gate-bucket），定义 4 个 indicator：

$$x^{\text{BF16}}_{l,R,B},\ x^{\text{FP8}}_{l,R,B},\ x^{\text{INT4}}_{l,R,B},\ x^{\text{DROP}}_{l,R,B}\ \in\{0,1\}$$

$$\text{s.t.}\quad x^{\text{BF16}}_{l,R,B}+x^{\text{FP8}}_{l,R,B}+x^{\text{INT4}}_{l,R,B}+x^{\text{DROP}}_{l,R,B}=1\quad \forall (l,R,B)$$

### 目标函数（按 receiver 入口流量加权）

$$\min\ \max_R\ \sum_{l,B} \text{freq}(l,R,B)\cdot \sum_{p\in\{\text{BF16,FP8,INT4,DROP}\}} \text{bytes}(p)\cdot x^p_{l,R,B}$$

> 用 $\max_R$ 直接对应 §4 的 $\max_r D_r$ 拥塞代理；如果用 LP 松弛，可以引入辅助变量 $z\ge\sum_{l,B}\dots$ 把 $\max$ 线性化。
> $\text{freq}(l,R,B)$ 是从 logged trace 上统计的 token 频次，让目标按真实分布加权。

### Accuracy 约束（**改为分层经验上界**）

离线 profile 出 per-layer 的 worst-case accuracy drop 表：

| layer $l$ | precision $p$ | 在该层独占近似时的 worst-case Δaccuracy |
|---|---|---|
| 0 | FP8 | $\delta_{0,\text{FP8}}$ |
| 0 | INT4 | $\delta_{0,\text{INT4}}$ |
| 0 | DROP-1 | $\delta_{0,\text{DROP}}$ |
| ... | ... | ... |

约束写成**线性可加上界**（保守假设，损失独立累加）：

$$\sum_{l,R,B} \text{freq}(l,R,B)\cdot \sum_p \delta_{l,p}\cdot x^p_{l,R,B}\ \le\ \epsilon_{\text{total}}$$

> $\delta_{l,p}$ 来自 §6.3 的 layer sensitivity profile；$\epsilon_{\text{total}}$ 是用户给的总精度预算（如 0.5 PPL）。
> 线性累加是上界（实际损失通常小于和，因为不同层的误差不完全独立），但满足上界一定满足约束，**这就是导师说的"accuracy 约束要和决策变量建立可操作的对应关系"**。

### Memory 约束（保留）

$$\sum_{l,R,B,p}\text{cache\_bytes}(l,p)\cdot x^p_{l,R,B}\le \text{HBM}$$

### 求解

变量数 $\approx 32 \times 4 \times 8 \times 4 = 4096$，加上 $\max_R$ 线性化的辅助变量后规模仍小，**Gurobi/CBC 几分钟即可解**。也可用贪心：按 $\frac{\delta_{l,p}}{\text{bytes 节省}}$ 排序逐层降精度，作为 ILP 的快速近似。

### Oracle 上界（保留，事后离线计算）

在 logged trace 上让每个 `(token, expert)` 自由选精度：

$$b^{*}_{t,e}=\arg\min_p\ \text{bytes}(p)\quad \text{s.t.}\ \text{per-token accuracy preserved}$$

报告 $\text{gap}=\frac{\text{static 收益}}{\text{oracle 收益}}$——越接近 1 说明静态策略越逼近理论上界。

---

## 7. 实验路线图（**新增，明确"先基础后优化"**）

> 导师全部 7 条反馈合起来传达的核心信号：**先把 evidence 打牢，再做优化**。

### Phase 0｜环境与 baseline（1–2 周）

- 选模型：DeepSeek-V2-Lite (top-6) / Mixtral-8x7B (top-2) / OLMoE。**top-k 越大尾部越长，先验上 DeepSeek-V2-Lite 最有戏**。
- 选 benchmark：MMLU / GSM8K + perplexity on WikiText / C4。
- 跑通原模型 baseline。

### Phase 1｜C0 整体精度损失实测（**最高优先级**）

- 实验 1.1：在 combine 阶段把所有 expert output 量化到 FP8 / INT4，测 PPL / MMLU。
- 实验 1.2：在 combine 阶段 drop 掉 top-k 中 gate 最小的 1/2/...个 expert（不重归一化 vs 重归一化两版），测 PPL / MMLU。
- 实验 1.3：layer sensitivity profile——逐层单独施加近似，测每层独占近似下的 Δaccuracy，得到 §6 的 $\delta_{l,p}$ 表。

**Go/No-Go**：如果 INT4 + drop-1 的端到端损失就 < 0.3 PPL，**整个 gate-aware 差分精度的卖点就弱了**——因为均一近似已经够好。这种情况下转向：用 receiver-port 拥塞模型 + 均一精度 + layer sensitivity。

### Phase 2｜C1 top-k 内部分布（**前置假设验证**）

- 实验 2.1：在 logged trace 上记录每 token 的 $\{g_{t,e}\cdot\|o_{t,e}\|\}_{e\in S(t)}$，画累积分布、计算尾部占比。
- 实验 2.2：用 AICB 模拟 MoE 通信流量，观察加 routing bias 后 expert-level 流量差异是否放大（呼应导师提到的"加 bias 后 expert 流量会有差异"）。

### Phase 3｜C2 routing drift 拆解（**辅助实验**）

- 实验 3.1：两遍近似前向（自由 routing vs 锁死 routing），测端到端损失差。

### Phase 4｜系统集成与求解器

- 实验 4.1：构建 logged trace，统计 $\text{freq}(l,R,B)$。
- 实验 4.2：用 Gurobi 解 §6 的 ILP，得到 LUT。
- 实验 4.3：runtime 查表，端到端跑 throughput / TBT / accuracy。
- 实验 4.4：对照 oracle 上界，报告 gap。

---

## 8. 修改清单（对照原版）

| # | 原版做法 | 修改版做法 | 触发的导师反馈 |
|---|---|---|---|
| 1 | link-level / spine-leaf 拓扑建模 | receiver-port 拥塞（INFOCOM 式简化） | 第 1 条 |
| 2 | 重点验证 routing drift（C2） | 先验证整体损失（C0），drift 拆解作为辅助 | 第 2 条 |
| 3 | C1 默认成立 | C1 标记为待验证假设，前置 AICB + 实测 | 第 3 条 |
| 4 | LUT key 含 tier / rank-group / link 多套候选 | 固定为 `(layer, receiver_group, gate_bucket)` 三维 | 第 4 条 |
| 5 | 决策变量 $b\in\{\text{BF16,FP8,INT4,drop}\}$ | 4 个 0/1 indicator + $\sum=1$ | 第 5 条 |
| 6 | accuracy 约束 Δacc≤ε（来源不清） | 离线 profile 出 $\delta_{l,p}$ 表，加权线性上界 | 第 6 条 |
| 7 | TBT≤SLO（孤立约束） | TBT 直接是 $\max_r D_r$ 的函数，目标=最小化最大 receiver 入口流量 | 第 7 条 |

---

## 9. 保留下来的"灵魂"（说明为什么修改没动这些）

1. **结构论点（dispatch 不能差分、combine 可以）**——导师没反对，这是整个方案的立题基础。
2. **三维决策框架（layer / 拓扑 / gate-bucket）**——导师反对的是"拓扑"维度的 link-level 复杂度，不是三维结构本身。把"拓扑"换成"receiver-group"，结构不变。
3. **静态查表 + oracle 上界**——这是 deployable 的关键，runtime 必须 O(1)。
4. **量化优先、丢弃次之**——量化路径定长、规整、无偏，落地最容易；drop 配重归一化抵消偏差。

---

## 10. 待和导师确认的问题

1. **receiver-port 拥塞模型的具体形式**：用 $\max_r D_r$ 线性近似还是 M/D/1 排队？前者好解（LP），后者更准（凸优化）。
2. **AICB 的具体使用方式**：是用 AICB 跑出来的 trace 作为 §7 Phase 2 的输入，还是先在真实模型上录 trace、用 AICB 做对照？
3. **layer sensitivity profile 的实验粒度**：每层只测 1 个精度档（INT4），还是 4 个档全测？后者实验量 $\times 4$。
4. **Phase 1 的 Go/No-Go 判定**：如果 INT4 已经够好，要不要把方案彻底转向 "receiver-port 拥塞 + 均一精度"——这等于砍掉 gate-aware 卖点，但故事仍然成立。
