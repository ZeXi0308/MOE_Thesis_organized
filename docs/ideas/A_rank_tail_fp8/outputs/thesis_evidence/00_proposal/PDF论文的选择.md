# MoE Serving 毕设选题方案

## 方案 A：Profile-Guided Receiver-Aware Rank-LUT Partial Combine


### 定位
现有 MoE 通信优化几乎都集中在 dispatch 和 expert placement 上；combine 阶段被忽视的不是"传输本身"，而是它独有的一个自由度——gate 权重作为重要性信号，免费可得、可差分化、且在 dispatch 阶段结构上根本做不到。

### 核心点

**结构论点**：在 MoE 里，combine 是唯一可以做 gate-aware 差分精度传输的位置。

- **Dispatch 阶段**：每个 token 被复制 $k$ 份发给 top-$k$ expert，所有副本是同一个 hidden state；要按 gate 给不同副本不同精度，就得做 $k$ 次不同精度的量化——要么浪费、要么破坏 collective 的规整性。
- **Combine 阶段**：每个 `(token, expert)` 对应一个独立 output $o_{t,e}$，且 $g_{t,e}$ 已知；按 gate 区分精度天然规整。

combine 的数学形式：$y_t=\sum_{e\in S(t)} g_{t,e}\cdot o_{t,e}$，线性可结合。
  - $y_t$：token $t$ 经 combine 后的 hidden state；$S(t)$：被 routing 选中的 expert 集合（top-k 时 $|S(t)|=k$）
  - $g_{t,e}$：expert $e$ 对 token $t$ 的 gate 权重（softmax 归一化，$\sum_{e\in S(t)}g_{t,e}=1$）；$o_{t,e}$：expert $e$ 的 d 维输出


### 论文的 claim

| # | 必须成立的前提 | 现有文献 & 我的判断 |
|---|-----------------|----------------------|
| **C1** | combine contribution $g_{t,e}\|o_{t,e}\|$ 在 top-k 内部呈现明显的长尾，而不只是"略微不均" | **没有直接支撑，只有间接旁证。** MoDES (2025) 和 Not All Experts are Equal (ACL 2024) 证明的是 expert-level 的不均（全 expert 池里多数 expert 不重要），是整体层面的结论，不是 top-k 内部的。characterization 可以做一个**top-k 内部的 per-token 贡献分布**。先验上，top-k 越大（如 DeepSeek-V2-Lite 的 top-6）尾部贡献越小，但必须实测才能下定论。**验证手段分两层**：(i) C1 本身（$g_{t,e}\|o_{t,e}\|$ 长尾）需在真实 MoE 模型（如 DeepSeek-V2-Lite、Qwen-MoE）上挂 forward hook 直接采样 per-(token, expert) 贡献，画分布图——这是 AICB 给不出的；(ii) 阿里开源的 [AICB (AI Communication Benchmark)](https://github.com/aliyun/aicb) 用来生成 / 分析通信 trace、估算 receiver 端口的负载分布与 $\text{freq}(l,r,R)$，为优化模型提供频次输入。两者职责正交。 |
| **C2** | 端到端精度退化的相当一部分来自 routing drift（误差扰动 hidden state、改变下游 routing、被层层放大），而非纯数值偏差 | **EAQuant (2025, arxiv 2506.13329) 间接支撑这一风险存在。** 该工作把 "Routing Fragility Under Quantization Noise" 列为 MoE 量化三大挑战之一，原文："router's top-k expert selection is highly sensitive to quantization-induced perturbations, causing misrouting and cascading degradation"，处理思路是"对齐路由、消除 drift"。但 EAQuant 没给出 drift 占总精度损失的比例。**本文实验的定位**：先量"应用方案后总精度差有多大"（baseline 数字），再借鉴其"对齐路由"思路把损失拆成 drift vs 数值误差两部分，回答 drift 在 combine 近似损失中的占比，而不是预先 claim drift 一定是主因。 |

### C2 实验设计

C2 实际上要量化两件不同的事，不能合并：

1. **总精度差量化（baseline）**：先做最朴素的对比——`不丢弃任何信息` vs `应用本方案的近似/丢弃` 两遍前向，直接测端到端精度差，告诉读者：应用这个 idea 后，总精度损失有多大。这是最被审稿人关心的数字，必须有。
2. **精度差归因（routing drift 拆解）**：在(1)的基础上做归因——
   - 实验 A：近似前向 + gate 自由选 expert
   - 实验 B：近似前向 + gate **锁死在原模型路由**上
   - 二者精度差**可作为 routing drift 贡献的近似估计**（锁死 routing 本身改变了模型执行语义，并非严格等号；但能给出 drift 影响的同阶量级），剩下的近似归为纯数值误差。
   - 这套拆解借鉴 EAQuant 的"对齐路由"思路，回答**精度到底是怎么崩的**。

### 静态可部署策略

把决策切成三个互相正交、可以离线确定的维度：

- **WHERE｜按 receiver 端口的拥塞情况启用**：**假设网络内部资源充足无拥塞，所有抢占只发生在 receiver 端口**。某个 receiver 端口越拥塞（共享它的 sender 越多、聚合到的 expert outputs 越多），就在该 receiver 上越激进地降精度——比如 receiver $r$ 上同时收到 expert 2、5、7 的 outputs，端口堵就把**编号最大的 rank**（即 $R=k$，gate 权重最低的那个 expert）降精度，端口闲就全精度。优化目标直接对到"各 receiver 端口的 P99 占用 / 排队长度"，比"总共省了多少字节"更有说服力。同节点的 NVLink 受拥塞影响小，默认全精度不动。
  > 落地时为避免 LUT 绑定具体集群 / placement，将 receiver 维度**离线聚类成 receiver group**（hot / warm / cold 三档，按聚合到的 expert 流量分位数划分），LUT 真正的 key 是 `(layer_id, receiver_group, rank)`。新部署只需重新做一次 receiver-group 分类即可复用 LUT。

- **WHICH-LAYER｜按敏感度筛**：离线跑端到端实验，对每一层单独施加近似、测 perplexity，得到一张层敏感度热力图。低敏感的层启用近似，高敏感的层全精度不动；进入优化模型时，高敏感层直接固定为 BF16，只在低敏感层上求解 Rank-LUT。

- **HOW｜Rank-LUT 静态查表（量化优先、丢弃次之）**：runtime 只查表得到精度（BF16 / FP8 / INT4 / drop），O(1)、无排序、无在线优化、无 per-token 决策。**采用 rank（即 expert 在该 token 的 top-k 中的位次，$R\in\{1,\dots,k\}$）作为重要性代理**——rank 在 routing 时已经确定，无需做 gate 分位数估计、无需在线维护阈值表，落地最简洁；rank 1 = 最高 gate，rank k = 最低 gate（论文中统称"编号最大的 rank / lowest-ranked expert / $R=k$"，避免"最低 rank"歧义），与 C1 长尾的方向一致。量化路径定长、规整、无偏，落地最容易；drop 只用在编号最大的 rank 位（$R=k$），并配 gate 重归一化 $y_t\approx\frac{1}{\sum_{e\in\text{kept}}g_{t,e}}\sum_{e\in\text{kept}}g_{t,e}o_{t,e}$ 作 best-effort 修正（注意：这是对丢弃带来的尺度偏差的近似补偿，并非数学无偏；其有效性依赖 C1 的长尾性质），零通信开销。**查表 key 为 `(layer_id, receiver_group, rank)`**，receiver_group 是 receiver 端口按聚合流量离线聚类后的类别（如 hot/warm/cold 三档），让 LUT 与具体集群 / placement 解耦，便于跨部署复用。**Gate 分桶版**作为 ablation / enhanced variant 在评估章节单独对比（rank 是 gate 的粗化代理；当 rank-LUT 已能贴近 oracle 上界时，gate 分桶并非必需）。

### 唯一的优化问题

#### 决策变量（one-hot 形式）

对每个三元组 `(layer l, receiver group r, rank R)`，引入一组 0-1 决策变量：
$$x_{l,r,R,p}\in\{0,1\},\quad p\in\mathcal{P}=\{\text{BF16, FP8, INT4, drop}\}$$
$$\sum_{p\in\mathcal{P}} x_{l,r,R,p}=1\quad \forall (l,r,R)$$

对高敏感层，固定 $x_{l,r,R,\text{BF16}}=1$；MILP 只在低敏感层上优化精度选择。

其中各下标含义：

| 下标 | 含义 | 取值范围 / 举例 |
|------|------|-----------------|
| $l$ | 第几层 MoE layer | $1,2,\dots,L$ |
| $r$ | receiver group（按端口聚合流量离线聚类） | $r\in\{\text{hot, warm, cold}\}$（三档），LUT 与具体 GPU id 解耦 |
| $R$ | rank（expert 在该 token top-k 中的位次） | $R\in\{1,2,\dots,k\}$，routing 时已确定 |
| $p$ | 精度档位（precision level） | $\mathcal{P}=\{\text{BF16, FP8, INT4, drop}\}$ |

$x_{l,r,R,p}=1$ 表示：**第 $l$ 层 combine 阶段，发往 receiver group $r$ 的 expert output 中，rank 为 $R$ 的那些 (token, expert) 对，统一采用精度 $p$ 传输。**

每个三元组在 4 种精度档中只能选一个。bytes 函数：
$$\text{bytes}(l,r,R)=\sum_{p\in\mathcal{P}}\text{size}(p)\cdot x_{l,r,R,p}$$
其中 $\text{size}(p)$ 是常量（BF16=2B / FP8=1B / INT4=0.5B / drop=0B）。

#### 优化问题

目标直接对应 receiver 端口拥塞——**最小化所有 receiver group 的最大利用率**

$$\min_{x,U}\ U\quad \text{s.t.}\ \lambda_r(x)/\mu_r\le U\ \forall r\in\{\text{hot, warm, cold}\}$$

其中 $\lambda_r(x)=\sum_{l,R,p}\text{size}(p)\cdot x_{l,r,R,p}\cdot \text{freq}(l,r,R)/T_{step}$ 为 receiver group $r$ 的字节到达率（按组内代表端口归一化），$\mu_r$ 为该组端口带宽，$\text{freq}(l,r,R)$ 来自离线 trace（层 $l$ 上、目标 receiver group 为 $r$、rank 为 $R$ 的 (token, expert) 对数）。引入辅助变量 $U$ 后整体仍是 MILP。

约束：

- **accuracy 约束（per-(layer, rank, precision) profile 表）**：离线 profile 时把 $\delta_{l,R,p}$ **定义为单位 (token, expert) pair 的边际精度退化贡献**——即"在层 $l$ 上让 rank 位 $R$ 的一个 (token, expert) pair 改用精度 $p$，相对全 BF16 baseline 端到端精度退化的边际增量（worst-case 上界，BF16 时 $\delta\equiv 0$）"。注意 $\delta_{l,R,p}$ 与 receiver group $r$ 无关——精度退化由 (层, rank 位, 精度) 决定，receiver group 维度只通过频次进入约束。约束写成线性形式：
  $$\sum_{l,r,R,p}\delta_{l,R,p}\cdot x_{l,r,R,p}\cdot \frac{\text{freq}(l,r,R)}{\sum_{l',r',R'}\text{freq}(l',r',R')}\le \epsilon$$
  即"决策诱导的全局加权平均边际精度退化 $\le \epsilon$"，量纲一致。这样 INT4 vs drop、rank=1 vs rank=k、敏感层 vs 不敏感层 的差异都落进 profile 表，决策变量可以专挑"$R=k$ + hot receiver group"的组合精确降精度。校准时离线对比"全层全 rank 启用 $p$ 的预测损失 $\sum w\cdot \delta$" vs "实测损失"，验证 (i) 量纲一致 (ii) 层间可加性偏差作为 sanity check。

- **TBT 约束（receiver 端口利用率上限）**：当目标已是 min-max 利用率时，本约束直接以 $U\le \rho^*$ 形式参与（$\rho^*$ 如 0.7，对应 P99 排队时延的经验上界）。端到端 TBT 拆解为 $\text{TBT}_{p99}=\bar T_{compute}+\bar T_{dispatch}+T^{queue}_{combine}(x)+\bar T_{other}$，前三项作为常量预算离线测得，combine 排队是唯一可调项；其中 $T_{other}$ 包括 attention / KV cache access、非 MoE FFN、layernorm、sampling 与 runtime overhead。同节点的 NVLink 受拥塞影响小，默认全精度不动。

变量规模 $O(L_{\text{low}}\times |\text{groups}|\times k\times|\mathcal{P}|)$，其中 $L_{\text{low}}$ 是通过 layer sensitivity 筛出的低敏感层数；高敏感层固定 BF16，不进入 MILP。receiver group 仅 3 档、rank 只到 $k$，整体仍是小规模 MILP。

#### oracle 上界（不进 runtime，事后离线计算）
在 logged trace 上让每个 `(token, expert)` 自由选精度
$$b^{*}_{t,e}=\arg\min_b\ \text{bytes}\quad \text{s.t.}\ \|y_t^{approx}-y_t^{full}\|_2^2\le \delta$$

**oracle 用 combine-output MSE（即近似后 combine 输出 $y_t^{approx}$ 与全 BF16 输出 $y_t^{full}$ 的 L2 距离）作为局部约束**——这是真正可分解到单个 `(token, expert)` 精度选择的量（每个 $o_{t,e}$ 的扰动直接映射到 $y_t$ 的扰动）。logit/KL 是经过后续层后的端到端结果，不天然可分解，**仅作为事后相关性校准**：报告"combine MSE oracle 收益"与"端到端 PPL 收益"的相关系数，验证局部代理的有效性。最终报告 $\text{gap}=\text{static 收益}/\text{oracle 收益}$——越接近 1 说明静态策略越逼近理论上界。runtime 仍只查 `(layer, receiver_group, rank)` 三个 key 的表，O(1) 零开销。

### 评估计划

- **Baselines**：(i) 全 BF16 combine（无压缩）；(ii) uniform FP8 combine；(iii) uniform INT4 combine；(iv) rank-only 启发式（无 receiver 感知）；(v) receiver-only 启发式（无 rank 区分）；(vi) Rank-LUT（本工作）；(vii) 离线 oracle（上界）。
- **指标**：端到端 TBT P99 / mean；WikiText-2 与一个长上下文 benchmark 上的 perplexity；combine 阶段字节节省率；gap-to-oracle。
- **模型与平台**：DeepSeek-V2-Lite (top-6) 与 Qwen1.5-MoE-A2.7B (top-4)；单节点 8×A100/H100，expert parallelism；vLLM 或 SGLang 作为 serving backend。

### 风险与回退

- **C1 长尾不够明显**（rank-$k$ 贡献 > 10%）：去掉 drop 精度档，只保留 BF16/FP8/INT4 差分量化；receiver-aware + Rank-LUT 主结构不变，bytes-saving 故事弱化但仍成立。
- **routing drift 占主导**：把 $\delta_{l,R,p}$ 收紧成 per-layer cap，考虑叠加 EAQuant 式 routing alignment 作为补丁。
- **MILP 在真实集群上求解时间过长**：退化到 LP-relaxation + 取整，或按 layer 独立求解。

---

## 方案 B：能效/SLO 双约束下的 MoE 推理联合优化

### 定位
现有 MoE 推理系统几乎都在优化 latency / throughput。Dense 模型的能耗优化已经做得相当多，但旋钮主要是"实例数 / 并行度 / GPU 频率"——MoE 专属的能耗模型基本是空白，多出几个 expert-level 的旋钮可以用。本方案建一个 MoE 专属能耗模型（静态 + 动态计算 + 通信 三部分），在 SLO 约束下联合优化 expert 放置和副本数，目标是最小化 J/token。

### 核心：MoE 比 dense 多出两个 expert-level 旋钮

Dense 那边的旋钮只有"实例数 / 并行度 / GPU 频率"；MoE 多出两个 dense 用不上的：

1. **expert 放置** —— 决定 all-to-all 的通信能耗；
2. **expert 副本数** —— 用"多副本带来的多卡静态功耗"换"更少通信、更均衡负载"。

### 三个判断
- **① 静态功耗**：expert 必须存在某张 GPU 上，放得越分散、副本越多 → 点亮的 GPU 越多 → 静态功耗越大。多副本带来更均衡负载和更少通信争用，代价是多烧静态功耗——这是 MoE 才有的 trade-off。需要先实测静态功耗在总能耗里占比多大，看看值不值得优化。
- **② 通信功耗**：MoE 计算稀疏但 all-to-all 密集，通信能耗是 MoE 独有的大头，dense 完全没有这一项，是能耗模型里最重要的动态项。
- **③ latency-最优 ≠ energy-最优**：延迟最低和能耗最低这两套配置通常不是同一套。靠实验把这件事 demonstrate 出来——画一张 latency-energy 散点图（横轴延迟，纵轴 J/token，扫多组 placement / 副本 / batch 配置）。

### 研究问题
1. **RQ1**：建 MoE 推理能耗模型（静态 / 动态计算 / 通信 三项），实测 latency-optimal 和 energy-optimal 配置的差距。
2. **RQ2**：energy-aware expert replication & placement——SLO 约束下决定每个 expert 放在哪、留几个热副本，目标最小化 J/token。

### 建模

#### 系统假设
- **拓扑**：2-tier spine-leaf 数据中心网络，同节点 NVLink、机架内 leaf+IB、跨机架 spine。
- **MoE 部署**：$L$ 层 × $E$ expert，按 expert parallelism 分布到 $G$ 张 GPU 上，每个 expert 可有多副本。
- **解码**：batch size $B$，每 token 每层激活 $k$ 个 expert（top-k routing）。

#### 决策变量
$x_{l,e,g}\in\{0,1\}$：第 $l$ 层 expert $e$ 是否放在 GPU $g$ 上。副本数 $r_{l,e}=\sum_g x_{l,e,g}$ 由 $x$ 导出；DVFS 频率档位作为 future work 暂不引入。

#### 能耗模型（每 decode step）

$$E_{\text{step}} = E_{\text{static}} + E_{\text{compute}} + E_{\text{comm}}$$

- **静态**：$\sum_g [P^{idle}_g + \rho_g\cdot(P^{TDP}_g - P^{idle}_g)]\cdot T$，$P^{idle/TDP}$ 来自 datasheet（H100 ≈ 70 W / 700 W）。
- **动态计算**：每层 $\alpha^{load}_l\cdot \mathbb{1}[\text{副本激活}] + \beta_l\cdot \text{token 数}$；$\alpha$ 解析可算（权重 / HBM 带宽 × 平均功率），$\beta$ 从 LLMCarbon 反推 J/FLOP；激活触发用 0-1 辅助变量做线性化，避免给"未分到 token 的多余副本"也算 weight load 能耗。
- **通信**：$\sum_l\sum_{(g,g')} D^l_{g\to g'}(x)\cdot c^{comm}_{g\to g'}$。$D^l$ 用 trace 的 expert pair co-activation 频率解耦（McCormick 线性化）；$c^{comm}$ 按 NVLink / leaf / spine 三档 pJ/bit 展开（NVLink ≈ 1.3、IB ≈ 10–20、QM9700 ≈ 1.5），跨机架与同节点比约 8× 高，给 placement 一个明确的拓扑梯度信号。

#### 优化问题

$$\min_{x}\ \frac{E_{\text{step}}(x)}{B}\ \text{(J/token)}$$

约束：TBT SLO（$T(x)\le \text{TBT}_{99}^{SLO}$，$T^{comm}_g$ 复用方案 A 的 receiver 端口排队上界）、HBM 容量、副本下界 $\sum_g x_{l,e,g}\ge 1$。$\rho_g\cdot T$ 双线性项用 $T=\text{SLO}$ 上界做保守线性化。求解：小规模 MILP（Gurobi），大规模拉格朗日松弛 / 贪心。

---
