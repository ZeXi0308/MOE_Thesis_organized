# 研究计划：MoE EP 系统侧深层优化方向调研（2026-07-20）

## 1. 任务拆解

**背景**：本硕士论文项目在 MoE 专家并行(EP)通信/调度优化方向已经系统性评估并判死5条候选机制（CreditReduce、RouteFidelity-EP、WaveCredit-EP、MassCover-EP、TokenRace-EP），并对receiver-aware(v1/v2)、MassCover-EP、TokenRace-EP做了knob/算法优化复查，均确认无法挽回。已确立的硬约束（子任务必须避免重复撞线）：

- uniform FP8/FP4 combine 是极强baseline，精度动态重分配天花板 5%-10%（多次验证）
- DeepEP/NCCL EP(arXiv:2603.13606)/MoRI-EP/TensorRT-LLM 已覆盖主流dispatch/combine通信原语（LL/normal/HT路径、one-sided all-to-all）
- UCCL-EP(sender-side in-flight bytes)、FAST(traffic-matrix incast-free staging) 已覆盖RDMA/网络层拥塞控制
- Lynx(arXiv:2411.08982)、ExpertFlow、Gimbal(arXiv:2606.15177)、From Tokens to Layers(MLSys2026)、AEP/AMoE(arXiv:2505.08944) 已覆盖MoE请求级/批级调度、dispatch阶段all-to-all barrier异步化
- TokenRace-EP（combine后layer-advance barrier异步化）被真实RTX 5090硬件数据判死：kernel launch开销(~57us)、重批开销(~26.6-87.5us/层)、CUDA Graph代价(~40us/次)系统性压垮收益；自适应触发这个knob也被证明不可行（收益几乎全部来自不可预测噪声，免费因果信号无预测力）
- MassCover-EP（故障域副本放置）算法已逼近oracle上限（差距<1%），oracle相对简单baseline的天花板本身只有2-7%
- receiver-aware（在线拥塞感知调度）价值被精确限定在"拥塞结构性可预测"场景，该场景下静态放置已够用；瞬时拥塞场景需要的信息在因果上不可得
- **元教训**：这套系统里简单静态信号(gate mass/frequency/uniform格式)普遍已经逼近可挖掘空间的oracle上限；任何"仿真时间模型"必须用真实硬件验证，因为固定开销/信息时效性容易被过度乐观建模

**用户诉求**：站在系统侧视角，找"更深层次、更系统、更有潜力"的优化方向和创新方式——不是重复上述已撞线的范式，而是识别真正还有空间的层面。

**期望输出形式**：一份结构化报告，覆盖多个候选方向，每个方向说明：核心机制、与现有工作的边界（是否已被覆盖）、预期收益量级、验证所需资源（是否单卡Mac/GPU可验证，还是需要真实多机RDMA集群）、与本项目已死路线的关系（是否是同一范式的变种）。

## 2. 查询类型判定

**广度优先(breadth-first)为主，深度优先为辅**：用户要找多个独立的系统层面（内存/权重管理、计算图/编译器、算法层面自适应计算、多租户/能耗调度、prefill-decode解耦架构等），这些是彼此独立可并行调研的子领域，但每个子领域内部也需要"从多个角度验证是否被现有工作占满"这种深度分析。

## 3. 子任务分工（4个并行research_subagent + 1个中文语料交叉验证）

### 子任务A：专家权重的内存层级管理与异构放置
覆盖：hot/cold expert分层缓存(HBM↔CPU↔NVMe/CXL)、跨代GPU异构EP放置、专家权重的预取/驱逐策略、KV cache与专家权重的联合内存管理。

### 子任务B：Prefill-Decode解耦架构 × 编译器/CUDA Graph协同设计
覆盖：MoE场景下的PD分离(disaggregated serving)、结合TokenRace-EP暴露的"CUDA Graph固定shape vs 动态调度"矛盾，是否存在可以两者兼顾的图捕获策略（如分桶shape、多graph切换）、Mooncake/DistServe类系统在MoE EP场景下的适配空间。

### 子任务C：算法层面自适应计算——动态top-k、专家合并、投机解码×MoE
覆盖：decode阶段动态调整激活专家数、专家合并/剪枝在serving时的在线应用、speculative decoding与MoE路由的结合、这些方法是否已被2025-2026工作覆盖。

### 子任务D：多租户SLO差异化调度与能耗/异构硬件协同
覆盖：MoE推理下的多租户资源隔离与SLO分级、能耗感知的专家放置/调度（复查Energy-SLO Precision EP这个此前搁置候选在2026年是否有新进展）、carbon-aware/power-capped serving。

### 子任务E（我本人执行）：微信公众号中文社区对MoE推理系统优化的最新讨论
用`wechat-article-search`技能检索中文技术社区(2026年6月-7月)对以上方向的讨论，交叉验证国内是否有相关工程实践或论文解读，补充英文arXiv/GitHub检索的盲区。

## 4. 信息来源与整合方式

- 子任务A-D：优先 arXiv(2025-2026)、NSDI/OSDI/ATC/MLSys/SOSP/ISCA接收论文、主流框架(vLLM/SGLang/TensorRT-LLM/DeepEP/Mooncake)的GitHub issue/设计文档
- 子任务E：微信公众号搜索，关键词包括"MoE 推理优化"、"专家并行"、"PD分离 MoE"、"专家权重卸载"，时间范围2026-06至2026-07
- 整合：按"是否已被覆盖"、"验证所需资源"、"与本项目资源现实（Mac+偶尔租GPU，无RDMA集群）的匹配度"三个维度交叉筛选，最终按"潜力×可验证性"排序输出候选清单
