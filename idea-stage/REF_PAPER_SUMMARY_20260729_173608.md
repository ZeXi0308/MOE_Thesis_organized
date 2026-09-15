# MoE Serving Idea Landscape（探索态）

**生成时间**：2026-07-29  
**状态**：`EXPLORATORY / NOT_CURRENT_MAINLINE`  
**当前权威**：`docs/current/README.md`

## 1. 约束边界

当前没有任何正式成立的新 MoE 系统机制。唯一已授权执行线仍是：在自然连续到达、exact model semantics 和完整 denominator 下，先验证 fragmentation、route-conditioned straggler 与 HoL 是否真正进入暴露关键路径；只有共同现象 Gate 和 exact Oracle 均成立，才允许选择一个最小 action space。本轮仅做 idea 探索，不改写当前权威，也不复活 PhaseMap、FJRC、ConfidenceGuard、fixed RankLane、CreditReduce、MassCover、TokenRace、QuotaEP、PLTB additive、RouteFidelity 或 Prefetch。

## 2. 本地论文扫描

本轮先阅读了本地论文 PDF 的前三页，并检查了首页渲染：

- **HOBBIT**：以混合精度 expert offloading 缓解显存和 PCIe 压力；会改变参数精度，不适合作为 exact-semantics 主线的直接 baseline。
- **AdapMoE**：按敏感度动态改变激活专家数与 gating；会改变模型执行语义，不能直接进入当前 exact-semantics action space。
- **Aurora**：联合专家部署与 all-to-all 通信调度，覆盖同构/异构集群中的放置和通信优化。
- **Lina**：以 expert popularity 做资源调度和 all-to-all 平衡，说明“热门度驱动放置/调度”不是空白方向。

## 3. 近期文献版图

### 3.1 负载均衡、异步执行与跨层调度已经拥挤

- **AMoE / AEP**（arXiv:2505.08944）用每层队列、动态 re-batching 和 defragging 取消 barrier-style 同步。
- **UltraEP**（arXiv:2606.04101）在 rack-scale 节点上按 microbatch、layer 的 post-gating exact load 实时重平衡。
- **METRO**（arXiv:2512.09277）指出 memory-bound decode 应平衡 activated experts，而不是 token count。
- **Gimbal**（arXiv:2606.15177）联合 frontend engine pressure、队列顺序、expert pressure、source-aware placement 和 migration stability。
- **ExpertPlex**（arXiv:2607.18002）在 prefill/decode 之间共享 experts，并用 adaptive persistent kernels 做 tile 级隔离与调度。

因此，“再做一个 expert-aware scheduler / balancer”本身不是可信新意。

### 3.2 激活模式、动态放置与重配置代价也已有直接工作

- **Scaling Multi-Node MoE Inference Using Expert Activation Patterns**（arXiv:2604.23150）在三类 frontier MoE 上收集超过 100k route traces，报告 domain-specific popularity shift 和 prefill/decode correlation，并据此做 grouping 与 placement。
- **Mixture-of-Experts Serving**（arXiv:2607.17880）直接形式化“随时间变化的 expert popularity + GPU assignment + reconfiguration cost”，给出在线与离线算法。
- **Gimbal** 同时覆盖 migration stability；因此 drift-aware placement 的新方法碰撞风险很高。

### 3.3 内存受限、异构与压缩路径同样成熟

- 本地 HOBBIT、AdapMoE，以及 **FloE**（arXiv:2505.05950）都在 offloading、压缩或动态 gating 上优化资源受限推理。
- **Achieving Cloud-Grade SLOs for Local MoE Inference**（arXiv:2606.10493）已覆盖双路 CPU、消费级 GPU、stream-loading、SmallEP 和 prefill/decode 混合执行。

这类方法往往改变精度、路由或硬件目标，与当前 exact-semantics 研究线不完全同域，但“资源受限 MoE 推理”本身不新。

### 3.4 多租户、安全与可靠性出现新近直接邻居

- **FaaSMoE**（arXiv:2604.26881）研究 serverless、多租户、按需专家资源。
- **RepetitionCurse**（arXiv:2512.23995）证明重复 token 可诱导 route concentration 并造成 MoE DoS。
- **EEP**（arXiv:2605.10670）把 EP membership 变成可变运行态，以从 rank failure 中局部恢复。

所以 tenant interference、timing channel、故障定位方向都必须证明与这些工作不同，不能只换一个公平性或弹性名称。

## 4. 仍值得查证的结构缺口

1. **完整 request-DAG 下的 Oracle 合法性**：大量工作报告端到端收益，但公开摘要通常没有给出“局部窗口 Oracle 对跨 layer/step 的失真边界”。这也正是当前工作区已承认的硬缺口。
2. **action-space existence，而不是 controller design**：高负载、热点或高通信占比不等于允许动作能删除关键依赖；缺少 fail-closed 的 absent certificate。
3. **局部不均衡到请求尾部的放大边界**：AMoE、UltraEP、METRO 和 Gimbal 各自给出机制，但跨模型、逐请求、完整 denominator 的 no-benefit boundary 是否公开建立仍不清楚。
4. **观测量的内生性**：wall-clock expert popularity 可能被 admission、batching 和 completion speed 反向塑造；这与“已知需求序列下如何追随 drift”不同。
5. **强静态 baseline 与负结果资产**：动态 placement 是否只是击败脆弱静态基线，以及 robust static 是否已经捕获绝大部分带迁移税 Oracle，值得先证伪。
6. **删失与分母完整性**：若 timeout、cancel 或 window-end unfinished request 与 route 相关，completion-only P99 可能改变 Gate 裁决；但其独立论文价值尚不确定。

## 5. 本轮探索原则

- 先允许候选结论为 `NO-GO` 或 `ACTION_SPACE_ABSENT`。
- 不把 5090 本地 expert 时间写成 EP/NCCL/RDMA/return-path 证据。
- 不用 synthetic skew 替代自然 workload 正式 cell。
- 不在现象 Gate 前实现 controller。
- 所有收益必须进入 full request-DAG、all-arrival denominator，并扣除求解、迁移、同步和调度成本。

