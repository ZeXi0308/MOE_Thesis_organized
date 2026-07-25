# MoE 10 个研究 Idea：Red-Team 严格评审与去留结论

> 本文已与另外两份候选/复审材料合并到[统一主文档](./MoE_研究方向统一梳理_17项去重筛选与Top3执行路线_2026-07-25.md)；本文保留为详细 Red-Team 附录。

> 评审日期：2026-07-25  
> 资源边界：1×RTX 5090 32 GB；正式实验仅 1 台 8×A100  
> 被审对象：[上一版 10 个候选](./MoE_严格Idea探索与Top3_2026-07-25.md)  
> 立场：以下数字不是收益承诺，而是用于尽早拒绝方向的上界、门槛和反证条件。

## 1. 总体判断

### 1.1 结论先行

上一版 Top 3 不能原样执行。本轮没有任何候选达到 **A. 立即验证**：

| 排名 | Idea | 强制结论 | 当前判断 |
|---:|---|---|---|
| 1 | I1 RankLane-Combine | **B. 先做 Profiling** | 直接 novelty 尚可，但真实 combine critical-path 占比未知；现有未融合 codec 在快链路上已是负收益 |
| 2 | I2 RouteCloak | **B. 先做 Profiling** | 问题新且重要，但真实观察者、攻击优势和低开销防御空间均未建立 |
| 3 | I3 ResumeSet | **C. 有条件保留** | 必须证明 pause/resume 产生独立于 ELDR/InferCept 的 expert cold-start；全驻留时上界为零 |
| 4 | I4 ScaleBridge | **D. 降级为子机制** | 适合作为所有多卡 Idea 的测量与外推工具，不足以独立成论文 |
| 5 | I7 JouleBatch | **D. 降级为子机制** | 可成为调度系统的能耗目标或消融，不足以单独支撑新机制论文 |
| 6 | I5 QualityDebt-SlowRank | **E. 淘汰** | Capacity-Aware Inference、ReaLB 等已覆盖关键机制；真实慢 rank 发生率未知 |
| 7 | I8 Cancellable Spec-MoE | **E. 淘汰** | 2025–2026 多篇 Spec-MoE 已正面覆盖，且取消信号通常来得过晚 |
| 8 | I10 Sensitivity Cache/Quant | **E. 淘汰** | HOBBIT、DynaExq、ProMoE、MODE 等高度重合；本地 cache/quant 路径已有 NO-GO 证据 |
| 9 | I9 Shadow-Price Placement | **E. 淘汰** | Aurora、Gimbal、MoEless、Mixture-of-Experts Serving 已覆盖在线放置/重配置；单卡无法验证 |
| 10 | I6 RouteShare-VTC | **E. 淘汰** | 本地匹配直方图后 latency 差异仅 2.35%–3.06%，oracle 端到端上界已接近淘汰线 |

最终资源配置不是“三条并行主线”：

- **首选主线：I1，但只获准做 1–3 天的 headroom/break-even profiling。**
- **次选备线：I3，仅在暂停恢复 cold-start oracle ≥15% 且跨模型成立时启动。**
- **高风险高收益：I2，仅在真实可达观察者上复现显著 route leakage 后启动。**

### 1.2 共同硬伤

1. 多数候选把“局部现象存在”误当成“端到端关键路径足够大”。
2. 多数候选没有先跑删除式 oracle，因此无法区分 2% 局部改进与可投稿系统贡献。
3. 8×A100 是单节点 scale-up，不等价于跨节点 scale-out。需要 InfiniBand、真实故障域或远端权重池才能成立的方向，应按资源不匹配淘汰。
4. [A100 数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)不包含原生 FP8。I1/I5/I10 若正式实验依赖 FP8 Tensor Core 加速，硬件假设错误；A100 上只能研究通信编码、INT8/INT4 或软件模拟，不能把逻辑字节下降写成计算加速。
5. 当前仓库中已有三类重要否定证据：receiver join phase 没有跨过 action boundary；cache/prefetch 在正确 `(layer, expert)` key 和 full top-k 下趋于饱和；未融合 quantize/codec 在快链路下没有正收益区。

本轮不是从零猜测，关键本地证据为：[Rank-tail 设计与两模型结论](../ideas/A_rank_tail_fp8/设计说明.md)、[Receiver codec 硬门槛](../01_current_status/Receiver_Codec硬门槛测量结论_2026-07-21.md)、[FJRC corrected Level-1](../ideas/receiver_aware/FJRC_Corrected_Level1_Result_2026-07-23.md)、[Phase1 v4 严格审计](../01_current_status/Phase1_v4_StrictReview_2026-07-23.md)和 [RouteShare Gate-0](../ideas/route_share/experiments/RESULT_2026-07-23.md)。这些结果只能支持各自的测量域；H2D proxy、fake quant、offline replay 均未被当成真实 NCCL/端到端结果。

### 1.3 数字纪律

统一使用

\[
S(p,s)=\frac{1}{(1-p)+p/s},\qquad S_{oracle}=\frac{1}{1-p}.
\]

文中“改善 x%”指 wall-clock 降幅，不把 speedup 倍数与百分比混用。没有实测的数字均标为估计区间；只用于设门槛，不能作为论文结果。

---

## 2. 每个 Idea 的严格 Review

## Idea 1：RankLane-Combine——固定双 lane 的 gate-rank 感知 combine 编码

### 2.1.1 中立复述

**可证伪假设：** 当 expert-parallel MoE 的 combine 通信处在真实 token 关键路径、且不同 gate rank 的误差敏感性稳定不同时，固定双 lane 编码相对“全量同精度通信”与“统一低比特通信”可降低 TPOT/P99 或提高 throughput，同时保持预设质量边界且不增加消息数。

- 决策变量：combine payload precision/layout；高低精度 lane 的 rank cutoff；producer/consumer 融合方式。
- 最终目标：TPOT、P99、throughput、通信字节和 energy/token；“逻辑 payload 减少”只是中间指标。

### 2.1.2 上界与场景分解

- 单 GPU：没有真实 EP all-to-all；核心端到端 headroom 为 **0**，只能测 codec tax、误差和 replay break-even。
- 8 GPU prefill、高并发、continuous batching：GEMM 较大，通信较容易被 overlap；残余 combine critical-path 粗估 3%–10%。
- 8 GPU decode、batch 1/低并发：消息启动和同步占比高，combine 可占 10%–30%，但 byte compression 对固定启动延迟无能为力。
- 8 GPU decode、高并发：payload 增大，编码更可能摊薄；combine critical-path 粗估 7.5%–20%。
- NVLink/NVSwitch compute-bound：更接近 3%–10%；PCIe 或跨节点 communication-bound：可达 15%–30%，但后者不在当前正式资源保证内。

边界计算：若 `p=7.5%`，完全删除 combine 的延迟极限为 7.5%，将其加速 1.5×只改善 2.6%；若 `p=30%`，完全删除极限 30%，1.5×只改善 11.1%。因此它不是“压缩 50% 字节就端到端快 50%”。平均、P99、throughput 的上界都受 `p` 限制；energy/token 还要扣除 codec kernel，可能不降。

**当前 Headroom：无法判断，必须先 profiling。** 本地历史 microbenchmark 已给出强负证据：FP8→INT4 同构 lane 在 rows 128/512、200–400 Gbps 等效链路下 0/8 可行，codec tax 约 58–61 μs，break-even 仅 4.5–68.6 Gbps。只有“每 step 一次且 producer/consumer 融合”的乐观重放出现正净值；H2D proxy 不能替代 NCCL/DeepEP。

### 2.1.3 Oracle、最强 baseline 与 Captured Headroom

- Oracle：从真实 8 卡 trace 中删除 combine critical-path 时间，另做“零成本按 rank 选精度”的离线质量 oracle。
- 最强简单 baseline：单消息、统一 INT8/INT4（或 BF16/INT8）+ static cutoff；与通信直接 overlap；不做动态预测。
- 必须报告 `CH=(T_base-T_simple)/(T_base-T_oracle)`。目前无真实 `T_oracle`，不能声称复杂双 lane 的增量价值。若 uniform INT8 或单 cutoff 捕获 ≥80% headroom，双 lane 论文故事结束。
- 配置消失测试：增大 batch、改变 EP/TP、启用 DeepEP/SwiftEP 类 overlap。若默认通信栈已把 combine 从关键路径移除，问题消失。

### 2.1.4 最近邻与创新边界

- [Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin)：以通信调度和优先级处理 MoE all-to-all；未研究 gate-rank 质量不对称编码。
- [ScMoE](https://arxiv.org/abs/2404.05019)：用压缩/系统协同降低通信，报告部分平台上通信占比；最危险之处是“MoE 通信压缩”并不新。
- [Comet](https://arxiv.org/abs/2502.19811)、[FlashDMoE](https://arxiv.org/abs/2506.04667)：以 kernel/通信计算 overlap 提高 EP 效率；可能直接吃掉 I1 的可优化关键路径。
- [SwiftEP](https://www.usenix.org/conference/nsdi26/presentation/li-xingyi)、[MixServe](https://arxiv.org/abs/2601.08800)：代表更强的生产级通信/调度 baseline。
- [NCCL-EP](https://arxiv.org/abs/2603.13606)：进一步抬高高性能 EP 通信 baseline。

Novelty 分类为 **新观察 + 新编码布局**；当前评价是“有潜力但边界不清”。最危险质疑是：`This is MoE communication quantization with a static rank cutoff; the claimed novelty is a layout choice.` 若没有“rank 顺序稳定地给出 Pareto 优于 activation magnitude、per-token scale、uniform quantization”的跨模型证据，差异不足。

### 2.1.5 因果链、开销和验证价值

因果链目前是：**rank 质量不对称（已有证据） → combine 是瓶颈（缺） → 双 lane 降低真实 wire time（缺） → 不破坏 kernel batching（缺） → E2E/P99（缺）**。替代解释包括：收益来自更小 batch/较慢 proxy；误差差异来自所选数据；逻辑 payload 没有映射成实际 NCCL wire time。

开销包括量化/反量化、scale 元数据、额外访存、lane partition、可能的 kernel launch 和融合 ABI 修改。若拆成两个 collective，固定启动成本会直接让 decode 负优化。内存开销小，但临时 buffer 与碎片必须计入。

- 5090：可真实验证质量、codec tax 和 fused local replay；不能验证真实 EP collective。
- 8×A100：必要，用于真实 all-to-all/combine；单节点 NVSwitch 可能掩盖 headroom，8 卡足以验证 scale-up，不足以声称跨节点扩展。
- 论文评分：当前 **3/5（边缘）**；若真实 `p≥15%`、融合净加速 ≥10%、跨 3 个模型且强 baseline 后仍成立，可升至 4。

**强制结论：B. 先做 Profiling。** 必测：dispatch/combine 独立关键路径占比、payload 分布、collective 启动/带宽分解、overlap 后残余、fused codec tax、P50/P99、quality Pareto、uniform baseline 的 Captured Headroom。

---

## Idea 2：RouteCloak——SLO 约束下的 exact-output 路由足迹防御

### 2.2.1 中立复述

**可证伪假设：** 当共享 GPU/通信域中的低权限观察者能从 MoE expert route 足迹显著恢复 token 或敏感属性时，bucketed dummy routing/padding 相对无防御、全 padding 和固定 bucket baseline 可把攻击优势压到预设阈值，同时保持 exact output 且 TPOT/P99 开销 ≤10%。

- 决策变量：dummy token/expert、bucket、padding 时机、批间混洗和隐私预算。
- 最终目标：攻击成功率/互信息、P99、throughput、energy/token；不是“route histogram 更平”。

### 2.2.2 上界与场景分解

这是隐私—性能问题，Amdahl 不能把隐私收益伪装成速度收益。防御的 latency/throughput 毛收益为负；其价值来自降低攻击优势。

- 单 GPU：若观察者可访问性能计数器、cache/TLB 或共享内存通道，可测攻击；若权限/隔离阻断观察，则 privacy headroom 为 0。
- prefill/high batch：天然聚合可能降低单 token 可辨识性，但 traffic volume 大；padding 更易摊薄。
- decode/batch 1/低并发：路由序列与 token 对齐最清楚，攻击面最大，dummy 固定成本也最难摊薄。
- continuous batching：请求混合可能是强免费 baseline，也可能被请求边界泄漏抵消。
- 8 GPU：真实 all-to-all traffic 暴露更强，但单节点管理员隔离、NCCL 可见性和共租模型决定观察者是否存在。

若 top-8/64 为隐藏 route 而执行全部专家，expert 计算最坏约增加约 8×，不可接受。全 padding 是隐私 oracle 但不是可部署 baseline。若仅做直方图 bucket，开销可能 5%–100%，必须实测。平均延迟、P99、throughput 和 energy 都可能恶化；没有性能正收益指标。

**Headroom：无法判断。** 只有在真实 observer 上无防御攻击相对随机基线的优势显著（建议 token top-1 提升 ≥20 个百分点，或敏感属性 AUC ≥0.75），且 ≤10% 开销内可移除 ≥80% 攻击优势，才有充分 headroom。

### 2.2.3 Oracle、最强 baseline 与 Captured Headroom

- Oracle：全 expert 执行/完全定长通信，使 route 对观察者信息为零，同时保持原输出；离线计算最大可移除攻击优势。
- 最强简单 baseline：跨请求 continuous batching + 固定时间窗混洗 + 静态 bucket padding；不训练预测器。
- Captured Headroom 用攻击优势计算：`CH=(Adv_base-Adv_simple)/(Adv_base-Adv_oracle)`。如果自然 batching/固定 bucket 已捕获 ≥80%，复杂自适应 cloak 无价值。
- 问题消失测试：MIG/进程隔离、禁用性能计数器、固定 collective API 是否直接阻断低权限观察。若可信隔离以更低代价解决，系统机制缺乏部署动机。

### 2.2.4 最近邻与创新边界

- [MoEcho](https://arxiv.org/abs/2508.15036)：展示 MoE 架构/微架构侧信道可泄露输入；支持问题重要性，但也可能已覆盖攻击建模。
- [Expert Selections Reveal Text](https://arxiv.org/abs/2602.04105)：证明 expert selection 能用于文本恢复；尚不等价于普通共租者能观测真实 route。
- [CryptoMoE](https://arxiv.org/abs/2511.01197)：从密码协议角度保护 MoE 推理，威胁模型和代价不同。
- [SecMoE](https://arxiv.org/abs/2601.06790)：select-then-compute 的安全 MoE 协议；说明“隐藏全部 selection 但保留稀疏性”已有强密码学路线。

Novelty 分类为 **新问题设定 + 新隐私/系统组合**；评价“创新性强，但威胁模型尚未落地”。最危险质疑：`The attacker assumes privileged observability unavailable to a normal tenant`; `padding is a textbook traffic-analysis defense`; `exact output is obtained only by doing useless work`。

### 2.2.5 因果链、开销和验证价值

因果链为：**route 与文本相关（已有文献） → 目标 observer 可看到足够精细的 route proxy（缺） → 可训练/迁移的攻击（缺本机证据） → cloak 降低可观测信息（缺） → ≤10% SLO 代价（缺）**。最大风险不是实现，而是第一跳和第二跳不同时成立。

开销包括 dummy expert GEMM、额外 A2A bytes、padding buffer、同步、随机化和更差 GEMM packing；P99 可能比平均更糟。静态 bucket 状态小，不需要复杂模型；一旦引入在线隐私预测器，关键路径和稳定性风险上升。

- 5090：能否验证取决于真实 observer 权限；可以先重放公开 route trace 做信息论上界，但这不能证明现实攻击。
- 8×A100：必要但未必充分。若攻击依赖跨租户/跨进程并发，必须搭建真实共租；若依赖跨节点 traffic，现有资源不足。
- 论文评分：当前 **3/5（高方差边缘）**；攻击面真实、跨模型迁移、≤10% 开销三者齐备时可达 4–5，否则为 1。

**强制结论：B. 先做 Profiling。** 必测：observer API/权限、时间分辨率、攻击优势、跨模型/模板迁移、自然 batching 泄漏、full-padding oracle、静态 bucket Captured Headroom、P50/P99/throughput/energy 开销。

---

## Idea 3：ResumeSet——Agent interruption 下的 KV–expert 双状态保留

### 2.3.1 中立复述

**可证伪假设：** 当 tool/human interruption 导致继续请求的 KV 保留但 expert 工作集被驱逐、且恢复阶段存在显著 expert cold-start 时，按 KV block 关联的近期 expert signature 保留相对 KV-only、LRU/LFU 和 last-window baseline 可降低 resume TTFT/前 N token TPOT，同时不显著降低普通请求 goodput。

- 决策变量：expert cache admission/eviction、pause TTL、保留集合大小、KV 与 expert metadata 关联。
- 最终目标：resume TTFT、前 N token P99、SLO/goodput、HBM 使用量；expert-set hit rate 只是代理指标。

### 2.3.2 上界与场景分解

- 全模型权重驻留 GPU：expert miss cost 为 0，所有场景 headroom **0**。这是 8×A100 跑中小 MoE 的首要反例。
- 单 5090 放不下模型、CPU/NVMe offload：若恢复窗口中权重 miss stall 占 20%，删除式 oracle 最多改善 20%（1.25×）；占 50% 时最多改善 50%（2×）。
- prefill：恢复通常已有 KV，因此关键是新增 decode；若需要重算长上下文，KV 策略反而主导。
- decode/batch 1：cold miss 最显著但搬运难以隐藏；高并发/continuous batching 可用其他请求计算覆盖搬运，也会产生更激烈 cache 竞争。
- compute-bound/high batch：权重搬运可被算力摊薄；memory/capacity-bound 才有足够 headroom。

本地既有 cache/prefetch 证据对其不利：正确 `(layer, expert)` key 和 full top-k 后，OLMoE/LLM-jp 工作集趋于饱和，safe-budget oracle 接近零。pause/resume 是可能逃出该 NO-GO 的唯一新条件，但必须单独证明“暂停造成的驱逐”将 miss 放回关键路径。

**Headroom：有条件。** 真实 pause trace 中 resume 前 N token 的可删 expert fetch stall 必须 ≥15%；否则淘汰。只在故意缩小 HBM 的极端合成场景达到该门槛不算通过。

### 2.3.3 Oracle、最强 baseline 与 Captured Headroom

- Oracle：已知恢复后前 N token 的 future expert set，在暂停期间零成本预取/永久保留；trace replay 删除对应 fetch stall。
- 最强简单 baseline：KV pause TTL 对应的 `last-W union` expert set + static quota；并比较 LRU、LFU、全局 hot set、随机和 KV-only。
- 若 last-W union 捕获 oracle ≥80%，学习型/联合策略无论文价值；它本身至多是工程 policy。
- 配置消失测试：增加 HBM、把模型完整放入 8×A100、提高 batch 隐藏搬运。若任一常见配置使 resume cold-start 消失，泛化性弱。

### 2.3.4 最近邻与创新边界

- [InferCept](https://arxiv.org/abs/2402.01869)：处理中断式 LLM serving 的 KV 缓存策略；I3 仅多了 expert state。
- [Fiddler](https://arxiv.org/abs/2402.07033)：在 CPU/GPU 间调度 MoE expert，覆盖权重驻留与搬运基本机制。
- [ProMoE](https://arxiv.org/abs/2410.22134)：预测/预取 expert，直接构成 strongest learned baseline。
- [ReMoE](https://arxiv.org/abs/2605.27081)：更近的专家驻留/内存管理系统，压缩独立机制空间。
- [ELDR](https://arxiv.org/abs/2607.00466)：用 prefill expert activation 构造 signature，和 KV block 共同索引，并据此预测 decode locality；这是对 I3 最大的新颖性打击。

Novelty 分类从上一版的“新状态抽象”下调为 **已有 KV/expert signature 组合在 pause/resume workload 上的延伸**。当前评价“增量式贡献”。最危险质疑：`This is InferCept's pause policy plus ELDR's KV-indexed expert signature`; `last-window retention is obvious`; `the problem vanishes when weights fit in HBM`。

### 2.3.5 因果链、开销和验证价值

因果链为：**暂停恢复存在（真） → expert 被驱逐且 fetch 是 resume 关键路径（缺） → 过去 route 可预测恢复 route（需跨任务/工具验证） → 保留不挤压更有价值状态（缺） → resume SLO 改善且总 goodput 不降（缺）**。

开销包括 expert/HBM 保留、普通请求 cache pollution、signature metadata、TTL 扫描、预取带宽和并发干扰。它可能改善 resume P99 却恶化全局 P99；必须以总 goodput/SLO 而非命中率评价。

- 5090：可用真实模型 offload + 受控 HBM budget 验证 oracle、last-W 和污染；仍有“人为制造容量压力”的外部有效性风险。
- 8×A100：只有运行超出总 HBM、或人为 partition HBM 的大模型才有价值；若模型全驻留，8 卡证明的是零 headroom。单节点足够测 weight transfer，不足以代表远端 expert pool。
- 论文评分：当前 **2/5（偏弱拒稿）**；只有证明独立的 resume cold-start、ELDR 不能覆盖、且复杂策略显著胜过 last-W 才可能到 3–4。

**强制结论：C. 有条件保留。** 条件：至少两个模型、两类 interruption trace 上，真实 resume fetch stall ≥15%，oracle E2E ≥15%，last-W 捕获 <70%，且在同等 HBM budget 下总 goodput 不降超过 2%。

---

## Idea 4：ScaleBridge——单卡 trace 到多卡 EP 的性能迁移模型

### 2.4.1 中立复述

**可证伪假设：** 当只有单卡 route/compute trace 和少量通信 microbenchmark 时，分解式模型相对简单 roofline/线性模型能预测 8 卡 EP 的 P50/P99/最佳配置，误差和配置 regret 足够低，且明显减少实际多卡 profiling 成本。

- 决策变量：模型中的 compute、A2A、overlap、排队、straggler 参数和配置选择。
- 最终目标：预测误差、配置决策 regret、profiling GPU-hours；它本身不直接改善 serving latency。

### 2.4.2 上界、Oracle 与 baseline

该方向没有传统 Amdahl 性能收益。它的 oracle 是对所有候选配置做完整 8 卡 sweep，成本最高、regret 为 0。价值上界是被替代的 GPU-hours 与错误配置造成的 regret。

- 单卡可采集 compute/route，但无法产生真实 A2A contention、rank skew、collective ordering 和 overlap interference。
- batch 1/decode 的启动/排队最难迁移；大 batch/prefill 更接近 roofline，简单模型反而更强。
- 单节点 NVSwitch 只能拟合一种 topology，不能支持跨节点或异构推广。

最强 baseline 是 analytical roofline + NCCL microbenchmark + 每模型 3–5 个校准点。若它的 TPOT 误差 <10%、最佳配置 regret <5%，复杂模型没有研究价值。复杂方案必须把 regret 降低至少 5 个百分点，并节省 ≥80% profiling 运行。

**Headroom：不是端到端 headroom，而是测量成本 headroom；当前无法证明足够。**

### 2.4.3 最近邻、因果链和论文性

- [EPS-MoE](https://arxiv.org/abs/2410.12247)：以性能模型选择 expert parallelism/配置。
- [Aurora](https://arxiv.org/abs/2410.17043)：用分析与搜索共同优化 MoE serving 配置。
- [MixServe](https://arxiv.org/abs/2601.08800)：包含现代 MoE serving 调度与系统优化，是必须解释的强系统对象。
- [FlashDMoE](https://arxiv.org/abs/2506.04667)：展示 kernel/通信 overlap 使简单可加模型失真，但也提供可校准结构。

Novelty 分类为 **新模型/测量工具**，当前是“工程实现或增量模型”。因果链是“单卡特征 → 足够决定多卡交互”这一步最脆弱；只报告平均误差会掩盖错误选配置的高 regret。

- 5090：只能产生输入特征和模拟；无法验证核心迁移准确度。
- 8×A100：是 ground truth，必要；但只有一个 topology，泛化主张有限。
- 论文评分：**2/5**。更适合作为 I1/I3/I2 的统一测量基础、实验规划器和 artifact，而不是主论文。

**强制结论：D. 降级为子机制。** 停止条件：简单 roofline 的 held-out 配置 regret ≤5%，或复杂模型跨模型/并发迁移后误差 >15%。

---

## Idea 5：QualityDebt-SlowRank——暂态慢 rank 的质量预算式降级

### 2.5.1 中立复述

**可证伪假设：** 当真实生产负载中存在短暂且可在线识别的慢 rank 时，在累计质量预算内对该 rank 的 token 做 drop/reroute/低精度执行，相对等待、静态超时和 least-loaded reroute 可降低 P99/SLO violation，同时保持长期质量约束。

- 决策变量：超时、reroute/drop、precision、每请求/租户质量债务。
- 最终目标：P99、SLO、goodput、quality；“慢 rank 次数”不是目标。

### 2.5.2 上界与场景分解

设慢事件只影响 `q` 比例 step，并在这些 step 中造成 `r` 的额外延迟，完全消除后的整体 latency headroom 近似 `p=q·r/(1+q·r)`。例如 q=10%、慢事件使 step 多 50%，整体可删部分约 4.8%；只有高频或严重慢事件才过 5%。平均值往往低，P99 可能高。

- 单 GPU：不存在 rank，只能注入模拟，不能证明真实发生率。
- 8 GPU：可以做进程/时钟/通信注入，但注入不是生产真实性证据。
- prefill/high batch：单 rank 延迟可能造成 barrier straggler；decode/batch 1 更敏感，但低精度/drop 的质量风险更直接。
- compute-bound：降精度可能有效；communication-bound：计算降级不一定碰到瓶颈。
- A100 没有 FP8 原生路径；动态低精度必须先证明 kernel 真加速。

Oracle 是零成本、零质量损失地完全消除慢 rank。最强 baseline 是 timeout + least-loaded exact reroute/replica；若简单策略捕获 ≥80%，quality-debt controller 无价值。没有真实 slow-event census 时，headroom **无法判断且先验偏低**。

### 2.5.3 最近邻、因果链和论文性

- [Capacity-Aware MoE Inference](https://arxiv.org/abs/2503.05066)：已用 token drop/reroute 缓解 straggler，并报告显著收益；正面覆盖核心动作。
- [ReaLB](https://arxiv.org/abs/2604.19503)：对 hot/slow rank 做精度自适应，直接覆盖“性能—质量”权衡。
- [EEP](https://arxiv.org/abs/2605.10670)：处理部分失效/弹性执行，虽非暂态降速，但覆盖故障下 expert 可用性。
- [Tarragon](https://arxiv.org/abs/2601.01310)：代表更广的 MoE 尾延迟/负载调节背景。

Novelty 仅剩 **新优化目标（跨时间 quality debt）**，不是新瓶颈或新动作；评价“基本被覆盖”。因果链缺真实慢事件频率、可预测性和“质量债务能弥补逐 token 不可逆错误”的证据。

开销包括每请求记账、同步质量预算、动作预测、备用 expert 计算、质量评估延迟；决策在 token 关键路径。错误触发会同时损害质量和性能。

- 5090：只能模拟 action/quality frontier，不能验证 rank 因果。
- 8×A100：可做注入实验，但若论文核心依赖合成 straggler，外部有效性不足；单节点也不代表跨节点故障域。
- 论文评分：**1/5**。

**强制结论：E. 淘汰。** 不因“quality debt”这个新名词保留。只有未来获得真实生产 trace，证明慢事件使目标 P99 至少恶化 15%、现有 exact reroute 失败且 ReaLB 不适用，才可重新立项。

---

## Idea 6：RouteShare-VTC——route coalition 感知的多租户公平

### 2.6.1 中立复述

**可证伪假设：** 当多个请求的 expert-route 重叠能在控制 token 数、序列长度和 arrival pattern 后稳定解释 GPU 服务成本差异时，coalition-aware virtual time 相对 token-count VTC、least-served 和 round-robin 可降低跨租户 slowdown/Jain 不公平，同时不显著降低 goodput。

- 决策变量：请求/租户调度顺序、virtual cost、route coalition 分组。
- 最终目标：公平性、P99 slowdown、SLO、goodput；route-overlap 分数只是代理变量。

### 2.6.2 上界、Oracle 与已有反证

本地 Gate-0 已给出近似淘汰证据：在匹配 route histogram 后，latency 差异仅 **2.35%–3.06%**；简单模型 held-out `R²=0.9971–0.9986`。这说明大部分成本已经由 token 数、长度和普通 batch 特征解释，coalition 额外因果 headroom 很小。

- batch 1/低并发：几乎没有 coalition packing，headroom 0。
- high concurrency/continuous batching：才可能出现共享 expert 批处理，但也最容易被 framework batching、GEMM shape 和队列顺序混淆。
- 单 GPU：可验证局部执行复用/批形状，无法验证 EP rank contention。
- 8 GPU：route overlap 可能转化为相同 destination rank 的 incast，而非计算共享；机制若只用集合重叠，可能调错。
- prefill 通常由 token 数/GEMM shape 主导；decode 更可能受小 GEMM batching 影响，但现有差异仍低于 5%。

Oracle 是知道每个候选 batch 的真实增量 GPU 时间并做最优公平调度。最强 baseline 是 measured-cost VTC（moving-average 服务时间）+ ordinary continuous batching。若 oracle 相对该 baseline 的端到端公平/P99 改善 <5%，直接淘汰；现有证据已使该结果高度可能。

### 2.6.3 最近邻、因果链和论文性

- VTC 类 LLM 公平调度：已建立 token/service-based virtual time；I6 只是换成本估计器。
- DLPM/基于预测长度的调度：覆盖用更精细请求成本改善公平/吞吐的基本路线。
- [LLMVisor](https://arxiv.org/abs/2502.02633)：代表多租户 LLM GPU 调度与隔离背景。
- [ExpertPlex](https://arxiv.org/abs/2607.18002)：通过 persistent kernel/tile isolation 处理共享 expert 资源，可能从执行层直接消除 I6 观察到的干扰。

Novelty 分类为 **已有调度器的新 cost feature**，评价“增量且 headroom 不足”。因果链在“route overlap → 稳定的增量服务成本”这一跳已有反证；即使预测更准，也不代表 fairness action 会改变。

开销包括在线 route feature、pairwise overlap、调度状态和潜在 head-of-line blocking；复杂 coalition 计算可能随 active requests 二次增长。简单 moving average 更便宜且已解释绝大多数波动。

- 5090：现象/上界已基本验证为 NO-GO。
- 8×A100：不应为救活 Idea 而花资源；只有单卡 oracle >5% 才有资格进入多卡，当前不满足。
- 论文评分：**1/5**。

**强制结论：E. 淘汰。** 不再写复杂调度器。最多保留匹配直方图的负结果，作为“route 特征不能自动推出调度价值”的方法学证据。

---

## Idea 7：JouleBatch——deadline-aware expert coalescing

### 2.7.1 中立复述

**可证伪假设：** 当小 expert GEMM 的低占用造成显著 energy/token 浪费且请求存在 SLO slack 时，按 expert/deadline 合并执行相对立即执行、固定等待阈值和普通 continuous batching 可降低 energy/token，同时保持 P99/SLO 和 goodput。

- 决策变量：等待时间、expert execution order、batch composition、功率/频率状态。
- 最终目标：energy/token、SLO、goodput；kernel occupancy 不是最终目标。

### 2.7.2 上界与场景分解

- batch 1/低并发：合并机会少；等待只增加延迟。
- 高并发 decode：小 GEMM 合并机会最多，但 ordinary continuous batching 已捕获大量收益。
- prefill：GEMM 本来较大，额外 coalescing headroom 小。
- compute-bound：可能提高利用率但功率也升高；energy 降幅不等于 latency 降幅。
- communication-bound：合并 expert compute 可能不改变 E2E，甚至制造更大 burst/incast。
- 单 GPU 可真实测能耗/批形状；8 GPU 还需计入等待导致的 A2A 同步和 rank 不均衡。

若 expert compute 占总能耗 30%–60%，而合并仅使该部分能效提升 10%–20%，E2E energy/token 粗上界约 3%–12%。完全免费理想 packing 是 oracle；平均 latency 不一定改善，P99 可能恶化。Headroom **中低且 workload-dependent**。

最强 baseline 是固定 slack threshold + per-expert FIFO + framework continuous batching。只有 learned/dynamic controller 相对该 baseline 再获得 ≥5% E2E energy/token、且 SLO violation 不升，才有独立价值；预计简单 baseline 会捕获 >70% packing headroom，但当前需 trace replay 量化。

### 2.7.3 最近邻、因果链和论文性

- [AMoE](https://arxiv.org/abs/2505.08944)：代表 MoE 自适应执行/服务优化，压缩“按负载改变执行”空间。
- [PALS](https://arxiv.org/abs/2605.21427)：面向功率/延迟约束的 LLM serving 调度，覆盖 energy/SLO 共同目标。
- [Festina](https://arxiv.org/abs/2606.30391)：进一步覆盖 deadline-aware LLM serving。
- [ExpertPlex](https://arxiv.org/abs/2607.18002)：用 adaptive persistent kernel 和 tile-level isolation 提高 MoE 小任务效率，是更强的执行层 baseline。

Novelty 分类为 **新优化目标 + 已知 batching 机制的 MoE 特化**，评价“增量式”。因果链缺口是：低 occupancy 是否真导致系统级能耗浪费，以及等待合并是否在加上静态功耗、通信和 P99 后仍净正。

开销包括 token 等待、deadline 维护、跨请求隔离、队列扫描、可能的跨 rank 同步；最大风险是改善平均 energy 却增加 P99 和泄漏租户信息。

- 5090：可以用 NVML/板卡能量积分真实验证单卡上界和 threshold baseline。
- 8×A100：有价值但会引入通信耦合；8 卡足以 scale-up，不足以声称数据中心级能耗调度。
- 论文评分：独立方向 **2/5**；作为 I1/其他调度系统的能耗目标与消融更合理。

**强制结论：D. 降级为子机制。** 若 5090 的理想 packing oracle 在 realistic arrival trace 上 energy/token <5%，或固定阈值捕获 ≥80%，立即停止。

---

## Idea 8：Cancellable Spec-MoE——取消已失效 speculative branch 的 expert work

### 2.8.1 中立复述

**可证伪假设：** 当 speculative decoding 的被拒分支仍有大量尚未开始或可安全抢占的 MoE expert 工作时，跨阶段 cancellation 相对普通 speculative decoding、减小 draft tree 和 admission threshold 可减少浪费计算并提高 TPOT/energy，同时保持 exact output。

- 决策变量：speculative branch admission、expert task priority/cancel、执行粒度。
- 最终目标：TPOT、throughput、energy/token、P99；“取消 task 数量”是代理指标。

### 2.8.2 上界与场景分解

总上界不是 rejected token 比例，而是 `被拒 × 判决后仍未完成 × 可抢占 × 位于关键路径` 的乘积。若拒绝工作占 30%，但判决时 70% 已完成、只有一半可抢占且其中一半在关键路径，可删比例仅 2.25%。

- batch 1/低并发：容易低延迟判决，但并行 branch 少，浪费较小。
- 高并发：浪费总量可能大，但 GPU 队列更深，取消粒度和同步代价更高。
- prefill 不适用；decode 是唯一主场景。
- 单 GPU 可测 task timeline，但不能验证跨 rank cancellation/collective 一致性。
- 8 GPU cancellation 可能因 collective 已提交而无法回收 wire time，并引入 rank divergence/deadlock 风险。

Oracle 是在知道最终接受集合的情况下，从不执行被拒 branch。最强 baseline 是调小 speculative width/depth、置信阈值 admission 和优先执行最可能接受的 branch。若这些捕获 ≥80% oracle，runtime cancellation 不成立。Headroom 可能从 <5% 到中等，但必须先画出“拒绝判决时间—expert 工作完成时间”CDF；当前无证据。

### 2.8.3 最近邻、因果链和论文性

- [SP-MoE](https://arxiv.org/abs/2510.10302)：MoE speculative execution 的直接近邻。
- [MoE-Spec](https://arxiv.org/abs/2602.16052)：正面研究 MoE speculative decoding。
- [MoE-SpAc](https://arxiv.org/abs/2603.09983)：继续覆盖 speculative acceptance/执行优化。
- [SpecMoE](https://arxiv.org/abs/2604.10152)：2026 年直接同题工作，极度压缩 novelty。

Novelty 分类为 **已有 Spec-MoE 的 runtime optimization**，评价“基本被覆盖”。因果链中最危险的一跳是“知道 reject 时工作尚未完成”；没有 timeline，取消机制只是 API 想象。

开销包括 branch bookkeeping、GPU task cancellation、collective 一致性、细粒度 kernel 导致的 batching 损失和同步。取消失败或过晚时净收益为负。

- 5090：可做 timeline oracle；不应直接实现复杂 cancellation runtime。
- 8×A100：核心机制需要，但已有工作密集、死锁/一致性工程风险高，不值得当前资源投入。
- 论文评分：**1/5**。

**强制结论：E. 淘汰。** 除非严格 prior-art audit 证明四篇近邻都没有“late work suppression/cancellation”，且离线 oracle 在三个模型上 E2E ≥15%、width/depth baseline 捕获 <70%，否则不重启。

---

## Idea 9：Shadow-Price Placement——动态 replica/migration/rerouting

### 2.9.1 中立复述

**可证伪假设：** 当 expert demand 变化快于静态 placement、但慢到足以摊销复制/迁移成本时，shadow-price controller 相对 topology-aware static placement、least-loaded reroute 和 periodic greedy replication 可提高 goodput/P99，同时受 HBM 与迁移预算约束。

- 决策变量：expert placement、replication、migration、reroute 和更新周期。
- 最终目标：goodput、P99、SLO、迁移字节、HBM；shadow price 本身不是贡献。

### 2.9.2 上界与场景分解

- 单 GPU：没有 placement/rank，核心 headroom 为 0。
- 8 GPU：可测单节点 skew；NVLink/NVSwitch 使迁移较快，也可能让远程访问/重路由足够好，减少复制价值。
- batch 1/低并发：demand estimate 噪声大，复制难摊销。
- high concurrency/continuous batching：hotness 稳定性更高，但静态/周期 greedy 也更强。
- prefill burst 可能改变 hot expert；decode 更持久，但 token-level route 波动快。
- 跨节点网络/故障域最能放大 placement 价值，现有资源没有。

Oracle 是已知未来需求、零成本迁移、无限副本，在 HBM 约束下最小化 max-rank load。其理论上界可达 10%–30% 甚至更高，但实际净值必须扣除复制、同步和重布局。最强 baseline 是 topology-aware static placement + periodic greedy replica + least-loaded exact reroute。复杂优化器若不能相对它再提升 ≥10%，不值得。

### 2.9.3 最近邻、因果链和论文性

- [Aurora](https://arxiv.org/abs/2410.17043)：自动化 MoE parallel/placement optimization。
- [Gimbal](https://arxiv.org/abs/2606.15177)：近期动态 MoE serving/资源编排近邻。
- [MoEless](https://arxiv.org/abs/2603.06350)：重构 expert serving/资源池化，覆盖动态供给路径。
- [Mixture-of-Experts Serving](https://arxiv.org/abs/2607.17880)：形式化 expert GPU allocation 与 reconfiguration cost，并给出 online/offline algorithms；几乎直接覆盖本 Idea 的数学核心。
- [ELDR](https://arxiv.org/abs/2607.00466)：用 locality/load-aware routing 在不迁移 expert 的情况下解决部分动态负载问题。

Novelty 分类为 **已被覆盖**。把已有 online allocation 写成 shadow-price 形式不足以构成新 insight。因果链还要求 hotness 的时间尺度大于迁移 break-even；若不是，控制器追逐噪声。

开销包括迁移 expert 权重、额外 HBM、副本一致性、同步、控制周期、路由表更新和瞬时 tail spike。最坏情况会 thrash。

- 5090：只能 trace/simulator；不能真实验证核心。
- 8×A100：必要但对 scale-out 论证不足；单 topology + 8 ranks 很容易被“规模太小”拒稿。
- 论文评分：**1/5**。

**强制结论：E. 淘汰。** 原因同时满足“已有工作覆盖”和“资源不足”。不可用 simulator-only 结果包装成系统论文。

---

## Idea 10：Sensitivity Cache/Quant——expert 敏感度、热度与精度/预取/缓存联动

### 2.10.1 中立复述

**可证伪假设：** 当 expert 的质量敏感度与未来 hotness 可稳定预测、且模型不能全驻留 HBM 时，联合 precision/cache/prefetch 相对 uniform quantization、LRU/LFU、static hot-set 和独立优化可降低 TPOT/内存占用，同时满足质量约束。

- 决策变量：expert precision、cache admission/eviction、prefetch、热度/敏感度估计。
- 最终目标：TPOT、P99、HBM、throughput、quality；预测准确率和 hit rate 是代理指标。

### 2.10.2 上界与已有反证

- 全驻留：权重 miss headroom 0；只剩 quant compute/memory benefit。
- 单 5090 offload：可人为产生高 miss，但要证明不是资源特设。
- 8×A100：中小模型全驻留后问题消失；超大模型才有价值，但当前可运行模型/CPU RAM/框架支持不确定。
- decode/batch 1 对 fetch stall 最敏感；高并发可 overlap prefetch。
- prefill 大 GEMM 下低比特可能有计算价值，但 A100 无 FP8，动态 INT4/INT8 kernel 必须真实快于 BF16。

本地已有两类 NO-GO：正确 cache key/full top-k 后工作集饱和、safe-budget oracle 近零；动态 FP8/QuantizeOnce 比 BF16 慢且没有实际加速区。联合优化不能凭组合自动恢复 headroom。

Oracle 是 100% future hit + 零成本最低比特且零质量损失。最强 baseline 是 uniform INT8/INT4 + static global hot set/LFU + double-buffer prefetch。若独立简单策略已捕获 ≥80%，联合 controller 只是调参。当前 Headroom 在全驻留场景为 **0**，在 offload 场景需重新测，但本地先验偏向 <5%。

### 2.10.3 最近邻、因果链和论文性

- [HOBBIT](https://arxiv.org/abs/2411.01433)：异构低比特 expert 与 offload，直接覆盖质量/内存/性能联动。
- [ProMoE](https://arxiv.org/abs/2410.22134)：expert 预测与预取。
- [DynaExq](https://arxiv.org/abs/2511.15015)：动态 expert quantization，直接覆盖敏感度/热度精度选择。
- [MODE](https://arxiv.org/abs/2606.17118)：近期动态 MoE 精度/执行优化近邻。
- [MC#](https://arxiv.org/abs/2510.10962)、[DuoServe](https://arxiv.org/abs/2509.07379)：进一步覆盖 MoE cache/驻留/服务组合。

Novelty 分类为 **已有机制组合**，评价“基本被覆盖”。因果链在两处断裂：hotness 是否带来未被 static set 捕获的新 hit；逻辑低比特是否映射成更快 kernel/搬运。

开销包括预测器、per-expert 元数据、多精度权重副本、量化/反量化、碎片、预取带宽竞争和更差 GEMM batching。三个控制环联合会使归因和稳定性都恶化。

- 5090：可做 oracle，但已有结果不支持继续工程实现。
- 8×A100：模型全驻留时无 cache 证据；A100 精度路径又受硬件限制。
- 论文评分：**1/5**。

**强制结论：E. 淘汰。** 不允许以“联合优化”绕过单机制已失败的上界。只有找到在常见模型/预算下 ≥15% fetch critical-path、static hot set 捕获 <70%、A100 实测低比特 kernel 净快，才可重开。

### 2.11 所有 Idea 的 1～3 天最小 Go/No-Go 协议

以下是“最小能否定实验”，不是完整 evaluation。所有实验固定模型 checkpoint、prompt、随机种子、arrival trace、质量预算和硬件功率状态；至少 5 次独立重复并报告 CI。没有真实多卡机制时，模拟只能筛选，不能算验证成功。

| Idea | 推荐对象与预计资源 | 1～3 天实验 / 最小代码 | Baseline 与 Oracle | 预注册 No-Go |
|---|---|---|---|
| I1 | OLMoE top-8 + LLM-jp top-16；PyTorch/CUDA；5090 约现有 runner + ≤2 GB buffer | 复用 rank-tail 与 codec runner，导出真实 shape；做 once-fused replay；可用时加 8 卡一层 collective | BF16、uniform INT8/INT4、static cutoff；删除 combine 与 zero-codec oracle | oracle <5%；simple CH ≥80%；codec tax >毛收益 30% |
| I2 | 同两模型 route trace；5090；另需普通权限共租进程 | 只实现 observer 数据采集和 attacker，不写防御；变量为并发、batch、模板、隔离模式 | 随机/频率攻击、自然 batching；ground-truth route 与 full-observation oracle | AUC <0.70、token 优势 <20pp、跨 run 不迁移或 observer 需管理员权限 |
| I3 | OLMoE/LLM-jp offload；agent/tool pause trace；5090 8/12/16/24 GB cache budget | 在现有 cache replay 加 pause/resume、fetch timeline、last-W/future set；N=32/64 resume token | KV-only、LRU、LFU、hot set、last-W；future-set oracle | oracle <15%；last-W CH ≥70%；仅 ≤25% 权重 cache 时成立；goodput -2% |
| I4 | I1/I3 单卡 trace + 8 卡少量 ground truth；CPU/5090，8 卡只跑校准点 | 实现 roofline 和分解模型；按模型/并发/EP degree held-out，不随机泄漏相邻点 | 线性/roofline + 3–5 校准点；全配置 sweep oracle | baseline regret ≤5%；复杂模型误差 >15% 或节省 profiling <80% |
| I5 | OLMoE/LLM-jp；5090 action replay；真实 slow trace 若无则实验自动失败 | 先做 slow-event census 与删除式 P99 replay；只模拟 drop/reroute/precision | wait、timeout、exact least-load reroute；zero-cost no-slow oracle | 无真实事件；P99 oracle <15%；exact reroute CH ≥80%；质量超界 |
| I6 | 已有 matched-histogram trace；5090；无需新大模型 run | 复核普通特征与 route feature 的 action regret，做 permutation/placebo | measured-cost VTC、moving average；真实增量 cost oracle | 现有 2.35%–3.06% 已触发；新样本 oracle 仍 <5% 即永久停止 |
| I7 | OLMoE decode arrival replay；5090；NVML/板卡能量 | sweep fixed wait 0–2 ms 与 perfect packing；测焦耳、P99、goodput | immediate、continuous batch、per-expert FIFO threshold；perfect packing oracle | energy oracle <5%；threshold CH ≥80%；P99/SLO 恶化 |
| I8 | 任一可运行 Spec-MoE/离线 execution trace；5090 | 只打 timeline：reject decision、queue、kernel/collective complete；不实现 cancellation | width/depth、confidence admission；never-execute-reject oracle | late-cancellable E2E <15%；simple CH ≥70%；需取消已提交 collective |
| I9 | 真实/公开 route demand trace；CPU simulator + 8 卡仅在先例差异成立后 | 先做 prior-art mechanism matrix；再测 hotness half-life / migration break-even | static topology placement、periodic greedy、least-load；future/zero-migration oracle | 与近期算法无实质差异；half-life <5× break-even；只靠跨节点才成立 |
| I10 | 现有 cache/quant runners；OLMoE/LLM-jp；5090 | 重算正确 key/full top-k 的 deletion oracle；测真实 low-bit kernel，不写联合 controller | LFU/hot set、uniform quant、double-buffer；100%-hit/zero-quant-cost oracle | 任一单项 oracle <5%；static CH ≥80%；目标 kernel 不比 BF16 快 |

### 2.12 Paper-story 审查

| Idea | 一句话核心 insight | 技术深度/一图故事 | 适合方向 | 预期 reviewer 分 |
|---|---|---|---|---:|
| I1 | gate rank 可能同时携带质量敏感度和固定 wire-layout 信息 | 若能画出 `rank→quality Pareto→wire time→E2E`，故事完整；否则只是 codec | MLSys/ASPLOS | 3 |
| I2 | MoE 稀疏 route 是模型输出之外的新 traffic secret | threat→attack→privacy/SLO frontier 可形成强一图；威胁失败则全盘失败 | OSDI/S&P/MLSys | 3（方差最大） |
| I3 | pause 保存 KV 却丢 expert 工作集，形成双状态不一致 | 容易被概括为 InferCept+ELDR；技术深度当前不足 | MLSys workshop / 边缘系统稿 | 2 |
| I4 | 单卡 route trace 可识别多卡配置决策 | 一图能讲但像 cost model；需误差界与低样本理论 | MLSys artifact/measurement | 2 |
| I5 | 暂态降级应受跨时间质量预算控制 | 新目标函数多于新机制；已有工作过近 | 不建议投稿 | 1 |
| I6 | route coalition 应进入公平服务成本 | 自身数据已否定核心 insight | 不建议投稿 | 1 |
| I7 | SLO slack 可换 expert GEMM 能效 | 容易是 threshold heuristic；适合作为子机制 | MLSys workshop | 2 |
| I8 | reject 信号可回收尚未完成的 expert work | 一图清楚但近邻拥挤、runtime 太难 | 不建议投稿 | 1 |
| I9 | shadow price 协调副本、迁移与重路由 | 技术可深，但近期理论/系统已覆盖且资源不足 | 不建议当前投稿 | 1 |
| I10 | 敏感度和热度应联合决定精度与驻留 | 三种已知机制的组合，消融难归因 | 不建议投稿 | 1 |

---

## 3. 每个 Idea 的 Oracle / Headroom 表

所有未实测区间只用来决定是否值得测。`p` 是被优化部分的真实 critical-path 占比，不是 profiler 中可重叠时间的简单求和。

| Idea | 可执行 Oracle | 当前 `p` / 极限 | Oracle 对 E2E 的最大潜在收益 | 预计实际净收益 | Headroom 结论 | 立即停止线 |
|---|---|---:|---:|---:|---|---|
| I1 RankLane | trace 中删除 combine；零成本 rank-aware 编码 | NVLink 粗估 3%–20%；PCIe/跨节点 15%–30% | NVLink 3%–20%；慢链路 15%–30% | NVLink 0%–8%；慢链路 5%–15% | **未知，需 profiling** | 删除式 oracle <5%；或 fused codec 净收益 <5%；或 uniform 捕获 ≥80% |
| I2 RouteCloak | 全 expert/完全定长 traffic，攻击优势归零 | 性能 `p` 不适用；隐私 headroom 取决于真实 attack advantage | 最多移除 100% 攻击优势；性能收益 ≤0 | 若成立，移除 ≥80% 优势、性能代价目标 ≤10% | **未知，需 threat profiling** | 真实 observer 无显著优势；或最低防御开销 >15%；隔离即可解决 |
| I3 ResumeSet | future expert 已知且零成本保留/预取 | 全驻留 0；offload resume stall 0%–50% | 0%–50% | 0%–15%，且可能污染普通请求 | **有条件** | 真实 trace oracle <15%；last-W 捕获 ≥70%；只在极端 HBM cap 成立 |
| I4 ScaleBridge | 全 8 卡配置 sweep，regret=0 | 非 Amdahl；价值是 GPU-hours/决策 regret | 最多省去全部 profiling；不直接加速 serving | 若简单 roofline 已好，≈0 | **独立论文不足** | roofline regret ≤5% 或复杂模型 held-out 误差 >15% |
| I5 QualityDebt | 零成本、零质量损失消除所有 slow rank | `p≈q·r/(1+q·r)`；真实 q 未知 | 常见假设约 <5%，P99 可更高 | 负值到低个位数 | **偏低/未知** | 无真实 slow census；exact reroute 捕获 ≥80%；真实 P99 headroom <15% |
| I6 RouteShare | 已知每个 batch 的真实增量服务成本 | 本地匹配后 2.35%–3.06% | 约 <5% | 约 0%–2% | **偏低** | 现有 Gate-0 已触发 |
| I7 JouleBatch | 零等待、完美 expert packing | expert 能耗 30%–60% × 能效提升 10%–20% | energy/token 约 3%–12%；P99 可为负 | 0%–8% energy，latency 不保证 | **中低** | energy oracle <5%；固定阈值捕获 ≥80%；SLO 恶化 |
| I8 Cancellable | 从不执行最终拒绝 branch | rejected work 可高，但 late-cancellable residual 未知 | 理论 5%–30%；可回收部分常 <10% | 负值到 5% | **未知但 novelty 已死** | timeline oracle <15%；width/admission 捕获 ≥70%；collective 不可安全取消 |
| I9 Shadow-Price | 已知未来、零成本迁移/复制 | skew 场景可 10%–30% | 10%–30% | 单节点 0%–10% | **理论充足，研究空间不足** | periodic greedy 捕获 ≥80%；迁移 break-even 长于 hotness 半衰期 |
| I10 Sensitivity | 100% hit + 零成本最低比特 + 零质量损失 | 全驻留 0；offload 可高 | 0%–50%（取决于人为容量） | 当前实测先验 ≤0%–5% | **常见配置偏低** | safe-budget oracle <5%；static set ≥80%；low-bit kernel 不快 |

三个容易被误读的结论：

1. I9 的 oracle 大，不代表值得做：已有工作和资源边界可以独立淘汰一个高上界方向。
2. I2 的价值指标是隐私而非速度；它必须接受明确的性能负收益预算。
3. I3/I10 的高上界依赖模型不驻留 HBM。用人为缩小 cache 制造 50% miss，不能证明常见部署有 50% headroom。

### 3.1 分指标上界约束

| Idea | 平均 latency | P99 | Throughput/goodput | Energy/token | 明确不会改善的指标 |
|---|---|---|---|---|---|
| I1 | 受 combine `p` 限制，NVLink 常见估计 3%–20% oracle | 若 slow rank combine 主导可高于平均；也可能被额外 launch 恶化 | 饱和时与关键路径上界同量级 | 最多为通信能耗份额，codec 可抵消 | TTFT 若 prefill compute-bound 可能几乎不变 |
| I2 | 防御只会持平或变差 | padding/dummy 常使其变差 | 只会持平或变差 | 只会持平或变差 | 所有性能指标；它优化 privacy |
| I3 | 全请求平均取决于 resume 请求占比，常远低于 resume 指标 | resume 前 N token oracle 0%–50% | cache pollution 后可能为负 | 少搬运时可降，保留/预取也可升 | 全驻留时全部性能指标不变 |
| I4 | 不直接改善 | 不直接改善 | 仅通过选配置间接改善 | 不直接改善 | serving 指标的直接贡献为零 |
| I5 | 真实事件稀少时常 <5% | 若 slow event 决定尾部可 >15%，但未证 | action 错误可降低 | 备用计算/低效动作可升 | TTFT 若事件只在 decode 不变 |
| I6 | 本地 oracle <5% | 未见足够残余 | 0%–低个位数 | 未针对能耗 | 模型质量不变但也非贡献 |
| I7 | 可能为负到低个位数 | 等待使其最可能恶化 | packing 可小幅提升 | oracle 约 3%–12% | TTFT/prefill 通常无益 |
| I8 | decode oracle 随无效 work 0%–30% | cancellation 同步可恶化 | 可回收部分通常小于拒绝比例 | 与真正省掉的 work 同量级 | prefill 无改善 |
| I9 | skew 下 oracle 10%–30% | 重配置瞬时可恶化 | 理论上同量级 | 多副本/迁移可能增加 | 单 GPU 全部为零 |
| I10 | 全驻留 0；offload 取决于 fetch `p` | 预测错会恶化 | 只有真实 kernel/fetch 快才改善 | 多精度副本和转换可升 | 全驻留 cache 指标为零 |

---

## 4. 每个 Idea 的最强简单 Baseline

| Idea | 最强简单 baseline | 最可能捕获的 headroom | 复杂机制必须额外证明什么 | 让问题消失的配置 |
|---|---|---|---|---|
| I1 | 单消息 uniform INT8/INT4 + static rank cutoff + communication overlap | 字节收益和大部分质量—精度 Pareto | 在相同消息数、相同 fused kernel 下再有 ≥5% E2E 或明显更优质量 | 更强 overlap、增大 batch、换 EP/TP、全 BF16 走 NVLink |
| I2 | continuous batching/mixing + 固定 bucket padding | 大部分 traffic 模糊化 | 相同 ≤10% 开销下再移除 ≥20% oracle attack advantage | MIG/权限隔离/固定 API 直接阻断观察 |
| I3 | pause TTL 内 last-W expert union + static quota | 大部分短期 route locality | last-W CH <70%，联合状态仍改善 E2E ≥10% | 模型权重全驻留 HBM |
| I4 | roofline + NCCL microbench + 3–5 个 8 卡校准点 | 大部分配置排序 | regret 再降 ≥5pp，profiling 运行减少 ≥80% | 直接跑小规模配置 sweep |
| I5 | timeout + least-loaded exact reroute/replica | 大部分 slow-rank tail | 在相同质量下 P99 再降 ≥10% | 多副本、修复硬件/网络、保守 SLO admission |
| I6 | measured-cost VTC / moving-average service time | 本地结果显示几乎全部 | route feature 在匹配普通特征后仍带来 ≥5% E2E | 普通 continuous batching + cost normalization |
| I7 | fixed slack threshold + per-expert FIFO | 大部分 packing | energy 再降 ≥5%，P99/SLO 不变 | 增大普通 batch 或 persistent kernel |
| I8 | 减小 speculative width/depth + confidence admission | 大部分无效工作避免 | late cancellation 仍有 ≥10% E2E 增量 | 更准确 draft/降低分支数 |
| I9 | topology-aware static placement + periodic greedy replication + least-load reroute | 稳定 hotness 下大部分收益 | 非平稳 trace 上再提升 ≥10%，且迁移税 <毛收益 30% | 增副本/HBM、使用 locality routing |
| I10 | uniform INT8/INT4 + LFU/static hot set + double-buffer prefetch | 大部分低比特和 locality 收益 | 联合 controller 再有 ≥10% E2E，质量相同 | 全驻留、加 HBM、静态量化 |

Captured Headroom 必须在相同质量、相同 SLO、相同 HBM、相同消息数和相同 arrival trace 下计算；任一约束不同，该比值无效。当前只有 I6 有足够本地反证可近似判断 simple baseline 已覆盖绝大部分可解释空间；其他方向不得虚构 CH 数字。

---

## 5. 每个 Idea 的 Red Team 拒稿意见

以下每条都附带“唯一能推翻它的证据”。没有该证据时，默认拒稿理由成立。

### I1 RankLane-Combine

1. **优化空间不足：** 单节点 NVLink 上 combine 可能仅占 3%–10%，完全删除也解释不了声称的两位数收益。推翻证据：强 overlap 后的真实 8 卡 timeline 显示 combine critical path ≥15%，跨 prefill/decode 与至少两个模型成立。
2. **codec 已有负结果：** 本地 58–61 μs codec tax 在 200–400 Gbps 等效链路下 0/8 可行。推翻证据：producer/consumer fused kernel 在真实 A2A 上净减少 wall-clock ≥10%，不是逻辑字节或 H2D proxy。
3. **创新性像静态量化：** gate-rank cutoff 可能只是 uniform quantization 的一个 feature。推翻证据：与 activation magnitude、per-token scale、uniform、random rank 在同质量约束下形成跨模型稳定 Pareto，并解释非显然原因。
4. **最强 baseline 未过：** DeepEP/SwiftEP/Comet 式 overlap 可能吞掉所有 headroom。推翻证据：在这些栈或等价最佳配置上仍有 ≥5% E2E 增量。
5. **实验资源偏置：** 5090 无 EP，8×A100 单节点又可能过快；Idea 只在 PCIe/跨节点成立。推翻证据：现有 A100 拓扑上通过；若只在跨节点通过，则当前资源直接不匹配。
6. **尾延迟可能恶化：** 双 lane/额外 launch 增加小 batch 固定开销。推翻证据：单 collective、融合实现下 P99 不升，所有 batch bucket 均报告净值。

### I2 RouteCloak

1. **威胁模型不现实：** 普通 tenant 可能看不到 expert-level traffic。推翻证据：明确低权限 observer、可复现 API/侧信道和访问控制，非管理员权限下跨进程攻击成功。
2. **从 route 可重构文本不等于从硬件 proxy 可重构 route。** 推翻证据：端到端 observer→proxy→token/attribute 攻击，而非把 ground-truth route 直接喂给攻击器。
3. **padding 是教科书机制：** novelty 可能只是在 MoE 上做 bucket。推翻证据：证明 route 结构产生新的 privacy–SLO frontier，且机制相对固定 bucket 有显著不可被调参解释的收益。
4. **防御开销不可接受：** top-8/64 的全隐藏最坏需约 8× expert work。推翻证据：≤10% TPOT/P99/energy 代价下移除 ≥80% 攻击优势。
5. **隔离 baseline 更强：** MIG、权限禁用或固定 collective 可能免费消除观察。推翻证据：现实部署中隔离不可用/不足，并在相同资源成本下比较。
6. **泛化性弱：** 攻击可能只记住所选模型/语料的 router。推翻证据：跨 checkpoint、语言、prompt template 和并发模式迁移。

### I3 ResumeSet

1. **问题在全驻留时消失：** 8×A100 可容纳目标模型后 expert fetch 为零。推翻证据：明确且常见的目标模型超过可用 HBM，或多租户 HBM partition 使 offload 成为正常部署而非人为限制。
2. **ELDR/InferCept 已覆盖抽象：** KV-indexed expert signature + interruption KV policy 的组合很直接。推翻证据：pause/resume 出现 ELDR 未处理的新 cold-start phase，并需要非显然的 eviction/admission 机制。
3. **last-window baseline 足够：** 短期 expert locality 自然由最近 W token 捕获。推翻证据：last-W CH <70%，新机制在相同 HBM 下仍有 ≥10% E2E 增量。
4. **命中率不等于恢复速度：** 预取可被 compute 隐藏，或 miss 不在关键路径。推翻证据：删除式 stall oracle 和 CUDA/NVTX timeline，把 hit 转化为 resume TTFT/P99。
5. **污染普通请求：** 为暂停请求保留权重会挤掉活跃请求热专家。推翻证据：全局 goodput/SLO 不降超过 2%，并报告受害请求 P99。
6. **workload 合成：** 固定 pause 长度/人工 cache cap 可制造结论。推翻证据：真实 agent/tool/human trace，多 pause 分布、多模型和敏感性分析。

### I4 ScaleBridge

1. **不是 serving 改进：** 预测器不直接改善 E2E，只减少 profiling。推翻证据：展示真实配置选择 regret 导致 ≥10% serving 损失，且方法可靠避免。
2. **简单 roofline 足够：** NCCL microbench + 少量校准点可能已把 regret 压到 5% 内。推翻证据：严格 held-out 比较，复杂模型显著降低配置 regret而非仅 MAPE。
3. **单 topology 过拟合：** 8×A100 只有一种互联。推翻证据：至少两类 topology/硬件；当前资源无法自然提供。
4. **单卡特征不识别多卡排队：** contention/overlap 不可由 route trace唯一推出。推翻证据：跨并发/EP degree 的因果分解和校准后外推。
5. **论文技术深度不足：** 容易沦为 curve fitting。推翻证据：可解释误差边界、可识别条件、失败域和显著低样本优势。

### I5 QualityDebt-SlowRank

1. **真实优化空间未知且可能 <5%：** 没有 slow-rank census。推翻证据：生产 trace 显示目标 P99 中 ≥15% 可归因于暂态 slow rank。
2. **Capacity-Aware Inference/ReaLB 已覆盖动作。** 推翻证据：明确二者无法处理的事件/质量约束，并有非显然新机制而非新目标函数。
3. **exact reroute baseline 可能无质量损失且足够。** 推翻证据：在同副本/HBM 下 exact reroute CH <70%，quality debt 再降 P99 ≥10%。
4. **质量债务不能恢复已错 token：** 跨时间平均质量预算不保证单请求语义正确。推翻证据：任务级、请求级 worst-case 质量约束，而非平均 perplexity。
5. **注入 straggler 不代表现实：** 8 卡实验可能只证明注入器有效。推翻证据：真实故障/节流分布与注入参数匹配。
6. **动态低精度未必更快：** A100 无 FP8，本地路径已有负收益。推翻证据：真实 A100 kernel 在目标 shape 上净快并计入转换税。

### I6 RouteShare-VTC

1. **本地 oracle 已低：** 匹配后 latency 差异 2.35%–3.06%。推翻证据：新的真实多租户 trace 在控制混杂后显示 ≥10% 增量服务成本。
2. **普通成本模型已解释 99.7% 以上方差。** 推翻证据：route feature 对 action regret 而非 R² 有显著额外贡献。
3. **创新性只是 VTC 换 feature。** 推翻证据：存在 route coalition 特有的非加性资源共享，并需要新的公平定义/算法。
4. **复杂度可能二次增长。** 推翻证据：O(active requests) 或更低实现，关键路径开销 <毛收益 10%。
5. **8 卡 incast 与单卡共享不是同一机制。** 推翻证据：分别拆出 compute coalition 和 destination-rank contention，并证明同一 policy 正确。
6. **收益可能来自 workload selection。** 推翻证据：匹配 token、长度、arrival、histogram、batch 后跨模型成立。

### I7 JouleBatch

1. **energy headroom 可能 <5%。** 推翻证据：理想 packing oracle 在真实 arrival trace 上 E2E energy/token ≥10%。
2. **固定等待阈值足够。** 推翻证据：动态机制相对 per-expert FIFO+threshold 再降 ≥5% 能耗且 SLO 相同。
3. **与 PALS/Festina/ExpertPlex 增量。** 推翻证据：清楚展示 MoE route-induced packing 的新约束和新算法，而非 deadline scheduler 特化。
4. **等待恶化 P99。** 推翻证据：完整 energy–P99 Pareto，在相同 violation rate 下比较。
5. **GPU 利用率升不等于能耗降。** 推翻证据：板卡级焦耳积分、静态功耗和通信能耗，而非 occupancy proxy。
6. **多卡 burst 可能恶化 A2A。** 推翻证据：rank-level timeline 显示 coalescing 不制造 incast/同步尾部。

### I8 Cancellable Spec-MoE

1. **近邻工作过密，机制可能已被覆盖。** 推翻证据：逐项对 SP-MoE、MoE-Spec、MoE-SpAc、SpecMoE 做机制表，证明 cancellation 尚无。
2. **知道拒绝时工作已完成。** 推翻证据：真实 GPU timeline 显示 ≥15% E2E 是“判决后尚未完成且可取消”的工作。
3. **减小 tree/admission 已足够。** 推翻证据：强 baseline CH <70%。
4. **collective 不能安全取消。** 推翻证据：无死锁、rank-consistent 的取消协议和故障测试。
5. **细粒度 task 破坏 batching。** 推翻证据：计入 launch/fragmentation 后仍有 ≥10% TPOT/energy 净收益。
6. **只在低接受率极端 workload 有效。** 推翻证据：多 draft model、多接受率、真实 prompt 分布均通过。

### I9 Shadow-Price Placement

1. **Mixture-of-Experts Serving 已形式化同一问题。** 推翻证据：新问题约束或理论结果，不是换求解器/变量名。
2. **periodic greedy 足够。** 推翻证据：非平稳 trace 上 shadow-price 相对 greedy 再有 ≥10% E2E。
3. **迁移税抵消收益。** 推翻证据：hotness 半衰期至少为迁移 break-even 的 5×，并计入 tail spike。
4. **单卡完全不可验证。** 推翻证据：无；这是资源事实。单卡 simulator 只能筛选，不能支持论文结论。
5. **8 卡不足以证明 scale-out。** 推翻证据：至少跨 topology/跨节点；当前资源缺失。
6. **复制 HBM baseline 可让问题消失。** 推翻证据：在相同 HBM budget 下比较，且静态冗余仍失败。

### I10 Sensitivity Cache/Quant

1. **已有工作高度覆盖。** 推翻证据：相对 HOBBIT、DynaExq、ProMoE、MODE 的新观察和不可替代机制。
2. **本地 safe-budget cache oracle 接近零。** 推翻证据：真实常见容量配置下 fetch critical-path ≥15%，不是人为极小 cache。
3. **静态 hot set/LFU 足够。** 推翻证据：跨 workload CH <70%。
4. **逻辑低比特没有 kernel 加速。** 推翻证据：A100 目标 shape 上端到端实测快，包含 quant/dequant 和 packing。
5. **联合三个控制环无法归因。** 推翻证据：完整 factorial ablation，联合项产生稳定非加性增益 ≥5%。
6. **全驻留配置问题消失。** 推翻证据：明确常见目标模型/多租户预算必须 offload，并跨两种容量比成立。

---

## 6. 淘汰名单及原因

### 直接淘汰（E）

| Idea | 首要淘汰原因 | 次要原因 | 不应继续做的工作 |
|---|---|---|---|
| I5 QualityDebt | Capacity-Aware Inference/ReaLB 已覆盖关键 action | 无真实 slow-rank census；A100 低精度路径不成立 | 不要先实现 quality ledger 或 8 卡注入器 |
| I6 RouteShare | 本地匹配后差异仅 2.35%–3.06% | cost model 已解释 99.7% 以上波动 | 不要实现 coalition scheduler |
| I8 Cancellable | 多篇 2025–2026 Spec-MoE 直接近邻 | late-cancellable residual 未知；collective 风险高 | 不要实现 GPU cancellation runtime |
| I9 Shadow-Price | 在线 allocation/reconfiguration 已被近期工作正面覆盖 | 单卡无证据，8 卡单 topology 不足 | 不要做 simulator-only 主论文 |
| I10 Sensitivity | cache/quant/prefetch 三条线均有强近邻 | 本地 cache oracle 与 dynamic quant 已有 NO-GO | 不要把失败组件组合成更复杂 controller |

### 降级而非独立立项（D）

- **I4 ScaleBridge：** 作为 I1/I2/I3 的 trace replay、删除式 oracle、配置 regret 和 8 卡预测工具。若写进论文，应是 methodology/measurement section，不宣称独立系统。
- **I7 JouleBatch：** 作为任何最终 serving 系统的 energy/token 目标、固定阈值 baseline 或 scheduler ablation。没有 ≥10% energy oracle 前不做新 controller。

### 不合并成“更大系统”的原因

把 I4、I7、I10 加到 I1 上不会自动提高论文价值。每多一个 controller 都增加因果混淆、调参空间和失败面。主线必须先用单一机制得到可解释净收益，再考虑把 I4 用作测量工具、I7 用作额外指标。I5/I8/I9/I10 不作为 future-work 模块保留。

---

## 7. 保留名单及前置条件

| 角色 | Idea | 获准继续的唯一前置条件 | 条件未满足时 |
|---|---|---|---|
| 首选候选 | I1 RankLane | 真实/等价 trace 的 combine 删除式 oracle ≥10%，fused codec 预计税 <毛收益 30%，uniform baseline CH <80% | 立即停止，不做 8 卡完整 backend |
| 高风险候选 | I2 RouteCloak | 非管理员 observer 的攻击优势显著，静态隔离不足，full-padding oracle 表明 ≤10% overhead 区间内存在可用 privacy frontier | 立即停止，不做 dummy runtime |
| 条件备选 | I3 ResumeSet | 真实 interruption 下 resume fetch critical path ≥15%，last-W CH <70%，非极端容量配置跨两模型成立 | 立即停止，不做联合 cache manager |

重复性检查：

- I3 与 I10 都依赖 expert hotness/locality；I10 淘汰后，任何 cache/prefetch 组件只能服务于 I3 的 pause-specific hypothesis，不能泛化回“动态热度系统”。
- I5 与 I9 都处理 rank imbalance；前者以质量降级，后者以资源重配。近期工作已覆盖两类 action，不合并。
- I1 与 I7 都可能改变通信/计算 batching。若 I1 进入实现，I7 只作为能耗和 batch-shape 消融，不能同时启两个控制环。
- I4 与所有 Idea 共享 trace/performance modeling，但它是测量基础，不是第四条研究线。

---

## 8. 横向评分表与排名

### 8.1 1–5 分评分

`Risk` 分数越高表示失败风险越高；其余维度越高越好。

| Idea | Importance | Headroom | Novelty | Mechanism | Baseline resistance | 5090 verify | 8A100 value | Measurement clarity | Feasibility | Generality | Paper | Risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I1 RankLane | 4 | 3 | 4 | 4 | 3 | 4 | 5 | 4 | 3 | 3 | 4 | 4 |
| I2 RouteCloak | 5 | 3 | 5 | 3 | 3 | 2 | 4 | 2 | 2 | 3 | 4 | 5 |
| I3 ResumeSet | 3 | 3 | 2 | 3 | 2 | 4 | 3 | 4 | 3 | 2 | 2 | 4 |
| I4 ScaleBridge | 3 | 2 | 3 | 3 | 2 | 2 | 5 | 4 | 4 | 2 | 2 | 3 |
| I5 QualityDebt | 3 | 2 | 1 | 2 | 1 | 2 | 3 | 2 | 3 | 2 | 1 | 5 |
| I6 RouteShare | 2 | 1 | 2 | 2 | 1 | 5 | 2 | 5 | 3 | 2 | 1 | 4 |
| I7 JouleBatch | 3 | 2 | 2 | 3 | 2 | 5 | 3 | 4 | 4 | 3 | 2 | 3 |
| I8 Cancellable | 4 | 3 | 1 | 2 | 2 | 3 | 4 | 3 | 1 | 3 | 1 | 5 |
| I9 Shadow-Price | 5 | 4 | 1 | 4 | 2 | 1 | 3 | 3 | 1 | 4 | 1 | 5 |
| I10 Sensitivity | 4 | 2 | 1 | 2 | 1 | 4 | 2 | 4 | 2 | 2 | 1 | 5 |

### 8.2 决策附表

| Idea | 最关键未知变量 | 最大拒稿风险 | 结论 |
|---|---|---|---|
| I1 | overlap 后 combine critical-path 占比与 fused codec tax | byte reduction 不转化为 E2E | B |
| I2 | 普通 observer 是否真实看到 route proxy | 威胁模型不可达 | B |
| I3 | pause 后 expert fetch stall 占 resume 时间比例 | ELDR/InferCept 组合式增量 | C |
| I4 | simple roofline 的配置 regret | 只是 curve fitting 工具 | D |
| I5 | 真实 slow-rank 频率与贡献 | 已有工作覆盖 | E |
| I6 | route overlap 的残余因果成本 | oracle <5% | E |
| I7 | perfect packing 的 E2E energy oracle | threshold baseline 足够 | D |
| I8 | 判决后可取消的 critical work 比例 | 近邻覆盖 + 取消过晚 | E |
| I9 | hotness 半衰期/迁移 break-even | 已有理论与系统覆盖、资源不足 | E |
| I10 | 常见容量下 fetch critical path | 组件各自已有 NO-GO/先例 | E |

### 8.3 六类排名

排名不代表末位之间有统计显著差异；它只是当前证据下的资源优先级。

1. **综合：** I1 > I2 > I3 > I4 > I7 > I5 > I8 > I10 > I9 > I6。
2. **Headroom（不考虑 novelty/资源）：** I9 > I3 > I1 > I8 > I2 > I7 > I5 > I10 > I6 > I4。I2 是隐私 headroom，I4 是 profiling cost，不能与 latency 完全同标尺。
3. **创新性：** I2 > I1 > I4 > I3 > I7 > I6 > I5 > I8 > I9 > I10。
4. **最容易快速得出 go/no-go：** I6（已 NO-GO）> I1 > I7 > I3 > I10 > I8 > I2 > I5 > I4 > I9。
5. **最可能写成完整论文：** I2（若威胁成立）> I1 > I3 > I4 > I7 > I5 > I8 > I9 > I10 > I6。
6. **继续投入最可能浪费时间：** I9 > I8 > I10 > I5 > I6 > I7 > I4 > I3 > I2 > I1。

---

## 9. 最终 Top 3

这里的 Top 3 是“仅允许进入下一道门槛”的三项，不是三条都开工。

### 9.1 首选方向：I1 RankLane-Combine

- 为什么选：本地已经有跨两个模型的 rank-tail 质量不对称证据；直接相同机制的近期工作尚未发现；5090 可以廉价完成 codec/oracle 门槛，8 卡能增加真实 A2A 证据。
- 为什么不是其他方向：相比 I3，它不依赖权重 offload；相比 I2，它不依赖难以获得的侧信道权限；相比 I9，它没有明显资源错配。
- 第一个实验：第 10 节的 `critical-path × codec break-even` 双门槛，不实现完整 runtime。
- 失败阈值：combine 删除式 oracle <10%，或 uniform baseline CH ≥80%，或 fused codec tax >毛收益 30%，任一即停止。
- 成功后：只实现单 collective、producer/consumer fused 的两 lane layout；先做 8 卡一层/多层 replay，再接 serving backend。

### 9.2 次选方向：I3 ResumeSet

- 为什么选：pause/resume 是明确 workload，5090 能用真实 offload 验证删除式 oracle，失败也可迅速发现。
- 为什么不是 I7/I4：I3 至少可能影响用户可见 resume TTFT；I7 headroom 偏低，I4不直接改善 serving。
- 第一个实验：从真实/半真实 agent interruption trace 提取恢复后 N=32/64 token，比较 KV-only、LRU、LFU、last-W 与 future oracle，在相同 HBM budget 下测 fetch critical-path。
- 失败阈值：oracle <15%；last-W CH ≥70%；全局 goodput 下降 >2%；只能在 ≤25% 模型权重 cache 的极端条件成立。
- 成功后：把问题严格限定为“pause-induced expert cold start”，与 ELDR 做逐项机制差异，先实现 TTL+quota，不上学习模型。

### 9.3 高风险高收益方向：I2 RouteCloak

- 为什么选：若现实 observer 成立，它是十个方向中问题新颖性和社会重要性最高的一个，且 exact-output + bounded-overhead 的系统 frontier 可形成强论文。
- 为什么不是主线：最关键的 observer 可能根本不存在或需要管理员权限；一旦 threat model 不成立，后续所有防御代码归零。
- 第一个实验：只复现攻击链。普通权限共租进程收集可达计数器/时间序列，预测 ground-truth route、token 或敏感属性；同时测自然 continuous batching baseline。
- 失败阈值：跨 run AUC <0.70、token top-1 优势 <20pp、跨模板/模型不迁移，或 MIG/权限隔离几乎零成本阻断；任一则停止。
- 成功后：先画 full-padding 与 fixed-bucket 的 privacy–P99 oracle frontier；只有存在 ≤10% overhead 可用区间才写 RouteCloak runtime。

### 9.4 Top 3 机会成本

| Idea | 1–3 天能获得的结论 | 1–2 周实现 | 最大工程风险 | 失败后可沉淀 | 选择它放弃的机会 |
|---|---|---|---|---|---|
| I1 | combine 上界、codec break-even、uniform CH | fused pack/unpack + 单 collective replay | backend/ABI 融合后仍被 NVLink 掩盖 | rank-quality + wire-time negative/positive measurement | 暂缓安全 observer 与 pause trace 工作 |
| I3 | resume cold-start 是否真实、last-W 是否足够 | cache TTL/quota、KV-signature metadata、offload replay | 人为容量压力和 cache pollution | interruption/workset locality measurement | 放弃直接通信机制的多卡优势 |
| I2 | threat model 是否存在 | attacker、fixed-bucket/full-padding runtime | 权限不可达、跨模型攻击失败 | 现代 GPU 隔离下的负测量，但论文价值不稳 | 放弃最确定的性能 profiling |

顺序应为：先完成 I1 门槛；I1 NO-GO 后启动 I3；I2 的 observer profiling 可在不占 GPU 主线实现时间时做，但不得先写防御系统。

---

## 10. 首选 I1 的 1～3 天 Go/No-Go 实验

### 10.1 目标

只回答三个问题：

1. 在目标 shape/链路下，combine 位于多少真实 critical path？
2. 零成本减少 combine payload 后，端到端 oracle 是否至少 10%？
3. 最简单 uniform 编码是否已经捕获 ≥80% oracle；融合 codec 税是否低于毛收益 30%？

这三问未通过前，不实现自适应策略、不写完整 NCCL plugin、不做大规模质量评测。

### 10.2 模型、框架与数据

- 模型 1：OLMoE（top-8），复用 `docs/ideas/A_rank_tail_fp8` 已有 route/quality runner。
- 模型 2：LLM-jp MoE（top-16），用于跨模型否证。
- 单卡框架：现有 PyTorch/CUDA 环境；复用 `run_idea_a_rank_lut_gpu_rigorous_verify.py` 和 `run_homogeneous_lane_codec_gate_gpu.py`，仅加 timeline/replay 输出。
- 8 卡若在第 3 天可用：优先 DeepEP/NCCL 当前可运行 backend；没有可用 backend 时，不用 H2D proxy冒充正式结果。
- 数据：已有 matched-quality 文档集 + 至少一组 serving-like prompt；route/activation shape 必须按 prefill、decode、batch/concurrency bucket 分层。
- 规模：每模型至少 200 个文档用于误差复核；每个性能 bucket ≥1,000 次 warm iteration，5 个独立 run；报告 bootstrap 95% CI。
- 预计 5090 显存：模型按现有 runner 的可运行配置，额外 activation/codec buffer 控制在 2 GB 内；禁止通过缩小到不代表真实 traffic 的 tiny tensor 获得结论。

### 10.3 实验矩阵

自变量：

- payload：BF16、uniform INT8、uniform INT4、固定 dual-lane（高 rank BF16/INT8 + 低 rank INT4）；
- rank cutoff：0、25%、50%、75%、100%；
- shape：真实 trace 的 P10/P50/P90 token-per-rank，至少覆盖 rows 32/64/128/256/512；
- 等效链路：25/50/100/200/400 Gbps，另加真实 8 卡链路若可用；
- 模式：unfused（负对照）、每 tile serialized、每 step once-fused replay。

控制变量：相同 router output、相同 token/expert assignment、相同随机种子、相同 scale metadata 计费、相同消息数、相同 warm-up、相同功率状态、相同质量样本。严禁 dual-lane 用一个 collective 而 baseline 用两个，或只给新机制做 overlap。

### 10.4 Baseline 与 Oracle

- B0：BF16 单消息通信，无 codec。
- B1：uniform INT8 单消息，使用相同 fused pack/unpack 假设。
- B2：uniform INT4 单消息。
- B3：最佳 static rank cutoff，不做 per-token/per-layer预测。
- B4：activation-magnitude cutoff，防止“任何 signal 都能分组”。
- Oracle-P：从 trace 中删除全部 combine critical-path 时间。
- Oracle-C：codec cost=0，但保留真实消息启动和不可压缩 metadata。
- Oracle-Q：逐样本选择满足质量约束的最低精度；只用于质量上界，不可作为在线系统结果。

### 10.5 指标与代码修改

必须记录：

- dispatch、expert GEMM、combine、overlap residual 的 CUDA event/NVTX 时间；
- codec pack/unpack P50/P95/P99、实际读写字节、临时 buffer；
- replay net saving、break-even bandwidth、每 step message count；
- E2E prefill latency、TPOT、P99、tokens/s、energy/token；
- quality delta 及文档级 paired bootstrap CI；
- `Captured Headroom` 和 `codec_tax / gross_saving`。

最小代码修改：

1. 给现有 rank-tail runner 导出 `(phase, layer, rank, tokens_per_dest, dtype_bytes)` shape histogram；
2. 给 codec runner 添加单 buffer dual-lane layout、统一 metadata 计费和 once-per-step fused lower-bound；
3. 新增纯离线脚本合并 shape、codec LUT 和带宽/启动模型，输出每 workload bucket 的 B0–B4、Oracle-P/C/Q；
4. 若 8 卡可用，只加 NVTX 分解和一层/多层 collective replay，不先改全模型执行图。

### 10.6 预注册判据

**GO 必须同时满足：**

1. 至少两个模型的代表性 decode/continuous-batching bucket 中，Oracle-P E2E ≥10%，且至少一个常见 bucket ≥15%；
2. once-fused dual-lane 的预计/实测净收益 ≥5%，95% CI 下界 >0；
3. codec tax ≤ gross byte-time saving 的 30%；
4. 相同质量约束下 dual-lane 相对最佳 uniform/static baseline 仍获得 ≥20% 剩余 oracle headroom，即 simple CH <80%；
5. P99 不恶化 >2%，消息数不增加，跨两个模型质量约束均通过。

**NO-GO 任一即触发：**

- Oracle-P <5%：永久淘汰；5%–10%：降级为工程优化，不作为论文主线；
- 只有 ≤50 Gbps 等效链路为正，而目标 8×A100 为 NVLink/NVSwitch：资源不匹配；
- uniform/static baseline CH ≥80%；
- codec tax >毛收益 30%，或只能通过不现实的零成本 fusion 通过；
- 收益仅在单模型、单 shape 或合成 batch 下成立；
- P99 恶化 >2% 或质量 CI 越过预设边界。

### 10.7 三天日程与退出点

- Day 1：导出真实 shape；完成 B0/B1/B2 与 Oracle-P；若所有常见 bucket Oracle-P <5%，当天结束。
- Day 2：once-fused lower-bound、dual-lane/B3/B4、质量配对复核；若 simple CH ≥80% 或 codec 税超限，结束。
- Day 3：仅在前两天通过时做真实 8 卡一层/多层 replay；若真实链路净收益与 replay 方向相反，判 NO-GO 并定位启动/overlap/访存误差。

---

## 11. 对当前研究路线的明确建议

1. **撤销上一版“直接保留 Top 3”的执行含义。** I1/I2 都是 B，I3 是 C；没有 A。
2. **未来一周只允许一个主 GPU 任务：I1 三天硬门槛。** 在通过 Oracle-P、simple baseline 和 codec-tax 三关前，不写完整系统。
3. **I6 立即归档；I5/I8/I9/I10 停止新代码。** 这些方向继续投入的边际信息价值低于其机会成本。
4. **把 I4 固化成公共实验方法。** 所有后续 Idea 必须先有删除式 oracle、真实 critical-path 分解、simple CH 和配置消失测试；这比再发散候选更重要。
5. **I2 先攻攻击链，不攻防御。** 没有现实 observer 的 defense paper 是空中楼阁。
6. **I3 必须正面回应 ELDR。** 若无法用“pause-induced eviction/cold-start”画出 ELDR 未覆盖的独立 phase，直接淘汰，不用换名字包装。
7. **8×A100 只用于增加因果证据。** 单卡 oracle 未过线的方向，不得用多卡“再试试看”；需要跨节点才能成立的方向按资源不匹配淘汰。
8. **论文主张必须限定到硬件事实。** A100 无原生 FP8；逻辑 payload、fake quant 和 H2D proxy 都不能写成 wire-level 或 Tensor Core-level 加速。

本轮唯一合理的下一步是：**先做 I1 的三天否证实验；通过才工程化，失败就切 I3 的 pause-specific oracle；I2 仅以低成本 observer profiling 作为高风险旁线。** 任何不满足预注册阈值的结果都应停止，而不是通过增加 controller、换 workload 或降低 baseline 来救活。
