# MoE 12 Ideas：Reviewer 级严格复审、Oracle 与判死实验

> 本文已与另外两份候选/复审材料合并到[统一主文档](./MoE_研究方向统一梳理_17项去重筛选与Top3执行路线_2026-07-25.md)；本文保留为 12 项 Reviewer 复审附录。

日期：2026-07-25

输入：[上一轮 12 个候选](./MoE推理新方向_文献空白与上界审查_2026-07-25.md)

资源：1×RTX 5090 32GB；单机 8×A100（40/80GB、PCIe/SXM/NVSwitch 待确认）

> 附件只给出了 Review 规范，没有另附 idea 清单；本报告将上一轮 I1–I12 作为被审对象。
> 所有未实测百分比都是用于判死的区间假设，不是预期结果。`[本地]` 表示仓库已有实验，
> `[论文]` 表示作者报告，`[推断]` 表示尚待实验验证。

## 1. 总体判断

严格标准下，**没有任何 Idea 达到 A. 立即验证**。上一轮把 ScaleTail 列为首选、把
RankLane 和 ResumeSet 列为 Top 3，仍然偏乐观：ScaleTail 更像所有多卡研究的测量工具，
不是天然成立的独立系统论文；RankLane 尚未证明真实 combine critical path；ResumeSet
在模型全驻留时上界严格为零。

| Idea | 强制结论 | 一句话判决 |
|---|---|---|
| I1 RouteShare | **E 淘汰** | 本地 matched-route 残差仅 2.35%–3.06%，简单模型几乎解释全部差异 |
| I2 Expert-set speculation | **E 淘汰** | EcoSpec、EVICT、MoE-Spec 已正面覆盖 |
| I3 Layered prefill | **E 淘汰** | From Tokens to Layers 已提出同一观察和机制 |
| I4 KV/Expert HBM | **E 淘汰** | FluxMoE 已把 expert residency 与 KV pressure 联合建模 |
| I5 Fused quantize | **D 子机制** | 需要作为 I12 codec/backend，而非独立论文 |
| I6 Route clocking | **D 子机制** | PALS/Festina 后只剩“route feature 是否有增量”的消融 |
| I7 DriftGuard quant | **D 子机制** | 没有可执行快区，且动态 expert quantization 已拥挤 |
| I8 ScaleTail | **D 子机制** | 是测量/配置工具；必须显著胜过 roofline+少量校准点才可能独立 |
| I9 RouteCloak | **B 先 Profiling** | 高风险高收益；先证明低权限真实观察者能稳定看到 route |
| I10 Critical sibling | **E 淘汰** | 与旧 FJRC/receiver rescue 重复，单卡无法验证核心因果 |
| I11 ResumeSet | **C 有条件保留** | 先证明 KV-only 后仍有 ≥15% 的 resume expert cold-start 上界 |
| I12 RankLane | **B 先 Profiling** | 首选，但先测 combine 删除式 oracle、uniform INT8 和 fused codec 税 |

资源顺序应是：**I12 必要条件实验 → I11 独立 cold-start oracle → I9 真实攻击面存在性**。
不应先写调度器、collective 或在线预测器。

## 2. 每个 Idea 的严格 Review

### Idea I1：RouteShare——route coalition 感知多租户记账

#### 准确复述

当 matched token/workload 下 route coalition 仍造成显著租户成本差异时，post-router
route-cost debt 相对 token-VTC 改善 worst-tenant slowdown/SLO，同时不降低总 goodput。
决策变量是请求 admission/service order；最终目标是公平性、P99 和 SLO，而不是 route
overlap 预测精度。

#### Headroom、Oracle 与场景

`[本地]` matched-route effect 仅 2.35%–3.06%，held-out 简单模型 R²=0.9971–0.9986。
若 expert 部分占 E2E 40%，把 route 残差完全消除，平均 E2E 降幅粗上界仅
`0.40×0.0306≈1.2%`，speedup≈1.012×。公平指标可能高于平均上界，但必须先证明该残差
系统性集中在某一租户，而不是噪声。prefill 大 batch 可能有 expert-union 效应；decode、
低并发和单卡下更弱；8卡排队可能放大，也可能被 continuous batching 平滑。

- Oracle：已知每个请求对真实 GPU service time 的边际贡献，用 Shapley/leave-one-out
  只做离线重排；不改变执行。
- 最强 baseline：token-VTC + measured batch-size/sequence-length cost bucket；再加一个
  线性 route histogram 特征。
- Headroom：**偏低，已足以淘汰**。simple 已捕获可观测方差的绝大部分。

#### 最近邻、创新与因果链

最近邻：[VTC, OSDI'24](https://www.usenix.org/conference/osdi24/presentation/sheng)、
[Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin)、
[Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html)、
[JANUS](https://qzweng.github.io/assets/pdf/2025.arXiv-Janus-Zhang.pdf)。差异只是把 route cost
加入公平记账，属于**已有系统的小幅延伸**。因果链在“route 差异是主要成本差异”处已被
本地证据否定；复杂 coalition accounting 只会优化代理指标。

#### Baseline、开销、可验证性与论文性

决策在调度关键路径，需 route 完成后才能记账，可能延迟 admission；coalition 成本非加性，
易引入 CPU/GPU 同步。5090 可以真实判死效应大小；8卡只可能重新测量，不能挽救已小的
单卡物理残差。预计论文评分 **1/5**。

#### Red Team：五条拒稿理由与所需反证

1. **上界不足：** E2E oracle 约≤1.2%；需 8 卡真实 trace 证明 P99/fairness headroom≥10%。
2. **强 baseline 已足够：** 线性模型 R²>0.997；需证明它在 request-disjoint holdout 上产生
   ≥10% scheduling regret。
3. **novelty 弱：** 是 VTC 加 route feature；需新的非加性资源分配定理或机制。
4. **因果倒置：** 排队/batch 可能解释 slowdown；需 matched queue/batch/route 因果实验。
5. **泛化窄：** 只在特定 top-k/skew 生效；需跨 3 模型、均匀与自然 route 均成立。

**结论：E. 淘汰。** 不再做 GPU 实验。

---

### Idea I2：Expert-Set-Aware Speculative Verification

#### 准确复述

当 speculative tree 的 expert union 成本显著时，联合 acceptance 与 marginal expert cost
的 draft/verification 决策相对 acceptance-only baseline 提高 TPOT/throughput，同时维持
质量。决策变量是 draft path、depth 和 expert budget；目标是端到端 decode 性能。

#### Headroom、Oracle 与场景

offload/大模型 memory-bound 时 union 可成为主要成本，oracle 可能>15%；全驻留、低 top-k
或 compute-bound 时趋近 0。prefill不适用；decode高 speculation width 时最大。Oracle
知道未来 acceptance 与全部 expert routes，枚举 verification tree；可离线 replay。

#### 最近邻、创新与结论

[EcoSpec](https://arxiv.org/abs/2607.12696) 已把 acceptance 与 marginal expert activation
cost 联合；[MoE-Spec](https://arxiv.org/abs/2602.16052) 已做 verification-time expert
budget；[EVICT](https://arxiv.org/abs/2605.00342) 和
[MoESD](https://papers.nips.cc/paper_files/paper/2025/hash/b637af7745d3ad4cb0b9cdaa056ab41e-Abstract-Conference.html)
覆盖 MoE speculative decoding。属于**已被覆盖**，不是“仍有 headroom 就仍有 novelty”。
最强 baseline 就是 EcoSpec/MoE-Spec，而不是 EAGLE-only。

开销包括 expert predictor、tree search、buffer 和可能的质量损失。5090可复现小模型算法，
但不能证明大模型 offload union 成本；8卡可验证性能，却不能创造差异。评分 **1/5**。

#### Red Team

1. 核心机制与 EcoSpec 相同；反证必须指出不同信息集和动作。
2. MoE-Spec 已有 expert budget；需证明 exact-quality 且机制不同。
3. 收益可能来自较浅 tree；需 matched accepted tokens/quality。
4. 5090模型过小、全驻留；需真实大模型 memory traffic。
5. learned predictor 可能比节省更贵；需端到端含 predictor 与 buffer。

**结论：E. 淘汰。** 仅可把相关实现作为其他实验 baseline。

---

### Idea I3：MoE-aware Chunked/Layered Prefill

#### 准确复述

当 chunked prefill 导致 MoE expert weights 反复装载时，layer/expert reuse-aware prefill
调度相对 fixed chunk 和 Sarathi-style chunking 降低 TTFT/JCT，同时不恶化 decode TPOT。
决策变量是 chunk/layer执行顺序；目标是 TTFT、TPOT、goodput与weight traffic。

#### Headroom、Oracle 与场景

CPU/NVMe offload、小 chunk 时 weight reload 可占>50%，删除上限>2×；全驻留 8×A100、
大 chunk 或 compute-bound 时接近0。Oracle令每个活跃expert只加载一次；trace replay可估。
最强简单 baseline是增大chunk到SLO允许上限，并把同层chunks合并。

#### 最近邻、创新与结论

[From Tokens to Layers](https://openreview.net/pdf?id=yyDbI3HXco) 已明确提出 layer-wise
prefill以减少权重读取；[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
处理 chunked prefill/decode干扰；[DistServe](https://www.usenix.org/conference/osdi24/technical-sessions)
分离prefill/decode；[MoE-Lightning](https://pschafhalter.com/papers/2025-asplos-moe-lightning.pdf)
处理MoE offload/pipelining。属于**被覆盖**。

#### Red Team

1. 主观察和layered mechanism已有论文；无法靠MoE标签构成差异。
2. 最佳large-chunk可能拿到大部分收益；需在相同SLO下显著超过。
3. 全驻留A100上weight reload消失；需说明正式资源的真实regime。
4. 改执行顺序可能饿死decode；需P99 TPOT而非prefill throughput。
5. 只在极端offload成立；需多模型、自然内存压力。

**结论：E. 淘汰。** 可作为I11的调度baseline，不再独立立项。

---

### Idea I4：KV/Expert HBM Exchange

#### 准确复述

当KV与expert residency争夺HBM且二者边际收益随负载变化时，联合预算控制相对固定split
提升SLO-goodput，约束迁移和质量不变。决策变量是KV/expert容量及paging；目标是goodput、
P99和HBM利用率。

#### Headroom、Oracle 与场景

大模型offload、长上下文高并发可能有15%–60%可删I/O；模型全驻留或低并发时≈0。
Oracle按未来route和KV需求枚举每epoch最优split，并令迁移零开销。最强baseline是
KV-first + static reserve + 双阈值/hysteresis。

#### 最近邻、创新与结论

[FluxMoE](https://arxiv.org/abs/2604.02715) 已以expert paging释放KV空间并在vLLM实现；
[HOBBIT](https://arxiv.org/abs/2411.01433) 管理混合精度expert cache/offload；
[MoE-Lightning](https://pschafhalter.com/papers/2025-asplos-moe-lightning.pdf) 做paged weights；
[KVCache Cache in the Wild](https://www.usenix.org/conference/atc25/presentation/wang-jiahao)
提供强KV policy；[PROBE](https://arxiv.org/abs/2602.00509)覆盖复制/预取/放置。
通用表述属于**基本被覆盖**。

#### Red Team

1. FluxMoE直接解决同一资源冲突；需证明不同决策变量而非自适应阈值。
2. fixed split可能捕获>80%；需真实trace oracle。
3. 收益可能只来自人为HBM cap；需自然32GB/8×A100配置。
4. migration与cache pollution未计会产生负优化；需真实I/O干扰。
5. 8卡全驻留时headroom归零；需超出总HBM的代表模型。

**结论：E. 淘汰。** request-lifecycle特例移入I11，不保留generic版本。

---

### Idea I5：Fused Quantize-at-Dispatch

#### 准确复述

当activation quantize/layout conversion在dispatch关键路径占显著比例时，将其融合进producer
相对独立codec kernel降低TPOT/P99，同时保持质量和collective规整性。决策变量是kernel
fusion/layout/dtype；最终目标是端到端性能，不是codec吞吐。

#### Headroom、Oracle 与场景

`[本地]`完整path上界约≤20%，折到E2E常≤10%；小decode消息时launch税高，高并发/大rows
更易摊薄。单卡只测producer→codec→consumer，8卡才测collective。Oracle删除quant/layout
时间；若目标占E2E 5%/10%，完全删除仅1.053×/1.111×。

#### 最近邻、创新与结论

[FlashMoE](https://arxiv.org/abs/2506.04667)、
[SwiftEP, NSDI'26](https://www.usenix.org/conference/nsdi26/presentation/li-xingyi)、
[Comet](https://arxiv.org/abs/2502.19811)、[MixServe](https://arxiv.org/abs/2601.08800)
均推进fusion/overlap/buffer路径。纯fusion是**工程实现**；只有作为I12固定双lane的必要
backend才有意义。

#### Red Team

1. 新意是kernel implementation detail；需新的representation/collective接口。
2. SwiftEP/FlashMoE可能已消除buffer tax；需同栈强baseline。
3. Amdahl上界不足；需真实combine/dispatch关键路径≥10%。
4. fusion会破坏GEMM batching或增加寄存器压力；需占用率和P99。
5. A100无原生FP8；需INT8真实快路径，不能只算逻辑字节。

**结论：D. 降级为子机制。** 仅在I12通过物理上界门后实现。

---

### Idea I6：Route-Intensity Clocking

#### 准确复述

当相同phase/batch下route histogram仍稳定改变算存强度时，route-aware power/clock policy
相对phase+batch policy降低J/token，约束P99/SLO。决策变量是clock/power cap；最终目标是
energy/token和goodput。

#### Headroom、Oracle 与场景

先验估计route residual energy 0%–10%，但未测。prefill大GEMM偏compute-bound；decode
偏memory/launch-bound。低并发有降频空间，高负载可能直接伤throughput。Oracle穷举每个
真实shape的可用clock并知道未来SLO；最强baseline是phase×batch bucket的离线最优clock。

#### 最近邻、创新与结论

[PALS](https://arxiv.org/abs/2605.21427) 联合power cap与batch；
[Festina](https://arxiv.org/abs/2606.30391) 联合placement、SM、operating point与SLO；
[GreenLLM](https://arxiv.org/abs/2508.16449)与
[The Illusion of Power Capping in LLM Decode](https://arxiv.org/abs/2605.11999)构成能耗基线。
创新仅为**新特征的增量消融**。

#### Red Team

1. PALS/Festina已覆盖主机制；需证明route是不可替代状态变量。
2. phase+batch可能捕获>80%；需matched shape不同route实验。
3. clock切换、温度和achieved clock混淆；需随机顺序控温重复。
4. 省board power却增加J/token；需完整能量积分和SLO-goodput。
5. 5090结果不迁移A100；需分别校准，无法声称通用控制器。

**结论：D. 降级为子机制。** 一天feature-value profiling可以做，不能作为主线。

---

### Idea I7：DriftGuard-MoE——expert quantization sensitivity drift

#### 准确复述

当expert低比特敏感度随请求分布发生可预测漂移时，稀疏audit与低频precision更新相对固定
offline profile、uniform quantization和EMA，在同一质量约束下降低J/token或提高goodput。
决策变量是expert precision/profile刷新；最终目标是quality-constrained系统收益。

#### Headroom、Oracle 与场景

质量oracle可以很大，但系统headroom必须先存在：若INT8/BF16两条真实执行路径等速，
正确分配precision也不会改善端到端。仓库已有负先验：risk persistence弱，quality proxy
不免费，动态FP8完整路径慢于BF16。A100无原生FP8使正式路径更受限。Oracle知道未来每个
expert/token的质量损失，选择最低成本precision；最强baseline是offline per-expert profile
+ uniform INT8/BF16 + 定期EMA更新。

若低精度expert执行占E2E的p=40%，真实快1.25×，最大E2E改善仅
`1-[(1-p)+p/1.25]=8%`；若executor没有快区则0。prefill大rows更可能有INT8快区，decode
小rows可能被launch/dequant主导。

#### 最近邻、创新与因果链

[MiLo](https://proceedings.mlsys.org/paper_files/paper/2025/hash/9032e5c9ec394ce768a2fa9bdc56af6c-Abstract-Conference.html)、
[MoEQuant](https://arxiv.org/abs/2505.03804)、[DynaExq](https://arxiv.org/abs/2511.15015)、
[HOBBIT](https://arxiv.org/abs/2411.01433)、[MoBiE](https://arxiv.org/abs/2604.06798)
覆盖expert-aware importance、动态精度、offload与量化。剩余差异是**sensitivity drift的
measurement**，不是已成立的新系统机制。因果链在“漂移稳定”“proxy便宜”“precision有
真实快区”三处同时缺证据。

#### Red Team

1. 动态expert quantization已拥挤；需证明漂移是新的一阶变量。
2. 没有真实低精度快区；需端到端INT8/BF16 operating region。
3. quality proxy可能比收益更贵；需不运行第二份模型的低成本audit。
4. EMA/offline profile可能捕获>80%；需request-disjoint drift oracle。
5. 分布漂移实验易人为构造；需自然多domain/time trace且跨模型复现。

**结论：D. 降级为子机制。** 可作为I12的quality稳定性消融；没有真实executor快区前不立项。

---

### Idea I8：ScaleTail-MoE——单卡trace到8卡fork-join P99模型

#### 准确复述

当单卡route与expert service distribution包含足够多的多卡tail信息时，分解式fork-join模型
相对roofline/mean-load模型，以≤10% P99误差和≤5%配置regret预测8卡EP，并减少≥80%
profiling GPU-hours。决策变量是模型参数和配置选择；目标是预测准确度/测量成本，不直接
改善serving。

#### Headroom、Oracle 与场景

不存在传统Amdahl speedup。oracle是完整8卡配置sweep，regret=0；价值上限是省掉的
GPU-hours和错误配置损失。单卡可真实测route、rows与kernel distribution，但不能产生
NCCL contention、rank synchronization、overlap、queue coupling。prefill大batch较接近
roofline，简单模型可能已够；decode小batch/fork-join tail更难，但也最难从单卡识别。

#### 最近邻、创新与因果链

[Vidur](https://proceedings.mlsys.org/paper_files/paper/2024/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html)、
[GenZ](https://arxiv.org/abs/2406.01698)、[EPS-MoE](https://arxiv.org/abs/2410.12247)、
[Aurora](https://arxiv.org/abs/2410.17043)、[MoE-Inference-Bench](https://arxiv.org/abs/2508.17467)
构成性能模型、配置选择和benchmark邻域。差异是**小规模到fork-join tail的迁移模型**，
有一定新意，但技术贡献可能只是更细的simulator。

- 最强baseline：analytical roofline + NCCL microbench + 3–5个8卡校准点 + `max rows/rank`。
- 机制成本：离线低；但每模型都需重新校准会消解节省。
- 5090：只能产生输入，不能验证核心claim；8×A100是ground truth且必要。一个topology不足
  以支撑跨平台泛化。
- 论文评分：当前 **2/5**；更适合作为所有候选的测量基础。

#### Red Team

1. 单卡无法验证核心迁移准确度；必须先拿8卡ground truth。
2. 简单roofline+少量校准可能已≤10%误差；需显著降低config regret。
3. 只有一个A100拓扑，泛化性弱；需至少PCIe/SXM或公开外部数据。
4. 预测误差低不等于决策有用；需best-config regret和GPU-hours。
5. 是artifact而非系统机制；需新的可解释tail law并纠正实际配置决策。

**结论：D. 降级为子机制。** 保留为I9/I11/I12统一的测量基础，不作为当前首选论文。

---

### Idea I9：RouteCloak——SLO-bounded MoE execution-footprint privacy

#### 准确复述

当共享GPU/通信域中的低权限观察者能从route执行足迹恢复token或敏感属性时，静态bucket/
dummy routing相对无防御、自然batching和full padding，在保持exact output下移除≥80%攻击
优势且TPOT/P99开销≤10%。决策变量是padding、dummy expert和batch mixing；目标是攻击
优势/互信息与SLO，不是平均延迟改善。

#### Headroom、Oracle 与场景

隐私收益不能用Amdahl伪装成speedup。无可达observer时privacy headroom=0；decode batch1
最易对齐、风险最大；prefill/high batching天然混淆但padding易摊薄。full expert execution
可作零泄漏oracle，但top-8/64时计算最坏约8×，不可部署。8卡all-to-all可能增加信号，
也可能因NCCL聚合和隔离而消失。

- Oracle：固定每层执行/通信到全expert或完全定长，对观察者互信息归零。
- 最强baseline：continuous batching + 固定窗口混洗 + static traffic buckets；以及MIG/权限
  隔离作为配置消失baseline。
- Headroom：**无法判断**。无防御攻击需比随机高≥20个百分点或敏感属性AUC≥0.75；否则
  直接淘汰。

#### 最近邻、创新与因果链

[MoEcho](https://arxiv.org/abs/2508.15036)展示CPU/GPU微架构side channels；
[Expert Selections Reveal Text](https://arxiv.org/abs/2602.04105)研究selection到文本恢复；
[CryptoMoE](https://arxiv.org/abs/2511.01197)与[SecMoE](https://arxiv.org/abs/2601.06790)
保护MoE选择/计算。差异是**共享推理系统中的exact-output、SLO-bounded traffic defense**，
创新性较强，但威胁模型是最大风险。

因果链：route与文本相关（已有）→低权限tenant可观测细粒度proxy（缺）→攻击跨模板/
模型迁移（缺）→padding减少信息（缺）→≤10%代价（缺）。决策可低频，但dummy work、
额外A2A和更差packing会恶化P99/energy。

#### 5090、8卡与论文性

5090只有在能搭真实低权限observer时才可验证；公开route trace的信息论重放不能证明攻击。
8卡必须构造真实共租和collective observer；若攻击依赖跨节点traffic，资源不足。当前论文
评分 **3/5，高方差**。

#### Red Team

1. attacker可能需要管理员权限；需最小权限、可复现实验。
2. route相关不等于可观测；需真实counter/TLB/timing信号。
3. padding是教科书式防御；需新Pareto或MoE特有机制。
4. natural batching/MIG可能更便宜；需强配置baseline。
5. ≤10%开销可能不可能；需同时报告P99、throughput、J/token和攻击优势。

**结论：B. 先做 Profiling。** 只做攻击面存在性；看不到真实route proxy就立即停止。

---

### Idea I10：Critical-Sibling Receiver Rescue

#### 准确复述

当top-k fork-join的最后未完成sibling可由receiver/join信息提前识别时，只复制或改道该
branch相对least-queue/replica baseline降低P99，额外work≤3%。决策变量是reroute/replicate
与执行顺序；目标是P99/SLO和goodput。

#### Headroom、Oracle 与场景

若join wait占P99的20%，完全消除上限1.25×；低负载或无straggler时0。单卡没有真实rank/
receiver，只能注入；8卡才可归因。oracle知道每个branch未来completion time，只对会改变
join完成时刻者采取动作。最强baseline是least-queue exact replica、EDF/join-remaining-work
和moving-P95 hedge。

#### 最近邻、创新与因果链

[JANUS](https://qzweng.github.io/assets/pdf/2025.arXiv-Janus-Zhang.pdf)、
[PROBE](https://arxiv.org/abs/2602.00509)、[AMoE](https://arxiv.org/abs/2505.08944)、
[Capacity-Aware Inference](https://arxiv.org/abs/2503.05066)覆盖异步队列、复制、路由和
容量动作；仓库旧FJRC已研究keyed sibling deficit。新版本属于**旧对象换名**。

#### Red Team

1. 与FJRC scientific object重复；需新信息集而非新名字。
2. 单卡注入不能证明自然receiver straggler；需8卡census先行。
3. least-queue可能捕获>80%；需matched q-map只改变join phase。
4. duplicate work在高负载制造拥塞正反馈；需全局P99/goodput。
5. action-flip先验弱；需receiver信息使首动作翻转≥10%。

**结论：E. 淘汰。** 未来只有真实8卡trace满足三个重开门槛才复议：join wait≥15%、
aggregate queue解释不了、join信息action flip≥10%。

---

### Idea I11：ResumeSet——暂停请求的KV–expert双状态保留

#### 准确复述

当tool/human pause导致恢复请求的KV仍有价值、但expert工作集被驱逐并造成显著cold-start时，
联合保留相对KV-only TTL、LRU/LFU和last-window expert set降低resume P95/JCT≥10%，同时
额外HBM≤5%、普通请求P99恶化≤3%。决策变量是KV/expert admission、TTL与quota；目标是
resume SLO和全局goodput。

#### Headroom、Oracle 与场景

模型全驻留时expert miss=0，headroom严格为0。单卡offload、大模型或HBM partition下，若
resume前N token中expert fetch stall占15%/30%/50%，完全删除的speedup为1.176×/1.429×/
2×。prefill重算时KV主导；decode batch1的cold miss最明显；高并发可隐藏搬运但增加污染。

- Oracle：知道resume时间和未来N token routes，在固定总HBM下最优保留KV/expert；分别
  删除KV restore与expert fetch，计算独立上界。
- 最强baseline：Continuum/InferCept式KV TTL + last-W union top-N experts同TTL；另比LRU、
  LFU、global hot set。
- Headroom：**有条件**。KV-only后expert独立E2E oracle必须≥15%，last-W捕获<70%。

#### 最近邻、创新与因果链

[InferCept](https://arxiv.org/abs/2402.01869)处理API pause和KV恢复；
[Continuum](https://arxiv.org/abs/2511.02230)处理tool duration与KV TTL；
[KVCache Cache in the Wild](https://www.usenix.org/conference/atc25/presentation/wang-jiahao)
给出真实KV retention规律；[HOBBIT](https://arxiv.org/abs/2411.01433)和
[FluxMoE](https://arxiv.org/abs/2604.02715)管理expert residency；
[ELDR](https://arxiv.org/abs/2607.00466)的KV-indexed expert signature是最危险邻近工作。
因此novelty从“新状态抽象”下调为**已有机制在pause/resume workload上的增量组合**。

因果链缺三环：pause后expert是否真被驱逐、过去route是否预测恢复route、保留是否挤压更
有价值的active state。决策不在token关键路径，但HBM opportunity cost和prefetch bandwidth
可能让全局goodput负优化。

#### 5090、8卡与论文性

5090可用真实offload/pause验证必要条件，不能只人为设极小cache。8卡只有模型/并发使expert
不能全驻留时才增加证据；否则它会正确地给出零上界。当前评分 **2/5**。

#### Red Team

1. InferCept + ELDR/HOBBIT的直接拼接；需resume-specific新观察。
2. 全驻留时问题消失；需两个代表模型的自然offload regime。
3. last-W union可能捕获>80%；需future oracle和强简单baseline。
4. 旧cache实验已显示working-set饱和；需证明pause改变这一事实。
5. 只优化resume请求会伤普通请求；需全局SLO-goodput与memory opportunity cost。

**结论：C. 有条件保留。** 条件是两个模型、两类真实/公开pause trace上，expert独立oracle
≥15%、last-W捕获<70%、全局goodput下降≤2%。

---

### Idea I12：RankLane-Combine——gate-rank固定双lane返回编码

#### 准确复述

当EP combine处于关键路径且低gate-rank contribution对误差稳定更不敏感时，固定
head-BF16/tail-INT8单消息双lane相对BF16和uniform INT8降低TPOT/P99≥5%，保持质量阈值，
且codec/metadata不抵消收益。决策变量是combine payload dtype/layout和per-layer cutoff；
目标是E2E TPOT/P99/goodput/quality。

#### Headroom、Amdahl 与场景

单GPU真实EP headroom为0，只能测必要条件。8GPU上若残余combine critical path
`p=5%/15%/30%`，完全删除speedup为1.053×/1.176×/1.429×；若双lane只把该段快1.5×，
E2E延迟仅降1.7%/5.3%/11.1%。prefill大payload容易摊codec但通信常被overlap；decode
小payload启动延迟高，压字节未必有效；高并发中等rows最可能有正区。NVSwitch会压低
headroom，PCIe/跨节点会增大但后者未必可用。

`[本地]` gate-rank质量不对称在OLMoE/LLM-jp matched-byte测试中存在；但未融合
FP8→INT4 codec在rows 128/512、200–400Gbps等效链路上0/8可行，codec tax约58–61μs，
break-even约4.5–68.6Gbps。该结果只证明朴素codec不行，不证明fused INT8双lane可行。

- Oracle 1：从真实8卡trace删除combine critical-path时间。
- Oracle 2：零codec、按理想压缩比缩放payload。
- Oracle 3：离线穷举per-layer cutoff并测质量。
- 最强baseline：单消息uniform INT8 + producer/consumer fusion；其次static global top-h。
- Headroom：**无法判断，必须先profiling**。零combine E2E必须≥5%，最好≥10%。

#### 最近邻、创新与因果链

[Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin)、
[ScMoE](https://arxiv.org/abs/2404.05019)、[Comet](https://arxiv.org/abs/2502.19811)、
[FlashMoE](https://arxiv.org/abs/2506.04667)、
[SwiftEP](https://www.usenix.org/conference/nsdi26/presentation/li-xingyi)、
[MixServe](https://arxiv.org/abs/2601.08800)构成通信压缩、fusion和强EP baseline。
差异是**combine-only的gate-rank质量结构 + 固定规整双lane**；有潜力但边界不清。

因果链：rank质量不对称（有本地证据）→combine是关键路径（缺）→双lane减少真实wire time
（缺）→不破坏collective/packing（缺）→E2E收益（缺）。开销为quant/dequant、scale、
额外访存和buffer；若拆成两个collective几乎必然伤小消息P99。A100不支持原生FP8，正式
方案必须是INT8/BF16或证明的软件路径。

#### 5090、8卡与论文性

5090可真实验证质量、codec tax和fused local pipeline，不能验证NCCL wire time。8卡必要；
先做无代码修改的trace/Nsight删除式oracle，再决定是否实现。当前评分 **3/5（边缘）**；
若真实p≥15%、强baseline后净收益≥8%且跨3模型，可升到4。

#### Red Team

1. “通信量化+static cutoff”可能只是layout choice；需跨模型Pareto优于magnitude/uniform。
2. SwiftEP/overlap可能已把combine移出关键路径；需最新backend profiling。
3. uniform INT8可能捕获≥80%；需captured-headroom结果。
4. codec/metadata可能吃掉全部收益；需producer-consumer fused实测。
5. 单卡限速proxy不代表NVLink/NCCL；需8卡真实wire/P99，不能外推。

**结论：B. 先做 Profiling。** 当前首选，但只获准执行必要条件和oracle实验。

## 3. Oracle / Headroom 汇总

| Idea | Oracle 定义 | 最大潜在收益 | 预计可实现净收益 | Headroom结论 | 最大未知量 |
|---|---|---:|---:|---|---|
| I1 RouteShare | 完美边际成本调度 | 平均E2E约≤1.2%；P99未知 | 0%–1% | 偏低 | route残差是否集中于坏租户 |
| I2 Expert-set SD | 知道acceptance与future route | offload regime可>15% | 可能10%–30%，但已覆盖 | 充足但无novelty | 与EcoSpec差异 |
| I3 Layered prefill | 每expert weight只读一次 | offload小chunk可20%–50%+ | 10%–30%，但已覆盖 | 充足但无novelty | 正式配置是否offload |
| I4 KV/Expert HBM | future-aware零迁移split | offload可15%–60%；resident=0 | 10%–30%，但已覆盖 | regime-dependent | FluxMoE后的独立动作 |
| I5 Fused quantize | 删除codec/layout | E2E多为≤10% | 0%–5% | 中低 | fused kernel真实tax |
| I6 Route clocking | per-shape最优clock | route residual估计0%–10% energy | 0%–5% J/token | 无法判断，先验偏低 | phase+batch后的独立增量 |
| I7 DriftGuard | future quality-aware最低precision | 无快路径时0；示例快1.25×时E2E≤8% | 0%–5% | 无法判断，三重前置条件 | executor operating region |
| I8 ScaleTail | 完整8卡sweep、regret=0 | 可省最多约80% profiling cost；无直接latency收益 | 未知 | 非Amdahl问题 | 简单模型config regret |
| I9 RouteCloak | full padding，攻击优势归零 | privacy优势可全部移除；性能收益为负 | 目标为≥80%泄漏移除、≤10%代价 | 无法判断 | 真实低权限observer |
| I10 Critical sibling | 知道future completion、零成本rescue | straggler占20%时P99≤20%；低负载=0 | 0%–5%先验 | 偏低 | 自然join wait比例 |
| I11 ResumeSet | future resume+route最优保留 | fetch占15%/30%/50%时1.176×/1.429×/2×；resident=0 | 0%–15% | 有条件 | KV-only后的expert独立stall |
| I12 RankLane | 删除combine/零codec | combine占5%/15%/30%时1.053×/1.176×/1.429× | 0%–8% | 必须profiling | residual combine critical path |

其中I2/I3/I4即使oracle很大也应淘汰：**headroom充足不等于可发表空白存在**。I9的
“收益”是隐私而非加速；不能与latency百分比直接排序。

## 4. 最强简单 Baseline

统一使用：

\[
CH=\frac{M_{simple}-M_{current}}{M_{oracle}-M_{current}}
\]

对latency/攻击优势等越低越好的指标，分子分母同时按“从current到改善方向”取差，避免
符号错误。`CH≥0.8`时复杂机制原则上停止。

| Idea | 最强简单baseline | 配置消失测试 | 判死门槛 |
|---|---|---|---|
| I1 | token-VTC + batch/length bucket + linear route feature | 调整batch/EP | 已判死：R²>0.997且物理残差小 |
| I2 | EcoSpec/MoE-Spec公开方案 | 降低spec width | prior art即baseline，淘汰 |
| I3 | 最大SLO-safe chunk + layer-major execution | 全驻留/增大chunk | baseline≥80% oracle或主机制重复 |
| I4 | KV-first static reserve + 双阈值 | 模型全驻留 | FluxMoE或static捕获≥80% |
| I5 | framework fusion + uniform INT8 | 启用SwiftEP/overlap | codec吃掉≥30%毛收益 |
| I6 | phase×batch static clock table | 默认power cap/批量调整 | route residual<5%或CH≥80% |
| I7 | offline per-expert profile + EMA | uniform INT8/BF16 | 无真实快区或CH≥80% |
| I8 | roofline+NCCL LUT+3–5校准点+max rows | 固定最佳EP/TP | baseline regret≤5% |
| I9 | batching+window mixing+static buckets；MIG隔离 | 禁counter/固定collective | 无攻击或static CH≥80% |
| I10 | least-queue exact replica + moving-P95 hedge | 增加static replica | action flip<10%或CH≥80% |
| I11 | KV TTL + last-W union top-N expert | 全驻留/加HBM | expert独立oracle<15%或CH≥70% |
| I12 | fused single-message uniform INT8 + static top-h | SwiftEP/更合适EP/TP | zero-combine<5%或uniform CH≥80% |

## 5. Red-Team 拒稿意见横向归纳

每个Idea上文已有至少五条具体拒稿理由。最致命的跨Idea问题是：

1. **用局部字节/命中率替代端到端关键路径：** I5、I11、I12均有此风险。推翻证据是
   删除式oracle与wall-clock decomposition，而非microbenchmark speedup。
2. **把已有论文换workload或多一个knob：** I2、I3、I4已直接命中；I6、I11接近该边界。
3. **单卡能跑但不能验证核心因果：** I8、I9、I10、I12的多卡claim都不能由5090证明。
4. **弱baseline制造收益：** I1必须面对线性模型，I11必须面对last-W，I12必须面对
   fused uniform INT8，I9必须面对隔离和自然batching。
5. **资源regime不匹配：** I3/I4/I11在全驻留8×A100上上界归零；I9若需跨节点observer、
   I12若只在InfiniBand慢链路成立，现有资源不足。
6. **机制税与毛收益同量级：** codec、migration、dummy work、audit和duplicate work都可能
   把5%–10%局部空间完全吃掉。
7. **配置不合理而非系统问题：** 增大chunk、换EP/TP、启用最新collective、增加HBM或
   static replica可能直接消除问题。
8. **论文故事过薄：** I5/I6/I7/I8容易成为“一条观察+一个heuristic/模型”；必须作为
   更大系统的组件，而非独立包装。

## 6. 淘汰名单及原因

### 直接淘汰

- **I1 RouteShare：** 本地oracle和简单模型已同时判死，继续投入不会产生新知识。
- **I2 Expert-set SD：** EcoSpec/MoE-Spec/EVICT已正面覆盖。
- **I3 Layered prefill：** 同一观察、同一执行顺序已公开。
- **I4 Generic KV/Expert HBM：** FluxMoE直接覆盖；resume特例已拆到I11。
- **I10 Critical sibling：** 与旧FJRC重复，且单卡不能验证receiver因果。

### 降级为子机制，不作为独立论文

- **I5 Fused quantize：** 只可能是I12的backend。
- **I6 Route clocking：** 只做PALS/Festina的route-feature消融。
- **I7 DriftGuard：** 只做I12或量化系统的quality稳定性审计。
- **I8 ScaleTail：** 作为所有8卡候选的统一测量/配置选择工具。

## 7. 保留名单及前置条件

| Idea | 状态 | 不满足即停止的前置条件 |
|---|---|---|
| I12 RankLane | B Profiling | 8卡zero-combine E2E≥5%；uniform INT8 CH<80%；真实fused codec净正；跨2模型质量稳定 |
| I11 ResumeSet | C 条件保留 | KV-only后expert独立resume oracle≥15%；last-W CH<70%；全局goodput下降≤2% |
| I9 RouteCloak | B Profiling | 真实低权限攻击优势≥20个百分点或AUC≥0.75；static防御≤10% P99代价 |

这三项也不等于获准建设完整系统。当前授权仅是1–3天判死实验。

## 8. 横向评分与排名

1–5分；Risk越高越危险，其余越高越好。

| Idea | Problem | Headroom | Novelty | Mechanism | Baseline resistance | Local verify | Scale value | Measurement | Engineering | Generality | Paper | Risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I1 RouteShare | 3 | 1 | 2 | 2 | 1 | 5 | 3 | 5 | 4 | 2 | 1 | 5 |
| I2 Expert-set SD | 4 | 4 | 1 | 4 | 1 | 3 | 5 | 4 | 2 | 3 | 1 | 5 |
| I3 Layered prefill | 4 | 4 | 1 | 4 | 1 | 4 | 4 | 4 | 3 | 4 | 1 | 5 |
| I4 KV/Expert HBM | 5 | 4 | 1 | 4 | 2 | 4 | 5 | 4 | 2 | 3 | 2 | 4 |
| I5 Fused quantize | 4 | 2 | 2 | 4 | 2 | 5 | 5 | 4 | 2 | 3 | 2 | 4 |
| I6 Route clocking | 4 | 2 | 2 | 3 | 2 | 5 | 4 | 4 | 4 | 3 | 2 | 4 |
| I7 DriftGuard | 3 | 2 | 3 | 3 | 2 | 4 | 3 | 3 | 2 | 2 | 2 | 5 |
| I8 ScaleTail | 4 | 3 | 3 | 4 | 2 | 2 | 5 | 5 | 4 | 3 | 2 | 3 |
| I9 RouteCloak | 4 | 5 | 4 | 3 | 3 | 2 | 4 | 2 | 2 | 3 | 4 | 5 |
| I10 Critical sibling | 3 | 2 | 2 | 3 | 1 | 1 | 4 | 2 | 2 | 2 | 1 | 5 |
| I11 ResumeSet | 4 | 3 | 2 | 3 | 2 | 5 | 3 | 4 | 3 | 2 | 2 | 4 |
| I12 RankLane | 4 | 3 | 4 | 4 | 3 | 5 | 5 | 4 | 2 | 3 | 4 | 4 |

### 多维排名

- **综合可投入排名：** I12 > I11 > I9 > I8 > I7 > I5 > I6；其余不排序，均已淘汰。
- **headroom排名（不考虑novelty）：** I9隐私 > I11 > I12 > I8测量成本；已覆盖的I3/I4
  虽有大物理空间，不重新进入候选。
- **创新性排名：** I9 > I12 > I8 > I7 > I11。
- **最容易廉价判死：** I6 > I7 > I12必要条件 > I11 > I9。
- **最可能形成论文：** I12 > I9 > I11 > I8。
- **最可能浪费时间：** I10 > I4 > I2 > I3 > I7。

### 重复性检查

- I4与I11都谈KV/expert状态；只保留**pause/resume特例I11**。
- I5是I12实现路径，不是独立idea。
- I7与I12都使用quality-aware precision；I7只作为跨时间稳定性消融。
- I8不与某个瓶颈竞争，应作为I9/I11/I12共用的measurement substrate。
- I10与历史FJRC/receiver-awareness是同一scientific object，合并后仍淘汰。

### Top候选机会成本

| Idea | 1–3天能得到什么 | 1–2周才需要什么 | 最大工程风险 | 失败后沉淀 | 放弃的机会 |
|---|---|---|---|---|---|
| I12 | quality Pareto、codec break-even、projected oracle | 8卡profile与fused collective prototype | NCCL/DeepEP ABI、A100 INT8路径 | codec/quality负结果和benchmark | 暂缓I11 workload trace |
| I11 | resume stall decomposition、last-W vs oracle | 真实offload cache manager与多session scheduler | 代表性pause trace、cache污染 | agentic MoE cold-start characterization | 暂缓I9安全实验 |
| I9 | observer可达性与攻击优势 | 真实共租防御和8卡traffic experiment | 权限/threat model根本不存在 | MoE side-channel复现或负面边界 | 安全方向与传统infra路线分叉 |
| I8 | 简单模型是否已足够 | 多配置ground truth与迁移模型 | 只有一个topology | 所有后续实验共用工具 | 不直接产出系统收益 |

## 9. 最终 Top 3

### 首选：I12 RankLane-Combine——B. 先做 Profiling

为什么值得：有本地跨模型质量结构证据，必要条件可在5090廉价验证，8卡会增加真实EP
因果证据。为什么不是I8：I8不直接改善系统，且简单模型可能足够。为什么不是I11：I11
依赖offload与pause workload，问题可能在8卡全驻留下消失。

首实验：quality Pareto + fused codec break-even + zero-combine projected oracle。

失败阈值：uniform INT8捕获≥80%、codec吃掉≥30%毛收益、或预计zero-combine E2E<5%。

成功后：只做一次8卡无侵入profiling；`p≥10%`才写collective。

### 次选：I11 ResumeSet——C. 有条件保留

为什么值得：5090可以用真实offload直接测因果，决策不在token关键路径，失败也能得到agentic
MoE resume的measurement结论。为什么不是I4：generic问题已被FluxMoE覆盖。

首实验：KV-only、last-W、future oracle的resume前N token stall分解。

失败阈值：expert独立oracle<15%、last-W捕获≥70%、或只有人工cache cap成立。

成功后：实现event-driven TTL，并测active-request goodput污染。

### 高风险高收益：I9 RouteCloak——B. 先做 Profiling

为什么值得：新问题和潜在安全价值最大；为什么不是普通receiver方向：它优化的是可验证的
攻击优势，而不是小幅平均延迟。

首实验：限定低权限observer，测route proxy与token/属性攻击。

失败阈值：攻击优势<20个百分点/AUC<0.75，或只在管理员权限成立。

成功后：先测自然batching/static bucket，不先写自适应controller。

## 10. 首选 I12 的 1–3 天 Go/No-Go 预注册实验

### 10.1 目标与证据边界

只回答三个必要问题：

1. gate-rank是否比uniform/magnitude更好地分配质量误差；
2. fixed dual-lane的真实codec税是否低于逻辑传输节省；
3. 在合理链路与shape分布下，是否存在值得上8卡测量的正区。

5090**不能**证明真实EP/NCCL收益；通过后也只获得一次8卡profiling资格。

### 10.2 模型、框架与数据

- 模型：OLMoE-1B-7B主实验；仓库LLM-jp MoE独立复现。若时间允许加一个top-2模型。
- 框架：现有route/combine hooks + PyTorch CUDA/Triton microbenchmark；不先改vLLM/DeepEP。
- 数据：WikiText-2/C4共128–256文档；GSM8K与HumanEval各100样本作任务质量sanity。
- 显存：预计<20GB；5090 32GB足够。运行6–18小时，开发/复核合计1–3天。

### 10.3 Baseline、Oracle 与变量

- Baseline：BF16；single-message uniform INT8；static global top-h；per-layer top-h；
  activation-magnitude aware INT8。
- Oracle：零codec理想payload；combine完全删除的trace投影；每层cutoff质量穷举。
- 自变量：rows=1/4/16/64/128/512/2048/4096，真实hidden/top-k，保护rank数，等效链路
  1/5/10/25/50/100/200GB/s，prefill/decode route分布；900GB/s只作为A100 SXM聚合物理
  峰值参考，不能当作单rank collective有效带宽。
- 控制变量：相同token、route、scale规则、随机种子、warmup、GPU clock/temperature；至少30
  个独立trial，报告bootstrap CI。
- 必须实现完整 `producer→pack/quant→transfer proxy→unpack/dequant→weighted combine`；
  header、scale、alignment、temporary buffer、launch全部计入。不得用逻辑bytes代替时间。

### 10.4 指标

KL/NLL、任务accuracy、真实bytes、codec μs、P50/P95/P99、kernel数、temporary HBM、
break-even bandwidth、uniform baseline的Captured Headroom，以及按真实route加权的projected
E2E oracle。proxy结果只用于排除，不作为多卡性能结论。

### 10.5 Go / No-Go

全部满足才 **Go到8卡profiling**：

1. 两个模型上固定rank policy在相同bytes下均显著优于uniform INT8，且任务质量不反转；
2. uniform INT8捕获<80%质量—bytes oracle；
3. fused/local dual-lane codec开销≤毛传输节省的30%，并在保守的25GB/s等效有效带宽仍有
   正区；最终阈值由8卡原始BF16 combine实测带宽替换；
4. 真实route加权后，只有在combine占E2E≥5%时才可能净正，且候选shape覆盖≥30%的tokens。

任一出现即 **No-Go**：

- 质量差异只在一个模型/单一dataset；
- uniform INT8捕获≥80%；
- break-even只低于10GB/s，或低于后续实测8卡BF16 combine有效带宽；
- codec开销≥50%毛收益；
- 正区只在rows≥2048但真实decode几乎不出现；
- 必须拆成两个collective；
- projected E2E oracle<5%。

5090通过后，8卡第一步仍不是实现：先用原始BF16 backend测dispatch/combine独立critical
path、overlap后的残余、payload与P99。若zero-combine E2E<5%，最终淘汰I12。

## 11. 对当前研究路线的明确建议

1. **停止把“保留5个候选”当目标。** 当前只有3个获准做判死实验，0个获准完整实现。
2. **首做I12，但把预期设为判死。** 5090先验证quality/codec必要条件；不要先占用8卡或改
   serving engine。
3. **把I8降为公共工具。** 建立统一的route/rows/kernel/NCCL trace schema，为I9/I11/I12
   服务；只有复杂模型显著降低config regret时再讨论独立论文。
4. **I12失败后转I11，不要修补codec。** I11的核心门是独立expert cold-start；若oracle
   <15%，立即转I9。
5. **I9先验证threat model。** 看不到低权限真实信号就结束；不要用公开route trace自证攻击。
6. **不重开receiver-awareness。** 除非8卡真实trace同时满足join wait≥15%、aggregate queue
   无法解释、receiver/join信息action flip≥10%。
7. **正式8卡实验前确认硬件。** A100 40/80GB、PCIe/SXM、NVLink/NVSwitch会改变I11容量
   regime和I12通信上界；未确认前不得选最终模型或写收益预期。

当前最合理的论文路径不是“直接实现Top 3”，而是用三次廉价oracle连续淘汰：

`I12 quality/codec → 8卡combine Amdahl门 → I11 resume独立stall → I9 observer存在性`。

只有某一方向跨过自己的硬门，才进入“系统设计—实现—正式评测”。

## 12. 本轮关键文献核验

- [FluxMoE](https://arxiv.org/abs/2604.02715)：generic expert residency/KV pressure已非空白。
- [EcoSpec](https://arxiv.org/abs/2607.12696)：acceptance与expert activation cost联合选择已实现。
- [SwiftEP, NSDI'26](https://www.usenix.org/conference/nsdi26/presentation/li-xingyi)：buffer fusion、
  TMA与NVLink优化抬高I5/I12通信baseline。
- [MoEcho](https://arxiv.org/abs/2508.15036)：route相关微架构side channel存在，但具体权限与
  本机可复现性仍需验证。
- [InferCept](https://arxiv.org/abs/2402.01869)：API pause/KV swap与恢复已有完整系统基线。
- [KVCache Cache in the Wild, ATC'25](https://www.usenix.org/conference/atc25/presentation/wang-jiahao)：
  真实KV reuse/eviction规律使I11不能只与LRU弱基线比较。
