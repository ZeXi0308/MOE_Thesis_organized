# MoE Serving 毕设选题方案

> **2026-07-14 主线重置（优先于下文历史方案）**：新的 backend 语义审计与双模型 grouped-owner formal 表明，旧 **FP8-First Rank-Segmented FP4 Combine** 不应继续作为默认方法。当前候选升级为 **QuotaEP-H：Placement-Conditioned Fixed-Quota Mixed-Precision Hierarchical Combine**。它面向 hierarchical/high-throughput EP：在 source topology domain 内先对同一 `(token, destination-origin)` 的 expert outputs 做 BF16 partial reduction，再依据发送 partial 的 output-aware criticality，在每个 peer/tile 内选择固定数量 FP8 vectors，其余进入 MXFP4 lane。严格结论是：output-aware late binding 在 top-8/top-16、contiguous/round-robin 四个 setting 中均显著优于 rank/gate/random；但 grouped cancellation 没有稳定独立收益，固定 peer quota 相对 global budget 有 `4.2%～22.3%` KL tax，真实 kernel/TPOT/P99 尚未验证。完整报告见 `experiments/idea_a_mac/outputs/paper_validation/QuotaEP_H_查重与Grouped生死实验结论_2026-07-14.md`。下文 R-layout/PLTB/receiver-aware 内容保留为研究演进与 baseline；与本更新冲突时，以本更新为准。

## 方案 A+（当前候选）：QuotaEP-H

### 当前可防守的一句话

> 在 hierarchical EP combine 的跨域 grouped-partial wire unit 上，利用 expert output 已知后的 criticality 做 late binding，再用 per-peer/tile fixed quota 将动态重要性编译为规整 FP8/MXFP4 双 lane。

当前贡献层级必须收缩为：

| 层级 | 内容 | 证据状态 |
|---|---|---|
| 核心机制 | grouped wire unit 上的 output-aware FP8/MXFP4 allocation | 双模型、双 placement 质量侧通过 |
| 系统 contract | per-peer/tile fixed quota、bounded buffer、homogeneous lane | 只有 logical-wire/correctness；待 native kernel |
| backend co-design | fused local reduce + score/top-q + pack + hierarchical send | 尚未实现，是能否摸到 CCF B 的核心 |
| 条件扩展 | bounded quota borrowing | 用来解决 top-16 的 global-vs-peer quality tax，待验证 |
| 不并列 | rank、cancellation、Graceful、receiver-aware、QTree | baseline / negative result / stage-gated future work |

预注册主检验不再是 `fixed rank vs gate`，而是：

1. `QuotaEP-H vs uniform FP8`：同 backend、同 arrival trace，报告 incremental actual bytes、codec/selector overhead 与 TPOT/P99；
2. `output-aware vs rank/gate/random`：同 grouped vectors、同 precision quota、同 logical wire；
3. `peer/tile quota vs global quota`：量化规则性带来的 accuracy tax，并测试 bounded borrowing；
4. `grouped HT vs backend native HT`：必须证明方法不是把已有 local reduction 重做一遍。

停止条件：真实 backend 不存在 reduce-before-cross-domain 的 combine wire unit；或 fused selector/codec 开销吞掉相对 uniform FP8 的增量；或固定 quota 在跨模型上无法在质量约束内工作。任一成立，都应把 QuotaEP-H 降为 characterization/negative result，而不是继续增加离线策略。

## 方案 A（主线）：FP8-First Rank-Segmented FP4 Combine

> **2026-07-11 硬件格式审计后的当前边界**：uniform FP8 combine 已有公开实现，不能作为创新；旧 PLTB 的 35.5%/19.7% KL 收益依赖 per-row symmetric INT4 proxy，在 MXFP4/NVFP4 对齐模拟下缩小或反转。因此主线只回答一个更硬的问题：**uniform FP8 之后，能否利用 routing 已排序 rank，把低 rank combine outputs 静态编码为 block FP4，构造规整 FP8-head / FP4-tail two-lane kernel，并相对 uniform FP8 combine 获得真实 accuracy-wire-latency Pareto？** PLTB 降级为 format-dependent optional enhancement；receiver-aware、MILP、oracle、routing drift 均不与核心并列。

### 1. 一句话定位

MoE expert parallelism 在每个 MoE layer 都包含 dispatch 与 combine 通信。本文选择 combine 作为主优化位置，不是因为其他阶段理论上绝对不能做差分精度，而是因为 combine 中每个 `(token, expert)` 已产生独立 expert output，gate/rank 已知，且最终通过线性加权聚合，因而更适合做 importance-aware mixed precision。

combine 的数学形式为：

$$
y_t=\sum_{e\in S(t)}g_{t,e}o_{t,e},\qquad \sum_{e\in S(t)}g_{t,e}=1
$$

- $S(t)$：token $t$ 选择的 top-$k$ experts；
- $g_{t,e}$：selected expert 的 gate 权重；
- $o_{t,e}$：expert output；
- rank 1 对应最高 gate，rank $k$ 对应最低 gate。

本文的保守结构论点是：

| 阶段 | 传输对象 | 差分精度的系统特征 | 本文定位 |
|---|---|---|---|
| Dispatch | 同一个 token hidden state 的 $k$ 个副本 | 可以量化，但按 rank 使用不同精度会让相同输入产生多份编码，并把误差送入 expert 非线性计算 | 不作为主线，只做对照 |
| Combine | $k$ 个独立 expert outputs | 小 gate 会缩放 tail output 的量化误差；rank 已由 routing 给出 | 主优化位置 |

### 2. 经过前置实验后的核心 claim

| Claim | 当前状态 | 可以写到什么程度 |
|---|---|---|
| **C1：top-k 内部存在可利用的 rank-dependent FP4 sensitivity** | **质量侧已成立** | OLMoE top-8 中，tail-half/head-half 的 KL 比在 MXFP4 下为 8.75×、NVFP4 下为 2.56×；结论不再只依赖旧 INT4 proxy |
| **C2：uniform FP8 是强 baseline 且已有系统实现** | **已成立，不是本文创新** | 本地 uniform FP8 KL 为 0.00472；DeepGEMM/DeepEP 生态已出现 uniform FP8 combine，因此本文必须报告相对它的增量收益 |
| **C3：FP8 + selective block-FP4 能继续扩展质量—wire frontier** | **fake-quant 质量侧初步成立** | tail-half MXFP4/NVFP4 KL 为 0.00684/0.00571；含 scale 后理论 wire saving 为 61.52%/60.69%，相对 uniform FP8 的 49.61% 仍多约 11～12 pp |
| **C4：fixed rank segmentation 可逼近动态 selector 并提供规整布局** | **质量侧成立，系统侧待实现** | MXFP4 gate/oracle 仅比 fixed rank 低约 10%/19%；NVFP4 几乎持平。能否转化为 kernel 净收益仍需真实 GPU 对照 |
| **C4b：PLTB 能进一步改善 fixed rank** | **仅旧 INT4 proxy 成立，当前降级** | 旧 proxy 下 OLMoE/LLM-jp 降低 35.5%/19.7%；MXFP4 下旧 allocation 仅改善约 7.6%，NVFP4 反转约 1.7%，MXFP4-specific profile 也未稳定改善 |
| **C5：方法能改善真实 TBT/TPOT/P99** | **尚未证明** | 当前 Mac 结果只证明数值质量和理论 payload；没有真实 all-to-all、pack/unpack、量化 kernel 或端到端 serving latency |
| **C6：关键端口感知的有限 INT4 预算可能改善拥塞 Pareto** | **需拆分为两条 claim，见下** | 2026-07-11 混淆剥离实验（见第 6.2 节）证明：`receiver_only`（调度器可提前知道的 token-origin 热度）在负载不均衡场景下几乎独占收益，可保留为可部署 claim；`sender_only`（专家所在 GPU 负载）虽在均衡负载下贡献更大，但跨样本 Spearman 相关仅 0.39~0.50，无法离线 profile，只能作为 oracle upper bound |

> 本文区分 bit-only payload 与 scale-aware wire estimate：当前 hidden size 下，uniform FP8、tail-half MXFP4、tail-half NVFP4 分别约为 `49.61%`、`61.52%`、`60.69%` scale-aware saving。它们仍不包含 padding、alignment、collective header、pack/unpack 与 overlap，不能写成端到端加速。

### 3. 主方法：Rank-Segmented FP8/FP4 Two-Lane Combine

默认方法不再依赖 layer allocator。对 routing 后已按 gate 排序的 top-k expert outputs，选择静态 split `h`：rank `1...h` 进入 FP8 head lane，rank `h+1...k` 进入 block-FP4 tail lane。每个 token 的 lane 长度固定，不需要逐 token mask、prefix-sum 或 precision tag。论文通过 rank split sweep 决定 `h`，并把 gate threshold/tail-mass 作为动态强 baseline。

PLTB 只作为可选二级增强：必须在格式特定 calibration、跨模型/跨域 held-out test 中稳定优于 fixed split，才恢复为方法贡献。以下 3.1 的 PLTB 设计作为历史候选保留，不代表当前默认主线。

#### 3.1 离线 profile

对每个 MoE layer $l$，离线确定 tail-INT4 数量：

$$
m_l\in\{0,1,\ldots,k\}
$$

$m_l$ 表示该层最后 $m_l$ 个 ranks 使用 INT4，其余 ranks 使用 FP8。高敏感层可以令 $m_l=0$。运行时主表简化为：

```text
LUT[layer_id] -> num_int4_tail_ranks
```

如果跨模型实验表明同一层内不同 rank 不能被一个 tail count 充分描述，再扩展为：

```text
LUT[layer_id, rank] -> {FP8, INT4}
```

离线选择采用受控 probe：以 uniform FP8 为 reference，每次只在一个目标层将 base tail ranks 升级为 INT4，测量端到端 per-token KL。单层 profile 只用于 sensitivity 排序，不相加成端到端 KL 预测。随后在固定全局 INT4 rank-slot budget 下生成少量 `m_l` 候选，并在独立 test 上整体复测。

OLMoE top-8 当前冻结候选为：

```text
m_l = [2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 6, 6, 2]
```

它保护 layer 0/1/2/15 等敏感层，把更多 INT4 slots 转移到低敏感层；总 slots 与全层 fixed-tail4 完全相同。calibration 从 8 条扩大到 16 条后，该布局没有变化。

#### 3.2 运行时数据布局

默认方法把每个 token 的 top-$k$ outputs 按固定 split $h$ 分成两条定长 lane：

```text
head lane: rank 1 ... h -> FP8
tail lane: rank (h + 1) ... k -> packed block-FP4
```

因为 $h$ 固定，FP8/FP4 元素数量和 buffer offset 可提前计算，不需要逐 `(token, expert)` 携带 precision tag。目标系统实现包括：

- FP8/block-FP4 双 lane buffer；
- E2M1 packing 与 block/global scale metadata；
- quant + pack 融合；
- communication + unpack/dequant + gate-weighted reduction 融合；
- 与 DeepEP 或 NCCL EP combine primitive 集成。

这部分是论文能否达到系统论文级别的核心，不可以用 Python fake quant 或理论字节数替代。

#### 3.3 Rank 与 gate 的当前决策门

rank 不是理论上最精确的重要性信号，gate threshold 和 $g\|o\|$ contribution 可能更接近质量 oracle。选择 rank 的论文理由必须是模型—系统协同：

- routing 后已经得到有序 top-$k$；
- 每个 token 固定相同数量的 FP8/FP4 outputs；
- buffer 大小和 offset 规则；
- 更容易实现定长 collective kernel。

因此加入了 `gate-value threshold`、`cumulative gate mass` 与 contribution oracle 对照。硬件格式对齐后的 OLMoE held-out 结果如下：

| 策略 | bit-only saving vs BF16 | scale-aware saving | mean per-token KL | KL 95% CI |
|---|---:|---:|---:|---:|
| uniform FP8 | 50.00% | 49.61% | 0.00472 | [0.00377, 0.00589] |
| fixed rank-tail4 MXFP4 | 62.50% | 61.52% | 0.00684 | [0.00565, 0.00849] |
| gate threshold MXFP4 | 62.90% | 约 61.9% | 0.00615 | [0.00489, 0.00790] |
| contribution oracle MXFP4 | 62.50% | 61.52% | 0.00551 | [0.00480, 0.00641] |
| head4 MXFP4 control | 62.50% | 61.52% | 0.05987 | [0.04811, 0.07652] |
| fixed rank-tail4 NVFP4 | 62.50% | 60.69% | 0.00584 | [0.00433, 0.00780] |
| head4 NVFP4 control | 62.50% | 60.69% | 0.01499 | [0.01224, 0.01835] |

这意味着 fixed rank 不是质量 oracle，但在两种 FP4 下都保留明显 tail/head 分离；动态 gate/oracle 的质量增益远小于旧 INT4 proxy 下的差距。论文下一步实现两个系统候选：

1. **R-layout**：固定 rank split 的定长 FP8/FP4 two-lane；
2. **G-layout**：gate threshold/tail-mass 的动态选择，显式计入 mask、prefix-sum、metadata、variable-size buffer 与 kernel divergence。

最终按 end-to-end accuracy-latency Pareto 选主方法。若 R-layout 没有稳定系统优势，主方法切换为 gate/tail-mass aware；PLTB 只有在格式特定复测中恢复稳定收益才加入。

### 4. 各模块的职责边界

| 模块 | 在论文中的角色 | 当前处理 |
|---|---|---|
| FP8-first + fixed rank-segmented block-FP4 | **核心方法候选** | 主线 |
| Layer sensitivity / PLTB | format-dependent 二级增强；当前硬件格式证据不足 | 降级，不进入标题 |
| Two-lane combine kernel | **核心系统贡献** | 下一阶段必须实现 |
| Gate threshold / cumulative mass | 强 baseline 或增强版 | 必须补 |
| Contribution oracle | 质量上界，不进入 runtime | 保留为分析 |
| Routing drift | 解释多层误差如何传播 | 支撑分析，不是独立主线 |
| Drop / gate renormalization | aggressive ablation | 不进入默认策略 |
| Additive-KL MILP | 已失败的旧路线 | 退出主线，只保留 negative result |
| Receiver-aware | Two-Tier Congestion-Safe Budgeting 的在线分配层候选 | 仅保留 receiver-side scheduling signal；sender-side 降级为 oracle/future work（见第 6.2 节混淆剥离结果） |

### 5. 已放弃的旧 MILP 边界

旧方案用单层 profile 的 $\delta_{l,R,p}$ 线性预测多层端到端 KL。OLMoE 实测中，$\epsilon=0.1$ 的 MILP predicted KL 为 `0.10`，actual KL 为 `9.41`，低估约 `94×`；MILP 与 rank-only 的实际质量和 saving 也接近。

因此当前不能声称：

- 单层边际 KL 可跨层线性相加；
- MILP 的 accuracy constraint 能约束真实端到端损失；
- MILP 是本文的核心创新。

这一失败结果可用于说明多层低比特误差存在非线性累积，但新主线改用小规模、可端到端复测的 layer-wise tail budget。

### 6. Receiver-Aware 扩展的正确边界

当前 Mac 代码按 expert id 做 contiguous/mod 分组，它更接近 `expert-owner/endpoint group proxy`，还不是真实 combine receiver-port。

真实 expert-parallel combine 中：

```text
sender   = expert_owner_rank
receiver = token_origin_rank
```

只有在 trace 中同时记录 `token_origin_rank`、`expert_owner_rank`、`layer`、`rank` 和 bytes，构建 $Traffic_l[sender,receiver,rank]$，才能讨论 hot receiver、queueing 和 P99。receiver-aware 只有满足以下条件才升级为主贡献：

1. 在相同总 payload 和质量约束下，真实 hot receiver 的完成时间低于 random/cold；
2. pack/unpack、collective 和 overlap 全部计入；
3. 在多种 workload / placement 下稳定；
4. 相对 rank-only 带来可重复的额外 P99 收益。

否则它只作为流量代理实验或 future work，不进入标题。

#### 6.1 多 MoE/多请求拥塞的新增解析回放

2026-07-11 已补充一个使用正确 combine 语义的解析型回放：

```text
combine sender   = expert_owner_rank
combine receiver = token_origin_rank
```

输入来自 OLMoE held-out selected-expert trace；模拟 EP=8、每节点 4 GPU、16 个 MoE layers、1/2/4/8/16 个并发请求，以及 balanced/hotspot origin 分布。在 16 并发下：

| origin | 全部 safe tail INT4 | 随机使用半数 safe-tail 预算 | critical-port 使用半数 safe-tail 预算 |
|---|---:|---:|---:|
| balanced：bottleneck proxy saving vs uniform FP8 | 23.02% | 11.66% | 约 22.90%～23.02% |
| hotspot：bottleneck proxy saving vs uniform FP8 | 24.20% | 11.99% | 24.20% |
| 全局 payload saving vs BF16 | 62.50% | 56.25% | 56.25% |

将质量 selector 换成 PLTB 后，16 并发下 balanced/hotspot 的 all-safe bottleneck proxy saving 分别为 `23.52%/24.83%`；critical 50% safe budget 在相同回放中保持该 proxy，而 random 仅为 `11.93%/12.42%`。该结果仍受 bandwidth-only 边界限制。

这提示了一个比 receiver-only 更完整、但仍属于扩展的方向：

> **Quality-Safe Critical-Flow Budgeting**：先用 rank/gate/tail-mass 得到质量安全集合，再把有限 INT4 预算投到当前关键的跨节点 sender/receiver flow。

它的潜在价值是将"压多少"与"压在哪里"分开：质量模型约束候选集合，拓扑/调度信息决定预算落点。但它会破坏固定 rank two-lane 的完全规整性，因此必须比较三种实现层级：

1. 离线 `(layer, sender, receiver)` profile，静态生成少量模板；
2. scheduler 提供 coarse receiver-hotness，按 request/batch 切换模板；
3. 在线 greedy 只作为 oracle upper bound，不直接宣称可部署。

当前回放是 bandwidth-only：没有 collective schedule、queueing、contention、pack/unpack、kernel launch、overlap 或随机 arrival process。表中的 bottleneck time 只是 `bytes / nominal bandwidth` 的解析值，不是实测 TBT、TPOT 或 P99。

> **⚠️ 2026-07-11 更新：本节以上三条策略（`tail_budget_profile_ports`/`tail_budget_scheduler_receiver`/`tail_budget_greedy_ports`）在 `run_ep_congestion_sim.py` 中共享一个未识别的 confound——打分后统一给所有 remote 候选加最高优先级 bonus，导致三者数值几乎完全相同（num_jobs=1 时均为 0.22550），这掩盖了"识别热点端口"这个信号本身的真实贡献。剥离该 confound 后的结果和结论修正见第 6.2 节。**

#### 6.2 混淆剥离结果：receiver 侧信号可部署，sender 侧信号不可离线化

用固定候选池（tail-rank ∩ inter-node，对 hot/cold/random 完全相同）重新检验，四组新实验（脚本与完整报告见 `experiments/idea_a_mac/outputs/paper_validation/receiver_isolation/`，数据来源同一份 OLMoE held-out trace）：

1. **剥离 remote-bonus 后，热点识别本身仍稳定超出 random 的 95% CI**（例如 hotspot/16 jobs/frac=0.5：hot=0.242 vs random=0.121±0.004），cold 几乎归零——receiver-aware 的基本方向站得住。
2. **把热度拆成 receiver 侧和 sender 侧**：hotspot 型负载下收益几乎全部来自 receiver 侧（占比 69%~100%）；balanced 型负载下收益主要来自 sender 侧（占比常年 75%~99%）。**"receiver-aware"这个命名不准确，真实驱动机制经常是 sender 侧。**
3. **sender 侧信号（专家热度）跨 disjoint calibration/test 样本的 Spearman 相关仅 0.39~0.50**，最热 sender-rank 跨集合一致率仅 25%——内容依赖、非平稳，不能像 PLTB layer sensitivity 那样离线 profile 一次复用。
4. **直接验证 `tail_budget_profile_ports` 依赖的离线 profile 信息**：剥离混淆后只比 random 高约 0.03，远低于 oracle 上界的 0.056 差距，个别配置下甚至低于 random。**第 6.1 节"profile_ports 是 deployable only if calibration transfers"的判断被证实为否——calibration 不 transfer，因此该策略目前不可部署。**

**对本节结论的具体修正**：

- 上文第 6.1 节报告的三条 critical-port 策略（profile/scheduler/greedy）效果相近，其结论应撤回；相近是 confound 造成的假象。
- Quality-Safe Critical-Flow Budgeting 需要拆分为两条独立强度不同的 claim：
  - **可保留、可防守**：只用 **receiver-side scheduling signal**（调度器已知、不需要等本层路由结果的 token-origin 热度）分配 INT4 预算，在负载不均衡场景下几乎免费拿到 oracle 上界的全部收益，且低成本、可部署；
  - **需降级为 oracle/future work**：sender-side 信号（expert-owner 负载）虽然在均衡负载下贡献更大，但依赖本层实时路由结果、跨样本不稳定，若要使用必须在 dispatch 之后插入跨 EP-rank 的专家负载同步，这是额外系统开销，不能默认为免费信号。
- **新的候选设计——Two-Tier Congestion-Safe Budgeting**：离线层复用 PLTB 的 layer-sensitivity profile 决定每层安全 INT4 预算总量；在线层只用 receiver-side 热度决定这份预算在 remote pair 间怎么分配。完全绕开 sender 侧不可部署的问题，且与 PLTB"离线定总量、在线定分配"的哲学一致，比"识别热点端口"更精确、更可防守，可作为本文除 PLTB 之外的第二个系统设计贡献点候选。

---

### 7. 评估计划

#### 7.1 Baselines

1. BF16 combine；
2. uniform FP8 combine；
3. uniform hardware-aligned FP4 combine；
4. FP8 + global fixed tail-rank block-FP4（本文）；
5. FP8 + gate-value threshold block-FP4；
6. FP8 + cumulative gate-mass block-FP4；
7. FP8 + format-specific PLTB（optional）；
8. head/random/anti-layer control；
9. contribution oracle；
10. 有条件时加入现有通信压缩或 EP kernel baseline。

#### 7.2 指标

- **质量**：per-token KL、正确累计的 corpus PPL、标准下游任务、置信区间；
- **payload**：包含 scale/metadata/padding 后的实际通信字节；
- **kernel**：quant、pack、all-to-all、unpack/dequant、combine reduction 的逐项耗时与有效带宽；
- **serving**：TPOT/TBT mean、P50/P95/P99、throughput；
- **鲁棒性**：跨数据域、batch、top-k、EP size、placement 和拓扑。
- **拥塞扩展**：多 MoE layer、多并发 request/replica、balanced/hotspot origin、真实 arrival trace，以及 sender/receiver 双侧 critical port。

calibration、development 与 test 必须分离，不能继续用 WikiText-2 validation 的同一批前 N 条同时 profile 和评估。

#### 7.3 模型与平台

- **当前质量主证据**：OLMoE-1B-7B top-8；
- **跨 top-k 支撑**：LLM-jp E32-k16 top-16；Mixtral-TinyMistral top-2 仅作辅助，不把极端 one-hot gate 比例当主证据；
- **后续补强**：至少一个更大、真实 serving 目标 MoE；
- **系统平台**：至少 8×A100/H100 的 expert parallelism，优先基于 DeepEP/NCCL EP 接入 serving backend。

### 8. 当前材料能证明与不能证明的内容

| 可以证明 | 不能证明 |
|---|---|
| uniform FP8 是强质量 baseline，且公开系统已开始实现 FP8 combine | uniform FP8 在本系统中一定达到理论字节收益 |
| MXFP4/NVFP4 下 tail rank 显著比 head 安全 | rank 一定优于 gate threshold |
| FP8 + tail block-FP4 在小模型上扩展了质量—scale-aware wire frontier | 真实 TPOT/TBT/P99 已改善 |
| PLTB 的旧 INT4 proxy 收益具有格式依赖 | PLTB 已在硬件 FP4 下成立 |
| 正确 EP 语义的 trace replay 支持 critical-flow budget 的研究动机 | critical-flow policy 已改善真实 collective completion time 或 P99 |
| 多层 INT4/drop 存在明显非线性误差放大 | routing drift 是全部放大的唯一原因 |
| receiver-side scheduling signal（token-origin 热度）在剥离 remote-bonus 混淆后仍显著优于 random | 真实 combine receiver-port 拥塞已缓解 |
| —— | sender-side 信号可离线 profile（跨样本 Spearman 仅 0.39~0.50，已证否，见第 6.2 节） |
| MMLU 200 题 pilot 未观察到下降 | 下游任务质量无损已被统计证明 |

### 9. Go / No-Go 门槛

- **质量门槛**：在 held-out 数据上，tail block-FP4 相对 uniform FP8 的 PPL/任务退化处于预设容忍区间，并显著优于 head/random；
- **信号门槛**：fixed rank 在同 scale-aware wire budget 下接近 gate/contribution 上界，或虽略差但端到端延迟更优；
- **selector 决策门**：若 fixed-rank two-lane 不能展示稳定 operator/TPOT 优势，则主方法改为 gate/tail-mass aware；PLTB 只有格式特定复测稳定后才恢复；
- **kernel 门槛**：相对 uniform FP8，mixed combine operator 净延迟稳定改善约 `10%～15%` 以上；
- **端到端门槛**：TPOT 改善约 `3%` 以上，或 P99 改善约 `5%` 以上；
- **receiver 门槛**：仅针对 receiver-side scheduling signal；若其相对 rank-only 的额外 P99 收益不足 `3%～5%`，从主线删除；sender-side 信号已因跨样本不稳定（见第 6.2 节）预先排除出可部署主线，不再单独设置门槛；
- **critical-flow 门槛**：只有在同质量、计入不规则布局与 metadata 后，相对 selector-only 仍有稳定额外收益，才升级为扩展贡献；
- **回退**：若 mixed kernel 开销吞掉 payload 收益，则论文收缩为 characterization / quality study，不继续堆叠优化器。

---

## 方案 B（备选，不与 A 合并）：能效/SLO 双约束下的 MoE 推理联合优化

> 方案 B 保留为独立备选，不作为方案 A 的扩展章节。除非获得可信的真实功耗、通信能耗和 placement 数据，否则不与 A 同时推进，也不把两个方向包装成一篇论文。

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

约束：TBT SLO（$T(x)\le \text{TBT}_{99}^{SLO}$）、HBM 容量、副本下界 $\sum_g x_{l,e,g}\ge 1$。方案 B 必须建立并校准独立的 placement/replication latency model，不能复用方案 A 当前尚未验证的 receiver-port proxy。$\rho_g\cdot T$ 双线性项用 $T=\text{SLO}$ 上界做保守线性化。求解：小规模 MILP（Gurobi），大规模拉格朗日松弛 / 贪心。

---
