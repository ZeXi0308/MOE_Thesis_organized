# MoE 推理研究方向：严格 Idea 探索、上界审查与 Top 3

> 本文已与另外两份候选/复审材料合并到[统一主文档](./MoE_研究方向统一梳理_17项去重筛选与Top3执行路线_2026-07-25.md)；本文保留为候选生成与初筛过程附录。

日期：2026-07-25  
硬件边界：1×RTX 5090 32GB；正式实验 1 台 8×A100（显存容量、PCIe/SXM 与 NVLink/NVSwitch 拓扑尚需盘点）  
结论强度：文献分析 + 已有本地测量 + 新实验设计；不是多 GPU 实测结论

## 0. 先给结论

第一轮生成 10 个候选。严格审查后，5 个进入“保留/有条件保留”，5 个淘汰；因此没有触发“保留少于 5 个时再凑第二轮”的规则。真正值得先投入 1–3 天的只有三项：

1. **RankLane-Combine：固定双 lane 的 gate-rank 感知 combine 通信编码。** 它已有本地跨模型质量结构证据，下一步只需回答一个物理问题：在真实 pack/unpack 与 A2A 会计后，是否仍有 ≥5% 端到端空间。
2. **RouteCloak：普通共享 GPU/EP serving 的 exact-output 路由侧信道防御。** 论文上限最高，但必须先证明真实观察通道存在，并击败静态 bucket padding。
3. **ResumeSet：面向 tool/human interruption 的 KV–expert 双状态保留。** 它把 InferCept 类 KV 管理与 MoE expert cache 联合起来；核心不是预测下一个 expert，而是判断暂停请求的 route working set 是否对恢复冷启动有独立价值。

两个有条件后备是：**ScaleBridge**（单卡到多卡的可证伪性能迁移模型，偏 measurement）与 **QualityDebt-SlowRank**（暂态慢 rank 下的质量预算式降级，创新性被 Capacity-Aware Inference/ReaLB 严重挤压）。

最不建议继续投入的是泛化 receiver-awareness。仓库已有 corrected FJRC 结果：加入 join phase 后，OLMoE 的 deadline-miss 绝对下降仅 1.56 个百分点、LLM-jp 为 0，未过冻结门；RouteShare 的 matched route-cost effect 也只有 2.35%–3.06%。这说明“更复杂 receiver state”或“route-aware fairness”在现有单卡证据下没有足够上界。

## 1. 证据纪律与硬件假设

全文使用四种标签：

- **[论文事实]**：论文明确陈述的机制或结果；只引用论文/会议主页等一手来源。
- **[本地观察]**：本仓库现有脚本和报告测得的结果；不外推为网络或生产 serving 结论。
- **[推断]**：由论文 profiling、硬件带宽或本地 trace 推导的判断。
- **[新假设]**：本报告提出、尚待实验否定的主张。

所有 Amdahl 上界均是筛选用粗估。若目标部分占端到端时间比例为 `f`，完全消除时最大加速为：

\[
S_{max}=\frac{1}{1-f}.
\]

若机制只消除该部分的 `r`，则 `S=1/(1-fr)`。不知道 `f` 时，不报告虚假精确值，而是给区间并把 profiling 设为 Gate 0。

8×A100 不能默认是 NVSwitch：正式实验前先记录 A100 40/80GB、PCIe/SXM、`nvidia-smi topo -m`、可用 NCCL/DeepEP backend、GPU 间带宽和是否能锁频。[NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)列出的 Tensor Core 格式包括 TF32/BF16/FP16/INT8/INT4，而非 FP8；因此所有可扩展精度机制以 **BF16/INT8 或纯通信编码** 为主，5090 上的 FP8 只作补充。

本地证据的主要来源是：[Idea A rank-tail 设计与证据边界](../ideas/A_rank_tail_fp8/设计说明.md)、[corrected FJRC Level-1 负结果](../ideas/receiver_aware/FJRC_Corrected_Level1_Result_2026-07-23.md)、[Phase 1 v4 严格复核](../01_current_status/Phase1_v4_StrictReview_2026-07-23.md) 与 [receiver codec 硬门槛勘误](../01_current_status/Receiver_Codec硬门槛测量结论_2026-07-21.md)。这些结果只在各自冻结协议内成立。

## 2. 文献地图：哪些空间已经拥挤

- 通信与 EP runtime：Lina、Comet、FlashDMoE、SwiftEP、DeepEP/NCCL-EP、MixServe 已覆盖 A2A 排序、细粒度重叠、单 persistent kernel、buffer fusion/TMA、专用 dispatch/combine primitive 与 TP–EP 混合。
- 异步 serving 与 phase：AMoE/AEP、ASAP、MegaScale-Infer、ExpertPlex 已覆盖 μ-queue、异步 prefill、attention/expert 解耦、phase sharing 和 adaptive persistent kernel。
- placement/replication/routing：Aurora、Gimbal、MoEless、Mixture-of-Experts Serving、Capacity-Aware Inference 已覆盖部署–通信联合优化、跨层 pressure、弹性专家、在线 GPU 分配以及 token drop/reroute。
- offload/cache/precision：Fiddler、ProMoE、HOBBIT、DuoServe-MoE、DynaExq、ReMoE 已覆盖 CPU/GPU 协同、预测预取、混合精度 miss、分 phase cache、动态 expert bit-width 和通过 router fine-tuning 增强 reuse。
- speculative MoE：SP-MoE、MoE-Spec、SpecMoE、MoE-SpAc 已覆盖 SD-aware prefetch、verification expert budget、自辅助推测和以 speculation 作为 cache lookahead。
- reliability/security：Tarragon、EEP、LUMEN 已覆盖 worker/rank failure 与恢复；MoEcho 和 Expert Selections Reveal Text 已证明 route footprint 泄漏，而 CryptoMoE/SecMoE 处理密码学 private inference。

因此，“换一个 predictor/controller”“把 queue length 加进 router”“动态量化”“热门 expert 复制”“speculative prefetch”本身不再足以构成选题。

## 3. 第一轮 10 个候选总览

| ID | 候选 | 第一轮直觉 | Review 结论 |
|---|---|---|---|
| I1 | RankLane-Combine | 利用已知 gate-rank，把 combine 返回流固定分成高/低精度 lane | **保留** |
| I2 | RouteCloak | 用 exact-output route-equivalence trace 防普通 serving 侧信道 | **保留** |
| I3 | ResumeSet | 暂停请求恢复时联合管理 KV 与 expert working set | **保留** |
| I4 | ScaleBridge | 用单卡可执行分解预测多卡 policy 排序与 headroom | **有条件保留** |
| I5 | QualityDebt-SlowRank | 暂态慢 rank 时只近似低 gate-mass contribution | **有条件保留** |
| I6 | RouteShare-VTC | 按 route coalition 的真实成本做多租户公平计费 | **淘汰** |
| I7 | JouleBatch | deadline-aware expert coalescing 降 J/token | **淘汰** |
| I8 | Cancellable Spec-MoE | 尽早取消最终会被拒绝的 speculative expert work | **淘汰** |
| I9 | Shadow-Price Placement | 根据热点/拓扑动态复制、迁移、重路由 | **淘汰** |
| I10 | Sensitivity Cache/Quant | 按 expert 敏感度动态精度、预取和缓存 | **淘汰** |

---

## Idea 1：RankLane-Combine——固定双 lane 的质量感知 combine 编码

### 1. 核心假设

**[新假设]** 对 top-k MoE，gate-rank 的质量不对称足够稳定，因此把 combine contributions 以固定 `head-BF16/tail-INT8`（5090 可另测 FP8/INT4）双 lane 发送，可在无需逐 token metadata/predictor 的情况下减少 combine wire bytes；当 combine 占 MoE step ≥15% 时，真实净 TPOT/P99 改善可达 5% 以上，且质量损失受冻结阈值约束。

### 2. 问题背景

EP 每层通常有 dispatch 与 combine 两次跨 rank 数据交换。dispatch 的 top-k 副本内容相同，rank-aware 编码并不自然；combine 返回的是独立 expert outputs，且 gate 权重在发送前已知，tail contribution 的误差还会被较小 gate 缩放。**[本地观察]** OLMoE top-8 与 LLM-jp top-16 的 matched-byte 量化中，压 tail 比压 head 安全数十倍；但现有结果只是 fake quant/逻辑 payload，不是 wire 或 TPOT。

### 3. 与已有工作的区别

- [Lina（ATC’23）](https://www.usenix.org/conference/atc23/presentation/li-jiamin) 平衡 A2A transfer size/bandwidth并降低推理尾延迟；不按 contribution 质量分配 combine 表示。
- [SwiftEP（NSDI’26）](https://www.usenix.org/conference/nsdi26/presentation/li-xingyi) 用 buffer fusion 与 TMA offload 优化 prefill EP；不改变表示精度。
- [FlashDMoE](https://arxiv.org/abs/2506.04667) 把通信与专家计算融合进 persistent kernel；提供潜在实现载体，但不是 gate-rank codec。
- [MixServe](https://arxiv.org/abs/2601.08800) 自动选择 TP/EP 并融合 AR–A2A；不利用 combine 线性聚合的质量结构。

差异不是“再压缩一次 A2A”，而是 **只在 combine、只用已知 gate-rank、固定 lane layout、端到端质量约束**。

### 4. 新机制

- 输入状态：`(layer, top-k rank, gate weight, token owner rank, row count)`；可选离线 layer sensitivity。
- 决策变量：每层保护的 head ranks 数 `h_l`；不做 token 级预测。
- 目标：最小化 combine wire bytes/critical-rank completion，在 `ΔNLL/KL/task score` 与 P99 不劣化约束下最大化 goodput。
- 约束：lane 定长、scale/header/padding 全计；同一 layer 的 `h_l` 低频更新。
- 频率/关键路径：route 后 O(1) rank partition；codec 在 combine producer/consumer 中，位于关键路径，必须 fused。
- 开销：`O(L)` 配置；两个 buffer descriptor；scale metadata；不得有 CPU–GPU 同步。

### 5. 单张 5090 的最小验证实验

- 模型：`allenai/OLMoE-1B-7B-0924` 主模型，LLM-jp E32K16 复现；WikiText-2/C4 文档 + GSM8K/MMLU 小样本。
- 修改：现有 combine hook 与 codec microbench；先不改大型 serving engine，只实现真实 GPU pack→copy/NVLink-proxy→unpack→combine microkernel。
- baseline：BF16、uniform INT8、现有 uniform FP8、固定 tail lane、gate-threshold variable mask、no-codec oracle。
- 自变量：rows 1–4096、hidden 512/2048/4096、top-k、head ranks、链路限速 25/50/100/200/400Gbps。
- 指标：真实 bytes（含 metadata/alignment）、codec μs、break-even bandwidth、KL/NLL/task score、captured oracle headroom。
- 规模/时间：两模型各 64–128 文档；microbench 10k 次；约 1–3 天。32GB 对 OLMoE/LLM-jp 足够。
- 模拟边界：限速/trace replay只能判 codec break-even，不能称 NCCL/RDMA 结果。

### 6. 8×A100 正式实验

EP=8，attention 可 DP/TP；先按真实 topology 分别报告同 NVLink island 与 PCIe path。接入 DeepEP/NCCL-EP 的 combine path，保持 dispatch 不变。负载覆盖 prefill 512–8192、decode concurrency 1–256、ShareGPT/LongBench 与合成热点。报告 1/2/4/8 GPU weak/strong scaling、A2A critical-path time、P99 TPOT/goodput。新增验证：固定双 lane 是否在真实 collective 中保持规整、是否因 NVSwitch 太快而失去上界、慢 rank 是否由 codec 而非 compute 决定。

### 7. 关键指标

TPOT、P95/P99、goodput、combine/A2A time、wire bytes、codec overhead、SM occupancy、HBM temporary bytes、KL/NLL/任务分数、SLO violation。

### 8. 失败判据

- 真实 combine 占端到端 <10%，或无限带宽 oracle 提升 <5%；
- codec+metadata 吞掉 ≥50% 毛收益，或 break-even 低于实际链路；
- uniform INT8/FP8 达到同等质量–性能；
- 固定 lane 明显输给简单 gate threshold，且 variable mask 开销并不高；
- 只在人工限速或极端 top-k 下有效。

### 9. 研究风险

创新性中高但实现风险高；A100 无 FP8，需要 INT8/BF16 版本；NVSwitch 上通信可能不是瓶颈；reviewer 会要求真正 backend integration，而非 H2D proxy；质量可能模型特定。

### 10. 论文潜力

若真实 EP 过门，属于 GPU communication/runtime co-design，可投 MLSys/ASPLOS/SC；若只得到质量结构和 break-even 曲线，则降为 characterization/工程短文。

### 11. 优化空间上界

ScMoE 在其环境中报告通信占 MoE 时间约 15%（NVLink）到 60%（PCIe），但不能直接当本机数字。[推断] 若 combine 约占一半，`f_combine≈7.5%–30%`。完全消除的 Amdahl 上限约 1.08×–1.43×；减少 50% combine 时间的上限约 1.04×–1.18×。对优化良好的 8×A100 NVSwitch，真实上界很可能靠近低端，故 Gate 0 必须先测 `f_combine`。

### 12. 最强简单 Baseline

`uniform INT8` 或 `uniform FP8` + 现成 collective；其次是固定 `top-h BF16/rest INT8`。若它们捕获 `(simple-current)/(oracle-current) ≥0.8`，不做动态 layer LUT。

### 13. Oracle 实验

三种 oracle：把 combine CUDA event 直接置零（无限带宽）；保留时间但按目标压缩比例缩放；使用真实质量 trace 穷举每层 head-rank 数但 codec 零开销。先估上界，再写 fusion。

### 14. 净收益判断

当前是 **有真实质量必要性、系统净收益未验证**。固定 lane 避免 predictor/scan，但 scale 计算、pack/unpack、额外 kernel 和两个小消息可能抵消 wire saving；只有 producer/consumer fusion 后才有论文意义。

### 15. Reviewer 反对意见

1. “NVLink/NVSwitch 上 combine 只占几个百分点，oracle 都不到 5%。”
2. “uniform INT8/FP8 已拿走绝大多数收益，rank 只是调参。”
3. “单卡 H2D/限速并不代表 NCCL/DeepEP；缺少真实 8-GPU P99。”

### 16. Review 结论

**保留：** 已有本地质量结构证据；立即做 codec/infinite-bandwidth oracle，过 5% 上界后再接 8×A100。

---

## Idea 2：RouteCloak——SLO 约束下的 exact-output 路由足迹防御

### 1. 核心假设

**[新假设]** 在攻击者只能观察 expert identity/load/timing 或相关 GPU/EP footprint、不能读取 activation 的场景中，将真实 route 映射到少量 route-equivalence buckets，并以 dummy rows/cover experts/mixing 补齐，可在模型输出完全不变、延迟和能耗开销各 ≤10% 时消除 ≥80% 的 held-out attacker advantage。

### 2. 问题背景

MoE 的输入相关路由使执行路径携带语义。共享 GPU、性能计数器、TLB/cache 或 EP message-size/timing 都可能暴露 route footprint。防御不能简单加 route noise，因为这会改模型输出；也不能默认上密码学方案，因为普通云 serving 的威胁和成本不同。

### 3. 与已有工作的区别

- [MoEcho](https://arxiv.org/abs/2508.15036) 提出 CPU/GPU 架构侧信道并完成 prompt/response/vision 攻击；解决的是攻击发现，不是低开销防御。
- [Expert Selections Reveal (Almost) As Much As Text](https://arxiv.org/abs/2602.04105) 从 expert selections 重构文本；说明 route 本身高度敏感。
- [CryptoMoE](https://arxiv.org/abs/2511.01197) 用密码学安全 dispatch/combine 与 balanced routing 做 private inference；威胁模型和成本区间不同。
- [SecMoE](https://ojs.aaai.org/index.php/AAAI/article/view/39721) 也面向安全推理协议，而不是普通 shared-GPU runtime cover traffic。

本 idea 的边界是 **普通 serving execution-metadata 防御 + exact model output + SLO Pareto**，不声称首次发现泄漏或首次 private MoE。

### 4. 新机制

- 输入：真实 per-layer route histogram、request slack、当前 bucket occupancy、攻击通道可见字段。
- 决策：row-count bucket、dummy expert set、短 mixing window、是否走强 cover path。
- 目标：最小化冻结攻击器的 attribute/reconstruction advantage；约束 TTFT/TPOT、J/token、dummy work、exactness。
- 频率：每层/每 microbatch；仅查表和 padding prefix sum 在关键路径。
- 开销：bucket table O(E×B)，dummy rows 与额外通信；不训练在线大模型。

### 5. 单张 5090 的最小验证实验

- 模型：TinyMixtral/OLMoE；OpenWebText/WikiText 加可控 domain/attribute 标签。
- 修改：route capture、dummy-row executor、至少一个真实 GPU observer（优先 performance-counter 或 TLB channel；权限不允许时明确只做 instrumentation upper bound）。
- baseline：no defense、batch aggregation、static bucket、fixed top-q cover、full padding、random route noise（非 exact 对照）。
- 自变量：bucket 数、padding granularity、mix window、concurrency、attacker train/test domain。
- 指标：attacker top-1/top-k、mutual-information proxy、TTFT/TPOT、J/token、dummy rows、logit bitwise/数值一致性。
- 规模/时间：先 50k–500k route tokens；1–3 天 Gate -1。32GB 足够；虚拟 EP 只能测 message trace proxy。

### 6. 8×A100 正式实验

EP=8；观察 intra-node message size/timing、rank load、可行的 co-tenant channel；覆盖并发 1–256、top-2/top-k、均匀/倾斜 route。报告 privacy–latency–energy Pareto、跨 GPU 数 scaling、对 NCCL/DeepEP buffer fusion 的影响。新增验证是真实 A2A observer 是否能恢复足够 footprint，以及 cover traffic 是否制造新的 incast/P99。

### 7. 关键指标

Held-out attacker advantage、reconstruction top-k、TTFT/TPOT P99、goodput、dummy FLOPs/bytes、J/token、output exactness。

### 8. 失败判据

真实 observer 无法恢复 route；攻击只在同模板/直接拿 ground-truth route 时成立；static bucket 已达同一 Pareto；达到 80% leakage reduction 需 >10% latency 或 energy；攻击者能读 activation 导致隐藏 route 无意义。

### 9. 研究风险

最大风险是威胁模型，不是算法；硬件计数器权限和隔离机制可能阻止复现；安全评测要求严格 train/calibration/sealed split；跨模型泛化未必成立。

### 10. 论文潜力

成功是 systems-security/architecture 论文，可考虑 ASPLOS、USENIX Security、CCS、MLSys security track；只有 route-only instrumentation 则仅够测量论文或 workshop。

### 11. 优化空间上界

这是 leakage–overhead 问题而非 speedup。全 expert/full histogram padding 是隐私 oracle，但计算上界可达 `E/k` 倍，通常不可用。可行 headroom 定义为：在 ≤10% 额外延迟/能耗下，静态 bucket 到隐私 oracle 之间是否仍有 ≥20% attacker-advantage gap。低负载 mixing 空间小，高负载更易隐藏但 queueing 成本也高。

### 12. 最强简单 Baseline

固定 row-count buckets + 普通 continuous batching。它无需在线优化，可能已获取 70%–90% 可用隐私收益；若是如此，自适应机制没有必要。

### 13. Oracle 实验

直接把不同输入映射为相同 full trace，测 attacker 降到 chance 的隐私天花板；再穷举小 trace 上最小 padding 的等价类，得到给定 5%/10% overhead 的最优 privacy curve。

### 14. 净收益判断

科学空间明确，但尚未证明可部署 observer。只有 Gate -1 成功且 static buckets 留下明显 gap 时，adaptive cover 才值得实现。

### 15. Reviewer 反对意见

1. “你使用的是 instrumentation route，不是攻击者现实可见的侧信道。”
2. “静态 padding 已足够；自适应策略只是增加攻击面和复杂度。”
3. “隐私提升来自弱攻击器或同分布数据，且 10% overhead 会被生产系统拒绝。”

### 16. Review 结论

**保留：** 论文上限高且单卡可快速做问题存在性 Gate；先复现真实 observer，再谈机制。

---

## Idea 3：ResumeSet——Agent interruption 下的 KV–expert 双状态保留

### 1. 核心假设

**[新假设]** tool call、人类等待或多轮会话暂停前的短窗口 expert working set，对恢复后前若干 decode steps 具有比全局 LFU/LRU 更高的条件命中率；在固定 HBM 预算下，联合保留 KV 与小型 per-request ResumeSet，可将 resumption P95/P99 冷启动 stall 降低 ≥15%，且不使非暂停请求 P99 增加 >3%。

### 2. 问题背景

暂停请求的 KV 暂时不用但恢复时很贵；MoE offload 系统同时还可能丢失该会话的 experts。现有 KV preemption 与 expert cache 通常独立优化，可能出现“保住 KV、恢复后每层仍连续 miss”或“保专家却挤掉更值钱 KV”的错误资源分配。

### 3. 与已有工作的区别

- [InferCept](https://arxiv.org/abs/2402.01869) 针对 augmented LLM interception，联合 discard/swap/preserve KV；不管理 MoE expert working set。
- [Fiddler](https://arxiv.org/abs/2402.07033) 以 CPU 计算 miss expert，减少权重搬移；不利用 pause/resume session state。
- [ProMoE](https://arxiv.org/abs/2410.22134) 用中间结果预测后续 expert 并预取；目标是连续推理，不是 interruption 恢复。
- [ReMoE](https://arxiv.org/abs/2605.27081) 通过 router fine-tuning 提高短期 expert reuse；会改模型/router，且不联合 KV retention。

### 4. 新机制

- 输入：暂停概率/预计时长、KV bytes、最近 W 步 `(layer,expert)` set、cache pressure、request age/SLO。
- 决策：preserve/swap/discard KV；为请求保留或在预计恢复前预取的 expert set；每请求 memory quota。
- 目标：最小化恢复 TTFT/首 N 步 TPOT 与全局 P99，约束总 HBM、starvation、额外 H2D bytes。
- 频率：pause/resume 边界和低频 cache pressure event，不在每 token critical path；在线只用 moving average/阈值。
- 开销：每请求 L×bitset/小 top-m list；无大 predictor。

### 5. 单张 5090 的最小验证实验

- 模型：OLMoE 或量化 Mixtral；ToolBench/AgentBench 风格 trace，也可从 ShareGPT 合成 pause 50ms–60s。
- 修改：expert cache cap + pinned host store + pause/resume trace replayer；不先改完整 agent framework。
- baseline：KV-only InferCept heuristic、LRU/LFU expert cache、last-window top-m、global popularity、Belady expert oracle、joint oracle。
- 自变量：cache cap 10/25/50%、pause duration、ResumeSet size、concurrency、route drift。
- 指标：resume TTFT、前 8/32 step TPOT、H2D bytes、expert hit rate、KV hit/recompute、non-paused victim P99。
- 规模/时间：500–5k pause events；trace oracle 先跑数小时，真实 H2D 1–3 天；32GB 通过显式 cap 制造但必须报告“人为 cap”。

### 6. 8×A100 正式实验

适用于模型/多租户状态不能全部驻留时：EP=8，KV 可按 attention rank 分片，experts 按 EP rank；pause/resume 请求经 gateway 重定向。测试 sticky routing、跨 replica 恢复、工具调用 burst。报告每 rank cache pressure、恢复 P99、H2D/GPU–GPU bytes、goodput。新增验证是 ResumeSet 是否因 rank 分片而更值钱，还是 8 卡总 HBM 使 miss 消失。若全驻留，正式实验应诚实报告零空间。

### 7. 关键指标

Resume TTFT、first-N TPOT P95/P99、expert/KV hit、H2D/GPU–GPU bytes、HBM、goodput、victim slowdown、starvation。

### 8. 失败判据

暂停前后 expert-set Jaccard/recall 不胜 global LFU；Belady 相对 LRU headroom <10%；expert miss stall 占恢复时间 <5%；last-window top-m 已捕获 ≥80% oracle；8 卡全驻留后收益消失。

### 9. 研究风险

本地 cache cap 可能被 reviewer 视为人为；需要真实 agent pause trace；大模型下载/CPU RAM 可能是资源瓶颈；与通用 KV scheduling 的差异必须用 joint action flip 和恢复冷启动证明。

### 10. 论文潜力

若 joint policy 在真实 agent trace 和受限/多卡两类环境均成立，可做 systems mechanism（MLSys/EuroSys/ATC）；若只发现 route locality，则是 characterization。

### 11. 优化空间上界

令恢复窗口中 expert miss stall 占比 `f_m`。100% hit 的上界为 `1/(1-f_m)`：`f_m=0.2` 时 1.25×，`f_m=0.5` 时 2×；若模型全驻留则 `f_m=0`。低负载下 resume latency 更显著，高负载下 cache contention 更大但 queueing 可能掩盖它。

### 12. 最强简单 Baseline

保 KV + 暂停前最后 W 步每层 top-m experts；用统一 TTL，恢复前按 layer 顺序预取。它很可能已吃掉大部分收益。

### 13. Oracle 实验

给 replay 未来 resume routes，用 Belady 同时选择 KV/expert bytes；再做“100% expert hit”“0 H2D latency”“无限 HBM”三个删除开销 oracle，分清瓶颈来自 expert 还是 KV/queue。

### 14. 净收益判断

动作在 pause 边界，不污染 token critical path，开销较可控；但只有 offload/capacity miss 真实存在时有价值。必须先过 trace oracle，不能直接建复杂 scheduler。

### 15. Reviewer 反对意见

1. “32GB 人为 cache cap 不代表 8×A100 生产部署。”
2. “简单 sticky session + last-window prefetch 已经一样好。”
3. “收益其实来自 KV 保留或请求 affinity，不是 MoE expert ResumeSet。”

### 16. Review 结论

**保留：** 两个割裂问题的联合建模有清晰 gap；先做无代码/少代码 trace oracle 与强简单 baseline。

---

## Idea 4：ScaleBridge——单卡 trace 到多卡 EP 的可证伪性能迁移模型

### 1. 核心假设

**[新假设]** 将 MoE step 分解为 route histogram、permute、per-expert service curve、dispatch/combine message matrix、overlap DAG 与 queueing，而不是用总 FLOPs/bytes roofline，可在只用单卡 executable traces 加少量链路 microbench 的条件下，对 8×A100 的 policy **收益符号与排序**达到 ≥0.8 Spearman，并把端到端收益误差控制在 ±10%；若做不到，也能界定哪些单卡 proxy 不可外推。

### 2. 问题背景

大量 MoE 论文方向在单卡模拟中制造队列/限速，然后假设多卡收益会放大。本仓库已经出现多次 proxy 正结果在严格会计或真实路径中消失。一个能预测“值得不值得占用 8 卡”的校准工具，本身可能形成 measurement contribution。

### 3. 与已有工作的区别

- [EPS-MoE](https://arxiv.org/abs/2410.12247) 用 load 决定 GroupGEMM/DenseGEMM 并重叠通信，含性能建模但目标是自身 scheduler。
- [Aurora](https://arxiv.org/abs/2410.17043) 对 deployment 与通信 scheduling 给出多场景优化；不是单卡→多卡可迁移性审计。
- [MixServe](https://arxiv.org/abs/2601.08800) 根据模型/硬件通信开销自动选 parallelism；不以预测误差、policy ranking 和 proxy validity 为研究对象。
- [FlashDMoE](https://arxiv.org/abs/2506.04667) 表明 kernel/runtime 结构可极大改变性能，正说明简单相加模型可能失效。

### 4. 新机制

输入为真实 route trace、单卡 service curves、A100/NVLink/PCIe microbench；变量是候选 placement/codec/batching policy；目标输出是可校准 latency distribution、critical-path attribution 和置信区间；约束因果可见状态与资源容量。离线运行，不在 serving critical path；存储 O(trace + E×shape LUT)。

### 5. 单张 5090 的最小验证实验

模型 OLMoE/LLM-jp；采集 prefill/decode route、kernel time、rows→service curve。baseline 是 bytes-only roofline、平均 expert load、简单 additive LUT。自变量为 batch、top-k、synthetic bandwidth、placement/codec/coalescing policies。指标为对留出单卡“伪多 rank”执行的时间误差和 policy ranking。1–2 天、32GB 足够；核心假设的最终部分仍需 8 卡，单卡只能验证 decomposition consistency。

### 6. 8×A100 正式实验

在 1/2/4/8 GPU、不同 batch/concurrency/topology path 上运行不少于 20 个 policy/config points；训练校准只用一半点，sealed 点测 MAPE、P95 error、ranking、bottleneck attribution。新增验证是 overlap/contention/collective startup 是否破坏单卡模型，以及需要多少真实多卡校准点。

### 7. 关键指标

Latency MAPE、P95/P99 error、policy Spearman/Kendall、收益正负判断 accuracy、headroom CI coverage、profiling cost。

### 8. 失败判据

简单 roofline 已达同等精度；需要大量 8 卡点才可校准；跨 batch/model/topology ranking 不稳定；只能预测均值不能预测 P99；误差大于机制间典型 5%–10% 差距。

### 9. 研究风险

容易被视为 simulator engineering；A100 单节点拓扑覆盖范围窄；FlashDMoE/DeepEP 等实现变化导致模型失效；需要公开可复现 trace 才有影响。

### 10. 论文潜力

偏 measurement/modeling，可投 MLSys/SC/ISPASS；若只能服务本论文实验，则是内部工具而非独立论文。

### 11. 优化空间上界

不直接提升 serving；价值上界是减少错误立项与 8 卡实验成本。可用 decision regret 衡量：oracle 总能选真实最优配置，简单 roofline 的 regret 与 ScaleBridge 的 regret 之差。若简单模型配置选择损失 <5%，研究空间不足。

### 12. 最强简单 Baseline

`T = max(compute_service, dispatch_bytes/BW, combine_bytes/BW) + launch`，加一个按 active-experts 修正的线性模型。

### 13. Oracle 实验

全量 8-GPU exhaustive surface 作为真值；比较只用 5090、5090+1个A100点、+5个点时的 regret 曲线。

### 14. 净收益判断

工程开销低、能复用所有方向，但独立论文 novelty 中等。适合作为所有 Top idea 的统一前置工具，是否独立立项取决于简单 roofline 是否明显失败。

### 15. Reviewer 反对意见

1. “只在一台 8×A100 上校准，谈不上 scale prediction。”
2. “模型只是常见 LUT/roofline 组合，没有新原理。”
3. “P99 由 serving queue 决定，单卡 kernel trace 无法预测。”

### 16. Review 结论

**有条件保留：** 先验证简单 roofline 的 decision regret；若它已足够，不作为论文，只作为实验基础设施。

---

## Idea 5：QualityDebt-SlowRank——暂态慢 rank 的质量预算式降级

### 1. 核心假设

**[新假设]** 在尚未 fail-stop、但某 EP rank 连续数步变慢的窗口中，仅对该 rank 上低 gate-mass contributions 做 drop/低精度，并用每请求 quality debt 限制累计近似，可相对等待、全量降级和无预算 token drop 把 P99 TPOT/SLO violation 降低 ≥15%，同时保持冻结质量阈值。

### 2. 问题背景

fail-stop 系统处理“死了怎么办”，load balancing 处理长期热点；现实中还会有 thermal throttling、共租干扰、瞬时 kernel/链路抖动。同步 EP 的一步由最慢 rank 决定，但为所有 token 重路由/迁移可能比抖动本身更慢。

### 3. 与已有工作的区别

- [Capacity-Aware Inference](https://arxiv.org/abs/2503.05066) 已做 overloaded token drop/reroute，并报告接近 1.94×；与本 idea 高度邻近。
- [ReaLB](https://arxiv.org/abs/2604.19503) 对多模态热点 rank 动态低精度以减轻 straggler；已覆盖当前-load→precision 主链。
- [EEP](https://arxiv.org/abs/2605.10670) 处理 partial rank failure，但明确不处理未触发 timeout 的 transient degradation。
- [Tarragon](https://arxiv.org/abs/2601.01310) 用 reconfigurable datapath/shadow experts 掩盖 worker failure；不是短时质量预算降级。

唯一可能的增量是 **per-request cumulative quality debt + transient slow-rank timescale + contribution-level而非 token-level动作**；若该增量不产生 matched-state action flip，方向即被已有工作覆盖。

### 4. 新机制

输入：当前 rank service residual、gate mass、request debt/slack；变量：wait、INT8、drop tail contribution、reroute；目标最小化 P99/SLO violation 与总 quality debt；约束每请求 debt、最大连续降级步、fairness。每 step O(tokens×k) prefix sum，在 GPU route/dispatch path；不能 CPU 同步。存储每请求一个 debt 标量和短 EWMA。

### 5. 单张 5090 的最小验证实验

真实 OLMoE/LLM-jp route 与质量 hook；注入 1.2–3× service delay，做 causal trace replay和实际 CUDA delay。baseline：wait、global threshold、Capacity-Aware token drop/reroute proxy、ReaLB-like hot-rank precision、oracle。自变量 slow duration/frequency、gate mass、concurrency；指标 P99 proxy、quality CVaR、debt violation、work。数小时至 2 天；32GB 足够；不能称物理网络。

### 6. 8×A100 正式实验

EP=8，注入锁频/低优先级干扰/受控 NCCL delay，区分 compute slow 与 link slow；真实 continuous batching 下测 P99 TPOT、goodput、质量与公平。新增验证是慢 rank 是否可及时检测、动作是否比抖动短、A100 INT8 path是否真快。

### 7. 关键指标

P99 TPOT、SLO violation、goodput、straggler stall fraction、quality CVaR/max、debt/fairness、动作率、检测/恢复开销。

### 8. 失败判据

真实暂态慢 rank 事件太少/太短；Capacity-Aware 或 ReaLB-like baseline 已捕获 ≥80% oracle；gate mass 与质量损失相关性弱；控制开销/错误降级抵消收益；只在人工 3× slowdown 有效。

### 9. 研究风险

创新风险极高，因两篇近邻已覆盖 drop/reroute/precision；质量 proxy 可能不代表自由生成；A100 动态 INT8 也可能无执行快区；reviewer 会质疑为何不直接等/复制。

### 10. 论文潜力

过严格 necessity gate 后可做 reliability×approximation systems paper；否则只是已有 load-aware routing 的增量。

### 11. 优化空间上界

只在 slow-rank 窗口生效。若该窗口占请求关键 steps 的 `p`，且等待 stall 占这些 step 的 `f_s`，完全消除的整体上界约 `1/(1-p f_s)`。例如 `p=.2,f_s=.5` 仅 1.11×；低负载/无抖动为 0。必须从真实 8 卡 trace 估 `p`，不能用注入频率冒充普遍性。

### 12. 最强简单 Baseline

检测到 rank 超过 EWMA 阈值后，对其最低 gate contribution 固定 drop 一个，最多连续两步；无 debt predictor。若它已足够，新 controller 无价值。

### 13. Oracle 实验

知道未来 slow duration 与每个 contribution 的真实质量损失，穷举 wait/drop/INT8/reroute；比较无质量限制与固定 quality budget 两个 oracle。

### 14. 净收益判断

物理上界可能存在，但 novelty 与 workload coverage 都弱。只有 matched `(load, gate mass, slack)` 下 debt state 确实改变动作且带来 >10% 额外收益，才继续。

### 15. Reviewer 反对意见

1. “Capacity-Aware Inference/ReaLB 已经做了同一件事。”
2. “你用人为 slowdown 和 teacher-forced KL 制造了收益。”
3. “瞬态窗口占比低，Amdahl 上界不足；简单 threshold 一样好。”

### 16. Review 结论

**有条件保留：** 先做 prior-art necessity/action-flip 与真实 slow-event census；不是 Top 3。

---

## Idea 6：RouteShare-VTC——route coalition 感知的多租户公平

### 1. 核心假设

相同 token 数/rows 下，tenant 与同批请求的 expert overlap 产生不可分离成本；按 realized route coalition 更新 virtual service 可把 worst-tenant slowdown 降低 ≥10%。

### 2. 问题背景

VTC 按 token 或自定义独立 cost 计费，MoE grouped execution 可能有共享 weight/launch 成本，导致 route-heavy tenant 把外部性转给他人。

### 3. 与已有工作的区别

[VTC](https://www.usenix.org/conference/osdi24/presentation/sheng) 提供公平界；[DLPM](https://arxiv.org/abs/2501.14312) 已联合 locality/fairness/load balance；[ExpertPlex](https://arxiv.org/abs/2607.18002) 用 tile 级 persistent kernel 做隔离；LLMVisor 已做 per-request GPU latency attribution。潜在差异仅剩 coalition-dependent MoE cost。

### 4. 新机制

输入 route histogram/tenant debt，变量 next-batch admission，目标 worst-tenant isolated-normalized slowdown，预算平衡成本份额；batch 完成后低频更新，不改 router；O(rows) ledger。

### 5. 单张 5090 的最小验证实验

OLMoE/LLM-jp matched total rows/active experts、不同 overlap；完整 expert path；baseline token-VTC、rows、rows+active-experts、leave-one-out、exact Shapley oracle；测 cost residual、action flip、victim slowdown。现有本地 Gate-0 已完成。

### 6. 8×A100 正式实验

理论上 EP=8 测 network coalition externality；但单卡 Gate 未过，不应占用正式资源。

### 7. 关键指标

matched latency contrast、model R²、worst-tenant slowdown、SLO、throughput。

### 8. 失败判据

route effect <5%；简单模型解释 ≥90%；oracle fairness gain <10%；自定义公平指标循环自证。

### 9. 研究风险

本地已出现决定性低 headroom；prior art 密集；多卡才可能出现的新 effect 无法先在 5090 验证。

### 10. 论文潜力

原本是 fairness systems paper；当前仅剩负面 characterization。

### 11. 优化空间上界

**[本地观察]** matched histogram effect 仅 2.35%–3.06%，最强简单模型 held-out R²=0.9971–0.9986。即使完全消除，单卡端到端收益也不超过约 3%，触发先验淘汰。

### 12. 最强简单 Baseline

`cost = α·rows + Σβ_e·1[expert active]` 接 VTC；已解释绝大部分变化。

### 13. Oracle 实验

exact leave-one-out/Shapley 与可执行 matched batches；已有结果显示 oracle gap 小。

### 14. 净收益判断

ledger/scheduling 开销与剩余 2%–3% 空间同阶，多卡 speculation 不能挽救单卡核心假设。

### 15. Reviewer 反对意见

1. “上界低于 5%。” 2. “简单模型 R² 已接近 1。” 3. “公平改进可能由你自己的计费定义产生。”

### 16. Review 结论

**淘汰：** 不进入 8 卡。

---

## Idea 7：JouleBatch——deadline-aware expert coalescing

### 1. 核心假设

相同 expert 的小 row kernel 在低利用率区存在 race-to-idle/静态功耗交叉点；按 deadline 聚合可在 P99 SLO 内降 J/token ≥10%。

### 2. 问题背景

decode expert GEMM 小且碎片化；等待可提高效率，却也增加静态能耗和 latency。

### 3. 与已有工作的区别

[AMoE/AEP](https://arxiv.org/abs/2505.08944) 已用 μ-queue/adaptive rebatching；[PALS](https://arxiv.org/abs/2605.21427) 联合 power cap/batch；[Festina](https://arxiv.org/abs/2606.30391) 联合 placement、SM/frequency 与 SLO；[ExpertPlex](https://arxiv.org/abs/2607.18002) 自适应 tile 资源。只加 energy objective 已不够新。

### 4. 新机制

输入 rows/deadline/power state，变量 wait/coalesce，目标 J/completed token，约束 P99；每 arrival 决策；可能需 GPU/CPU scheduler，开销在 μs–ms。

### 5. 单张 5090 的最小验证实验

真实 expert kernel+NVML，Poisson/bursty arrivals；immediate、fixed timeout/rows、EDF、oracle；负载 0.3/0.6/0.9；1 天，32GB 足够。

### 6. 8×A100 正式实验

EP=8 测 rank-local coalescing 与 barrier；但已有工作覆盖且 A100 power telemetry/控制粒度需核实。

### 7. 关键指标

J/token、P99、deadline miss、GPU active/idle、goodput。

### 8. 失败判据

固定 timeout 捕获 ≥80% oracle；能耗差 <5%；只改善平均值；等待恶化 P99。

### 9. 研究风险

novelty 低；NVML 分辨率和完整能量窗口难；容易重复 AMoE/PALS/Festina。

### 10. 论文潜力

工程优化或测量，不足以独立成为强系统论文。

### 11. 优化空间上界

只有 expert execution 能耗/时间占比乘以 batching efficiency gap；若 expert stage 30%、oracle 降 20%，端到端最多约 6%。低负载可能省能但吞吐不重要，高负载 queueing 掩盖。

### 12. 最强简单 Baseline

固定 row threshold + 最大等待时间；PALS-style batch/power lookup。

### 13. Oracle 实验

已知未来 arrivals 的 batch packing；0 wait tax 与真实 wait energy 分开。

### 14. 净收益判断

物理现象可测，但论文增量不足；复杂 controller 不值得。

### 15. Reviewer 反对意见

1. “AMoE/PALS/Festina 已覆盖。” 2. “能量会计不完整。” 3. “fixed timeout 一样好。”

### 16. Review 结论

**淘汰：** 可保留为其他系统的 baseline/measurement，不独立立项。

---

## Idea 8：Cancellable Spec-MoE——推测树的 expert work 取消

### 1. 核心假设

draft tree 中最终被拒绝分支的 expert work 可在 verifier 足够早的层被识别并取消，从而降低 HBM/A2A work ≥15%，且不改接受分布。

### 2. 问题背景

SD 将多个 token/分支并行验证，MoE 会激活更多 unique experts；被拒分支浪费计算与通信。

### 3. 与已有工作的区别

[SP-MoE](https://arxiv.org/abs/2510.10302) 已做 SD-aware expert prefetch/流水；[MoE-Spec](https://arxiv.org/abs/2602.16052) 已做 verification-time expert budgeting；[SpecMoE](https://arxiv.org/abs/2604.10152) 自辅助 speculative decoding；[MoE-SpAc](https://arxiv.org/abs/2603.09983) 把 speculation 当 cache lookahead。剩余“early cancel”很窄。

### 4. 新机制

输入 verifier partial logits/branch state，变量 cancel time/expert budget，目标 accepted token/s，约束 exact acceptance；逐 layer critical path，需复杂 branch bookkeeping。

### 5. 单张 5090 的最小验证实验

小 MoE + EAGLE/draft trace；离线 oracle 标记 rejected branches；删除其后续 work；baseline vanilla SD、MoE-Spec budget、shorter draft tree；测 wasted work与理论 TPOT。1–2 天 trace replay。

### 6. 8×A100 正式实验

EP=8 测跨 rank cancellation message、in-flight kernel不可取消和 batch fragmentation；但需先证明现有方法未覆盖。

### 7. 关键指标

Accepted tokens/s、acceptance rate、unique experts、A2A bytes、cancelable work、quality exactness。

### 8. 失败判据

reject signal到来时 work已发出；缩短 tree 同样好；取消使 batch变小/效率下降；MoE-Spec 覆盖。

### 9. 研究风险

prior art 极新且密集；实现复杂；单卡无法验证通信取消。

### 10. 论文潜力

若有硬件/通信 primitive 新机制才可能成文；当前更像增量。

### 11. 优化空间上界

上界是 rejected-branch 中尚未执行的 expert share；acceptance 高或 rejection 很晚时接近 0。即使 wasted expert work 30%，attention/accepted work 保留，端到端上限通常低于 1.43×。

### 12. 最强简单 Baseline

自适应缩短 draft depth / fixed expert budget；很可能获取大部分 oracle。

### 13. Oracle 实验

事先知道 accepted prefix，删除所有 rejected expert work；与只缩 tree 的 oracle 对照。

### 14. 净收益判断

oracle 可能大，但可实现 causal gap 很小且已被最新工作包围。

### 15. Reviewer 反对意见

1. “MoE-Spec/SP-MoE 已覆盖。” 2. “oracle 使用未来 acceptance。” 3. “取消开销和 fragmentation 未计。”

### 16. Review 结论

**淘汰：** 看起来新颖，但截至 2026 已被相邻工作实质覆盖。

---

## Idea 9：Shadow-Price Placement——动态复制/迁移/重路由

### 1. 核心假设

根据 expert 热度、source DP、receiver queue 与拓扑在线调整 replica，可在迁移开销后降低 P99 ≥15%。

### 2. 问题背景

热点 expert 造成 rank imbalance；静态 placement 对 workload drift 反应慢。

### 3. 与已有工作的区别

[Aurora](https://arxiv.org/abs/2410.17043) 联合 deployment/通信 scheduling；[Gimbal](https://arxiv.org/abs/2606.15177) 用 source-DP→expert statistics 与 migration stability；[Mixture-of-Experts Serving](https://arxiv.org/abs/2607.17880) 已形式化 GPU assignment 与 reconfiguration cost；[MoEless](https://arxiv.org/abs/2603.06350) 用弹性 serverless experts。剩余空间极窄。

### 4. 新机制

输入热度/queue/topology，变量 replicas/placement/reroute，目标 P99+迁移 cost；秒级控制面，不在 token path；需额外 HBM和迁移通信。

### 5. 单张 5090 的最小验证实验

route replay + min-cost flow oracle；baseline static balanced、periodic rebalance、least-queue；只能估机会，无法验证物理 migration/collective。

### 6. 8×A100 正式实验

EP/replica hybrid；真实 migration、route tables、CUDA graph invalidation；但不是先单卡后自然扩展，而是核心因果只在多卡。

### 7. 关键指标

P99、imbalance、migration bytes/pause、HBM、goodput。

### 8. 失败判据

periodic rebalance ≥80% oracle；热度变化快于迁移；额外副本内存过高；8卡规模不足。

### 9. 研究风险

novelty 低、工程高、单卡证据弱，直接违反优先约束。

### 10. 论文潜力

成功也像已有工作增量。

### 11. 优化空间上界

完美 balance oracle 仅消除 max-rank 与平均 rank 的差；若现代 router/EPLB 已平衡，gap 小。迁移成本在短 drift window 可超过全部收益。

### 12. 最强简单 Baseline

静态 topology-aware placement + moving-average periodic replication。

### 13. Oracle 实验

未来热度已知、零迁移、无限 replica；若该 oracle <10%，立即停。

### 14. 净收益判断

即使有空间，已有论文和实现路径更成熟；不值得在硕士资源上重做。

### 15. Reviewer 反对意见

1. “与 Gimbal/Aurora/在线 MoE Serving 重复。” 2. “单卡模拟不能证明。” 3. “迁移/graph invalidation 未完整会计。”

### 16. Review 结论

**淘汰。**

---

## Idea 10：Sensitivity Cache/Quant——专家敏感度驱动的动态精度与预取

### 1. 核心假设

expert sensitivity/hotness 有稳定时间相关性，可低频决定 precision/cache residency，在质量阈值内降 miss stall/能耗 ≥10%。

### 2. 问题背景

offload miss 很贵，统一低精度伤质量；看似可按专家异质性分配资源。

### 3. 与已有工作的区别

[HOBBIT](https://arxiv.org/abs/2411.01433) 已做 token/layer/sequence 三级 mixed-precision offload/cache；[DynaExq](https://arxiv.org/abs/2511.15015) 把 expert precision 作为动态资源；[ProMoE](https://arxiv.org/abs/2410.22134) 预测预取；[DuoServe-MoE](https://arxiv.org/abs/2509.07379) 分别优化 prefill/decode。泛化 sensitivity controller 已被覆盖。

### 4. 新机制

输入 sensitivity/hotness/cache pressure，变量 bits/residency/prefetch，目标 latency+quality+energy；低频控制但切换/加载昂贵；额外多精度权重和 metadata。

### 5. 单张 5090 的最小验证实验

OLMoE/LLM-jp；expert cache cap、fake/真实量化；baseline uniform quant、LFU/LRU、HOBBIT-like；测 Belady/lowest-bit oracle。仓库已有相关 prefetch 与动态 FP8 审计。

### 6. 8×A100 正式实验

专家分片与 offload；但 A100 无 FP8，需 INT8/W4A16；若模型全驻留，cache 机制无空间。

### 7. 关键指标

miss stall、H2D bytes、P99、quality、J/token、precision-switch cost。

### 8. 失败判据

Belady headroom <10%；simple LFU/last-window ≥80% oracle；低精度没有真实快区；跨模型 sensitivity 不稳定。

### 9. 研究风险

prior art 最拥挤；fake quant 不代表 runtime；容易得到“质量统计正、系统负”。

### 10. 论文潜力

当前更像已有系统复现或工程调参。

### 11. 优化空间上界

**[本地观察]** 正确 `(layer,expert)` cache key/full top-k 后 working set 很快饱和，safe-budget prefetch oracle 接近 0；dynamic FP8/QuantizeOnce 在真实 decode path 也没有低于 BF16 的 operating region。对应上界低于 5% 或缺执行杠杆。

### 12. 最强简单 Baseline

uniform低精度 + per-layer LFU/last-window prefetch；已有工作已很强。

### 13. Oracle 实验

100% cache hit、未来 routes、无损最低 bit、零 precision-switch；本地相关 oracle 已显示不足。

### 14. 净收益判断

机制开销与剩余 headroom 同阶；不应复活旧 prefetch/precision 对象。

### 15. Reviewer 反对意见

1. “HOBBIT/DynaExq/ProMoE 已覆盖。” 2. “fake quant 无真实速度。” 3. “缓存压力是人为 cap。”

### 16. Review 结论

**淘汰。**

---

## 4. 淘汰结果与二阶段检查

### 被淘汰的 5 项

| Idea | 决定性淘汰原因 |
|---|---|
| I6 RouteShare-VTC | 本地 matched effect 2.35%–3.06%，simple model R²≈0.997–0.999；oracle 端到端空间低于 5% |
| I7 JouleBatch | AMoE/PALS/Festina/ExpertPlex 邻近覆盖；fixed timeout 很可能吃掉大部分空间 |
| I8 Cancellable Spec-MoE | 2025–2026 的 SP-MoE/MoE-Spec/SpecMoE/MoE-SpAc 已把关键动作空间占满 |
| I9 Shadow-Price Placement | Aurora/Gimbal/MoE Serving/MoEless 直接覆盖，且单卡不能验证核心因果 |
| I10 Sensitivity Cache/Quant | prior art 密集；本地 cache oracle 与真实低精度 path 均没有足够系统上界 |

### Review 后保留的 5 项

I1、I2、I3 为保留；I4、I5 为有条件保留。数量恰好为 5，因此按附件规则 **不生成为了凑数的第二轮新 idea**。其中 I5 高重合，仅作为“必要性一票否决”的后备，不能未经查重直接实现。

## 5. 评分与综合排名

所有维度 1–5；Engineering cost 越高表示越容易实现，Risk 越高表示失败风险越高。综合分不把 Risk 当奖励：

`Score = Novelty + Importance + Local feasibility + Scale-up value + Engineering cost + Measurement clarity + Paper potential - Risk`。

| 排名 | Idea | Novelty | Importance | Local | Scale-up | Eng. easy | Measure | Paper | Risk | 综合 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | I1 RankLane-Combine | 4 | 4 | 5 | 5 | 2 | 5 | 4 | 3 | **26** |
| 2 | I2 RouteCloak | 5 | 4 | 4 | 5 | 2 | 3 | 5 | 5 | **23** |
| 3 | I3 ResumeSet | 4 | 4 | 5 | 3 | 3 | 4 | 4 | 4 | **23** |
| 4 | I4 ScaleBridge | 4 | 3 | 4 | 5 | 3 | 5 | 3 | 4 | **23** |
| 5 | I5 QualityDebt-SlowRank | 2 | 4 | 4 | 5 | 2 | 3 | 3 | 5 | **18** |
| 6 | I7 JouleBatch | 2 | 4 | 5 | 4 | 4 | 4 | 2 | 3 | **22**† |
| 7 | I8 Cancellable Spec-MoE | 2 | 4 | 3 | 5 | 1 | 3 | 3 | 5 | **16** |
| 8 | I6 RouteShare-VTC | 3 | 3 | 5 | 3 | 3 | 5 | 2 | 5 | **19**† |
| 9 | I9 Shadow-Price Placement | 1 | 4 | 2 | 5 | 1 | 2 | 2 | 5 | **12** |
| 10 | I10 Sensitivity Cache/Quant | 1 | 3 | 5 | 2 | 4 | 4 | 2 | 4 | **17** |

† I6/I7 的机械分数不低，但被“oracle <5% / prior-art 覆盖”硬门淘汰；硬门优先于加权总分。这正是不用总分替代 reviewer judgment 的原因。

## 6. 最终 Top 3 与为什么优于其余方向

### Top 1：RankLane-Combine

优点是已有跨模型、matched-byte 的本地质量结构证据，不依赖预测模型；单卡可在 1–3 天内给出 codec break-even 与无限带宽 oracle；成功后可自然落进 8×A100 的真实 combine path。它优于 ScaleBridge，因为能产生直接 runtime mechanism；优于 SlowRank，因为 prior-art 边界更干净。

最大的反面是 NVLink/NVSwitch 上 Amdahl 上界可能 <5%。因此它排第一并不表示“最可能发表”，而是 **最快获得高信息量 go/no-go**。

### Top 2：RouteCloak

它发现的是被优化文献较少处理的变量——execution metadata privacy，而非再调 load/latency；攻击论文已经证明问题可能真实，系统防御仍有空间。它优于 placement/quant/speculation，是因为 novelty 轴更独立。风险是 threat model：若真实 observer 不能恢复 route，整个方向立即结束。

### Top 3：ResumeSet

它把“暂停请求的 KV 管理”和“MoE expert working set”两个割裂问题联合起来，动作发生在 pause/resume 边界，控制开销不在 token critical path；最小 oracle 实验便宜。它优于 JouleBatch/SlowRank，因为不需要构造持续拥塞，也没有被 2026 MoE 调度论文直接覆盖。弱点是 8×A100 总 HBM 可能消除 expert cache pressure，因此论文故事必须同时覆盖 resource-constrained 与多租户大模型场景。

## 7. Top 3 的 1–3 天 go/no-go 实验

### I1：RankLane-Combine（预计 2–3 天）

1. 从现有 OLMoE/LLM-jp trace 生成固定双 lane buffers，真实计 scale/header/padding/alignment。
2. 实现/复用单 kernel pack+unpack；测 rows×hidden×link-rate surface，与 uniform INT8/FP8 和 gate-threshold mask 对照。
3. 同时做三个 oracle：combine=0、bytes-only缩放、zero-overhead quality-optimal lane。

**Go：** 在至少一个代表性 A100 链路带宽区间，codec 后 combine 毛收益 ≥20%，预测端到端 oracle ≥5%，固定 lane 捕获 ≥70% zero-overhead oracle，且两模型质量过门。  
**No-go：** 实际 200–400Gbps/NVLink 区域全部净负，或端到端 oracle <5%。

### I2：RouteCloak（预计 1–3 天）

1. 冻结一个 route-only attribute/text reconstruction attacker，严格按 prompt/domain 分 train/calibration/test。
2. 在 5090 上复现至少一种攻击者现实可见的 footprint；若权限不允许，仅 instrumentation 结果不能开 systems-security 主线。
3. 跑 no-defense、static bucket、full padding 三点 Pareto。

**Go：** 真实 observer 的 held-out advantage 明显高于 chance；static bucket 在 ≤10% latency/energy 下仍与 full-trace oracle 有 ≥20% leakage gap。  
**No-go：** observer 无法恢复 route，或 static bucket 已消除 ≥80% advantage。

### I3：ResumeSet（预计 1–2 天）

1. 用现有 route capture 生成 500–5k 个 pause/resume event；先不写 scheduler。
2. 比较恢复前后 route-set overlap、global LFU、last-W top-m、Belady 与 joint KV/expert oracle。
3. 用真实 pinned H2D LUT把 miss转成恢复 stall，并做 10/25/50% cache cap sensitivity。

**Go：** 在至少两个模型/trace 中，Belady 相对 KV-only+LFU 的 resume P95 headroom ≥15%，且 last-W ResumeSet 捕获 50%–80% oracle、对 non-paused victim 的预测损失 ≤3%。  
**No-go：** last-window不胜 global LFU，expert miss stall <5%，或只有 10% 极端 cap 才有空间。

## 8. 最可能早期淘汰、以及“看似新颖但已被覆盖”的方向

最可能早期淘汰的是 **RouteCloak**：不是因为概念弱，而是实际 GPU/EP observer 的可达性可能不成立。其次是 RankLane-Combine：在 8×A100 NVSwitch 上 combine fraction 可能太低，oracle 会直接判死。

看似新颖但大概率已覆盖：

- “慢 expert 就 drop/reroute/降精度”：Capacity-Aware Inference 与 ReaLB 已直接覆盖。
- “speculative tree 只加载必要 experts”：SP-MoE、MoE-Spec、MoE-SpAc 已覆盖。
- “动态复制热门 experts并考虑迁移成本”：Gimbal、MoEless、Mixture-of-Experts Serving 已覆盖。
- “动态 expert precision/cache”：HOBBIT、DynaExq、DuoServe-MoE、ReMoE 已覆盖。
- “异步收集 expert tokens提高 batch”：AMoE、ASAP、ExpertPlex 已覆盖。
- “receiver 根据 queue/带宽发 credit”：网络侧 SIRD/Pyrrha 与 MoE 侧 Gimbal 已让这一空间非常狭窄；本地 FJRC necessity test 也未过。

## 9. 对 receiver-awareness 的客观结论

receiver-awareness 并非完全无效，而是当前可辩护空间被压缩成三个前提同时成立的窄问题：真实 temporal incast 存在；receiver 额外信息会改变早期动作；这种 action flip 能胜过 request-FCFS/EDF/SRPT/Gimbal-like pressure，并转化为端到端 P99/goodput。仓库现有结果只证明 route identity 有 many-to-one 结构，未证明物理时间重叠；corrected FJRC 又没有跨模型 deadline headroom。因此它不应进入 Top 3。

若未来 8×A100 timed trace 自然出现高比例 receiver busy periods，可把它作为 measurement 结果重新评估；在此之前，不再增加 queue feature、shadow price、RL 或复杂 credit controller。

## 10. 建议的执行顺序

1. **并行思维、串行资源：** 先做 I1 codec oracle 与 I2 observer Gate -1；二者互不依赖，且都能在 3 天内判死。
2. 若 I1 过门，优先拿 8×A100 做真实 combine fraction/collective；不要先写完整在线策略。
3. 若 I2 过门，再冻结 threat model 与 static-padding strongest baseline；否则立即结束 security pivot。
4. I3 只先跑 trace/Belady，不做复杂 joint scheduler；有 ≥15% resume headroom 后再实现。
5. I4 ScaleBridge 作为三条线共用的测量基础；只有简单 roofline 明显失败时再考虑独立论文。

最终决策标准很简单：**Top 3 中任何一项的 oracle 端到端空间低于 5%，或简单 baseline 已捕获 80% 以上，就立即停止。**

## 11. 主要一手参考文献（按主题）

- MoE communication/runtime：[Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin)、[Comet](https://arxiv.org/abs/2502.19811)、[FlashDMoE](https://arxiv.org/abs/2506.04667)、[SwiftEP](https://www.usenix.org/conference/nsdi26/presentation/li-xingyi)、[MixServe](https://arxiv.org/abs/2601.08800)。
- Serving/scheduling：[AMoE](https://arxiv.org/abs/2505.08944)、[Gimbal](https://arxiv.org/abs/2606.15177)、[ASAP](https://arxiv.org/abs/2606.22541)、[ExpertPlex](https://arxiv.org/abs/2607.18002)、[Mixture-of-Experts Serving](https://arxiv.org/abs/2607.17880)。
- Cache/offload/precision：[Fiddler](https://arxiv.org/abs/2402.07033)、[ProMoE](https://arxiv.org/abs/2410.22134)、[HOBBIT](https://arxiv.org/abs/2411.01433)、[DuoServe-MoE](https://arxiv.org/abs/2509.07379)、[DynaExq](https://arxiv.org/abs/2511.15015)、[ReMoE](https://arxiv.org/abs/2605.27081)。
- Speculative MoE：[SP-MoE](https://arxiv.org/abs/2510.10302)、[MoE-Spec](https://arxiv.org/abs/2602.16052)、[MoE-SpAc](https://arxiv.org/abs/2603.09983)、[SpecMoE](https://arxiv.org/abs/2604.10152)。
- Reliability/security：[Tarragon](https://arxiv.org/abs/2601.01310)、[EEP](https://arxiv.org/abs/2605.10670)、[LUMEN](https://arxiv.org/abs/2606.17787)、[MoEcho](https://arxiv.org/abs/2508.15036)、[Expert Selections Reveal Text](https://arxiv.org/abs/2602.04105)、[CryptoMoE](https://arxiv.org/abs/2511.01197)。

需要进一步核实：部分 2026 arXiv 工作尚未经过正式同行评审；ExpertPlex、Mixture-of-Experts Serving 等在本报告日期前仅发布数日。立项前应下载全文与代码，逐项核对实现边界，而不能只依据摘要声称 novelty。
