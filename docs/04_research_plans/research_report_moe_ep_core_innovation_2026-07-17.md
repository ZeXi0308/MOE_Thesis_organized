# MoE EP 唯一核心创新严格研究报告：从精度策略转向可证伪的 Wire-Semantics Differential Profiling

## Executive Summary

[Observed] 2026-07-17 的全新 sealed holdout 已在 OLMoE top-8 与 LLM-jp top-16 上同时杀死 CreditReduce：碰撞机会充足，但 clean early-BF16 不违反质量预算，uniform early-FP8 又以显著更低逻辑字节通过同一质量门。[Inferred] 因此不能再把 rank-aware、QuotaEP-H、receiver-aware、CreditReduce 或其他 mixed-precision 变体包装为默认主创新；当前真正反复出现、且尚未被解决的系统问题，是逻辑 routed pairs、backend records、application-frame bytes、transport/NIC bytes 与 exposed completion time 被混为一谈。[Inferred] 本报告选择唯一主方案 **EP-WireScope**：对相同 MoE route/expert-output trace，在多个 backend-native dispatch/combine contract 下进行语义差分重放和分层计数，自动生成“某个优化 claim 在哪个证据层成立、在哪个层失效”的可复核证书与 break-even envelope。[Hypothesis] 它的现实投稿上限是高质量 ML Systems workshop 或 CCF-C；只有在至少两个真实 backend 上发现可复现的语义/性能反例，并用小型自动 fallback 证明实践价值时，才有更高系统论文潜力。

## A. 当前 Idea 的真实技术核心，以及最大的新颖性缺口

### A.1 已成立的 observation

[Observed] 两个模型、两种 placement 的 grouped-owner surrogate 显示 local grouping 的逻辑机会真实存在：OLMoE routed-pair/grouped-vector 约 1.455–1.475，LLM-jp 约 2.106–2.118。

[Observed] output-aware late-binding signal 在旧 reused-data grouped 实验中稳定优于 rank/gate/random，但四组 mixed treatment 均劣于 uniform FP8；peer/tile fixed quota 还有 4.2%–22.3% KL tax。因此它只证明“output-aware signal 更有信息”，没有证明 mixed precision 是 Pareto winner。

[Observed] CreditReduce 的干净 P0-1 使用两个模型各 64 篇全新文章。OLMoE 的 `p_eligible=99.01%`、`rho_credit=74.66%`；LLM-jp 分别为 100% 和 87.50%。但 clean early-BF16 在两个模型上均为 `NONINFERIOR`，uniform early-FP8 也均为 `NONINFERIOR` 且 accounted bytes 约为 PD-Full 的四分之一。CreditReduce 按预注册规则正式失败。

[Observed] 当前所有正负实验仍停留在 Torch/CPU 的 numerical 与 logical-payload 层；没有 native FP8/FP4 parity、actual wire、GPU kernel completion、RDMA、TPOT 或 P99 证据。

### A.2 当前最强 inference

[Inferred] 当前最稳的 baseline 不是某个复杂 selector，而是 backend-native 路径上的 uniform FP8（若 combine 路径原生支持）或公开 benchmark 所采用的 FP8 dispatch + BF16 combine。

[Inferred] 工作区多次方向反转的共同原因不是缺少 selector，而是缺少一个统一、可执行的“证据层映射”：surrogate 把 routed pair 当 wire record、把 hidden-vector payload 当 frame、把 frame 当 NIC bytes、再把 bytes/BW 当 completion。这个缺口比再发明一种精度策略更基本。

[Inferred] DeepEP、NCCL EP、MORI 和 TensorRT-LLM one-sided 的数据布局、handle、同步、LL/HT 与 dtype contract 不同；同一逻辑优化可能在一个 backend 中已被 rank-major 去重或 local aggregation 吸收，在另一个 backend 中才有真实增量。

### A.3 尚未验证的 system claim

[Hypothesis] 跨 backend 的语义差分能揭示目前论文/模拟常见且可复现的三类错误：wire unit 假设错误、reserved/application/NIC bytes 混算、overlap 后 logical saving 不暴露。

[Hypothesis] 自动生成的 backend-specific envelope 可以比静态“总是启用 FP8/某策略”减少负收益配置，并使 operator completion 或 P99 不退化。

[Hypothesis] 这些发现是否足以构成 CCF-C 论文，取决于真实 backend 上能否复现至少两个非平凡反例，而不是工具本身代码量。

### A.4 已经失败的旧叙事

[Observed] PLTB 是 format-dependent historical baseline；fixed-rank R-layout 不是质量 Pareto winner；additive-KL MILP 被端到端结果否定；Graceful、receiver-aware 与 QTree 没有真实 queue/deadline/topology system 证据；QuotaEP-H mixed treatment 未击败 uniform FP8；CreditReduce 已按双模型 sealed holdout 判死。

### A.5 最大新颖性缺口

[Inferred] EP-WireScope 不能声称发明 tracing、profiling、differential testing 或 auto-tuning。可辩护的新颖性只能是它们在 MoE EP 中的特定交集：以 route handle 和 backend contract 为语义锚点，对 dispatch/combine 的同一 counterfactual trace 同时重建 logical contributions、deduplicated rank/domain records、application frames、transport counters 与 exposed completion，并自动拒绝跨层 claim。

## B. 最强的 3 个候选创新核心

| 候选 | 一句话定义 | 核心机制与时间窗口 | 本质差异 | 最低资源 | 致命风险 / 可证伪实验 | 潜力 |
|---|---|---|---|---|---|---|
| **EP-WireScope** | [Inferred] 对同一 MoE trace 做 backend-contract-aware 语义差分，生成 claim-validity 与 break-even 证书 | offline 捕获 route/output；运行时仅做可选计数与时间戳；dispatch/combine 后对齐 handle、record、frame、counter、event | 不是新 codec，而是把“逻辑优化是否落到真实 wire/critical path”变成可执行验证问题 | Mac 完成 reference；1×GPU 做 kernel replay；2–4 GPU 做第一份 actual counter | [Hypothesis] 若 reference 不能预测 native record/frame，或所有 backend 均无显著 claim gap，则价值不足 | **7/10 条件性** |
| **Composable-FP8 hierarchical partial** | [Hypothesis] 用共享可组合 scale 让 hierarchy 中的 FP8 partial 少做 DQ/RQ | local subtotal 后编码；receiver 一次反量化/FP32累加 | 不分配不同 bit，尝试优化 uniform FP8 聚合本身 | Mac rate–distortion；1×H100 kernel | EQuARX、ZCCL、FP8-Flow/Practical FP4 高度相邻；若相对 optimized FP8 kernel <10% 或质量更差即死 | **5/10，高碰撞** |
| **PhaseFence** | [Hypothesis] 用离线实测的 exposed-cost envelope，在 LL/decode 与 HT/prefill 间选择 native protocol/representation | phase/batch/topology 进入 operator 前 O(1) 查表 | 只选择 backend-native contract，不发明预测器 | 至少1×GPU；可信结果需多GPU | 很像 AutoCCL/普通 autotuning；若 oracle gap <5% 或 native heuristic 已覆盖即死 | **4.5/10** |

## C. 唯一主方案：EP-WireScope

[Inferred] 选择 EP-WireScope，不选择新的 mixed precision。它最符合硕士资源约束：Mac 可以完成 semantic oracle、artifact schema、counterfactual replay 和污染检查；单 GPU 可验证 pack/codec/reduction 语义；多 GPU 只用于把关键层连接到 actual counters，不必一开始实现全新 collective。

[Inferred] 它不是 DeepEP/NCCL/MORI 的简单包装，因为核心产物不是又一个 profiler，而是一个 backend-independent、route-handle-anchored 的 differential contract：同一请求、同一路由、同一 expert output，在不同 backend/协议下必须能追溯到五层证据，并对不合法的跨层结论 fail closed。

[Inferred] Composable-FP8 降为受控 case study：其 novelty 与 EQuARX、ZCCL、Practical FP4、FP8-centric dataflow 太接近，且需要 GPU 才能判定。PhaseFence 降为 EP-WireScope 的演示应用，不单列贡献。

## D. 主方案完整设计

### D.1 系统模型与真实数据流

对请求 `r`、层 `l`、token `t`，记录 home rank、top-k experts、expert owner、source domain、route slot、gate weight、phase 与 backend handle ordinal。定义五个证据层：

\[
L_0=\text{routed contributions},\quad
L_1=\text{backend records},\quad
L_2=\text{application frames},\quad
L_3=\text{transport/NIC bytes},\quad
L_4=\text{exposed completion}.
\]

[Observed] TensorRT-LLM one-sided dispatch 会按 target rank 去重，同一 token 命中该 rank 的多个 experts 只发送一次；combine 以 pull 方式从 symmetric memory 读取并归约。[Observed] DeepEP 由 `EPHandle` 连接 dispatch 与 combine，公开 benchmark 是 FP8 dispatch + BF16 combine。[Observed] NCCL EP 区分小 token 的 LL 与大批量 HT；HT 会做 NVLink-domain aggregation，但仅凭摘要不能断言数学 partial reduction。[Observed] MORI README 也主要公开 FP8 dispatch + BF16 combine。EP-WireScope 因而不能假设统一 wire unit。

### D.2 状态、wire unit 与决策窗口

Sender 保存 route ordinal、token home、owner/domain 与 payload dtype；receiver 保存预期 record ordinal、ready epoch 和 reduction order；network 层只采集 message/QP/NIC counter，不推断语义。wire unit 由 backend adapter 明确定义为 expanded expert record、rank-deduplicated token、domain partial 或 raw frame，未知时标为 `UNRESOLVED`。

EP-WireScope 的核心决策发生在实验分析阶段，不进入每 token 性能关键路径。可选 PhaseFence 仅在 operator 发起前按冻结表选择 native LL/HT/FP8/BF16 path。

### D.3 可观测信号与决策变量

信号包括 route handle、buffer offsets、actual send length、message count、CUDA event、NIC counters、SM/HBM counters和 request timestamps。决策变量是证据映射 `M_b(L_i→L_{i+1})` 与 enable bit `z[bucket,backend,path]`，不是每 vector 精度。

优化目标：

\[
\min_z\;T_{exposed}(z)
\]

subject to semantic parity、质量非劣、counter coverage、P99 no-regression 和 measurement overhead 上限。

### D.4 Offline 与 online 阶段

Offline：冻结 request manifest、route trace、expert placement、backend revision；用纯 Python oracle 生成 `L0/L1`；由 adapter 解析 backend handle/layout 生成预期 `L2`；将 GPU/NIC 日志关联到 `L3/L4`；输出差分和 claim certificate。

Online：默认只采样少量 requests，记录固定大小事件；不在线训练、不预测未来拥塞。PhaseFence 使用冻结 LUT，异常时回退 backend native default。

### D.5 伪代码

```text
for request in frozen_trace:
    logical = reference_route_semantics(request)
    for backend_path in registered_paths:
        expected = adapter.lower(logical, placement, phase)
        observed = run_or_import_native_trace(backend_path, request)
        diff = compare(expected.records, observed.frames, observed.counters)
        exposed = causal_event_analysis(observed.events)
        certificate = classify_claim(logical, expected, observed, exposed)
        emit(request, backend_path, diff, certificate)

for bucket in deployment_profile:
    choose only among semantically valid native paths
    enable path iff CI(T_exposed(path)-T_default) < -epsilon
    otherwise fallback native default
```

### D.6 Buffer layout 与 integration

统一 artifact 不是统一通信 buffer：

```text
TraceHeader: version | backend_revision | request_id | phase | epoch
RouteTable: token | home | expert | owner | domain | slot | weight_hash
RecordTable: logical_id | backend_record_id | dtype | payload_len | offset
FrameTable: peer | message_id | app_bytes | padding | ready_event
CounterTable: nic_tx/rx | cuda_start/end | overlap_parent | completion
ClaimCertificate: VALID_L0...VALID_L4 | UNRESOLVED | FAIL_REASON
```

DeepEP adapter 挂接 `EPHandle` 与 `EventOverlap`；NCCL EP adapter 读取 LL/HT handle/device events；TensorRT adapter 读取 rank-major slot、epoch flag 和 symmetric buffer offsets；MORI adapter读取对应 dispatch/combine trace。若无法取得 NIC counter，`L3` 必须保持 `NOT_MEASURED`。

### D.7 复杂度与 break-even

Reference 复杂度为 `O(R·L·T·K)`，空间可流式降为 `O(TK+F)`；native tracing 增加 `O(E)` 固定事件记录，禁止逐元素日志。通信复杂度不改变 backend 数据路径。

PhaseFence 的 break-even：

\[
T_{profile\ amortized}+T_{lookup}+T_{instrument}
< T_{default}-T_{selected}.
\]

若 instrumentation 使 operator median 增加超过2%，或预测 `L2` 与真实 application frame 误差超过1%，主方案失败。Fallback 是关闭 instrumentation 和 PhaseFence，使用原生 backend。

## E. 三条可独立成立的贡献

第一，[Inferred] **EP evidence-level taxonomy and contract**：observation 是本地多个方向因证据层混淆反复失效；mechanism 是 L0–L4 分层、fail-closed claim certificate；evidence 是跨 backend 的语义对齐与反例集。

第二，[Hypothesis] **Route-handle-anchored differential oracle**：observation 是相同 top-k trace 在 rank-major、expanded、hierarchical 路径中对应不同 wire records；mechanism 是以 handle ordinal 将 counterfactual reference 与 native frame/counter对齐；evidence 是 deterministic replay、record/frame parity 与注入错误检测。

第三，[Hypothesis] **Measured break-even envelope**：observation 是 logical-byte gain 只有暴露在 critical path 才有系统价值；mechanism 是基于 causal events 的 exposed-time envelope和 fail-safe native-path selection；evidence 是单GPU、多GPU与两节点逐级测量。

## F. 核心可证伪假设

| 假设 | 直接杀死结果 | 仅缩小范围 |
|---|---|---|
| H1 backend adapter 能恢复真实 record/frame | 两个目标 backend 任一长期无法使 frame bytes 误差≤1% | 只能支持一个 backend，则不声称通用性 |
| H2 现有 logical claim 存在可复现跨层失真 | 所有 case 的 logical ranking 与 L4 ranking一致且无新反例 | 只在LL或HT出现，则限定phase |
| H3 tracing overhead足够低 | operator median overhead>2%或P99>5% | full tracing太贵但sampling可行 |
| H4 envelope可改善决策 | 相对native heuristic E2E收益<3%或CI含0 | 只保留verification，不保留selector |
| H5 跨backend有论文价值 | 发现仅是已知padding/计数常识，无新的错误或边界 | workshop/tool paper，而非CCF-C |

即使 H4 失败，L0–L4 taxonomy、公开 trace schema 和跨 backend negative result 仍可作为硕士论文与 artifact 贡献。

## G. 实验矩阵

模型使用 OLMoE top-8、LLM-jp top-16、Qwen MoE top-4；top-2 作 no-op/control。workload 覆盖 prefill 128/512/2K/8K、decode active sequences 1/8/32/128，balanced/natural/hotspot，contiguous/round-robin/random/coactivation-aware placement。

强基线为 backend native LL/HT、BF16 combine、optimized uniform FP8（后端支持时）、TensorRT rank-major one-sided、MORI/DeepEP公开最佳路径。所有策略固定同一 request、route、expert output、buffer capacity和arrival trace。

matched-quality：同模型输出语义和预注册 NLL/KL margin；matched-wire：分别报告 logical payload、record bytes、application frames、NIC counters和message count；matched-system：同一 backend revision、streams、warm-up、CUDA graph、QP和频率。

指标包括 record/frame parity、actual bytes、operator P50/P95/P99、exposed completion、TTFT/TPOT/TBT、throughput/goodput、SM/HBM/NIC利用率和J/token。统计单位为document/request；质量和系统使用paired cluster bootstrap 10,000次；主比较Holm校正。

Mac 可完成 L0/L1 oracle、manifest、route/placement sweep、已有 artifact 回放、故障注入和逻辑差分，不能支持actual wire或latency。单GPU可完成adapter正确性、codec/pack/reduction replay和tracing overhead，不能支持inter-GPU通信。单机多GPU可支持NVLink/PCIe frame、completion与overlap，不能支持RDMA。两节点至少2×2 GPU+RDMA才可支持L3 actual NIC bytes；2×4 H100/H200、约48–72 GPU-hours可形成可信系统闭环。没有两节点时不得写RDMA、TPOT/P99系统优化。

## H. Prior-art collision matrix

| 工作 | 真实机制 | 重叠 | EP-WireScope差异 | 风险 |
|---|---|---|---|---|
| DeepEP | EP dispatch/combine、handle、async event、FP8 dispatch/BF16 combine benchmark | 提供trace对象和native baseline | 不替代backend；验证其contract到wire/critical path的映射 | 若官方已有完整counter/certificate工具，novelty下降 |
| NCCL EP | LL direct mesh、HT domain aggregation、统一API | phase/path semantics | 对相同route跨LL/HT做分层差分，不发明LL/HT | 需正文/代码确认record语义 |
| TensorRT one-sided | symmetric memory、rank-major去重、push dispatch/pull combine | exact route coalescing和buffer trace | 把其作为语义不同的adapter和反例来源 | 只做wrapper则无贡献 |
| MORI-EP | intra/inter-node dispatch/combine，公开FP8 dispatch/BF16 combine | backend baseline | AMD/NVIDIA跨backend contract comparison | API tracing可访问性未知 |
| Practical FP4 MoE | FP4 EP通信与直接FP8↔FP4转换 | codec case study | EP-WireScope不认领codec | 若报告退化为codec benchmark则失败 |
| ZCCL / EQuARX | 压缩collective、量化AllReduce与pipeline | evidence层和break-even测量 | 面向稀疏EP route/handle和非对称dispatch/combine | taxonomy必须体现MoE特异性 |
| AutoCCL/一般autotuning | 选择collective/kernel参数 | PhaseFence重叠 | 主贡献不是搜索，而是semantic certificate | PhaseFence不能单独算创新 |
| Nsight/NCCL profiler | kernel/network tracing | instrumentation重叠 | route-handle到logical claim的语义对齐 | 若只做可视化，论文价值不足 |

## I. 最关键的前置实验（按信息增益）

第一，**artifact-only semantic replay**。Mac上将旧 grouped、CreditReduce 和R-layout结果统一映射到L0/L1，注入expanded/rank-dedup/domain-partial三种contract。成功标准是自动复现已知混算错误并生成不同certificate；若工具只重述人工结论则终止。

第二，**TensorRT rank-major或DeepEP单GPU adapter dry-run**。最低1×支持CUDA的GPU，约8–12 GPU-hours；验证route ordinal、buffer slot和frame长度。若adapter无法做到record/frame误差≤1%，主方案终止。

第三，**单机多GPU counter closure**。2–4 GPU、约16–24 GPU-hours，固定相同trace比较native BF16/FP8、rank-major与expanded control。成功是L2与实际send/copy counters闭合，tracing overhead≤2%；失败则降为离线工具。

第四，**LL/HT counterfactual inversion test**。在至少两个batch bucket中寻找logical bytes更少但exposed completion更慢的配置；若不存在反例，PhaseFence删除，但verification主线仍可存活。

第五，**两节点最小RDMA closure**。2×2 GPU+RDMA、24–48 GPU-hours；L2/L3误差≤3%，并报告overlap前后ranking。失败则不声称actual wire或系统优化。

## J. 推荐标题、摘要与完整叙事

推荐标题：**EP-WireScope: Claim-Safe Differential Profiling for Mixture-of-Experts Communication**。中文：**EP-WireScope：面向MoE通信优化的分层语义差分与安全证据验证**。

摘要：

[Observed] MoE通信研究常以路由对数或张量载荷估计收益，但真实后端会采用rank去重、层级聚合、对称内存、固定buffer与异步重叠，使逻辑字节无法直接推出实际通信或服务延迟。我们提出EP-WireScope，[Hypothesis] 以dispatch/combine route handle为语义锚点，将同一请求追踪到logical contribution、backend record、application frame、transport counter和exposed completion五层，并对跨层主张生成fail-closed证书。我们计划在DeepEP、NCCL EP、TensorRT one-sided与MORI路径上进行counterfactual replay和差分验证，量化何时逻辑优化被后端吸收、被padding/metadata抵消或被overlap隐藏，并构建只选择语义有效native路径的保守envelope。

Problem：MoE优化 claim 与backend wire现实脱节。Observation：本项目的多个复杂机制均在更严格的数据、baseline或语义审计下失败。Mechanism：L0–L4分层和route-handle differential oracle。Implementation contract：backend adapter、统一artifact、counter/event closure与fail-closed certificate。Evaluation：先Mac复现错误，再单GPU验证adapter，最后用多GPU/RDMA闭合actual bytes和completion。Limitation：它首先是验证/测量系统，不保证发现新的加速机制；没有多GPU时不能声称服务加速。

## K. 最终判决

[Inferred] 当前不应再宣称已有 CCF-B/C 级优化机制。EP-WireScope 当前创新成熟度约 **4/10**：问题真实、contract清楚、可充分复用已有负结果，但还没有任何native adapter或GPU counter closure。

[Inferred] 作为硕士毕业论文，它比继续追逐第六个mixed-precision selector更稳健：即使PhaseFence没有收益，仍能形成严谨的backend semantics、差分方法、公开artifact和negative-result研究。投稿上限：Mac-only 为硕士论文/negative-result workshop；单GPU+两个backend adapter为高质量ML Systems workshop；单机多GPU并发现多个非平凡反例，有CCF-C可能；两节点闭环并证明工具能避免真实P99回退，才有更高潜力。当前冲CCF-B概率低。

最低资源：Mac完成P0；1×H100/H200约8–12小时完成adapter与overhead生死；2–4 GPU约16–24小时完成L2 closure；可信RDMA claim至少2节点×2GPU，建议2×4 GPU，48–72 GPU-hours。

完全没有多GPU时，应把claim收缩为“MoE EP logical/record semantics differential verifier + 已有artifact上的错误复现”，不得写actual wire、completion、TTFT/TPOT/P99或backend speedup。

接下来两周：第1–3天冻结L0–L4 schema并导入三组历史artifact；第4–6天实现expanded/rank-dedup/domain-partial reference adapters和故障注入；第7天做硬Gate——若不能自动复现至少两个已知语义错误则停止；第8–10天争取一张GPU实现TensorRT或DeepEP adapter dry-run；第11–12天测tracing overhead与frame closure；第13–14天按PASS/FAIL/NOT TESTED写判决并向导师汇报。不要同时恢复QuotaEP-H、CreditReduce或receiver-aware。

## References

1. [DeepEP: An Efficient Expert-Parallel Communication Library](https://github.com/deepseek-ai/DeepEP)
2. [NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL](https://arxiv.org/abs/2603.13606)
3. [MORI: Modular RDMA Interface](https://github.com/ROCm/mori)
4. [TensorRT-LLM: Optimizing MoE Communication with One-Sided AlltoAll Over NVLink](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.html)
5. [Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs](https://arxiv.org/abs/2603.02731)
6. [ZCCL: Improving Collective Communication with Error-Bounded Lossy Compression](https://arxiv.org/abs/2502.18554)
7. [EQuARX: Efficient Quantized AllReduce in XLA](https://arxiv.org/abs/2506.17615)
8. [Occult: Optimizing Collaborative Communications across Experts](https://www.hanruiwang.com/projects/occult)
9. [Communication-Aware Placement and Pruning for Efficient MoE Inference](https://arxiv.org/abs/2607.05116)
10. [MSCCL++: Rethinking GPU Communication Abstractions](https://arxiv.org/abs/2504.09014)
