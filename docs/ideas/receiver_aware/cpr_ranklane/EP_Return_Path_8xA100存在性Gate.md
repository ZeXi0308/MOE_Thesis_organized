# EP Return-Path 8×A100 存在性 Gate（2026-07-25）

> **专项协议，不是当前总入口。** 本文保留 optimized 8×A100 EP return-path existence Gate；它是硬件条件分支。当前单卡执行顺序见[当前研究状态](../../../current/README.md)。

> 目标：先证明自然系统问题，再决定是否允许机制实现。  
> 资源边界：1×RTX 5090 32GB；8×A100 的显存、PCIe/SXM、NVLink/NVSwitch、
> NCCL/DeepEP/vLLM 状态尚未确认。  
> 证据优先级：最新 formal/sealed 结果 > 当前代码与产物 > 旧设计说明 > 假设。

## 0. 直接裁决

- **存在值得进入实验的科学现象吗？** 有且只有一个首选：在优化后的真实 EP serving 中，
  `expert-ready → return all-to-all → receiver unpack/combine` 仍有多少 **exposed critical-path**。
  这是现象级 profiling，不是 RankLane 已成立。
- **存在值得进入机制实现的候选吗？** **没有。**
- **裁决：** `NO_CANDIDATE：当前候选池没有通过 Gate 0–4、值得进入机制实现的对象。`
- **唯一下一动作：** 若取得 8×A100，先运行无新机制的 `EP Return-Path Existence Gate`；
  若没有 8 卡，不用 5090 的 H2D、逻辑 bytes 或解析带宽模型替代这个 Gate。

当前不允许设计 RankLane codec、Receiver controller、ResumeSet cache manager、RouteCloak
defense 或 Energy-SLO controller。第一笔 GPU 预算只用于问题存在性。

## 1. 最新证据账本

| 类别 | 最新证据 | 最高可信表述 | 不能推出 |
|---|---|---|---|
| 已确认观察 | OLMoE/LLM-jp matched-byte head-vs-tail KL 差异显著；见 `A_rank_tail_fp8/outputs/idea_a_rank_lut_gpu_verify_2026-07-20_{olmoe,llmjp}/report.md` | gate rank 与贡献质量敏感度有关 | 不等于 rank-aware codec 胜 uniform INT8，也不等于 TPOT/P99 收益 |
| 已确认观察 | 5090 上未融合 homogeneous codec：rows 128/512、200–400Gbps 解析链路共 0/8 正区，codec 约 58–61us | 朴素 pack/H2D/unpack 路径不能支持快链路收益 | H2D 不是 NCCL/RDMA；once-fused 仍未实测 |
| 已否定主张 | PhaseMap formal selection 停在 `BLOCKED_UNINFORMATIVE_DEADLINE_GRID`，`holdout_opened=false`；各模型自己的 50% 点仍 `B0=Q=J=R` | closed-pair、work-conserving reorder 中 queue/join 信息无增量动作价值 | 不是否定所有 receiver-awareness |
| 已否定主张 | corrected FJRC：OLMoE miss 绝对下降 1.5625pp，LLM-jp 0pp，双模型门失败 | keyed join bitmap 机制停止 | OLMoE secondary CVaR 不能升级为 physical receiver 结论 |
| 代理证据 | native route identity 中存在 many-to-one fan-in；冻结 64 waves/model；5090 RR-credit 仅 `LOCAL_CLONE` smoke | schema、identity 和守恒路径可运行 | 未证明 temporal incast、NCCL ingress、receiver busy period、TPOT/P99 |
| 真实单卡 inference-time | 5090 上 16 层 LLM-jp 完整 KV decode；本地 MoE block 占 profiled decode 约 82.8%–90.2%，最慢单层仅占各层 median 之和 6.36%–6.58%；粗分解中 expert loop 已占约 74.9%–85.7% | 多 MoE 层成本在完整 inference denominator 中近似均匀累积，8 卡 Gate 必须覆盖全层；主要单卡成本当前来自本地 expert loop | 本地 router/expert/combine 不是 return A2A；不得写成 `p_return`、Receiver congestion 或 serving TPOT/P99；通信仍需单独证明 exposed headroom |
| 真实单卡证据 | RouteShare matched-histogram residual 仅 2.35%–3.06%，强简单模型 held-out `R²=0.9971–0.9986` | 当前 route-coalition cost 对象判死 | clustered/current-route oracle 不能冒充 causal scheduler |
| 真实单卡证据 | RouteShare causal previous-token union reduction：OLMoE 3.08%，LLM-jp 0.285%；scheduler 税吃掉净收益 | RouteShare/VTC 不进入 serving scheduler | 不用 predictor 或 topology simulation 救活 |
| 已否定主张 | ConfidenceGuard v3 sealed decision 为 `NO_GO_PREFILL_RISK_RANKING_FOR_AUDIT_ALLOCATION` | 当前 prefill risk ranking 不能产生所需 audit allocation 增量 | engineering pass 不是 scientific pass |
| 受限 characterization | Energy-SLO fixed full-forward batch 与预量化 FP8 GEMM-core 有硬件效应；23 个 route-row/quantize-once summary 中 15 个无 robust FP8 fast region、2 个 quantize-once no-fast-region | 只能保留 microbenchmark characterization | 不能写 KV decode、arrival/P99、联合 controller 或 Energy-SLO Pareto |
| 会计纠错 | additive 旧 `3.77×` 是重复累加伪影；正确 incremental ratio=1.076，CI=[0.983,1.165] | 可加性未决 | 禁止引用 `3.77×` / `94×` |

### 不得复活

1. PhaseMap closed-pair / FJRC join-bitmap：不得换 deadline、pool 模型、次要指标、
   predictor、bandit 或 MILP 重开。
2. RouteShare/VTC：不得用 clustered oracle、未来 route 或 scheduler-free speedup 重开。
3. dynamic cache/quant/prefetch：不得在 full top-k working-set 饱和、执行快区不存在时把失败组件
   组合成“联合控制”。
4. Energy-SLO：不得把 full-forward batch、pre-cast GEMM 或拍定 idle power 写成 serving joint Pareto。
5. additive：不得再引用作废的 `3.77×` / `94×`。

## 2. 现象级候选池

### P1 — 优化后 EP return path 的 exposed criticality

现象表达：

> [Hypothesis] 在自然 continuous-batching decode / mixed prefill-decode、真实 EP 配置下，
> 第二次 all-to-all 加 receiver unpack/combine 仍占 TPOT/P99 critical path 的至少 10%，
> 删除该 exposed span 存在可测 E2E 上界。

- scientific object：优化后 `expert-ready → combine-complete` 的不可 overlap 时间。
- 自然 workload：真实 serving arrival；不能用单层同步 wave、人工 barrier 或 sleep 制造。
- 用户指标：TPOT P50/P95/P99、TTFT、goodput、tokens/s。
- 现有证据：gate-rank 质量结构、单卡 codec 反证，以及 16 层本地 MoE 成本在完整
  KV decode 中均匀累积；但后者无 EP/return A2A，**Gate 0 仍未通过**。
- 删除式 Oracle：在依赖 DAG 中把 return-path service 置零，重算 request/token completion；
  不能把 profiler span 直接相加。
- 配置消失风险：SwiftEP、Comet、tile-level second-A2A overlap、改变 EP/TP、增加 batch 或
  NVLink/NVSwitch 可能把 exposed span 隐藏掉。
- 5090：只能导出 route shape、质量与 codec 税，不能验证核心现象。
- 8×A100：可以验证单节点 EP/NCCL/NVLink；不能外推跨节点 RDMA。
- 初步结论：**唯一允许进入 Gate 0 的对象；不是机制候选。**

### P2 — pause-specific expert cold-start

现象表达：

> [Hypothesis] 在自然 tool/human interruption、KV 被保留且模型/多租户压力使 expert 被自然驱逐时，
> 恢复前 N token 的 expert fetch 独立占 resume latency 至少 15%。

- scientific object：KV-preserved resume phase 中独立 expert miss/fetch critical path。
- 自然 workload：公开或真实 agent pause trace；模型或多租户工作集自然超过 HBM。
- 用户指标：resume TTFT、前 N token TPOT、全局 goodput/SLO。
- 现有证据：**未验证**；仓库没有自然 pause-induced eviction 结果。
- Oracle：未来 N token experts 已知、零成本保留/预取；KV policy 固定。
- 配置消失风险：模型全驻留、增加 HBM、batch 隐藏 fetch 时上界为零。
- 5090：可做真实 offload，但不能靠人为极小 cache；缺公开 pause trace 与完整 runtime。
- 8×A100：模型全驻留时反而可能归零。
- 初步结论：条件备线；当前筛选资产不足，不能成为唯一下一实验。

### P3 — 普通低权限共租者可见的 MoE route side channel

现象表达：

> [Hypothesis] 在真实共享 serving 隔离下，非管理员共租进程能从允许读取的 GPU/traffic
> 信号把 token/属性识别优势提高至少 20pp 或达到 AUC≥0.75。

- scientific object：部署权限模型下的 route-dependent hardware/traffic leakage。
- 自然 workload：真实并发 tenant、正常 continuous batching 和平台默认隔离。
- 用户指标：攻击 advantage/AUC；后续才是 defense P99/throughput。
- 现有证据：MoEcho 已证明若干 CPU/GPU architectural channels；仓库本地 threat model 未验证。
- Oracle：直接读取 ground-truth route；用于定义最大可恢复信息，不是可部署 attacker。
- 配置消失风险：counter 权限禁用、MIG、进程隔离、自然 batching 可能消除信号。
- 5090：只有平台允许真实共租与低权限 observer 时才可测；独占云 GPU 不成立。
- 8×A100：可测共享/EP footprint，但仍需真实权限边界。
- 初步结论：高风险旁线；MoEcho 使“攻击存在”不再新，剩余 novelty 只能是 exact-output、
  SLO-bounded defense，必须先证明部署 threat。

### P4 — physical temporal receiver incast

现象表达：

> [Hypothesis] 在自然 EP execution 中，至少 10% receiver busy periods 同时包含 ≥2 joins
> 与 ≥2 independent senders，且 busy-period P95 超过单 contribution service P95 的 2 倍。

- scientific object：物理 receiver ingress/return-path queueing，而非 route fan-in。
- 自然 workload：真实 expert-ready timestamps、NCCL/NVLink transport、连续到达。
- 用户指标：先测 exposed busy/join wait；尚无可接受的 SLO 机制 claim。
- 现有证据：route fan-in 结构存在；单卡 smoke 明示
  `ONE_GPU_ALL_RANKS_FOLDED_LOCAL_CLONE_NOT_NETWORK_NOT_INCAST`。
- Oracle：删除 receiver wait；不能使用人工 queue depth。
- 配置消失风险：更强 overlap、collective aggregation、流控和 batch 会消除 busy period。
- 5090：不可验证；至少 4 GPU，EP8 首选 8 GPU。
- 8×A100：可与 P1 共用 instrumentation，但现有 runner 是受 barrier 控制的 microbenchmark，
  不能单独作为 serving Gate 0。
- 初步结论：作为 P1 的机制诊断一并采集；不单独重开 FJRC/PhaseMap。

### P5 — route-conditioned Energy-SLO residual

现象表达：

> [Hypothesis] 在 phase、batch、clock、arrival 与质量固定后，route shape 仍解释至少 5%
> 的 J/token 或 SLO-feasible energy residual，并给动态动作留下 ≥10% Oracle。

- scientific object：MoE route 对能耗的独立增量，而非 generic power/batch control。
- 自然 workload：真实 KV decode、Poisson/MMPP arrival、board energy、P99 SLO。
- 用户指标：J/completed token、TPOT/P99 violation、goodput。
- 现有证据：只有 full-forward 与 GEMM-core characterization；JouleQueue formal path 被 P0 阻断。
- Oracle：在相同完成 token identity 与质量地板下穷举 route-aware action。
- 配置消失风险：phase×batch static table、PALS/GreenLLM/Festina 已捕获主要空间。
- 5090：可测 Gate 0，但当前 route-row dynamic path没有 robust fast region。
- 8×A100：可测通信能耗，但单节点不能代表集群 energy。
- 初步结论：**淘汰为独立候选**；只保留为任何最终系统的测量维度。

## 3. Gate 淘汰过程

| 现象 | Gate 0 | 删除式 Oracle | 最强简单 baseline | Captured Headroom | Prior-art collision | 决策 |
|---|---|---|---|---|---|---|
| P1 exposed return path | 未验证 | 未测；目标 ≥10%，常见 cell 最好 ≥15% | optimized overlap + BF16；single-message fused uniform INT8；最佳 EP/TP | 未验证 | **高**：SwiftEP、Comet、CoCoQuant、最新 tile-level overlap | **保留一次 Gate 0** |
| P2 resume cold-start | 未验证 | 未测；要求独立 ≥15% | KV TTL + last-W expert union + static quota；LRU/LFU | 未验证 | **高**：InferCept、ELDR、HOBBIT、FluxMoE | 条件备线，不先实现 |
| P3 low-privilege leakage | 本地未验证；文献已有特定 channel | full route observation；防御 oracle 为 full padding | 默认隔离/MIG、自然 batching、static bucket | 未验证 | **中高**：MoEcho 已覆盖攻击，route-to-text leakage 已明确 | 仅 threat profiling |
| P4 temporal incast | 仅 structural fan-in；physical fail to test | 未测 | optimized collective/aggregation；RR/FCFS/EDF 仅在存在后比较 | 未验证 | receiver 调度拥挤且本地 FJRC 已失败 | 并入 P1 census，不独立立题 |
| P5 route-energy residual | 未通过 system Gate 0 | 粗估 3%–12%，未闭环 | phase×batch static clock/slack table | 未测，预计高 | **很高**：PALS、GreenLLM、Festina | 淘汰独立机制 |

### Gate 结论

没有任何一项具有已观测的 Oracle 和 `CH<70%` 证据，因此没有对象可以进入机制实现。
P1 只因“可用一次真实 8 卡测量廉价判死”而排第一，不是因为论文主张已经站住。

## 4. Prior-art collision（截至 2026-07-25）

| 工作 | 状态 | 输入信息 | 决策/系统动作 | 目标与硬件 | 对本项目的约束 |
|---|---|---|---|---|---|
| [SwiftEP](https://www.usenix.org/conference/nsdi26/presentation/li-xingyi) | NSDI 2026 | token placement、buffer/transport 状态 | buffer fusion、TMA offload、NVLink/RDMA zero-copy path | 16/32 GPU MoE prefill，服务容量 | P1 必须面对强 fused/zero-copy baseline；朴素 staging 不是 baseline |
| [Comet](https://arxiv.org/abs/2502.19811) | MLSys 2025 | tile dependency、compute/communication workload | fine-grained overlap 与 adaptive assignment | MoE layer/E2E | 不能把 overlap 缺失制造成 compression headroom |
| [Fine-grained second-A2A overlap](https://arxiv.org/abs/2607.19539) | 2026-07-21 preprint，未确认 peer review | expert tile ready signal、SM partition | persistent producer/consumer，segment transfer | 4×A100；报告 71.9%–99.9% second-A2A hiding | 新且直接：P1 必须测 **优化后 residual**，不能测裸 A2A |
| [CoCoQuant](https://openreview.net/forum?id=Bxyc3JZtAB) | ICML 2026 | hardware roofline、relative sensitivity | global MILP bit allocation、graph rewrite、comm/compute fusion | dense + MoE multi-GPU distributed inference | “混合精度通信 + MILP + fusion”已被覆盖；RankLane 只剩 gate-rank 固定 lane 的增量空间 |
| [ELDR](https://arxiv.org/abs/2607.00466) | 2026-07 preprint | prefill expert signature、load、KV-block identity | decode-worker routing + signature cache | 至多 40 GPU；median TPOT 5.9%–13.9% | P2 的 KV-indexed expert signature 不新；只能研究 pause-induced cold-start phase |
| [FluxMoE](https://arxiv.org/abs/2604.02715) | 2026 preprint | KV pressure、expert demand | transient expert paging | vLLM、memory-intensive MoE | generic KV/expert HBM exchange 已碰撞 |
| [InferCept](https://arxiv.org/abs/2402.01869) | 2024 preprint | API interruption、KV waste、queue | pause KV swap/recompute/scheduling | augmented LLM inference | P2 必须固定 KV baseline，只证明 expert 独立增量 |
| [MoEcho](https://arxiv.org/abs/2508.15036) | CCS 2025 | cache/page/TLB/performance-counter signals | CPU/GPU architectural side-channel attacks | MoE LLM/VLM | P3 不能声称首次发现 MoE side channel；必须明确更现实权限和 defense frontier |
| [Expert Selections Reveal Text](https://arxiv.org/abs/2602.04105) | 2026 preprint | expert selection sequence | token reconstruction model | OpenWebText route traces | route 本身含敏感信息已明确；本项目缺口只能是可达 observer 与系统防御 |
| [PALS](https://arxiv.org/abs/2605.21427) | 2026 preprint | power、batch、throughput | power cap + batch feedback controller | vLLM、多 GPU、dense + MoE | generic Energy-SLO controller novelty 已很薄 |
| [Festina](https://arxiv.org/abs/2606.30391) | 2026 preprint | phase、slack、placement、SM、GPU state | placement、batch、SM、operating point、consolidation | TTFT/TBT SLO；shared GPU | P5 只有 route residual 超过强 static/phase baseline 才可能有增量 |

结论只能写“没有发现直接覆盖 gate-rank fixed combine lane 的完全同构实现”；不能写“证明创新”。
CoCoQuant 已覆盖其大部分优化抽象，最新 overlap 工作又可能吃掉物理上界，所以 novelty
和 headroom 都必须由 Gate 0/1 重新建立。

## 5. 唯一候选

当前没有机制候选。条件式命题只能写成：

> [Hypothesis] 当优化后的真实 EP serving 中 return-path 删除式 Oracle 在两个模型的共同自然
> workload cell 均达到 ≥10%、至少一个常见 cell 达到 ≥15%，且 fused uniform INT8/static
> cutoff 的 `CH<80%` 时，gate-rank fixed dual-lane 才可能相对该强 baseline 改善 TPOT/P99，
> 同时满足 matched-quality、相同消息数和 P99 非退化约束。

- `[Observed]`：gate-rank quality asymmetry；朴素 codec 在常用快链路点 0/8 viable。
- `[Inferred]`：gate rank 可能是比 activation magnitude 更稳定的 lane signal。
- `[Hypothesis]`：强 overlap 后仍有 exposed combine；fused dual-lane 在真实 wire path 净正；
  uniform/static baseline 捕获不到 80%。
- `[Not Testable Here]`：5090 单卡不能验证 NCCL all-to-all、receiver ingress 或 serving E2E。

## 6. 最小 Go/No-Go 实验：EP Return-Path Existence Gate

### 6.1 实验只回答什么

1. 优化后 return path 对真实 TPOT/P99 有多少不可 overlap 的贡献？
2. 零 return-path 的删除式 E2E Oracle 是否超过 10%/15%？
3. 现象是否只存在于某个不自然 topology、低负载或裸 baseline？

本轮不比较 RankLane，不写 codec/controller，不做质量 Pareto。

### 6.2 代码入口与当前资产

- **待新增正式入口：**
  `docs/ideas/receiver_aware/cpr_ranklane/experiments/profile_ep_return_path_gate0.py`
- **现有可复用但不能直接出结论：**
  `experiments/ric_clean_v2/run_multirank_rr_census.py` 可复用 identity、message conservation、
  NVTX 与 topology manifest；它当前使用 trial barrier、随机 expert weights、rank-local clock，
  并明确输出 `RAW_TRACE_AWAITING_NSYS_CUPTI_BINDING_NOT_HEADROOM_RESULT`，因此不是 serving Gate。
- **完整 inference denominator 参考：**
  [`../inference_time_5090/`](../inference_time_5090/) 已提供全 MoE 层 KV-decode 计时、逐层 census
  和 observer-tax 会计；可复用其全层覆盖和端到端 denominator 思路，但不得复用其
  单卡 MoE 占比当作 return-path 占比。
- **分析入口：**
  `docs/ideas/receiver_aware/cpr_ranklane/experiments/analyze_ep_return_path_gate0.py`（待新增）；必须从
  Nsight/CUPTI 同一时间轴重建依赖 DAG，不能拼 rank-local wall clock。

若必须先改完整 vLLM/DeepEP runtime 才能打点，则筛选成本超过 2 天，P1 降级；不允许因此先写
RankLane backend。

### 6.3 硬件预检

先记录并冻结：A100 40/80GB、PCIe/SXM、NVLink/NVSwitch、driver/CUDA/PyTorch/NCCL、
vLLM/DeepEP 版本、`nvidia-smi topo -m`、NCCL transport、power/clock。要求 8 个独立 GPU。

- 若只有单卡 5090：`BLOCKED_RESOURCE_NOT_TESTABLE_HERE`。
- 若 NCCL 回退 socket/SHM：该 cell 不得写 NVLink/RDMA claim。
- 单节点 A100 结果只支持 scale-up；不支持跨节点 RDMA。

### 6.4 模型与 workload

- M1：仓库 OLMoE，用于与既有 route/quality identity 闭合；
- M2：一个能在 8×A100 上以 BF16/INT8 正常 EP serving 的 serving-scale open MoE，优先
  Mixtral-8×7B；若环境只支持 LLM-jp，则结果只能算小模型复现，不能满足论文 Gate。
- traces：ShareGPT/chat + 一组长 prompt；只做 serving，不使用 calibration route replay。
- cells：decode-heavy continuous batching 与 mixed prefill/decode；负载为实测最大稳定容量的
  30% 与 70%；输入长度 128/1024，输出 128。
- 每 cell：3 分钟 warmup + 5 分钟 measured，5 个独立 arrival seeds；两个模型、两 workload、
  两 load，共 40 个 measured runs。

arrival seed、prompt split 与 tuning split 分离。第一次运行只用于栈/容量 calibration；正式
prompt IDs 与 arrival seeds 冻结后不得更换。

### 6.5 Baseline、Oracle 与指标

Baseline 必须是当前环境可运行的最强 unmodified backend：

1. 最佳合法 EP/TP 配置；
2. 开启 backend 已有的 overlap/fusion/zero-copy；
3. BF16 单消息语义；不得故意关闭 overlap 制造 P1。

主指标：

- per-request TTFT、per-token TPOT、P95/P99、goodput；
- `expert_ready → first_return_send → receiver_visible → unpack/combine_complete` 同轴时间；
- 上述事件必须覆盖每次正式 forward 的全部 MoE 层；任一层缺失即为 census 不闭合；
- exposed return-path critical-path fraction；
- receiver busy-period/join-wait 仅作 P4 诊断；
- message count、actual payload bytes、transport/topology、GPU utilization。

Oracle：对每个 token/request 的 precedence DAG 将 return-path service time 置零，保留 arrival、
expert compute、依赖和其他 fixed cost，重算 completion。报告 median 和 tail Oracle improvement。
不得使用 `sum(return spans)/E2E` 作为 Amdahl 比例。

统计：以 request 为样本、run/arrival seed 为 cluster，做 paired block bootstrap 5,000 次；
主表先报两个模型独立 95% CI，再报 pooled 仅作辅助。

### 6.6 冻结 PASS / NO-GO

**Gate-0 PASS** 必须同时满足：

1. 两个模型至少一个共同自然 cell 的 zero-return Oracle point estimate ≥10%，95% LCB >5%；
2. 至少一个常见 cell 在两个模型都达到 ≥15%；
3. 不是仅在 30% load、裸 NCCL、关闭 overlap 或异常小/大 payload 出现；
4. timeline conservation、message identity、transport 与 token completion 全闭合。

**NO-GO：**

- 所有共同常见 cells Oracle <5%：永久淘汰 RankLane/return-path receiver 优化；
- 5%–10%：只保留 measurement/工程优化，不进入论文机制；
- 只在慢 PCIe、关闭 overlap、tiny model 或 injected synchronization 成立：资源/配置不匹配；
- 任何 timeline/identity/transport 不闭合：`INVALID/REPEAT SAME PROTOCOL`，不是科学 NO-GO。

**Gate-0 PASS 也不授权写机制。** 下一步才是 Gate 1/2：在相同 stack 上测 fused uniform INT8、
static cutoff、activation-magnitude cutoff、zero-codec Oracle 与 `CH`。只有 `CH<80%` 且
gate-rank 剩余增量存在，才进入 RankLane 机制设计。

预计耗时：硬件和 serving stack 已就绪时 0.5–2 天；环境不就绪时立即报 resource blocker，
不把环境搭建周期开支伪装成 candidate 验证。

当前缺失资产：8×A100 规格与访问；serving-scale MoE 权重；可运行 EP backend；统一
Nsight/CUPTI timeline exporter；正式分析脚本；独立 serving prompt/arrival split。

## 7. 条件式机制设计

**现在不允许设计机制，下一步只能执行存在性实验。**

P1 通过后，也只能先做 Oracle 与 simple baseline；不能直接实现 dual-lane collective。

## 8. Red-Team 拒稿审查

| 最强拒稿理由 | 当前是否成立 | 需要什么证据推翻 |
|---|---|---|
| “CoCoQuant 已做 mixed-precision distributed inference + MILP + fusion，你只是换 gate-rank signal。” | **成立，高风险** | 同 stack、同质量、同消息数下 gate-rank 相对其等价 uniform/hardware-aware baseline 仍有 ≥5% E2E 与 `CH<80%` |
| “第二次 A2A 已能隐藏 71.9%–99.9%，没有 exposed critical path。” | 未在本 workload 直接验证，但外部反证强 | 优化 backend 的真实 serving trace 上 zero-return Oracle ≥10%/15% |
| “收益来自关闭 overlap、弱 BF16 baseline 或两个 collective 对一个 collective。” | 当前尚未排除 | strongest overlap/fusion/one-message baseline；相同 launch/message/layout 税 |
| “A100 没有原生 FP8，你的精度路径与正式硬件不一致。” | **成立** | A100 上真实 INT8/INT4 producer→collective→consumer 路径；或把最终 claim 限定到支持目标格式的硬件并取得对应多卡证据 |
| “OLMoE/LLM-jp 太小，route shape 不能代表 serving-scale MoE。” | **成立** | 至少一个 serving-scale open MoE + 一个独立模型复现；真实 continuous arrival |
| “你把 overlapped profiler spans 相加，Oracle 会计错误。” | 高风险 | precedence-DAG replay 与请求 completion 闭合；counterfactual 对 synthetic fixtures exact |
| “gate rank 的收益被 static cutoff/uniform INT8 捕获。” | 未验证 | 最强 simple baseline `CH<80%`，跨模型 held-out quality/performance 同时成立 |
| “8×A100 单节点不能证明跨节点 receiver congestion/RDMA。” | **成立** | 只写 single-node scale-up claim；若要 RDMA，另做跨节点实验 |
| “RouteCloak 已被 MoEcho 覆盖，ResumeSet 被 ELDR/FluxMoE/InferCept 拼出。” | 基本成立 | 明确新的部署权限/phase，且 Oracle 与 simple baseline 留下独立 E2E gap；否则不立项 |

## 9. 最终执行决策树

```text
P1: optimized EP return-path exposed criticality
├─ 无 8×A100 / 无真实 EP serving stack
│  └─ BLOCKED_RESOURCE；停止，不用 5090 proxy 替代
└─ Gate 0: zero-return precedence-DAG Oracle
   ├─ <5% → STOP：RankLane/return-path receiver 优化永久淘汰
   ├─ 5%–10% → measurement / 工程优化，不做论文机制
   └─ 两模型共同 ≥10%，且常见 cell ≥15%
      └─ Gate 1/2: fused uniform INT8 + static cutoff + activation magnitude
         ├─ simple CH≥80% → STOP / 采用简单 baseline
         └─ CH<80% 且真实 fused codec 净正
            └─ Gate 3: CoCoQuant/SwiftEP/最新 overlap 逐项 collision
               ├─ 只剩换 signal/solver → STOP：novelty 不足
               └─ gate-rank fixed lane 留下独立机制与 E2E gap
                  └─ 才允许冻结机制协议并实现一个机制
```

P1 失败后才考虑 P2 pause-specific cold-start Oracle；P3 只做低成本 threat profiling。
P4 随 P1 采集但不复活 FJRC；P5 维持子指标，不作为主线。

## 10. 自检

1. 先证明问题，再设计机制：是；当前明确禁止机制设计。
2. 代理收益写成 E2E：否；H2D、logical bytes、单卡 smoke 均被限定。
3. 最强简单 baseline：已列 optimized overlap、uniform INT8、static cutoff。
4. 人工 workload：禁止 barrier/sleep/queue-depth 作为 Gate 0。
5. 历史 NO-GO：PhaseMap/FJRC/RouteShare/cache/quant 未复活。
6. preprint：ELDR、FluxMoE、PALS、Festina、最新 overlap 均明确标为 preprint；
   SwiftEP/Comet/CoCoQuant/MoEcho 分别按会议状态列示。
7. 未为了 Top 1 放宽门槛：机制结论为 `NO_CANDIDATE`。
8. 当前能做/不能做：只有 8 卡 P1 profiling；5090 不可替代核心因果。
9. 所有候选未过 Gate 0–4：诚实停止在存在性实验。
