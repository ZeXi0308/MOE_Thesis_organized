# CCF-C 候选研究报告：RouteFidelity-EP

> **2026-07-18 sealed P0-B 状态更新**：该候选已按预注册规则判为 **`KILL_CCFC_MAINLINE`**。OLMoE 与 LLM-jp 均为 0/20 S1-R seeds 达到 5% selected-placement regret，seed median=0、Holm lower=0；S3/S4 packed size 分别为 109.72%/106.62%。低 tau 没有转化为错误配置选择。完整方法、结果与边界见 `RouteFidelity_EP_严谨实验路径与P0B验证_2026-07-18.md`。本文以下内容保留为实验前候选设计，不再代表当前推荐主线。

> 日期：2026-07-18  
> 定位：严格的候选方向筛选，不是已成立的论文结论  
> 证据标签：`[Observed]` 为已测结果，`[Inferred]` 为受证据支持的推断，`[Hypothesis]` 为待证伪命题

## Executive verdict

[Inferred] 当前最值得投入的 CCF-C 候选不是再设计一个 mixed-precision selector，也不是继续扩展 Idea B 的能耗 MILP，而是 **RouteFidelity-EP：面向配置决策保真的、backend-conditioned MoE EP 路由工作负载合成与重放系统**。

一句话核心创新：

> **RouteFidelity-EP 针对不同 EP backend 的 record/layout contract，自动识别保持 placement、buffer 和 protocol 配置排序所需的最小路由充分统计量，再将其编译为 backend-native workload。**

[Observed] Mac 上的 P0-A 只给出了探索性信号：architecture-only uniform route 对 logical receiver P99 的最大低估在 OLMoE/LLM-jp 上分别为 43.0%/13.3%；但在精确保留每层 expert degree 后，差异变为 10.0%/0.8%，且所有 placement regret 均低于 2.5%。因此当前 verdict 是 **EXPLORATORY_SIGNAL**，不是 PASS。

[Hypothesis] 该方向只有在两个语义不同的真实 EP backend 上，找到至少一个可复现的 `>=5%` actual latency/configuration-ranking inversion，并由 RouteFidelity 将 selected-config regret 降至 `<=2%` 时，才具备 CCF-C 潜力。当前创新成熟度约 **4.5/10**，通过全部 gate 后的论文潜力约 **7/10**。

## 1. 为什么不再强行复活 Idea A / Idea B

### 1.1 Idea A 可复用的是路由资产，不是已成立的 selector

[Observed] PLTB、rank/head control、QuotaEP-H、CreditReduce 等实验已经建立了一套不错的 route capture、exactness、sealed split、paired KL/NLL 和 logical accounting 资产；但它们没有证明复杂 mixed precision 能稳定击败 uniform FP8。

[Observed] 新跑的 temporal-residual EP P0 也失败：OLMoE 的 expert revisit rate 为 36.24%、逻辑字节节省 17.05%，但相对同格式同字节的 direct-MXFP4，端到端 KL 反而是 `1.164x`；LLM-jp smoke 中该比值为 `3.182x`。这直接否定了“路由重访就意味着 expert output 可预测”。

[Inferred] Idea A 最值钱的现有资产是两个模型的 token→expert hypergraph、placement sweep 和实验严谨性 harness，它们可直接转化为 RouteFidelity 的 trace corpus 和 reference oracle。

### 1.2 Idea B 的原始能耗置放叙事不足以做 CCF-C

[Inferred] 在固定 EP 部署中，GPU 通常已经处于 active 状态，“增加 expert replica 会打开额外 GPU”不是普遍成立的因果链；`bytes × pJ/bit` 也不能取代 GPU/node 的实测能耗。新工作 [PALS](https://arxiv.org/abs/2605.21427) 已将 power cap 与batch/runtime 联合控制用于 MoE serving，[PROBE](https://arxiv.org/abs/2602.00509) 则已联合预测、replication 和 token assignment。

[Inferred] Idea B 可留作备选的只有 **BarrierJoule-EP**：以减少同步等待能耗，而非以减少通信字节，决定 replica admission。但这条线需要 4–8 GPU 的真实 power trace，并且很可能退化成 latency-only placement，不适合当前资源约束下作为主线。

## 2. 候选比较与唯一选择

| 候选 | 一句话机制 | 现有资产复用 | 最大风险 | 当前判决 |
|---|---|---|---|---|
| **RouteFidelity-EP** | 为每种 backend contract 选择能保持配置排序的最小 route statistics | Idea A 的 route trace、placement、sealed methodology；Idea B 的拓扑/configuration 空间 | 可能只是 MLSynth/Chakra extension；真实 ranking inversion 可能不足 5% | **唯一主候选** |
| BarrierJoule-EP | 按 replica 对 barrier waiting energy 的边际收益分配 HBM/拷贝预算 | Idea B 的 placement/replica 建模 | 必须有真实 power/latency；与 PALS、PROBE、GreenMoE 相邻 | 备选，当前 3.5/10 |
| RouteShape-EP | 按实际 route contraction 在 expanded、rank-dedup、domain-partial lowering 中选择 | Idea A 的 owner grouping trace | layer-level contraction 变化小，易退化为 static mode；NCCL EP/HybridEP 已有相邻路径 | 不足 CCF-C，<=5/10 |

[Inferred] RouteFidelity-EP 胜出的原因不是它已经成立，而是它有清晰的低成本 problem gate，且能把 Idea A 的 negative results 和 EP-WireScope 的 backend semantic oracle 转化为方法资产。

## 3. 核心 observation 与研究问题

对于一个 serving window `$w$`，将路由表示为 token–expert 超图：

\[
G_w=(T_w,E,H_w),\qquad h_t=\{e_{t,1},\ldots,e_{t,k}\}.
\]

给定 backend contract `$c$` 和系统配置 `$\theta$`（expert placement、LL/HT protocol、buffer size、domain grouping），真实代价是：

\[
y_c(\theta,G)=\text{native records / bytes / completion / P99}.
\]

[Hypothesis] 常见的 architecture-only 或 marginal-only workload 使用合成路由 `$\hat G_S$` 时，即使保持了平均 expert load，也可能因丢失 token-level co-activation、source/owner grouping 或 windowed burst，而错误排序配置：

\[
\arg\min_{\theta} y_c(\theta,\hat G_S)
\ne
\arg\min_{\theta} y_c(\theta,G).
\]

但“保留完整 trace”不是论文创新，也不是最低成本。真正的研究问题是：

> **对于每一类 backend wire/layout contract，哪一级 route invariant 已足以保持系统配置决策？**

## 4. RouteFidelity-EP 完整设计

### 4.1 Backend-conditioned sufficiency lattice

RouteFidelity 定义一个逐步增强的 route invariant lattice：

| 级别 | 保留信息 | 有意丢弃的信息 |
|---|---|---|
| `S0 Architecture` | batch/tokens、experts、top-k | expert popularity、co-activation、ordering |
| `S1 Degree` | 每层每 expert 精确出现次数 | token hyperedge、ordering |
| `S2 Coactivation` | expert pair/sketch 或 hyperedge multiset | 时间顺序 |
| `S3 Windowed hyperedge` | request/phase/window 内的 hyperedge 与 burst | 精确 token 内容与模型密码 |
| `S4 Exact ordered route` | 完整 top-k route oracle | 无 |

不同 contract 可能有不同的最小充分层：

- expert-major expanded records 可能只需 `S1`;
- rank-major unique-owner records 可能需 `S2`;
- receiver tail/buffer 可能需 `S3`;
- hierarchical domain partial 还需 source-domain-conditioned hyperedges。

### 4.2 Decision-fidelity objective

对配置集合 `$\Theta_c$`，同时评估：

\[
\tau_c(S)=\operatorname{KendallTau}
\big(y_c(\Theta_c,G),y_c(\Theta_c,\hat G_S)\big),
\]

\[
R_c(S)=
\frac{y_c(\hat\theta_S,G)-\min_{\theta\in\Theta_c}y_c(\theta,G)}
{\min_{\theta\in\Theta_c}y_c(\theta,G)},
\quad
\hat\theta_S=\arg\min_{\theta}y_c(\theta,\hat G_S).
\]

离线选择最小表示：

\[
S_c^*=\arg\min_S \operatorname{Cost}(S)
\quad \text{s.t.}\quad
\tau_c(S)\ge0.95,\ R_c(S)\le0.02.
\]

`Cost(S)` 同时考虑 trace 体积、采集开销和隐私泄露。这使方法与“精确 replay 肯定最真实”产生本质区别。

### 4.3 RouteIR 与 backend lowering

```text
request/phase/window
        |
        v
RouteIR: token -> {expert, gate, owner, source_domain}
        |
        +--> expanded expert-major reference
        +--> rank-major unique-owner reference
        +--> domain-partial/hierarchical reference
        |
        v
backend adapter -> native records/frames/events -> decision certificate
```

RouteIR 只保存经 `$S_c^*$` 允许的信息。EP-WireScope 不再是独立主线，而是 adapter 正确性 oracle：检查 reference records 与 native frames/counters 是否闭合。

### 4.4 Offline/online 边界

RouteFidelity 默认是 offline benchmark/configuration system，不在每 token 路径上运行复杂控制器。部署阶段仅应用经 holdout 验证的配置 LUT；路由分布漂移或证书失效时，fallback 至 backend native default。

### 4.5 复杂度

精确 route 处理为 `O(LTK)`；degree/pair/windowed sketch 可流式维护。对 `|Theta|` 个配置和 `|C|` 个 contract，reference evaluation 为 `O(|C||Theta|LTK)`，可按配置并行。在线不增加 EP 通信量。

## 5. 当前 P0-A：有信号，但主假设未通过

[Observed] 已实现 degree-preserving double-edge rewiring，确保每层 expert occurrence count 完全不变；reference lowerer 按 token 命中的 unique owner 构造 logical records。实验代码有两个 reference tests。

| model | surrogate | max logical receiver-P99 underestimate | max placement regret |
|---|---|---:|---:|
| OLMoE | architecture-only uniform | 43.0% | 2.1% |
| OLMoE | exact-degree shuffle | 10.0% | 2.5% |
| OLMoE | hyperedge multiset + order shuffle | 10.2% | 2.1% |
| LLM-jp | architecture-only uniform | 13.3% | 1.5% |
| LLM-jp | exact-degree shuffle | 0.8% | 0.7% |
| LLM-jp | hyperedge multiset + order shuffle | 0.8% | 0.0% |

严格解读：

- [Observed] architecture-only uniform route 过于粗糙，可明显改变 logical receiver occupancy；
- [Observed] higher-order/temporal structure 只在 OLMoE 当前 dev grid 中有约 10% 信号，在 LLM-jp top-16 上几乎消失；
- [Observed] placement regret `<=2.5%`，没有达到预设 5% problem gate；
- [Inferred] 当前不能声称 buffer overflow、backend tail 或 configuration inversion。P0 对大量 cell 取 maximum，还有 winner's curse，只用于决定是否值得跑 sealed P0-B。

当前产物：

- `experiments/idea_a_mac/run_route_fidelity_p0.py`
- `experiments/idea_a_mac/test_route_fidelity_p0.py`
- `experiments/idea_a_mac/outputs/route_fidelity_p0_2026-07-18/dev_v3/`

## 6. Prior-art collision audit

| 工作 | 已覆盖内容 | RouteFidelity 可辩护差异 | 碰撞风险 |
|---|---|---|---|
| [MLSynth](https://marioskogias.github.io/docs/mlsynth.pdf) | 合成 ML traces，其公开 MoE 模型以 uniform distribution 生成 token→expert assignment | 不是增加一个 MoE template，而是 backend-conditioned sufficiency + configuration-ranking certificate | **最高**；若只是更真实 trace generator，就是 MLSynth extension |
| [MLCommons Chakra](https://arxiv.org/abs/2605.11333) | 通用 execution trace schema、采集、合成、分析和 replay | 可兼容 Chakra，但增加 MoE route semantic sidecar、contract lowering 与 decision fidelity | 不能把 schema/replay 本身算创新 |
| [Megatron Router Trace](https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.transformer.moe.router_trace.html) | training/inference 的 per-layer top-k route 采集，可保存 hidden/logits sidecar | 路由采集只是输入；贡献必须是最小充分表示与跨 backend 配置保真 | **致命碰撞**：exact capture/replay 没有新颖性 |
| [AICB](https://github.com/aliyun/aicb) | DeepSeek/Qwen3-MoE inference workload generation；当前公开推理仿真主要依赖 analytical `autobusbw`，DeepEP simulation 仍在开发 | 测量 token-level route statistics 对 native record/config ranking 的增量 | 必须作为强 baseline，不能笼统说“AICB 不真实” |
| [Scaling Multi-Node MoE Using Expert Activation Patterns](https://arxiv.org/abs/2604.23150) | 用真实 expert activation pattern 做 request grouping 和 placement | 不发明 activation-aware placement；而是判定哪些 activation statistics 对不同 backend decision 足够 | 宽泛的“路由相关性影响 placement”已被覆盖 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | EP dispatch/combine、LL/HT、FP8、native benchmark | 作为真实 contract/adapter，不替代 backend | 若只加 custom route input，只是工程 PR |
| [NCCL EP](https://arxiv.org/abs/2603.13606) / [TensorRT one-sided](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.html) | LL direct mesh、HT hierarchy、rank-major receive layout 等不同 contract | 为“backend-conditioned”提供必要的语义差异 | 若不能证明不同 contract 的最小充分统计确实不同，主线失去意义 |

[Inferred] 唯一安全的 novelty intersection 是：**backend-conditioned sufficient statistics + configuration decision fidelity + compact/private synthesis**。任何“首次采集真实 MoE route”、“首次重放路由”或“更真实的 benchmark”都不应出现在 claim 中。

## 7. 三条可独立成立的论文贡献

1. **Backend-conditioned route sufficiency observation**：系统描画不同 EP record/layout contract 对 expert degree、token co-activation、windowed burst 的依赖，并给出“哪一级统计量才足够”的可证伪结论。
2. **Decision-fidelity synthesis method**：不优化 trace 字段还原误差，而是直接以配置 Kendall 排序和 selected-config regret 为约束，选择最小 route invariant 并合成 workload。
3. **Cross-contract empirical boundary**：在至少两个真实 backend 上定量 architecture/marginal abstraction 何时会误导 placement、buffer 或 protocol 决策，以及 RouteFidelity 何时能以更小 trace 恢复决策。

第三条必须有非平凡实测结果；“实现了两个 adapter”不能单独算贡献。

## 8. 核心可证伪假设与杀死条件

| 假设 | 晋级条件 | 直接杀死结果 |
|---|---|---|
| H1 边际路由抽象会误导系统决策 | 至少两个 `model×contract` cell 中 `tau<=0.8` 或 regret `>=5%`，CI 支持 | exact-degree/marginal 在 sealed holdout 中 regret 全部 `<5%` |
| H2 windowed hyperedge 是更小的充分表示 | `tau>=0.95`、regret `<=1–2%`，且 trace/采集成本明显低于 exact replay | 只有 S4 exact trace 能恢复排序 |
| H3 逻辑 ranking gap 会暴露为 backend 代价 | actual operator latency regret `>=5%`，95% CI 不跨 0 | logical gap 进入 backend 后 `<3%` 或排序不变 |
| H4 方法具有 backend-conditioned 价值 | 两种语义不同 backend 的最小充分层或 failure mode 不同 | 一个通用 marginal summary 已在所有 backend 充分 |

H1 失败时，立即终止 CCF-C 主线；H3/H4 失败时，只保留 benchmark/workshop 定位。

## 9. 完整实验矩阵

### 9.1 模型与workload

- OLMoE top-8、LLM-jp top-16；新增 Qwen/Mixtral 系 top-2/top-4 作结构对照；
- 自然文本、数学/代码、合成 hotspot；
- 真实 prefill chunk 与 continuous-batching decode step，不再用 `token_position // B` 冒充 serving batch。

### 9.2 合成强对照

- architecture-only uniform；
- per-layer exact expert degree；
- degree + pair/coactivation sketch；
- hyperedge multiset + shuffled time；
- windowed hyperedge；
- full ordered trace oracle；
- AICB/MLSynth-style workload。

### 9.3 configuration pool

- contiguous、round-robin、frequency-balanced/EPLB-style、coactivation-aware；
- 至少 128 个 balanced random/hierarchical placements；
- LL/HT/direct/hierarchical protocol 与buffer/capacity headroom；
- 每个 backend 只比较其真实支持的配置，不用虚构 wire model。

### 9.4 指标与统计

- Kendall tau、top-1/top-5 agreement、selected-config normalized regret；
- native record/frame/bytes closure，operator P50/P95/P99，exposed completion；
- trace size、capture/synthesis time、instrumentation overhead 和隐私攻击面；
- `>=20` synthesis seeds，request/article-level paired cluster bootstrap，预注册 primary cells，不以全矩阵 maximum 作主结论。

### 9.5 硬件渐进路径

| 资源 | 可信 claim | 不可 claim |
|---|---|---|
| Mac | route invariant、CPU reference contract、ranking/regret、sealed statistics | actual wire、GPU latency、TTFT/TPOT/P99 |
| 1 GPU | capture overhead、pack/layout/reference parity | inter-GPU EP communication |
| 2–4 GPU | 单机 backend records、operator latency、NVLink/PCIe | RDMA/WAN |
| 2 节点 × 2–4 GPU | NIC bytes、hierarchical EP、backend-conditioned latency ranking | 超出实际拓扑的普遍性 |

## 10. 论文叙事、标题与摘要

推荐标题：

> **RouteFidelity-EP: Backend-Conditioned, Decision-Faithful Workload Synthesis for Mixture-of-Experts Communication**

中文：

> **RouteFidelity-EP：面向 MoE 专家并行通信配置决策保真的后端感知工作负载合成**

摘要草案：

> MoE 通信仿真通常由模型结构、平均 expert load 和集合字节数生成工作负载，但真实 EP backend 会使用 expert-major、rank-deduplicated 或 hierarchical records，使 token–expert co-activation 与时间突发可能改变 placement、buffer 和 protocol 的最优选择。我们提出 RouteFidelity-EP，将路由表示构造为从 architecture、expert degree、co-activation 到 windowed hyperedge 的充分性格，并以配置排序相关性与 selected-config regret 而非 trace 字段还原误差为目标，为每个 backend 选择最小路由表示。系统将该表示编译为 backend-native records，并输出决策保真证书。评估将检验边际路由抽象何时会产生配置排序反转，以及紧凑的 windowed-hyperedge 表示能否在两类真实 EP backend 上恢复决策。

完整叙事：

```text
Problem
  现有 workload 对 MoE route 的抽象级别与 backend contract 脱节
        ↓
Observation
  同一边际负载不一定产生同一 owner co-activation / receiver burst
        ↓
Mechanism
  sufficiency lattice + decision-fidelity objective + backend lowering
        ↓
Evaluation
  sealed CPU problem gate → two native backends → ranking inversion and recovery
```

## 11. 最终判决

- **可冲等级**：有条件的 CCF-C；当前尚不是 CCF-C-ready。
- **当前成熟度**：4.5/10。
- **最大不确定性**：在 exact-degree/coactivation 强对照下，是否还有 `>=5%` 的配置或 actual latency regret。
- **当前最重要的事**：先跑 fresh sealed P0-B，不要先写 GPU adapter，更不要先宣称新主线已成立。
- **停止原则**：P0-B 若没有至少两个 `model×contract` cell 达到 5% regret/0.8 tau problem gate，则将本方向降为硕士论文的 benchmark/negative-result artifact，不再追逐 CCF-C 叙事。
