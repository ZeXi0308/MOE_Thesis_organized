# MoE Serving Idea Landscape — Run 20260729_210201

> 状态：`EXPLORATORY / NOT_CURRENT_MAINLINE`  
> 检索截止：2026-07-29  
> 权威边界：本文不改写 `docs/current/README.md` 与已封存裁决。

## 1. 当前资产与证据边界

- RouteSlack 仍是 `MEASUREMENT_ONLY / Gate 0 FAIL`；不允许用新 controller 或调门槛复活。
- BCRD 本地 simulator correction 有62/62项单测/反例与随机 exactness，但仍缺 formal continuous-decode producer、完整 service surface、all-arrival denominator 与 full counterfactual request-DAG。
- 单张 RTX 5090 能验证真实 BF16 cached decode、route identity、实际模型语义和单卡故障注入；不能证明 EP/NCCL/RDMA、真实 rank barrier、TPOT/P99 或多卡恢复效果。
- 可用两个开放 MoE：OLMoE-1B-7B（E64/top-8）与 llm-jp optimal-sparsity（E32/top-16）。

## 2. 本地 PDF 先行扫描

| 工作 | 覆盖 | 对本轮的含义 |
|---|---|---|
| Lina, ATC 2023 | all-to-all 瓶颈与 dynamic expert resource scheduling | “按 popularity 调资源”不是空白 |
| AdapMoE, ICCAD 2024 | adaptive gating/prefetch/cache | 会改变执行语义，不进 exact-semantics action space |
| HOBBIT, 2024 | mixed-precision expert offloading | 面向 edge/offload，与当前 exact BF16 问题不同 |
| Aurora, 2024 | deployment + communication scheduling | placement/communication 联合优化已有直接先例 |

## 3. Primary-source 版图

### 3.1 MoE 调度、布置和 kernel 已高度拥挤

| 工作 | 已覆盖的核心面 | 本轮不应重复的表述 |
|---|---|---|
| [AMoE](https://arxiv.org/abs/2505.08944) | per-layer micro-queue、asynchronous EP、adaptive rebatching | 取消 barrier/异步执行本身 |
| [UltraEP](https://arxiv.org/abs/2606.04101) | per-microbatch/per-layer exact post-gating load balancing | “用真实 route 实时均衡” |
| [METRO](https://arxiv.org/abs/2512.09277) | memory-bound decode 下平衡 activated experts，带 exact local solver | 只把 token count 换成 expert count |
| [CRAFT](https://arxiv.org/abs/2603.28768) | layer-granular replication + exact MCKP surrogate | 逐层副本预算分配 |
| [Gimbal](https://arxiv.org/abs/2606.15177) | frontend pressure、queue order、expert pressure、placement、migration | 一般的跨层联合 scheduler |
| [Director](https://arxiv.org/abs/2607.08782) | pending-request route prediction + proactive placement | 用早期 route 预测迁移 |
| [Mixture-of-Experts Serving](https://arxiv.org/abs/2607.17880) | dynamic demand、GPU allocation、reconfiguration cost 的在线/离线形式化 | popularity + migration 的抽象算法模型 |
| [ExpertPlex](https://arxiv.org/abs/2607.18002) | shared experts + tile-level adaptive persistent kernels | 单独以 tile/SM 调度为新意 |

结论：“再做一个 expert-aware balancer/controller”不足以支撑新论文。

### 3.2 泛化的 C01/C02 已被全局图、因果仿真与等价改写工作包围

| 工作 | 直接碰撞 | 仍未覆盖的窄面 |
|---|---|---|
| [Frontier](https://arxiv.org/abs/2605.21312) | 支持 MoE/EP 的 closed-loop discrete-event graph，显式 scheduler-batch-engine 依赖，指出 proxy 会导致 decision drift | 无针对特定 MoE action 的 one-sided ranking certificate |
| [CausalSim](https://www.usenix.org/conference/nsdi23/presentation/alomar) | 处理 algorithm-induced trace bias 与 counterfactual replay 偏差 | 不是 identity-complete request-DAG/action-equivalence compiler |
| [Vidur](https://arxiv.org/abs/2405.05465) | event-driven LLM serving 仿真与配置搜索 | 不报 local/full action-sign reversal |
| [Proteus](https://arxiv.org/abs/2306.02267) | distributed execution graph 与 strategy-order preservation | 主要是 training strategy，非 request SLO |
| [dPRO](https://proceedings.mlsys.org/paper_files/paper/2022/hash/b422680f3db0986ddd7f8f126baaf0fa-Abstract.html) | global dataflow graph、critical path、partial replay | 不是 continuous-serving request DAG |
| [Parrot](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan) / [Agentix](https://www.usenix.org/conference/nsdi26/presentation/luo) | application/program dependency-aware LLM scheduling | 没有 local-vs-full Oracle 符号证书 |
| [Unity](https://www.usenix.org/conference/osdi22/presentation/unger) | graph rewrite + theorem-proved semantic equivalence | 不是 decision-time serving action 在 future queue 上的等价性 |

结论：“补一个 full-DAG simulator”的 C01 泛化版本已不成立。只有能给出非平凡 theorem/certificate、并明确比 Frontier/CausalSim 多出什么的窄化问题才能留在候选集。

### 3.3 可靠性已有强近邻，但 sub-timeout route-aware 诊断仍有窄缺口

| 工作 | 已覆盖 | 对候选问题留下的边界 |
|---|---|---|
| [GEM](https://arxiv.org/abs/2605.19945) | 已知、持久 GPU latency variability 的 expert-to-GPU mapping | 不从 request residual 反演新出现的 slow component |
| [GREYHOUND/FALCON](https://www.usenix.org/conference/atc25/presentation/wu-tianyuan) | hybrid-parallel training fail-slow detection + active microbenchmark localization | 不是 dynamic MoE inference，也不使用 request-to-expert incidence |
| [EEP](https://arxiv.org/abs/2605.10670) | partial EP rank fail-stop 恢复 | 不处理 5–20% 的 slow-but-responsive degradation |
| [Tarragon](https://arxiv.org/abs/2601.01310) | expert-worker fail-stop、timeout/liveness probe、shadow expert | 已知 peer + timeout 解决显式失效，不求解 sub-timeout 反问题 |
| [StriaTrace](https://www.usenix.org/conference/osdi26/presentation/wu-haonan) | online LLM tracing/diagnosis，识别 rank/phase/kernel 异常 | 论文未展示基于完整 EP route identity 的 expert localizer |
| [NetBouncer](https://www.usenix.org/conference/nsdi19/presentation/tan) / [deTector](https://www.usenix.org/conference/atc17/technical-sessions/presentation/peng) | active path probing、coverage/identifiability、网络故障定位 | 如果候选方法只是 binary incidence + NNLS，就只是 tomography 迁移 |
| [Link Delay Estimation via Expander Graphs](https://arxiv.org/abs/1106.0941) | sparse additive delay inversion | 要成立新意，必须解决 MoE 的 top-k、multi-layer、batching、barrier/max-plus 与 route drift |

## 4. 去重后的 gap map

| Gap | 现在的证据 | 主要风险 |
|---|---|---|
| MoE sub-timeout silent expert/rank slowdown 的 route-aware localization | 未找到同时使用 natural route incidence + request/token latency residual + identifiability certificate 的直接工作 | 被视为 NetBouncer/NNLS 在 MoE 上的套用 |
| max-plus/barrier 语义下的可辨识 probe codebook | 通用 tomography 多为 additive/binary path；MoE 每层是并行 fan-out/join，多层又串联 | 需要非平凡可辨识定义/算法，不能只换符号 |
| 区分 slow expert、slow rank、slow link 与正常 route/load imbalance | 相邻论文通常已知故障类型，或用 intrusive component benchmark | 只看 E2E latency 可能根本不可辨识 |
| 针对特定 action pair 的 causal-closure ranking certificate | Frontier/CausalSim 解决泛化仿真/因果偏差，未给 MoE-specific one-sided rank witness | universal bound 可能过宽而永远 abstain；容易重蹈 C10 |
| full-DAG action-equivalence quotient | 尚未找到横跨 admission/batch/assignment/seal/release/placement 的 sound+complete absence certificate | Unity/形式化调度强碰撞，工作量大且 5090 不能补多卡证据 |

## 5. Phase-1 裁决

1. 泛化 C01/C02 不进入 GPU pilot；只允许窄化 certificate 类候选进入后续独立方法评审。
2. C09 不按“route-aware sparse regression”继续；只保留能处理 MoE max-plus/barrier 语义、输出可辨识证书、并能分离正常负载混淆的方法版本。
3. 后续最多运行 3 个冻结门槛的资格性 pilot；单卡结果只标记 `QUALIFICATION_PROXY`。

