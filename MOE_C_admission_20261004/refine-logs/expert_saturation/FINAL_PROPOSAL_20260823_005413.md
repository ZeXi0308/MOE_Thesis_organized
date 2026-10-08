# 最终研究提案：Expert Pressure 何时构成 MoE Serving 的独立并发控制变量？

> English working title: **When Does Expert Pressure Add a Usable Control Dimension Beyond Token/KV Counters?**  
> Date: 2026-08-23  
> Repository HEAD before this planning run: `b141c1d587fe2c918643c3c7c3a8f5f5157d4c8a`  
> Final method review: `8.7/10, REVISE`  
> Protocol status: `PROTOCOL_EXECUTABLE`  
> Scientific status: `UNVERIFIED / NO_METHOD_GO / NOT_VENUE_READY`

## 0. 结论先行

这个方向**适合作为三个月硕士论文探索**，但不应把“Expert-Saturation-Aware Controller”预写成已成立的贡献。当前更强、也更诚实的主问题是：

> 在固定 placement、router、scheduler order 与候选请求语义后，已完成 epoch 的 Expert/EP-rank Pressure，是否会稳定改变“提高下一 epoch decode concurrency”带来的完整请求 SLO 风险？

论文中心应是 **pressure-conditioned marginal-capacity boundary**；Controller 只是该边界被实验证明可执行后的最小 supporting mechanism。

当前最重要的阻塞不是缺少 Predictor，而是仓库尚无：

- native serving 中的现象迁移；
- same-prestate、policy-specific budget branches；
- request-level TPOT/TTFT/SLO-goodput 分母；
- action Oracle headroom；
- 8×A100 EP 证据。

因此，**现在唯一应执行的是 N0 native qualification**。不能提前实现 Controller、做 8×A100 大矩阵，或把 custom runtime 的 route/logit difference 写成容量收益。

## 1. 开始前执行合同

| 项目 | 冻结内容 |
|---|---|
| Authority files | `docs/current/README.md`；`docs/ideas/README.md`；Route Capacity / execution-conformance 的 latest status、report、addendum；本提案的 action-level novelty report |
| 已继承事实 | 单一 OLMoE revision、RTX 5090、custom cached-decode runtime 中存在 repeat-stable batch-dependent execution-conformance difference；证据层级止于 `CUSTOM_CONTINUOUS_RUNTIME / MEASUREMENT_ONLY` |
| 已测 | 同一 target state 的 A/B/C/D 局部执行差异、route/logit/KV 传播与 bounded prevalence |
| 未测 | native request latency、TPOT/TTFT/P99、capacity、SLO-goodput、Controller、EP/A2A、第二模型泛化 |
| 一个研究问题 | pressure 是否修饰 decode-budget 的 request-level treatment effect，而非仅预测慢 step |
| 一个最弱因果环节 | paired budget effect 在控制完整 `Z_t`、上一 budget、总 routed tokens 和 recent latency 后，是否仍随 pressure 稳定变化 |
| 一个决定性实验 | N0 通过后，从同一 `Z_t` 分叉三个真实 budget，并让每条 policy 独立演化到同一固定 request cohort 完成 |
| 当前 claim ceiling | proposal / protocol only；不允许 method GO、production SLO、EP 或收益数字 |
| 继续条件 | native qualification 通过；paired interaction repeat-stable；Oracle SLO-goodput headroom `>=3%` 作为探索继续信号 |
| 停止条件 | invalid state matching、sign flip、无 common support、完整分母不闭合、Oracle `<3%`、ordinary + recent-latency baseline 吸收 residual |
| Reopen 条件 | 新 native runtime/hardware、可复现 action residual、或旧实验被证明 invalid；不能只换阈值/seed/名字 |

## 2. 对研究问题和创新点的客观 Review

### 2.1 值得做的部分

1. **问题是真实的。** MoE decode 的 exposed path 可能受 token-to-expert/rank load、expert kernel shape、A2A 与 slow rank 影响；普通 Token/KV state 未必完整描述它。
2. **动作足够小。** 只调节 active decode concurrency，不改 router、top-k、precision、placement 或 KV policy，能把因果链压缩到一个旋钮。
3. **正负结果都有论文出口。** 正结果可形成最小 Controller；无 action headroom 可形成 measurement boundary；信号被普通 counters 吸收也能形成明确退化规律。
4. **现有仓库给出了正确性警告。** Batch composition 会改变 KV、physical shape、route 与后续状态，因此固定 future route replay 不是合法的 action counterfactual。

### 2.2 最容易失败的地方

1. `active expert union` 可能在很小 batch 就饱和，随后没有区分度。
2. Expert Pressure 可能只是 batch、上一 budget、总 routed tokens 或 recent step latency 的代理。
3. 即使 Pressure 能预测慢 step，也未必改变增减 budget 的**边际** SLO 代价。
4. 局部 MoE/A2A 差异可能被 attention、KV、queue 或 overlap 吸收，无法进入 request completion。
5. 单卡 max-expert load 与 EP max-rank/A2A pressure 不是同一个运行域，不能直接外推。

### 2.3 创新性 verdict

- “Expert pressure 用于 serving scheduling”：**LOW novelty**。
- “Active-expert saturation 是新现象”：**LOW novelty**。
- “Quantile predictor + EMA + hysteresis”：**LOW novelty**。
- “同一 pre-state 下，用 policy-specific execution 识别 pressure-conditioned request-SLO budget effect”：**MEDIUM，结果依赖**。
- “跨 batch/KV/EP regime 给出 pressure 何时增加、何时不增加控制维度的稳定边界”：**最适合的论文贡献候选**。

Same-prestate rerun 首先是实验正确性；只有它揭示了稳定而反直觉的系统边界，才能成为 contribution。

## 3. 与现有工作的区别

| 邻近工作 | 已覆盖内容 | 本研究不能重复声称 | 仍可检验的 residual |
|---|---|---|---|
| [Gimbal](https://arxiv.org/abs/2606.15177) | recent expert/rank token pressure、KV/prefill/queue；跨 DP engine dispatch 与 placement | “首次用 expert pressure 调度请求” | 固定 placement/router 的 engine-local budget treatment effect |
| [SCORPIO](https://arxiv.org/abs/2505.23022) | SLO feasibility、VBS admission、credit batching | “首次做 TPOT-aware admission” | MoE pressure 对同一 action space 的 held-out causal increment |
| [SLOs-Serve](https://arxiv.org/abs/2504.08784) | performance model、soft admission、dynamic batch/token allocation | “首次以 cost model 控制 batch” | endogenous route 下的 policy-specific feasible boundary |
| [METRO](https://arxiv.org/abs/2512.09277) | activated-expert-aware MoE decode balancing | activated experts 本身的 novelty | 固定逻辑 routing、只调 active request cap |
| [ELDR](https://arxiv.org/abs/2607.00466) | expert signature 超越普通 worker load并驱动 worker dispatch | expert state 可用于 serving decision | same-engine、previous-epoch pressure 对 budget marginal cost 的影响 |
| [TAPER](https://arxiv.org/abs/2605.06914) | candidate-width externality、slack-aware per-step admission与fallback | admission 控制结构的新颖性 | MoE route endogeneity 与 expert-pressure boundary |
| [ReXpert](https://arxiv.org/abs/2608.13962) | SLO-limited batch、activated-expert union、decode saturation | “SLO 小 batch 下 expert union 问题首次出现” | 标准 GPU/EP、完整 request denominator、无硬件/placement修改 |

最接近的碰撞是四块拼合：

```text
Gimbal                        → MoE pressure sensing + scheduling
SCORPIO / SLOs-Serve / TAPER → SLO-aware admission / budget action
METRO / ReXpert              → activated-expert saturation
ELDR                          → expert state beyond ordinary load
```

所以差异不能写成“控制位置不同”，而要由以下完整链条建立：

```text
strong route-free baseline
→ same-Z_t real budget branches
→ independent future state
→ fixed request cohort
→ request-level risk difference
→ pressure-conditioned effect heterogeneity
```

## 4. 冻结问题、动作和因果量

### 4.1 唯一 action

```text
b_t = 下一固定 control epoch 中，每次 decode iteration 最多选择的 sequence 数
```

固定：

- Ready Queue 顺序：`FCFS + frozen aging tie-breaker`；
- prefill policy 与 arrival trace；
- router、top-k、expert identity；
- placement/replication、precision、KV policy；
- sampling 参数与 seed；
- continuation policy。

Controller 只能截断固定顺序的前 `b_t` 个请求。第一阶段不同时控制 Token Budget、prefill/decode ratio、placement 或 batch delay。

### 4.2 共同 pre-action state

```text
Z_t = {
  complete active/ready queue snapshot,
  frozen request order and request identities,
  generated token histories,
  logical/physical KV lengths and padding,
  KV occupancy / block mapping,
  frozen prefill and scheduler state,
  previous budget and scheduled tokens,
  previous total routed tokens,
  recent request ITL / step-latency EMA,
  model/backend/dtype/RNG identity
}

X_t(b) = g(Z_t, b)
```

`Z_t` 是 branch matching unit；`X_t(b)` 是 action 前可计算的 candidate-specific ordinary state。

### 4.3 Pressure 不是 intervention

首版唯一在线 MoE scalar：

```text
p_load(t) = max over completed epoch / layer / expert-or-EP-rank
            of routed_token_count
```

- 单卡：max expert token count；
- EP：max rank routed-token load；
- 只读已完成 epoch，绝不读取 candidate action 的 future route；
- previous total routed tokens、previous budget 与 recent latency 必须进入 route-free baseline，防止 pressure 只是历史 action 的代理。

### 4.4 Treatment effect

```text
tau(Z_t, p_t)
  = E[Risk_t(b_high) - Risk_t(b_low) | Z_t, P_<=t = p_t]

tau_hat(p-bin)
  = mean over same-Z_t states s
      [Risk_s(b_high) - Risk_s(b_low)]
```

主结论来自 paired risk difference；quantile model 中的 interaction coefficient 只作解释，不能替代 `tau_hat`。

### 4.5 与 treatment 无关的完整 request cohort

在 G1 calibration split 冻结墙钟窗口 `W` 和统一 timeout `T_max`：

```text
C = branch point 前 active/ready request IDs
    ∪ 在固定窗口 [t, t+W] 按预注册 trace 到达的 request IDs
```

每个 branch：

1. replay 完全相同的 `C` 与 arrival timestamps；
2. `t+W` 后不再加入主评价 cohort；
3. 一直运行至 `C` 全部完成或统一 `T_max`；
4. timeout、OOM、failure 与未完成请求全部按预注册 miss 计入，不能从分母删除。

这样不会因不同 budget 的完成速度改变纳入的 request IDs。

## 5. Runtime Cost Model 候选

### 5.1 先写完整会计

```text
T_step(b)
  = T_scheduler
  + T_attention(B, token_count, physical_KV, padding)
  + T_router
  + critical_path_over_ranks(
        T_dispatch_r + T_expert_r(load_r, working_set_r) + T_return_r)
  + T_combine
  + T_sampling
  + T_exposed_idle_or_barrier
  + T_telemetry
```

单卡把 rank critical path 换成本地 exposed expert path。若 stages overlap，只记 wall-clock exposed time，不把 kernel、A2A 与总 step time重复相加。

```text
T_request
  = queue + prefill + wall-clock decode completion
    + sampling/finalization
```

局部 step/model 只解释机制；最终 action 由完整 request SLO 判定。

### 5.2 两层模型，而不是一个复杂 Predictor

**D0 — profiling diagnostic：**

```text
Q_q(T_step | X_t(b), b)
Q_q(T_step | X_t(b), b, p_t)
```

用于回答 Pressure 是否有 held-out latency residual，不授权 Controller。

**D1 — action-risk model：**

```text
Q_q(Y_request | X_t(b), b, p_t)
  = f_route_free(X_t(b), b)
  + alpha * relu(p_t - k)
  + gamma * b * relu(p_t - k)
```

- `k` 只在 calibration split 冻结；
- 主比较是 route-free model 与加入 `pressure × budget` interaction；
- `gamma` 是诊断，不是 treatment estimator；
- 若 one-threshold policy 足够，不使用 Ridge/GBDT/RL。

### 5.3 Feature ladder 与采集成本

| Feature | 作用 | 采集成本/风险 | 首版决定 |
|---|---|---|---|
| decode sequences/tokens、physical KV mean/max、padding、KV occupancy、queue/age、prefill state | ordinary state | 低，多数已有 | 必须 |
| previous budget、scheduled tokens、total routed tokens | confounder control | 低 | 必须 |
| recent request ITL / step-latency EMA | strongest simple feedback | 低 | 必须 baseline |
| completed max expert/rank routed-token load | primary pressure | 低–中；需 GPU aggregate、epoch-boundary async export | 唯一首版 MoE input |
| active expert union / working-set | saturation diagnosis | 中；可能快速饱和 | 只做一次曲线 |
| load CV/P95/HHI | mechanism diagnosis | 同一 histogram 可派生，但有 feature-search 风险 | 附录/失败定位 |
| A2A bytes/time EMA | EP mechanism diagnosis | 中–高；可能同步扰动 | 只在 G4 |
| attention/MoE/communication spans | critical-path survival | 高、profiling 扰动 | sampled profiling only |

## 6. Scheduler 状态、流程与伪代码

### 6.1 最小数据流

```text
completed route counts
→ GPU-side reduce
→ epoch pressure + validity
→ Token/KV/queue baseline budget b0
→ one-level pressure correction
→ native scheduler active-decode cap
→ new request/KV/route/completion trajectory
→ request-level metrics and fallback audit
```

### 6.2 需要维护的状态

```text
current_budget
token_kv_baseline_budget
completed_pressure_ema
frozen_pressure_knee
high_threshold / low_threshold
dwell_counter
telemetry_valid
common_support_or_OOD_flag
request ages / max-wait guard
```

### 6.3 伪代码

```text
on_epoch_boundary(state):
    b0 = token_kv_queue_policy(candidate_state)

    if telemetry_missing(state):
        return b0
    if pressure_uncalibrated(state):
        return b0
    if outside_frozen_common_support(state):
        return b0

    p = completed_pressure_ema(state)

    if p >= high_threshold:
        target = max(b_min, b0 - one_budget_level)   # fast down
        reset_dwell()
    elif p <= low_threshold and dwell_passed():
        target = min(b0, current_budget + one_budget_level)  # slow up
    else:
        target = current_budget

    target = enforce_kv_feasibility(target)
    target = enforce_max_wait_and_aging_guard(target)
    return target
```

这个 pressure-threshold correction 就是最终最小 Controller；不再人为添加一个复杂 proposed predictor。

## 7. 三个月执行路线

| 周 | 唯一目标 | 继续/停止决定 |
|---|---|---|
| W1 | N0b native integration、telemetry OFF/ON parity | parity/identity 不闭合则只修测量 |
| W2 | N0a 一 steady + 一 bursty A/C/D，process-isolated repeats | 输出 qualification verdict；不产 capacity claim |
| W3 | G1 actuator、`W/T_max`、SLO、budget knee、common support calibration | 无合法 branch/common support 则停 |
| W4 | I1 same-`Z_t` replay、digest、三 budget dry run | policy state 不独立即 `INVALID` |
| W5 | I1 steady/bursty matched branches 与 repeats | 只测 paired effect 和 Oracle |
| W6 | fresh holdout、负控、唯一 verdict | negative 转 boundary/no-go；positive 才 Controller |
| W7 | 实现 one-level pressure correction | 不做 GBDT/RL/多 action |
| W8 | 单卡端到端、强基线、TTFT/fairness/overhead | G3 不过则不去 8×A100 |
| W9 | 8×A100 EP preflight、telemetry parity、冻结 4 anchor cells | EP 测量不闭合则停 |
| W10 | G4 signal/controller confirmatory | 决定是否允许 EP claim |
| W11 | controlled repeats、边界图、最小 ablation | 不新增机制 |
| W12 | 写作、claim audit、Resurrection Card、seal artifacts | 不用补跑掩盖负结果 |

负路径应在 W6 前结束 Controller 搜索；正路径才使用 W7–W10。

## 8. Profiling 与端到端实验矩阵

### 8.1 模型与硬件

| 阶段 | 模型/runtime | 硬件 | 角色 |
|---|---|---|---|
| N0–I1 | `allenai/OLMoE-1B-7B-0924`，沿用冻结 revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`；优先采用 [vLLM 官方支持列表](https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md)中的 native `OlmoeForCausalLM` path | RTX 5090 | qualification、profiling、paired action Gate |
| G3 | 同一 frozen primary stack | RTX 5090 | 最小 Controller 与 full-request evaluation |
| G4 | 首选候选为 `Qwen3-30B-A3B` 的 [vLLM Expert Parallel](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) path；若 preflight 不闭合则不强行运行。Exact model revision、dtype、EP degree、placement、A2A backend 与 batch-invariance 配置在运行前冻结 | 8×A100 | 只确认 signal/Controller transfer，不作 feature search |

第二模型或不同 routing regime 只在正信号后做 confirmatory；单卡结论不替代 EP。`Qwen3-30B-A3B` 当前只是资源与软件兼容性的候选，不是已经通过 preflight 的正式实验对象。

### 8.2 非笛卡尔矩阵

**G1-A：Budget knee**

- median KV、同质 workload、steady；
- budget 候选 `{1, 2, 4, 8, 16, 32}`，按显存/engine 可行性裁剪；
- 只冻结 `b_low/b_mid/b_high`，不作主结果。

**G1-B：KV sensitivity**

- `b_low/b_mid/b_high`；
- short/medium/long physical KV；
- 防止 Pressure 代理 physical KV/padding。

**I1：决定性 matched cells**

| Arrival | Pressure stratum | Budgets | 初始 states | Repeats |
|---|---|---|---:|---:|
| steady | low / high | low, mid, high | 每层 8–12 | 至少 3 次 process-isolated |
| bursty | low / high | low, mid, high | 每层 8–12 | 至少 3 次 process-isolated |

只有 ordinary-state common support 中的 natural episodes 进入主结果；synthetic skew 只作 mechanism sanity。

**G3：代表性端到端 episodes**

1. chat：短 prompt / 中输出 / steady；
2. chat：长 prompt / 短输出 / bursty；
3. code：中 prompt / 长输出 / steady；
4. code：长 prompt / 长输出 / bursty；
5. math：中 prompt / 中输出 / steady；
6. chat/code/math 混合、长短混合、bursty；
7. 一个同质 workload negative control。

**G4：4 个 EP anchor cells**

- 最强 positive regime；
- 第二个不同 KV 的 positive regime；
- boundary regime；
- 应退化为 Token/KV baseline 的 negative regime。

### 8.3 Baseline ladder

1. native/default maximum concurrency；
2. calibration split 上的 fixed best budget；
3. Token/KV/queue-only adaptive budget；
4. Token/KV/queue + recent-latency EMA feedback；
5. Token/KV baseline + one-threshold/one-level pressure correction；
6. future-known action-conditioned Oracle。

Oracle 只能从真实执行过的 budget branches 中选，不能共享 future route/KV/completion。

## 9. 指标与判定

### 9.1 主指标

- TTFT P50/P95/P99；
- TPOT 与 ITL P50/P95/P99；
- request-level SLO attainment；
- SLO-goodput：单位 wall-clock 时间内完成且满足冻结 TTFT + TPOT/ITL SLO 的请求数；
- request completion time；
- paired `Risk(b_high)-Risk(b_low)`；
- dangerous underprediction；
- Oracle 相对 strongest route-free baseline 的 headroom。

### 9.2 Guards 与诊断

- queue waiting P95/P99、max age、starvation count；
- token throughput、GPU/KV utilization；
- scheduler + telemetry overhead；
- max expert/rank load、active union；
- sampled attention/MoE/A2A exposed share。

SLO 数值、`W`、`T_max`、pressure knee 和 holdout split 在 G1 后冻结；当前没有 native 数据，故本提案不伪造毫秒阈值。

### 9.3 探索和正式门槛

| 层级 | 决策规则 |
|---|---|
| 弱探索信号 | Oracle headroom `1%–3%`：只允许 controlled repeat 或 boundary analysis |
| 继续信号 | Oracle headroom `>=3%`，paired interaction 同方向，负控不复现 |
| 正式 model claim | fresh holdout P95 pinball `>=5%` 或 dangerous underprediction `>=15%` 改善，并在冻结 regime 同方向 |
| 正式 action claim | 同 SLO 的 SLO-goodput `>=5%`，或同 throughput violation rate `>=20%` 下降 |
| Controller claim | 捕获 `>=40%` Oracle；相对 strongest ordinary baseline 净 SLO-goodput `>=3%`；overhead `<1%`；steady/bursty同方向 |
| 简化规则 | threshold policy 捕获 `>=90%` Oracle 时直接保留它，不上复杂模型 |

探索阈值用于决定是否继续，不自动升级论文主张。

## 10. 七类核心风险、验证与降级

| 风险 | 验证 | 降级 |
|---|---|---|
| Active union 很快饱和 | G1 union-vs-budget curve；控制 Token/KV 后看 residual | union 不进 Controller；max load 也饱和则退回 ordinary policy |
| Attention/KV 主导 | wall-clock + sampled exposed-share，按 KV 分层 | 只在 MoE-exposed regime 启用，否则 fallback |
| EP 真瓶颈是 load/A2A，不是 active count | G4 max-rank load、imbalance、A2A、slow-rank survival | 只保留 max-rank load；A2A作诊断，不在线 feature search |
| Feature/decision overhead | telemetry OFF/ON parity 与 overhead | GPU aggregate、异步/降采样；总开销不达标则删除 Controller |
| TPOT改善但TTFT/等待恶化 | full-request SLO-goodput、TTFT、max-age guard | aging/max-wait override；guard不通过判净收益失败 |
| 模型/top-k/kernel差异 | 第二 model/routing/runtime confirmatory | per-regime calibration；不写普遍规律 |
| Burst 下预测失准 | burst holdout、dangerous underprediction、OOD coverage | OOD/telemetry invalid 时回退 Token/KV；fast-down/slow-up |
| Action endogeneity/future leakage | same-`Z_t` 三分支、state digest、独立 policy state | 任一 future-state 共享立即 `INVALID_EXPERIMENT` |
| Native nondeterminism | process isolation、same-arm repeats、版本绑定 | sign flip 先受控重复，不先解释机制 |

## 11. 建议图表

主文优先四张：

1. **Pressure-conditioned treatment-effect plot**：横轴 completed pressure，纵轴 `Risk(b_high)-Risk(b_low)`；steady/bursty 分面，episode bootstrap CI。
2. **Empirical feasible-capacity phase map**：横轴 ordinary Token/KV load，纵轴 Pressure，颜色为实测可行 budget 或 SLO miss risk；单卡/EP 分开。
3. **SLO-goodput–attainment Pareto**：native max、fixed best、Token/KV、last-latency、pressure correction、Oracle；全部含 overhead。
4. **Feature/baseline ladder**：route-free、+pressure main effect、+interaction，同时展示 pinball、dangerous underprediction 和 calibration。

附录：

- active union saturation curve；
- burst timeline：pressure、budget、queue、ITL miss；
- attention/MoE/A2A exposed wall-clock breakdown；
- telemetry overhead 与 fallback coverage；
- positive/boundary/negative regime map。

## 12. 三种允许的研究结论

### 12.1 正向：Boundary + Minimal Controller

条件：native paired interaction、material Oracle、simple policy、full-request净收益和 EP confirmatory 均成立。

允许表述：

> 在冻结的 `[model/runtime/regime]` 中，completed max-rank/expert load 对 decode-budget 的 request-level marginal SLO risk 提供了强 route-free counters 之外的稳定增量；单级 correction 捕获了 `[X%]` Oracle，并以 `[Y%]` 净 SLO-goodput 改善通过 guard。

### 12.2 边界：Measurement Boundary Only

可能情形：interaction 只在小 batch、长 KV、特定 EP/A2A 或 bursty regime 存在；或 Oracle 存在但 simple baseline/overhead 吸收收益。

允许表述：

> Expert Pressure 不是普适调度变量；它只在 `[明确区间]` 穿透到完整请求 critical path，在 `[退化区间]` 应回退到 Token/KV/last-latency policy。

### 12.3 负向：Signal or Action Space Dead

可能情形：

- pressure residual 被 ordinary state/recent latency吸收：`SIGNAL_DEAD_IN_TESTED_REGIME`；
- residual 存在但 Oracle `<3%`：`MEASUREMENT_ONLY / ACTION_SPACE_DEAD`；
- Oracle 存在但 online policy 失败：`PHENOMENON_ALIVE_MECHANISM_DEAD`；
- only EP 未测：不能把单卡 negative 扩写成 family NO-GO。

## 13. MVP 与扩展边界

### 13.1 当前最小可执行版本

当前只做 N0：

- native adapter；
- one steady + one bursty A/C/D replay；
- telemetry OFF/ON；
- snapshot/replay/branch actuator eligibility；
- process isolation 与 raw artifact retention。

### 13.2 N0 通过后的最小科学 MVP

- ordinary-state collector；
- GPU-side completed pressure aggregate；
- frozen-budget branch actuator；
- same-`Z_t` replay/digest；
- request-level metrics recorder；
- offline paired estimator + action Oracle。

这一阶段仍不需要在线 Predictor 或 Controller。

### 13.3 只有 I1 正信号后的 Controller MVP

- Token/KV/queue baseline；
- one pressure threshold；
- one budget-level correction；
- hysteresis、dwell、aging/max-wait guard；
- OOD/telemetry fallback。

### 13.4 条件扩展

- 8×A100 EP max-rank/A2A confirmatory；
- 第二模型/第二 routing regime；
- 多档 pressure penalty；
- joint prefill/decode control、placement、migration 或 online learning 均不属于三个月 MVP。

## 14. 标题、摘要和简历候选

### 14.1 标题

**当前最稳妥：**

- 中文：`Expert Pressure 何时构成 MoE Serving 的独立并发控制变量？`
- 英文：`When Does Expert Pressure Add a Usable Control Dimension Beyond Token/KV Counters?`

**只有正结果后：**

- `Expert-Pressure-Conditioned Concurrency Control for SLO-Constrained MoE Serving`

**边界/测量结果：**

- `Measuring Feasible MoE Decode Concurrency under Endogenous Expert Routing`

**稳定负结果：**

- `When Expert Pressure Fails to Improve MoE Serving Capacity Control`

### 14.2 当前可用的 proposal abstract

现有 LLM serving scheduler 主要依据 Token、KV Cache、队列与历史延迟控制并发，但这些状态未必完整描述 MoE decode 中 expert/rank load、kernel shape 与通信所形成的关键路径。本研究不预设 Expert Pressure 必然有用，而是检验已完成 epoch 的 max expert/rank routed-token load，是否会在强 route-free baseline 之外稳定修饰 decode-budget 对完整请求 TPOT/ITL SLO 风险的边际影响。为避免路由内生性造成伪反事实，我们设计从同一 pre-action state 分叉、让各 budget 独立演化 request set、KV、route、queue 与 completion 的 native-runtime 实验，并以固定 request cohort 的 SLO-goodput 为主分母。若该 interaction 与 action Oracle headroom 同时成立，将实现 Token/KV baseline 上的单级 pressure correction；否则给出 Pressure 信号失效、action space 消失或仅在特定 batch/KV/EP 区间有效的边界结论。

### 14.3 当前证据下的简历表述

> 设计 MoE Serving 的 action-conditioned 并发容量实验：冻结 decode concurrency 单一动作，以 same-prestate、policy-specific 执行避免动态 batching 下的 KV/route 反事实泄漏，并构建 Token/KV/queue/recent-latency 强基线与完整 request SLO 分母；当前完成协议与查新，native capacity/Controller 收益尚待验证。

### 14.4 正结果后的条件模板

> 在 `[model/runtime/hardware]` 上实现 Expert-Pressure-Conditioned Decode Controller；通过 same-prestate action branches 验证 max expert/rank load 对 budget marginal SLO risk 的独立增量，在 telemetry 开销 `[X%]` 下相对最强 Token/KV/latency baseline 提升 SLO-goodput `[Y%]`，并明确 `[positive regime]` 与 fallback boundary。

`[X]`、`[Y]` 必须由 sealed native artifact 填写，当前不得替换成估计值。

### 14.5 负向/边界结果后的简历模板

> 构建 MoE Serving 的 policy-specific concurrency Oracle 与 full-request accounting，证明 Expert Pressure 在 `[regime A]` 提供独立 action residual、但在 `[regime B]` 被 Token/KV/历史延迟或 attention critical path 吸收，从而给出 Expert-aware admission 的启用与退化边界。

## 15. 最终固定字段

```text
Verdict:
  PROTOCOL_EXECUTABLE / SCIENTIFICALLY_UNVERIFIED / NO_METHOD_GO

Evidence type:
  Existing evidence stops at CUSTOM_CONTINUOUS_RUNTIME / MEASUREMENT_ONLY.
  This document is a reviewed experiment protocol, not new experiment data.

What was measured:
  Existing same-prestate A/B/C/D execution-conformance behavior in one OLMoE
  revision on one RTX 5090 custom cached-decode runtime.

What was not measured:
  Native request latency, pressure-conditioned budget effect, SLO-goodput,
  Controller gain, second model, 8xA100 EP/A2A.

Strongest baseline:
  Token/KV/queue/padding/age + previous budget/tokens + total routed tokens
  + recent request ITL / step-latency feedback.

Oracle/headroom status:
  NOT_RUN / ACTION_ORACLE_STILL_PAUSED.

Claim ceiling:
  Research proposal and executable protocol only.

Failure category:
  None yet for the new formulation; critical science Gate is UNRUN.

Resurrection condition:
  Native qualification plus a stable, matched, request-level pressure-budget
  interaction or a new EP regime that changes the denominator.

One next smallest experiment:
  N0a one steady + one bursty native A/C/D transfer, together with N0b
  telemetry parity, overhead, snapshot/replay and branch-actuator eligibility.
```
