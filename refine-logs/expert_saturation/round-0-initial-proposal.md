# 研究提案：MoE Serving 中 Expert Pressure 的安全并发边界与条件化控制

## Problem Anchor

- **Bottom-line problem：** 在三个月内判断：控制 Batch/Token/KV/队列等普通 serving state 后，低开销的 MoE Expert Pressure 是否仍能稳定解释和改变 SLO-safe decode concurrency；只有该 residual 与 action headroom 都成立，才实现最小在线 Controller。
- **Must-solve bottleneck：** 当前“Expert Saturation 会改变安全并发”只是合理假设。仓库现有单卡 custom cached-decode 证据发现 batch-dependent execution conformance，但 route-augmented latency diagnostic 跨未受控运行发生符号翻转；`running_set_budget` 的 action-conditioned Oracle 从未运行。若不先关闭这一链条，直接实现 Controller 会把 route/pressure 的内生变化当作外生标签。
- **Non-goals：** 不改变 Router、top-k、expert identity 或模型语义；不做 token-level rebatching、KV migration、expert placement/replication、跨请求迁移、RL、多 Controller 联合优化，也不从零实现 serving engine。
- **Constraints：** 本地一张 RTX 5090 用于 OLMoE 开发与单卡存在性测试；最终可用 8×A100 做 native Expert Parallel；总周期约 12 周；所有论文级结果必须采用完整 request denominator、保留原始 runs，并与最强简单策略比较。
- **Success condition：** 先在 representative native runtime 中证明 MoE-specific pressure 对 fresh holdout 的 tail-latency / dangerous-underprediction 有稳定增量；再以 action-conditioned rerun 证明 safe-concurrency Oracle 有 material SLO-goodput headroom；最后一个只调节 active decode concurrency 的小 Controller 在完整成本后超过 Token/KV-only baseline。任一前置 Gate 失败都产出精确适用边界并停止后续机制。

## Repository Reality Snapshot

- 当前 HEAD：`b141c1d587fe2c918643c3c7c3a8f5f5157d4c8a`；初核时工作树干净。
- 当前科学状态：formal method/system GO 数为 0。已有 `MIXED_SOURCE_IDENTIFIED / MEASUREMENT_ONLY` 只覆盖一个 OLMoE revision、一张 RTX 5090 和 custom cached-decode runtime。
- 已测：同 pre-state A/B/C/D source localization；batch/KV/padding 与 companion identity 可沿 attention/MoE output 产生 repeat-stable numerical/route difference。
- 未测：native runtime transfer、第二模型、request-level latency、TPOT/P99、8×A100 EP、capacity action、action Oracle、Controller gain。
- 当前唯一合法前驱：先完成 native runtime conformance/telemetry qualification；Route Capacity 观察分支仍为 `NEEDS_CONTROLLED_REPEAT`，因果 action 分支为 `ACTION_ORACLE_STILL_PAUSED`。

## 事实、假设与推断分层

### 已有证据支持的事实

1. 普通 token parity 不保证 custom runtime 中 serial/batched expert assignment 一致；batch execution condition 会改变数值轨迹。
2. 现有 custom-runtime M3-vs-M2 latency diagnostic 曾出现 `+9.63%` 与 `-24.04%` 的未受控符号翻转，不能作为稳定 capacity signal。
3. 每个 candidate concurrency/budget 会改变 admitted request set、batch/KV/padding、route、expert load 和 completion timeline；固定 future-route replay 不是合法 action Oracle。
4. 8×A100 EP 的 All-to-All、慢 rank 与 request-level SLO 后果尚未测，单卡结果不能替代。

### 当前提出的假设

1. 在部分 decode operating regime 中，`max-rank routed-token load`、`max-rank active working-set` 或历史 A2A pressure 在普通 workload state 后仍有 tail-latency residual。
2. residual 主要存在于 saturation knee 附近，而不是所有 batch size；Active Expert Count 可能快速饱和，只是候选信号，不是默认主角。
3. 当 residual 存在时，一个低维、单调、分段模型可能足以选择下一 scheduling tick 的 active decode concurrency。

### 尚未验证的推断

1. Expert Pressure 能提高 TPOT/P99/SLO attainment 或 Goodput。
2. 单卡 OLMoE 的规律能迁移到 fused native runtime 或 8×A100 EP。
3. `active expert union` 比 `max expert/rank token load`、A2A time EMA 或最近 step latency 更有价值。
4. Controller 收益不会被 TTFT、等待时间、公平性与 telemetry overhead 抵消。

## Technical Gap

现有工作已分别覆盖两侧：一侧用 token/KV/queue/预测延迟做 SLO admission、batching 或 autoscaling；另一侧用 expert activation/load 做 placement、replication、prefetch、redistribution、request dispatch 与 expert balancing。最强直接碰撞 Gimbal 已把近期 EP-rank expert token pressure反馈到 DP-engine request dispatch，同时联合 expert placement。

因此，“加入 Expert Pressure 做调度”不具备独立新颖性。当前只剩一个可检验的窄 gap：

> 在固定 placement、routing rule 与 execution semantics 下，MoE-specific pressure 是否定义了一个传统 Token/KV/queue counters 无法表达的 **within-engine SLO-safe decode concurrency boundary**；这一边界能否通过 action-conditioned execution 被最小 Controller 捕获？

如果没有跨模型/运行域 residual，本项目应收敛为负向 measurement thesis，而不是继续换 predictor。

## 两条候选路线与选择

### Route A：最小、可证伪路线（选择）

1. native runtime 先采集 aligned workload / route / expert / rank / timing ledger；
2. 用 workload-only quantile model 对比加入最多两个低开销 MoE features 的同类模型；
3. 只有 residual 稳定后，执行每个 budget 独立演化的 action Oracle；
4. 只有 Oracle material 后，实现一个 monotone quantile + hysteresis Controller。

### Route B：复杂预测路线（拒绝）

按 request/domain 预测 future route，叠加 Gradient Boosting/RL，同时控制 admission、prefill ratio、placement 与 token budget。它增加未来信息、action endogeneity、feature search 和系统集成风险，也与已有 route-prediction / coordinated scheduling 工作高度碰撞；三个月内无法形成清楚的因果主张。

## Method Thesis

- **一句话 thesis：** Expert Pressure 只有在通过“普通 workload residual → action-conditioned safe-capacity headroom → minimal feedback recovery”三门 Gate 后，才是 MoE serving 的有效并发状态变量；否则系统应自动退化为 Token/KV-aware scheduling。
- **最小充分干预：** 只调节 `active_decode_budget`，不更改 router、expert placement、batch composition policy 或 precision。
- **Frontier primitive：** 无需 LLM/RL/深度模型。核心是 serving runtime instrumentation、因果 action rerun 与低维 tail-risk model；强行加入新模型只会掩盖信号是否存在。

## Contribution Focus

- **Dominant contribution：** 定义并实测 MoE-specific pressure 对 within-engine SLO-safe decode concurrency 的条件性容量边界，严格区分普通 workload、历史压力与 action-conditioned future state。
- **Optional supporting contribution：** 在边界成立的 regime 中，用一个可回退的低维 quantile Controller 捕获部分 Oracle headroom。
- **Explicit non-contributions：** 不主张新 Router、expert load balancer、placement、replication、A2A collective、general-purpose SLO scheduler 或复杂 latency predictor。

## Proposed Method

### Complexity Budget

- **Frozen/reused backbone：** vLLM native OLMoE path用于 5090 transfer/profiling；8×A100 使用冻结版本的 vLLM native EP path与一个其正式支持的 MoE model；arrival replay、request ledger、metrics recorder 尽量复用现有资产。
- **New components：** (1) 低开销 pressure collector；(2) 单一 `active_decode_budget` Controller。
- **Intentionally excluded：** per-request route predictor、GBDT/RL 主策略、dynamic placement、token reshuffle、KV migration、precision action、多动作联合优化。

### System Overview

```text
Ready Queue + Active Decode Set
            |
            v
ordinary state x_t -----------------------------+
(queue, active, token, physical KV, prefill)    |
                                                 v
router/EP counters -> pressure p_t -> EMA -> quantile cost model
                                                 |
candidate active_decode_budget b ----------------+
                                                 v
                             safety margin + uncertainty gate
                                                 |
                      max safe b / Token-KV fallback
                                                 |
                          next native scheduling tick
                                                 |
                  request/step/route/cost ledger (append-only)
```

### Pressure Representation

不一次接入全部 features。用 feature ladder 决定保留项：

1. **Workload baseline `x_t`：** scheduled decode tokens / running sequences、queue depth、mean/max physical KV length、padding ratio、KV occupancy、prefill tokens、recent step-latency EMA。
2. **Load candidate `p_load`：** 每层/每 rank routed-token count 的最大值或 max/mean；单卡退化为 `max expert tokens`。
3. **Working-set candidate `p_ws`：** 每层/每 rank active expert count 或对应 active weight working-set；必须显式测试是否很快饱和。
4. **Communication candidate `p_comm`：** 上一已完成 step 的 A2A time/bytes EMA，只在 EP 中使用；同 step 的 future A2A 不可作为 pre-action feature。

最终 Controller 最多保留两个 MoE-specific scalars。优先级是 `p_load`，其次在 fresh holdout 确有增量时才加入 `p_ws` 或 `p_comm`。

### Runtime Cost Model

主模型是 P95/P99 条件分位数的单调分段线性模型：

```text
Q_tau(T_step[t+1] | H_t, b)
  = beta_0
  + beta_x * workload_state[<=t]
  + beta_b * proposed_budget[b]
  + beta_p * pressure_EWMA[<=t]
  + beta_h * relu(pressure_EWMA - learned_knee)
  + beta_i * proposed_budget * relu(pressure_EWMA - learned_knee)
```

其中 `H_t` 只含 `t` 时刻前已完成的观测。`b` 的训练样本必须来自实际执行过该 budget 的 trajectory；不使用同一 observed future route 伪造不同 `b` 的标签。

对比模型：

- M0：last-step EMA / fixed budget；
- M1：Token/KV/queue-only Ridge 或 quantile regression；
- M2：M1 + current max-rank/per-expert load（最强简单 MoE baseline）；
- M3：M2 + 一个候选 working-set/communication feature；
- M4：future-known、action-conditioned Oracle，只作 headroom 上界。

小型 depth-limited tree/GBDT 仅作诊断。如果 M3 的收益只存在于复杂模型而线性/分段模型不稳定，不实现在线 Controller。

### Full-request Accounting

```text
request time
  = ready-queue wait
  + prefill wait + prefill execution
  + sum(decode inter-schedule wait + model step + communication)
  + scheduler/telemetry overhead
```

主结论用 request-level TTFT、TPOT/ITL、SLO attainment 与 SLO-goodput；MoE kernel/A2A/step latency 只解释机制，不代替完整分母。

### Minimal Controller State

```text
budget_current
budget_min / budget_max
pressure_ema
step_latency_ema
quantile_model + calibration_residual
confidence / out_of_distribution flag
safe_margin
hysteresis_up / hysteresis_down
dwell_steps
waiting_queue arrival times
last_budget_change_step
```

### Decision Logic

```text
on scheduling_boundary(t):
    x = observe_ordinary_state(up_to=t)
    p = observe_completed_pressure(up_to=t)

    if telemetry_missing or model_uncalibrated or OOD(x, p):
        return token_kv_baseline_budget(x)

    candidates = local_budget_grid(budget_current, budget_min, budget_max)
    feasible = []
    for b in candidates:
        q = predict_tail_step_time(x, p, proposed_budget=b)
        q = q + uncertainty_margin(x, p, b)
        if q <= decode_step_slo and kv_feasible(x, b):
            feasible.append(b)

    target = max(feasible) if feasible else budget_min
    target = apply_hysteresis_and_dwell(target, budget_current)
    target = apply_aging_guard(target, waiting_queue)
    return target
```

`aging_guard` 只能在已冻结的公平性规则内改变等待队列优先级，不能绕过 KV 或 decode safety bound。高不确定度、冷启动、telemetry 缺失和 signal-dead regime 都回退到同 action space 的 Token/KV-only baseline。

## Claim-Driven Validation Sketch

### Claim 1：MoE pressure 存在稳定的增量容量信息

- **Minimal experiment：** native runtime 中按相同 token/physical-KV/prefill/queue cells 配对，比较 M1、M2、M3；document/arrival-episode disjoint holdout；steady + bursty；保留 controlled repeats。
- **Baselines：** workload-only quantile model、current max-rank/per-expert-load simple model、last-step EMA。
- **Metric：** P95 pinball loss、dangerous underprediction、calibration error；并报告 residual 对完整 step/request critical path 的 survival。
- **Positive：** 至少两个冻结运行域方向一致，且增量不是由未来 route、padding/backend change 或 instrumentation 解释。
- **Negative：** Active Expert/working-set 很快饱和或 residual 消失，则停止 Controller，形成适用边界。

### Claim 2：最小 Controller 能捕获 action-conditioned headroom

- **Minimal experiment：** 从同一 pre-action state 对每个 candidate budget 独立推进 request set、KV、route、queue 与 completion；先计算 Oracle，再比较 fixed max、Token/KV-only、simple pressure threshold 与最终 Controller。
- **Metrics：** 同一 TPOT/TTFT SLO 下的 Goodput、P95/P99 TPOT、TTFT、等待时间、公平性、overhead。
- **Positive：** Oracle material；简单 threshold 未覆盖绝大部分；最终策略在完整成本后净正。
- **Negative：** Oracle 小、简单策略已覆盖、或 TTFT/公平性税抵消，停止复杂化并报告 formulation/regime NO-GO。

## Failure Modes and Diagnostics

1. **Active Expert Count 饱和：** 画 `active_experts` 对 batch 的 coverage curve；若早饱和，删除该特征，保留 rank load/imbalance。
2. **Attention/KV 主导：** 分解 whole-step critical path；若 MoE exposed share 太小，退化为 Token/KV-only。
3. **A2A/慢 rank 才是主因：** 在 EP 中优先 `max-rank tokens + A2A EMA`，不坚持 active expert count。
4. **Telemetry overhead：** paired OFF/ON、token/logit/completion parity、GPU-side reduction；超过 1% 或改变 execution path 即不进入 Controller。
5. **TPOT 改善但 TTFT/等待恶化：** 用双 SLO + aging/fairness guard；Goodput 按同时满足 SLO 的请求计数。
6. **跨模型/kernel 不一致：** 将结论写成 regime boundary；只在 feature selector 的信号有效区启用，其他区 fallback。
7. **突发下预测失准：** quantile calibration + uncertainty margin + fast-down/slow-up hysteresis；禁止事后换 canonical run。

## Novelty and Elegance Argument

本提案不把 expert pressure、admission 或 quantile model 分别包装成新组件。Gimbal 已证明 backend expert pressure 可反馈到 DP-engine dispatch；SCORPIO、SLOs-Serve、Chiron 等已覆盖 SLO-aware admission/token allocation/backpressure；MoE-Infinity、Sem-MoE、Lina 等已把 activation/load 用于 prefetch、placement 或 redistribution。

潜在 residual 只在完整链条成立时存在：

```text
fixed placement/router
  + within-engine decode active-set action
  + ordinary-counter residual
  + endogeneity-safe action rerun
  + request-level SLO-goodput
```

若链条失败，贡献应改写为“Expert Pressure 何时不能提供安全容量信息”的 measurement/boundary result，而不是更换 predictor 抢救。

## Compute & Timeline Estimate

- 5090 native transfer + profiling：约 20–60 GPU-hours，视 runtime instrumentation 重编译次数而定。
- 5090 action-grid pilot：仅在 residual Gate 后，约 20–40 GPU-hours。
- 8×A100 EP confirmatory：约 80–160 node-GPU-hours；只在单卡/小规模 Gate 通过后申请。
- 代码规模目标：在已有 runtime 上新增 collector + controller，探索阶段新增核心逻辑尽量小于约 500 行；超过前必须先证明更小 A/B probe 不足。
- 时间：12 周；任何时刻只有一个 active Gate。

## Initial Verdict

`REVISE / PROCEED_AS_MEASUREMENT_FIRST`。

该方向适合作为三个月硕士论文的条件化探索，但“Expert-Saturation-Aware Controller”本身与已有工作碰撞严重。当前最合理的论文中心不是 Controller，而是 **MoE pressure 是否形成独立、可执行的 SLO-safe concurrency boundary**。Controller 是 Gate 通过后的 supporting contribution，不是预设结果。
