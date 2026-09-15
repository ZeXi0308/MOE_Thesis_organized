# Round 2 Refinement

## Problem Anchor

- **Bottom-line problem：** 在三个月内判断：控制 Batch/Token/KV/队列等普通 serving state 后，低开销的 MoE Expert Pressure 是否仍能稳定解释和改变 SLO-feasible decode concurrency；只有 residual 与 action headroom 都成立，才实现最小在线 Controller。
- **Must-solve bottleneck：** 当前“Expert Saturation 会改变可行并发”只是合理假设；route/pressure 与 action 后的 request set、KV、batch、route 和 completion 共同演化，固定 future trace 不能构成 counterfactual。
- **Non-goals：** 不修改 Router/top-k/expert identity/precision/placement；不做 token reshuffle、KV migration、RL、复杂 predictor 或多动作联合优化。
- **Constraints：** RTX 5090 用于 OLMoE native qualification 与小规模 probe；8×A100 仅在 Gate 通过后用于 EP confirmatory；总周期约 12 周；一次只推进一个 Gate。
- **Success condition：** 在 representative native runtime、完整 request denominator 和 action-conditioned branches 上，证明 pre-action expert pressure 是提高 decode budget 所致 SLO risk 的稳定 effect modifier，并且 Oracle headroom material；随后最小 correction Controller 在完整成本后超过 Token/KV-only baseline。

## Anchor Check

- **Original bottleneck：** 普通 Token/KV/queue counters 是否遗漏一个 MoE-specific、可执行的 concurrency state variable。
- **为什么修订后仍解决它：** 修订把“预测慢 step”收紧为“pressure-conditioned budget treatment effect”，直接检验该变量是否改变同一个 budget action 的边际 request-level 代价。
- **拒绝的 drift：** future-route prediction、dynamic placement/replication、router/top-k/precision 修改、prefill ratio 联控、execution-conformance repair method。

## Simplicity Check

- **Dominant contribution：** 一个经 action-conditioned execution 识别的 `pressure × budget` feasible-capacity boundary。
- **删除/降级：** 在线路径删除 active-working-set 主特征、GBDT 和 A2A predictor；active expert union 只做一次饱和诊断，A2A EMA 只作 EP 解释变量。
- **最小机制：** Token/KV baseline 上减一个 pressure penalty；没有 residual/headroom 就没有 Controller。

## Changes Made

1. **冻结共同 pre-action state。** 以完整 `Z_t` 匹配分支，`X_t(b)=g(Z_t,b)` 仅作为候选 budget 的 action-before 特征，不再把两个不同 `X_t(b)` 混写成同一条件变量。
2. **冻结 paired 主估计器。** 主要因果量直接由 same-prestate branch 的 request-risk difference 得到；quantile interaction 只用于解释和在线控制，不承担 treatment-effect 结论。
3. **冻结完整分母。** 每个 branch 从分叉点运行至预动作 cohort 全部完成；分叉后 arrivals 按同一冻结 trace 接纳并计入相同 observation cohort，不用短 horizon 截断未完成请求。
4. **补强 route-free controls。** previous budget、previous scheduled tokens、total routed tokens 与 recent latency 均进入强普通状态基线，防止 `p_t` 只是上一动作的记忆变量。
5. **合并最小机制。** pressure-threshold correction 就是最终 Controller，不再人为制造一个更复杂的 proposed model。
6. **拆开资格 Gate。** N0a 记录 native A/C/D transfer；N0b 独立记录 telemetry parity、overhead 与 branch actuator eligibility。两者都不是 capacity evidence。

## Revised Proposal

# 研究提案：MoE Serving 中 Expert-Pressure-Conditioned Feasible Decode Capacity

## 1. Evidence-bounded Problem

当前仓库只支持：一个 OLMoE revision、一张 RTX 5090、custom cached-decode runtime 中存在 repeat-stable batch-dependent execution-conformance difference；不支持 native serving、request latency、EP、capacity 或 Controller 结论。

本研究不预设 expert pressure 有用，而问：

> 在固定 scheduler order、placement、router 和 arrival episode 下，控制候选 budget 对应的 Token/KV/padding/queue state 后，已完成 epoch 的 max expert/rank load 是否稳定改变“提高下一 epoch decode concurrency”所带来的 request-level SLO risk？

## 2. Central Estimand

唯一 action：

```text
b_t = 下一 scheduling epoch 内，每次调度迭代最多选择的 decode sequences 数
```

固定请求顺序为 `FCFS + frozen aging tie-breaker`；Controller 只能截断该顺序。Prefill scheduling、router、top-k、expert placement/replication、precision 和 KV policy 均不变。

共同 pre-action state 冻结为：

```text
Z_t = {
  complete ready/active queue snapshot,
  frozen request order and request identities,
  per-prefix Token/KV/padding/age summaries,
  KV occupancy and frozen prefill state,
  previous budget, previous scheduled tokens,
  previous total routed tokens and recent latency
}

X_t(b) = g(Z_t, b)

tau(Z_t, p_t)
  = E[ Risk_t(b_high) - Risk_t(b_low) | Z_t, P_{<=t}=p_t ]
```

- `Z_t`：所有 paired branches 共享的完整 action-before state 和 matching unit。
- `X_t(b)`：由 `Z_t` 与候选 `b` 确定的 candidate-specific ordinary state，包括前 `b` 个固定顺序请求的 decode tokens、physical KV lengths、padding、queue age、KV occupancy 与 frozen prefill state。
- `P_{<=t}`：只来自已完成 epoch 的 expert pressure。
- `Risk_t(b)`：从 branch point 运行至预动作 cohort 全部完成时，对同一 observation cohort 计算的 request-level TPOT/ITL miss risk；分叉后 arrivals 使用完全相同的冻结 trace 和 admission rule。TPOT 分母是 cohort 中全部 at-risk decode-token intervals；TTFT guard 分母是同一 arrival cohort；未完成或失败请求按预注册 miss 计入，不从分母删除。

Action `b` 的分支是因果的；pressure 是观测到的 effect modifier，不主张人为提高 pressure 会导致风险。

## 3. Minimal Signal Surface

### Online input

只保留一个 MoE-specific scalar：

```text
p_load(t) = max over completed epoch/layer/rank of routed_token_count
```

单卡时退化为 `max expert token count`；EP 时为 `max rank routed-token load`。通过 GPU-side aggregation 在 epoch 边界异步导出，避免 per-layer host synchronization。

### Diagnostics only

- `active expert union / working-set`：只画 saturation curve，检验是否快速失去区分度；不进入首版 Controller。
- `A2A bytes/time EMA`：只在 EP 解释 residual 来源；不作为首版 action feature。
- attention/MoE/kernel breakdown：只做 critical-path survival 分解。

### Ordinary state baseline

```text
scheduled decode sequences
candidate physical KV mean/max and padding ratio
KV occupancy
queue depth and candidate ages
frozen prefill tokens/state
recent request-level ITL / step-latency EMA
```

## 4. Response Model and Interaction Test

I1 的主估计器是 same-prestate paired difference，而不是预测模型系数：

```text
tau_hat(p-bin)
  = mean over matched states s [Risk_s(b_high) - Risk_s(b_low)]
```

低压与高压 bin、matching/covariate-adjustment、bootstrap unit 和 canonical repeat 规则均在运行前冻结。Quantile response surface 只用于 residual 解释和条件性 Controller，而非因果主结论：

```text
Q_tau(Y | x_t(b), b, p_t)
  = f_TK(x_t(b), b)
  + alpha * relu(p_t - k)
  + gamma * b * relu(p_t - k)
```

- `f_TK` 是 Token/KV/queue-only baseline；
- `k` 只在 calibration split 冻结；
- `gamma` 只做 pressure-by-budget response 诊断；核心 interaction 由 paired `tau_hat` 及其 fresh-holdout 稳定性裁决；
- 数据必须来自实际运行过的 budget branches，不能来自 current scheduler 的单一 action trace。

Feature ladder：

```text
M0 fixed / EMA
M1 route-free strong baseline: Token/KV/queue/padding/ages + previous budget/tokens + total routed tokens + recent latency
M2 M1 + completed max expert/rank load main effect（diagnostic only）
M3 M1 + pressure x budget hinge interaction
M4 future-known action-conditioned Oracle
```

M1 是 strongest route-free baseline；M2 仅检验 pressure 是否预测当前慢态，不能证明 action relevance。只有 paired `tau_hat` 随 pressure 稳定变化，且 M3 相对 M1 在 fresh holdout 的 dangerous underprediction、calibration 与 request-level action effect 上产生 material 增量，才存在候选容量轴。

## 5. Empirically SLO-Feasible Boundary

在冻结 arrival episode、scheduler 与 SLO 下：

```text
b*(x,p) = max b such that
  TPOT/ITL SLO attainment >= target
  TTFT guard does not regress beyond tolerance
  fairness/maximum-wait guard passes
  KV is feasible
```

没有 request-level evidence前不使用 “safe capacity”。Step latency、MoE kernel 和 A2A 只解释为什么 boundary 移动。

## 6. Minimal Controller

只有 interaction 与 Oracle Gate 都通过才实现：

```text
on_epoch_boundary:
    b0 = token_kv_baseline(candidate_state)
    p  = completed_max_expert_or_rank_load_ema()

    if telemetry_missing or uncalibrated or OOD:
        return b0

    penalty = one_budget_level if p > frozen_knee else 0
    target = clamp(b0 - penalty, b_min, b_max)
    return fast_down_slow_up_hysteresis(target)
```

维护状态只有：当前 budget、Token/KV baseline budget、pressure EMA、frozen knee、上下阈值、dwell steps、telemetry validity、request ages。Controller 不预测 future route，也不搜索多动作计划。

## 7. Strong Baseline Ladder

1. native/default maximum concurrency；
2. fixed best budget；
3. Token/KV/queue-only adaptive budget；
4. recent-step-latency feedback；
5. simple pressure-threshold correction，即最终最小 Controller（最强简单/最接近 Gimbal-style feedback）；
6. future-known action-conditioned Oracle。

若 simple pressure threshold 捕获 >=90% Oracle，论文只保留简单策略，不增加模型。

## 8. One Decisive Experiment

### Eligibility sub-gate N0（当前唯一先执行项）

- **N0a — conformance transfer：** 在 vLLM native OLMoE path 重放仓库冻结的一 steady + 一 bursty A/C/D event，保持 target token/position/KV 与 matched companions，只记录 native runtime 是否迁移该现象。
- **N0b — harness eligibility：** 验证 pressure telemetry OFF/ON parity、开销、state snapshot/replay 与 branch actuator。

N0a 失败只说明既有 custom-runtime conformance 现象未迁移，不证伪 pressure-capacity hypothesis；依照当前仓库过程权威，应停止并显式决定是否 reopen I1，不能自动绕过。N0a/N0b 任何结果都不计作 capacity evidence，也不扩展 operator localization。

### Action-conditioned branch Gate I1（仅 N0 通过后）

1. 从 Token/KV profiling knee 周围选择三个 budgets：`b_low, b_mid, b_high`。
2. 从 matched replay prefix 重建同一 pre-action request/queue/KV state。
3. 三个 branch 分别运行至预动作 cohort 全部完成；branch 后独立维护 request set、KV、route、pressure、queue 与 completion timeline，分叉后 arrivals 使用同一冻结 trace。
4. 预动作只读取 `P_{<=t}`；为每个 `b` 计算 `X_t(b)`。
5. steady 与 bursty 各包含低/高 pressure 且 ordinary-state common-support 的 states；process-isolated repeats 全保留。
6. 负控：within-matched-cell pressure permutation 或 decorrelated pressure pairing。
7. 主估计器：pressure bin 内 same-`Z_t` paired risk difference `tau_hat`；主结果为 request-level TPOT/ITL SLO attainment、SLO-goodput 与 dangerous underprediction；次指标为 TTFT guard、公平性与 overhead。
8. 分母冻结：所有 branch 共享相同 pre-action cohort、arrival trace、SLO definition 和 failure/censor rule；任何未完成请求不得静默删除。

继续信号：interaction direction repeat-stable，探索性 Oracle SLO-goodput headroom `>=3%`，且 ordinary-state + last-latency baseline 不能吸收。正式 claim threshold 在 holdout 前另行冻结并沿用现有 RouteShape-SLO 边界。

停止：sign flip、无 common support、dangerous underprediction 不改善、Oracle `<3%`、simple baseline 捕获 >=90%、或完整 request denominator 不闭合。

## 9. Claim Map

- **C1（主张）：** 在预注册的强 route-free baselines 之外，completed max expert/rank load 在 fresh holdout 上对 decode-budget 的 request-level paired action effect 提供稳定且 material 的增量；只有跨 frozen regime 或 EP confirmatory 后仍成立，才称为新的 empirical feasible-capacity boundary。
- **C2（条件 supporting）：** 一个 Token/KV baseline 上的单-level pressure correction 能在完整成本后捕获 material Oracle headroom。
- **Anti-claim：** active expert count 本身不新；pressure 预测 latency 不等于支持 action；within-engine 控制位置不同不自动产生 novelty。

## 10. Three Allowed Outcomes

1. **Boundary + Controller positive：** interaction、Oracle、simple policy 和 8×A100 EP confirmatory 均成立。
2. **Measurement boundary only：** interaction只在特定 batch/KV/EP regime 存在，但 Controller 收益被简单 baseline或成本覆盖。
3. **Signal-dead negative：**普通 state/last-latency 已解释风险，或 active expert/load很快饱和；停止 Controller并报告退化条件。

## 11. Novelty Position

Gimbal 已使用 recent expert-token pressure + KV/prefill/queue 做 DP-engine dispatch并联合 placement；SCORPIO 与 SLOs-Serve 已覆盖 predictive SLO admission/token allocation；arXiv:2608.13962 已明确描述 decode SLO cap 与 activated-expert union。因而以下都不能作为新颖性：expert pressure、active expert saturation、quantile admission、hysteresis或“控制位置不同”。

唯一可保留 residual 是：

> fixed placement/router/native execution 下，经 same-prestate action branches 识别 pressure-conditioned decode-budget treatment effect，并用完整 request SLO 分母定义其 operating boundary。

没有该结果就不写独立 method claim。

## 12. Feasibility and Timeline

- Week 1–2：只做 N0 native transfer/telemetry qualification。
- Week 3：最小 native profiling，冻结 Token/KV knee、epoch horizon 与三个 budgets。
- Week 4–5：执行 I1 branch Gate；先回答 interaction/headroom。
- Week 6：决定论文路径；negative 则停止 Controller，转 boundary analysis。
- Week 6–8（仅 positive）：实现并评估单-level correction Controller。
- Week 9–10（仅 positive）：8×A100 native EP confirmatory；先复核 signal，再评估 Controller。
- Week 11：controlled repeats、风险/退化分析、主图表。
- Week 12：论文写作与 claim audit。

计算预算以 Gate 为单位申请；8×A100 不在 N0/I1 前运行。探索实现只新增 collector、branch actuator 和最小 correction，避免超过约 500 行核心逻辑。

## Final Method Verdict

`PROTOCOL_FROZEN_FOR_FINAL_REVIEW / MASTER-THESIS-VIABLE / CONTROLLER-CONDITIONAL`。

提案已经从 feature-plus-controller 组合收紧为一个可被单实验证伪的 pressure-conditioned action-effect hypothesis，并冻结了 matched state、paired estimator 与完整 request denominator。当前仍没有方法 GO；唯一授权的下一步是 N0 native qualification，随后才是 action-conditioned I1。
