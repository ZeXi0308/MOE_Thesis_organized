# Round 1 Refinement: HorizonFence

> 题目：**When the Scheduler Creates the Hotspot**  
> 修订方法名：**HorizonFence — A No-Prediction Futility Certificate for Slow Expert Migration**  
> 上一轮：`5.20 / 10, RETHINK`  
> 状态：`EXPLORATORY / NOT_CURRENT_MAINLINE / NO_EMPIRICAL_RESULT`  
> 日期：2026-07-29

## Problem Anchor

- **Bottom-line problem**：在固定自然 arrival、request identity、完整 route ledger 和 exact model semantics 的条件下，判定反应式 MoE expert load balancer 使用的有限窗口 post-scheduling expert statistics 是否会被 scheduler-induced backlog composition / right censoring 显著改变，进而触发错误 reconfiguration；若该现象成立，给出不伪造未来 route 的可识别证据边界和最小证书。
- **Must-solve bottleneck**：必须把真正的外生 workload shift 与策略诱导的 exposure shift 分开，并把归因差异连接到 hotspot/change-point 决策反转和 full request-DAG 净损失；只展示不同时间索引的曲线不算解决问题。
- **Non-goals**：不提出新的 expert placement optimizer、prefetch、router predictor、route/weight/precision 修改或 receiver controller；不声称 scheduler 永久改变长期真实需求；不复活 PhaseMap、FJRC、ConfidenceGuard、RankLane、CreditReduce、MassCover、TokenRace、QuotaEP、PLTB、RouteFidelity 或 Prefetch。
- **Constraints**：保持当前 `docs/current/README.md` 的共同现象 Gate 为唯一正式执行线；本方向仅为隔离探索。实验必须 exact semantics、自然 continuous arrival、document-disjoint、至少两个模型、冻结 detector/gate/denominator；失败后不得改 workload、阈值、cohort 或索引救活。现有 OLMoE/LLM-jp 与 RTX 5090 只够资格性 pilot；正式 EP/TPOT/P99 主张需要至少一个 serving-scale open MoE 和真实 optimized multi-GPU EP。
- **Success condition**：全文 prior-art 查杀后仍不存在等价的 fixed-arrival/fixed-route scheduler intervention 与 partial-identification certificate；两个模型自然 workload 均出现预注册的策略诱导假热点或假变点；certificate 不退化为 always-abstain，并保留真实热点；错误 reconfiguration 在完整路径上的影响达到预注册系统显著性门；最终在真实 multi-GPU EP 上复核，且 fresh same-family CCF-B mock review 无 fatal novelty/method objection。任一硬门失败即 NO-GO。

## Reviewer Objection Accepted

上一版把 arrival-cohort popularity 当成 action target，这是不充分的。scheduler-induced backlog 可以制造真实的近期 executed load：对快速 token assignment/replica selection，立即响应可能正确；只有对生效慢、必须回本的 expert migration，短时热点才可能在动作生效前消失。

因此本轮做四个收缩：

1. arrival-cohort decomposition 只保留为**归因变量**，不再把 policy-induced load 自动称为 false load；
2. 删除 generic change-point、generic detector 和 generic LP 支持；
3. 只冻结一个 per-expert threshold detector 和一个 stop-and-copy slow migration action；
4. 证书不再判断“真实 popularity”，只给出一个更窄的 action verdict：在所有合法 future routes 下，该迁移是否都不可能在冻结 horizon 内回本。

若这个最窄问题仍只是通用 accounting，或证书在动作 deadline 前几乎从不触发，C10 直接 KILL，不加入 predictor、OPE 或新 controller。

## Revised Method Thesis

**One-sentence thesis**：对一个已被现有 EPLB 触发的慢 expert migration，HorizonFence 利用 MoE 每 token、每 layer、top-k 无重复的 route-mass 约束和已知 logical progress，计算动作生效后所有可能受益 route event 的上界；若即使把全部未知 future routes 朝最有利于迁移的方向分配，其最大可能收益仍小于不可避免的 migration cost，则在不预测 future route 的前提下对该动作出具 `PROVABLY_FUTILE` veto。

这不是 beneficial-action selector：

- `PROVABLY_FUTILE`：所有可行未来中该动作净效用 `< 0`；可以安全 veto；
- `UNRESOLVED`：存在可回本未来，也存在不可回本未来；保持现有 policy，不宣称动作好坏；
- `INVALID`：route/progress/admission/cost/semantics ledger 不闭合。

快速负载均衡不在 scope 内。现有 service-window count 继续服务 fast path；HorizonFence 只审计 slow migration。

## Frozen Operational Object

### Detector

只支持一个冻结 detector：在长度为 `W` 的 post-service window 中，expert `e` 的 executed share 超过 `theta`，且 capacity excess 超过 `b`，产生候选动作 `a_e`：

```math
D_e(t)=1\{N_e(t-W,t]/N(t-W,t] \ge \theta\}
       \land 1\{N_e(t-W,t]-c_eW \ge b\}.
```

`W, theta, b` 只在 calibration split 冻结；evaluation 后不得修改。论文不声称支持 arbitrary detector 或 change-point family。

### Slow action

`a_e` 是一个冻结的 stop-and-copy expert replica migration：

- 触发时刻 `t`；
- 最长审计 holdoff `d_max`；
- 实际 holdoff `d <= d_max`；
- 不可隐藏的 apply latency `ell_a`；
- 下一次允许 placement decision 的绝对 horizon `t+H`；
- 迁移 expert set `E_a`、源/目标 GPU、参数 bytes、copy/synchronization path 全冻结；
- 只改变物理放置，不改变 expert identity、top-k、gate weight、token 或 output。

有效受益区间为：

```math
I_a(d)=[t+d+ell_a,\ t+H].
```

若 `d+ell_a >= H`，动作按定义无法在本 horizon 生效，直接为 `PROVABLY_FUTILE`。

### Utility

主效用不是“最终 popularity”，而是冻结 action horizon 内相对 stay-put 的 full-path completion-cost 改善：

```math
U_a = J_{stay}([t,t+H]) - J_a([t,t+H]) - C_a,
```

其中 `J` 是预注册、单调、1-Lipschitz 的 request completion-lateness 总和；`C_a` 是 stop-and-copy 对当前 DAG 造成的不可隐藏 barrier/copy 成本，二者使用同一时间单位。SLO-goodput、TPOT、P99 是 formal validation outcomes，不被这个 scalar certificate 偷换。

## MoE-Specific Future-Route Envelope

对 request `i` 在时刻 `t` 的 logical decode progress 记为 `z_i(t)`，冻结最大长度为 `L_i^max`。在 layer `l`，一个 token 的 top-k set 不含重复 expert，因此对单个 expert `e`，每个尚未生成 token 在该 layer 最多贡献一次 route event：

```math
u_{i,l,e}(t) = max(0, L_i^max-z_i(t)) * 1{e is eligible at layer l}.
```

对当前 admitted cohort：

```math
R^old_{l,e}(t) <= sum_i u_{i,l,e}(t).
```

对 `I_a(d)` 内新 admission，不假设具体 route；只使用预冻结 admission envelope `Lambda_max(I_a)`、最大生成长度和系统物理 token-step capacity。于是：

```math
R^new_{l,e}(I_a) <= min(
  Lambda_max(I_a) * L_max,
  K_l^max(I_a)
).
```

总受影响 route-event 上界：

```math
R^+_{l,e}(t,d,H)=min(
  R^old_{l,e}(t)+R^new_{l,e}(I_a),
  K_l^max(I_a)
).
```

这个 envelope 使用三项 MoE execution structure，而不是预测 route identity：

1. per-layer logical progress；
2. top-k set 对单 expert 的无重复上界；
3. layer-specific execution-capacity ceiling。

如果 admission 没有可审计上界、`L_i^max` 可被运行时扩张，或 batch-dependent numerical behavior 改变 fixed route/output ledger，certificate 为 `INVALID`。

## Action-Sign Upper Bound

从冻结 service surface 取得一次 `(l,e)` route event 因迁移最多能带来的 exposed completion-cost 改善上界 `delta_bar_{l,e,a}`。该值必须是对所有冻结 batch sizes、queue states 和 source-destination paths 的保守上界，而不是平均值。

因此任何合法 future route 下，gross benefit 都满足：

```math
G_a <= G_a^+(t,d,H)
    = sum_{e in E_a} sum_l R^+_{l,e}(t,d,H) * delta_bar_{l,e,a}.
```

从真实 stop-and-copy path 的 calibration trials 取得不可隐藏 migration cost 的 one-sided lower confidence bound `C_a^-`。最终证书为：

```math
G_a^+(t,d,H) < C_a^-
  => U_a < 0 for every feasible future route ledger
  => PROVABLY_FUTILE.
```

否则输出 `UNRESOLVED`。HorizonFence 从不因点估计或平均 prediction 发出 veto，也不把 `UNRESOLVED` 当成 beneficial。

### Soundness obligation

formal proof 只需证明：

1. `R^+` 覆盖所有合法 current/future route ledgers；
2. `delta_bar` 的逐 event 上界在冻结 action/path 上成立；
3. 1-Lipschitz `J` 使逐 event exposed-latency 改善可加地上界 completion-cost 改善；
4. `C_a^-` 是不可隐藏动作成本的合法下界；
5. 严格不等式成立时，动作净效用对所有可行未来为负。

不再声称一个 generic LP 可以验证 arbitrary detector。这个 special-case certificate 由闭式 bound 计算，trigger-time complexity 为 `O(number_of_active_requests * number_of_migrated_experts)`。

## Phase -1: Analytic Non-Vacuity Gate

实现 ledger 前先计算证书可能出现的最早时间。定义：

```math
d_cert(t)=inf { d in [0,d_max] : G_a^+(t,d,H) < C_a^- }.
```

对单 expert、统一每-event上界 `delta_bar_a`，必要条件可写成：

```math
sum_l R^+_{l,e}(t,d,H) < C_a^- / delta_bar_a.
```

冻结 analytic kill rules：

- 在 OLMoE 与第二模型各自的合法参数区间中，若所有 detector trigger 的 `d_cert > d_max`，C10 KILL；
- 若 certificate 只能在 `H-(d+ell_a)` 已小于一个 rebalance interval 时出现，视为“等到动作自然失效”，C10 KILL；
- 若 sound bound 必须令 `delta_bar` 或 admission envelope 取无穷，C10 INVALID/KILL；
- 若 certificate 只因把 future arrivals 设为 0 才非空，C10 KILL；
- 不允许缩短 `H`、降低 `theta`、缩小 `L_max`、改 workload/cohort 或换成 predictor 救活。

只有至少 10% 的 calibration triggers 在 `d<=d_max` 且剩余有效 horizon 至少两个 migration latencies 时可出证书，才进入 Phase 0。这个 10% 只是工程 non-vacuity 门，不是论文成功门。

## Phase 0: One-Detector, One-Action Paired Replay

冻结同一 natural arrival/request/route/output ledger，只改变 FCFS 与 pressure-aware scheduler；负控只平移时间、不改变 active-set order。

对每个 raw detector trigger 记录：

- service-window trigger 是否随 scheduler 改变；
- arrival-cohort backlog attribution；
- `PROVABLY_FUTILE / UNRESOLVED / INVALID`；
- certificate compute time 与 `d_cert`；
- sealed future 下实际 action sign；
- simple longer-window/hysteresis、no-migration、robust static 是否同样避免动作；
- full request-DAG replay 中 stay/migrate 的 completion cost 与 SLO outcomes。

Phase 0 的预注册 kill rules：

- 两个模型自然 cell 的 scheduler-dependent trigger disagreement `<5%`：KILL；
- 两个模型中实际 harmful slow migration 比例均 `<5%`：KILL；
- HorizonFence 对 harmful action 的 recall `<50%` 或 veto precision `<95%`：KILL；
- `PROVABLY_FUTILE` 覆盖全部 triggers `<10%`：KILL；
- best tuned-before-evaluation hysteresis/longer-window 捕获 HorizonFence 收益的 `>=90%`：KILL；
- full-DAG 相对 raw migration 的 completion-cost 或 SLO-goodput净影响 `<5%`：KILL；
- exact ledger、negative control、route/output equality 任一失败：INVALID。

阈值只授权 qualification；正式论文仍需第二模型、serving-scale MoE 和 optimized multi-GPU EP。

## Contribution Boundary After Revision

### 允许主张的最窄贡献

若所有 Gate 通过，dominant contribution 才可能是：

> 首个针对 slow MoE expert migration 的 no-prediction futility certificate：利用 layer progress、top-k route-mass conservation 和 action horizon，在未知 future routes 下证明某些 reactive migrations 不可能回本，并在 matched fixed-ledger serving 中暴露 scheduler-induced harmful actions。

### 不能主张

- active-set churn、historical EPLB lag 或 popularity volatility 的首次发现；
- arrival-cohort popularity 是唯一“真实 load”；
- fast load balancing 应忽略 service-window counts；
- generic partial identification、performative prediction 或 censored-demand theory的新算法；
- 新 placement optimizer、predictor 或 controller；
- 单卡结果等于 EP/TPOT/P99 系统证明。

### 仍然存在的 fatal-risk

即使 soundness 成立，方法仍可能被评价为“显然的 worst-case benefit-versus-cost accounting”。只有同时观察到跨模型自然 harmful decisions、非平凡的 early veto coverage、相对 hysteresis/robust-static 的增量，以及真实 EP full-path 收益，系统审计组合才可能达到 CCF-B 深度。当前全部为 `UNVERIFIED`。

## Route Comparison

| Route | 核心 | 裁决 |
|---|---|---|
| A0：原 CohortFence | generic arrival-cohort detector unanimity | 淘汰；estimand 与 action 不一致，always-abstain 风险高 |
| A1：HorizonFence | action-specific slow-migration futility certificate | 保留一次；先过 analytic non-vacuity 与 novelty gate |
| B：future-route predictor | 预测未来以收窄区间 | 拒绝；与 PROBE/Director 拥挤且扩张方法 |
| C：OPE / randomized exploration | 估计 counterfactual scheduler utility | 拒绝；positivity/action-space/状态定义不成立 |

## Updated Hard Gates

| Gate | 通过条件 | 当前状态 |
|---|---|---|
| H0 Authority | exploratory，不修改 current authority | PASS（文档级） |
| H1 Novelty | no-prediction action-futility certificate 不被近邻/通用理论完全吞没 | OPEN；fatal risk 仍高 |
| H2 Analytic non-vacuity | 合法 admission/capacity bounds 下，`>=10%` trigger 可在 deadline 前出证书 | UNVERIFIED |
| H3 Natural phenomenon | 两模型 scheduler-dependent trigger disagreement `>=5%` | UNVERIFIED |
| H4 Action harm | 两模型 harmful slow migration `>=5%`，至少一个 common cell full-path effect `>=5%` | UNVERIFIED |
| H5 Certificate utility | harmful recall `>=50%`、precision `>=95%`、coverage `>=10%`，增量不被 hysteresis 捕获90% | UNVERIFIED |
| H6 Exactness | identity-complete ledger、admission envelope、route/output/action hashes 全闭合 | OPEN |
| H7 Formal systems | serving-scale + independent model + optimized multi-GPU EP | BLOCKED_CURRENT_RESOURCES |
| H8 Fresh review | 无 fatal novelty/method objection | OPEN |

## Compute and Handoff

- Phase -1 只需现有 trace metadata、冻结 expert size/migration cost range 和 service-surface bounds；预计 1–2 天，无新 controller。
- Phase 0 需要 identity-complete continuous producer 与 fixed-ledger replay；RTX 5090 只给 route/service qualification，预计 10–20 GPU-hours。
- full-DAG 和真实 EP 仍是 formal blocker；在 H1/H2 未过前不实现。
- 下一轮 reviewer 必须优先判定：这个 special-case theorem 是否仍只是 rename；`C_a^-` 是否在可重叠迁移中退化为 0；future-admission/capacity bound 是否 sound；10% coverage 是否在 analytic level 已不可能。

## Current Verdict

`RETHINK-ONCE / ANALYTIC-GATE-ONLY / NOT CCF-B READY`。

本轮没有把 C10 提升为 candidate，只把它从不正确的 demand-label certificate 收缩为一个 action-specific futility certificate。若 fresh reviewer 仍给 fatal novelty objection，或 Phase -1 表明 certificate 只能在动作自然失效后出现，C10 按冻结规则终止并 pivot。
