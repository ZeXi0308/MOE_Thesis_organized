# Round 2 Method Review

> Reviewer: GPT-5.6-Sol, xhigh  
> Independence: same-family / provisional  
> Verdict: `REVISE`  
> Weighted score: `7.7 / 10`

<details>
<summary>Full raw reviewer response</summary>

## 第二轮总评

**Verdict：REVISE**  
**加权总分：7.7/10**  
**review_independence：same-family**  
**acceptance_status：provisional**

本轮修订保留了 Problem Anchor，并解决了上一轮最主要的概念错误：action 已冻结，`X_t(b)` 可在 action 前计算，pressure 被正确降为 effect modifier，“safe”也降级为 empirical feasibility。dominant contribution 现在是清楚、可证伪的系统假设，不再是 feature-plus-controller 拼装。

剩余阻塞集中在三个执行定义：共同条件变量、主效应估计器、request-level horizon/denominator。它们不要求增加实验，只需在 I1 协议中冻结清楚。

## 评分

| 维度 | 分数 | 权重 | 第二轮判断 |
|---|---:|---:|---|
| Problem Fidelity | 9.5 | 15% | 完整保留原 anchor，且没有滑向 route predictor、placement 或多动作系统 |
| Method Specificity | 6.8 | 25% | action 已清楚，但 `tau` 的条件变量、估计器和 horizon denominator 仍未完全闭合 |
| Contribution Quality | 7.2 | 25% | dominant contribution 已锐化为 pressure-conditioned treatment-effect boundary，但新颖性完全取决于实测边界 |
| Frontier Leverage | 8.5 | 15% | 使用最小模型、真实 action branches 和完整状态演化，选择合理 |
| Feasibility | 7.5 | 10% | Gate 化时间线可执行；只要不把 N0 扩成机制修复项目 |
| Validation Focus | 8.5 | 5% | 一个 eligibility Gate、一个 decisive Gate、一个条件 Controller，范围克制 |
| Venue Readiness | 5.5 | 5% | 尚无 native/action/request-level 结果，直接文献邻域仍拥挤 |
| **Weighted Overall** | **7.7** |  | **REVISE** |

## 低于 7 分项

### Method Specificity — 6.8

**弱点 1：`tau(x,p)` 的条件写法不适合 paired treatment effect。**

`X_t(b_high)` 与 `X_t(b_low)` 是两个不同的 candidate-specific summary，不能统一写成条件 `X_t(b)=x`。否则 common support 和 matched-prestate 的含义不明确。

**修正：** 定义共同 pre-action state：

```text
Z_t = {
  complete ready/active queue snapshot,
  frozen request order,
  per-prefix Token/KV/padding/age summaries,
  KV occupancy,
  frozen prefill state,
  previous budget and scheduled-token count
}

X_t(b) = g(Z_t, b)

tau(Z_t, p_t)
  = E[Y_t(b_high) - Y_t(b_low) | Z_t, p_t]
```

branch matching 基于同一个 `Z_t`；`X_t(b)` 只是各 budget 的预测输入。

**优先级：CRITICAL**

**弱点 2：quantile interaction `gamma` 不能直接充当 treatment effect。**

`gamma` 描述条件分位数模型中的交互，不等同于 request miss-rate、SLO-goodput 或风险差上的 `tau`。

**修正：** I1 的主估计器应直接使用 same-prestate paired branch difference：

```text
tau_hat(p-bin)
  = mean_s [Risk_s(b_high) - Risk_s(b_low)]
```

quantile response surface 只用于解释和后续 Controller；不能承担主要因果结论。另将 `previous budget`、previous scheduled tokens/total routed tokens 纳入 route-free baseline，避免 `p_t` 只是 `b_{t-1}` 的记忆变量。

**优先级：CRITICAL**

**弱点 3：`H` 内 request-level risk 仍可能产生删失分母。**

短固定 scheduling horizon 未必包含请求完成，因而不能直接计算完整 SLO-goodput；未调度请求的等待也可能被漏掉。

**修正：** 在 I1 前冻结以下二选一：

- 从 branch point 运行至预动作 cohort 全部完成，并计入期间固定 arrivals；或
- 使用固定 wall-clock episode，对未完成请求采用预注册 censor/failure 规则。

同时定义 TPOT/ITL 的 at-risk token、TTFT 的 arrival cohort，以及所有 branch 完全相同的 denominator。

**优先级：CRITICAL**

### Venue Readiness — 5.5

**弱点：** policy-specific rerun 本身只是实验正确性。Gimbal、METRO、ELDR、TAPER、ReXpert、SCORPIO 和 SLOs-Serve 已占据 signal、activated-expert phenomenon 与 SLO action 邻域。

**修正：** 将 C1 中“ordinary counters 不能表达”改为：

> 在预注册的强 route-free baselines 之外，对 fresh holdout 提供稳定、material、request-level action-effect 增量。

只有跨 frozen regime 或 EP confirmatory 后仍成立，才能进一步讨论新的系统边界。

**优先级：CRITICAL**

## CALIBRATION

**CALIBRATION：anchored**

当前方案像一个严格收缩后的“Gimbal-style pressure sensing + SLOs-Serve/SCORPIO-style single-knob capacity control”研究。它的中心不再是 Controller，而是验证是否存在一个不能被普通状态和历史 latency 吸收的 pressure-conditioned marginal-cost boundary。

## GAP

剩余 gap 已经很准确：不是缺少新的 expert feature，而是缺少一个稳定的系统规律——在同一完整 prestate 上，真实提高 decode budget 后产生的 request-level risk difference，是否随已完成 expert/rank pressure 系统性变化。Action rerun 只保证结果不是假 counterfactual；只有该 effect heterogeneity 跨 holdout、repeat 和运行域稳定，才形成贡献。

## Simplification Opportunities

1. 将 baseline 5“simple pressure-threshold correction”和 baseline 6“final Controller”直接合并。当前 Controller 本质上就是前者，不应人为制造 proposed-vs-simple 差别。
2. `M2` 仅保留为 pressure 主效应诊断；论文中心只比较 route-free model 与带 `pressure × budget` interaction 的模型。
3. N0 只输出 runtime/telemetry qualification verdict，不扩展新的 operator localization 点。

## Modernization Opportunities

**NONE。** 当前不需要更现代的预测模型。paired causal estimator、完整 denominator 和强 route-free baseline 比任何在线学习或复杂校准更重要。

## Drift Warning

最大 drift 风险是 N0 失败后转向 execution-conformance repair，或 N0 成功后继续深挖 kernel/source localization。两者都偏离 pressure-conditioned capacity anchor。

另一个风险是把 `p_t` 改造成 future-route predictor；修订版已经明确禁止，应继续冻结。

## 新颖性碰撞矩阵

| 工作 | 已占据的邻域 | 与本提案的碰撞 | 尚可能保留的 residual |
|---|---|---|---|
| Gimbal | expert-token pressure、KV/prefill/queue、在线 dispatch 与 placement | **最强 broad collision** | 固定 placement/router 下，pressure 是否改变 within-engine budget 的 request-level treatment effect |
| METRO | activated-expert / MoE serving 邻域 | 削弱 activated-expert statistics 本身的新颖性 | policy-specific feasible-capacity boundary，而非 activated-expert signal |
| ELDR | expert activation prediction到 decode-worker routing | 历史 activation 已是可执行 serving telemetry | 不预测 future route，只用 completed pressure 调节同 engine 单一 budget |
| TAPER | context/slack/externality 驱动的 per-step SLO admission | 控制形式与逐步容量动作接近 | MoE pressure 是否提供 route-free 状态之外的 budget-effect heterogeneity |
| ReXpert / arXiv:2608.13962 | activated-expert union/saturation 与 decode SLO cap | 现象与 action 两侧均直接逼近 | active union 不再可主张；只剩 max-load-conditioned treatment boundary |
| SCORPIO | latency/SLO prediction、admit/reject、queue/batch selection | 高 action collision | MoE load 对同类 action 的稳定增量 |
| SLOs-Serve | token allocation、soft admission、batch size 与 SLO | 高 action collision | 固定其他动作后的 MoE-specific marginal capacity axis |
| Generic Token/KV controller | candidate Token/KV/queue state 到 decode cap | **精确行动基线** | pressure interaction 必须改善 held-out risk，并产生实际 Goodput 增量 |

结论：修订稿没有明显把已有组件直接写成创新，但 C1 的“ordinary counters 不能表达”仍略强，应按上面的 held-out incremental wording 收紧。

## N0 → I1 顺序审查

顺序本身 **coherent**：

```text
N0：只验证 native runtime、telemetry parity、开销与执行资格
→ I1：唯一 decisive action experiment
→ positive 才实现 Controller
```

但 N0 的 pass/fail 语义仍应拆开记录：

```text
N0a: native A/C/D conformance transfer result
N0b: pressure telemetry + OFF/ON parity + branch actuator eligibility
```

A/C/D 不复现可以证明 custom-runtime conformance 现象未迁移，却不在科学上自动证明 expert-pressure capacity 不存在。根据当前仓库过程权威，N0 仍是唯一先执行项；若 N0a 失败，应停止并触发是否 reopen I1 的显式决定，而不要写成 pressure hypothesis 已被证伪。N0 的任何结果都不能计作 capacity evidence。

## 最弱因果环节

当前最弱环节已经从 action endogeneity 转移为：

> paired branch 得到的 budget treatment effect，能否在控制完整共同 prestate、previous budget 和普通状态后，随 pressure 稳定变化。

也就是必须证明 pressure 不是前一 budget、总 routed tokens、recent latency 或未匹配请求身份的替代变量。

## 唯一决定性实验

**I1 仍是唯一决定性因果实验；N0 是其当前唯一授权的资格前置，不是决定性容量实验。**

I1 不需要扩展实验面，只需在现有设计中冻结：

1. 共同状态 `Z_t` 与 `X_t(b)=g(Z_t,b)`；
2. same-prestate paired risk-difference estimator；
3. previous-budget/total-load route-free controls；
4. request cohort、branch horizon 和删失规则；
5. pressure permutation 负控。

若 paired `tau` 没有稳定 pressure interaction，即使 M3 pinball loss 改善，也应判 `MEASUREMENT_ONLY / NO CONTROLLER`。

## 最终结论

**REVISE，但已经是可执行前的协议修订，而不是方法重构。**

Problem Anchor 保持完整；dominant contribution 已足够锐利；action endogeneity 的执行路径基本关闭。当前仍需在 I1 前修正 treatment-effect notation、主估计器和 request denominator。完成这些文本级冻结后，研究计划可进入执行，但当前唯一授权动作仍只是 N0。没有 I1 正结果前，proposal 不能升级为方法 GO 或 top-venue-ready。

</details>
