# Round 1 Method Review

> Reviewer: GPT-5.6-Sol, xhigh  
> Independence: same-family / provisional  
> Verdict: `REVISE`  
> Weighted score: `7.0 / 10`

<details>
<summary>Full raw reviewer response</summary>

## 总评

**Verdict：REVISE**  
**加权总分：7.0/10**  
**review_independence：same-family**  
**acceptance_status：provisional**

方向与仓库真实状态高度一致，也正确把 Controller 降为条件性 supporting contribution。当前阻塞点不在模型复杂度，而在因果对象尚未定义清楚：提案证明了“历史 pressure 可能预测风险”的路径，却尚未严格定义并识别“pressure 是否改变 budget 的边际 SLO 代价”。这一区别决定它最终是新容量边界，还是 Gimbal 式 pressure telemetry 与已有 Token/KV admission 的组合。

仓库一致性核对通过：现有证据只到单卡 custom runtime 的 `MIXED_SOURCE_IDENTIFIED / MEASUREMENT_ONLY`；native transfer、request-level latency、EP、action Oracle 和 Controller 均未验证。提案没有越过这一证据上限。

## 评分

| 维度 | 分数 | 权重 | 评价 |
|---|---:|---:|---|
| Problem Fidelity | 9.0 | 15% | 问题窄、可证伪，正确继承 action endogeneity 与证据边界 |
| Method Specificity | 6.5 | 25% | 组件具体，但 action、候选状态和 SLO-safe estimand 仍有关键歧义 |
| Contribution Quality | 6.0 | 25% | conditional boundary 有潜力，当前仍可被概括为 Gimbal signal + 已知 admission action |
| Frontier Leverage | 8.0 | 15% | 不上 RL/复杂 predictor 是正确选择，因果 rerun 比模型复杂度重要 |
| Feasibility | 6.5 | 10% | Gate 设计合理，但两套硬件/runtime、EP telemetry 与 Controller 集成对 12 周偏满 |
| Validation Focus | 8.5 | 5% | baseline、完整分母、停止规则和独立 trajectory 都较扎实 |
| Venue Readiness | 5.0 | 5% | 当前无 native residual、action headroom 或方法收益，且直接碰撞很强 |

加权计算：

```text
9.0×0.15 + 6.5×0.25 + 6.0×0.25 + 8.0×0.15
+ 6.5×0.10 + 8.5×0.05 + 5.0×0.05
= 7.0
```

## 低于 7 分项及修正

### Method Specificity — 6.5

- **弱点：** `active_decode_budget`、`active_token_budget`、`running_set_budget` 混用。候选 `b` 会改变被选请求和 KV/padding，但当前公式只加入 `proposed_budget[b]`，没有定义候选对应的普通状态 `x_t(b)`。`current max-rank load` 也可能被误读为同 step 的 post-route 信息。
- **修正：** 冻结唯一 action：例如“下一 scheduling epoch 内最多调度的 decode sequences 数”。请求选择顺序固定为现有 FCFS/EDF，Controller 只能截断该顺序。对每个候选计算可提前知道的 `x_t(b)`；pressure 一律写成已完成 epoch 的 `p_{<=t}`。训练数据必须来自随机化或分块执行过的预算，而非当前 scheduler 自选择的数据。
- **优先级：CRITICAL**

- **弱点：** 决策约束是 `Q(T_step) <= decode_step_slo`，但主张是 request-level TTFT/TPOT/SLO-safe。排队和未被选中的请求没有进入该安全判据。
- **修正：** 将边界定义为固定 scheduler 和 arrival episode 下、同时满足预注册 TTFT/TPOT 约束的最大 budget。Step quantile 只能作为控制代理，不能直接命名为 “SLO-safe”；证据不足前建议称为 **empirically SLO-feasible boundary**。
- **优先级：CRITICAL**

### Contribution Quality — 6.0

- **弱点：** “expert pressure + quantile admission + hysteresis”本身没有明显方法新颖性。within-engine 与 cross-engine 的控制位置差异，也不会自动形成贡献。
- **修正：** 把中心贡献冻结为一个可检验的因果量：

```text
在控制候选 budget 对应的 Token/KV/queue state 后，
pressure 是否稳定改变提高 budget 所带来的 request-level SLO risk？
```

只有 pressure–budget interaction 在 fresh holdout、独立 action trajectory 和完整请求分母上成立，才主张新的容量轴。Controller 只证明该轴可用。
- **优先级：CRITICAL**

### Feasibility — 6.5

- **弱点：** 5090 native transfer、单卡 action grid、8×A100 EP、三个 pressure family、request-level Controller，在 12 周内组合风险偏高。
- **修正：** 在线路径只保留 `max expert/rank routed-token load`。`active expert union` 仅做一次饱和诊断；A2A EMA 仅作为 EP 解释变量。先完成一个 native runtime 的 interventional boundary probe，Gate 通过后才进入 8×A100 与 Controller。
- **优先级：IMPORTANT**

### Venue Readiness — 5.0

- **弱点：** 当前没有任何 native residual、action-conditioned headroom 或 request-level gain；“safe”措辞高于证据，并且 Gimbal 是强直接碰撞。
- **修正：** 在论文叙事中明确三种允许结果：新容量边界、仅 measurement boundary、signal-dead negative result。没有稳定 action interaction 和 material Oracle 前，不写独立方法贡献。
- **优先级：CRITICAL**

## CALIBRATION

**CALIBRATION：anchored**

它最像“瓶颈测量 → 单旋钮反馈控制”的 ML systems 工作：信号侧接近 Gimbal 的 recent expert-token pressure，动作侧接近 SCORPIO/SLOs-Serve 的 admission/token-capacity control。可成立的差异不是换了控制位置，而是证明固定 placement/router 下存在此前普通 counters 无法表达的、within-engine、action-conditioned 容量边界。

## GAP

真正缺失的不是另一个 pressure feature，也不是另一个 tail predictor，而是以下因果事实：在候选请求集合、Token/KV、padding、queue 和 prefill 状态都按 budget 重新计算后，历史可观测 pressure 是否仍稳定改变“增加一个 decode slot”的 request-level SLO 风险。如果 pressure 仅能预测当前慢 step，却不能区分不同 budget 的边际代价，它不能支持 Controller；如果它只代理 batch shape、近期 step latency 或 A2A 状态，则贡献也会被普通 serving counters 吸收。

## Simplification Opportunities

1. 在线 Controller 只保留 `p_load`；删除 `p_ws` 主路径，`p_comm` 仅作 EP 诊断。
2. 删除 GBDT 诊断分支。先用一个分段单调响应面和一个 pressure-threshold baseline。
3. 将最终策略简化为 Token/KV baseline 上的单一 pressure correction，而不是独立搜索完整候选网格：

```text
b = b_token_kv - pressure_penalty(p_t, knee)
```

## Modernization Opportunities

**NONE。** 当前缺口是因果识别和 action semantics，不是模型现代性。引入 LLM、RL、复杂在线学习或更多特征只会降低可解释性。

## Drift Warning

以下均属于 drift：预测 future route；动态 placement/replication；修改 router/top-k/precision；token reshuffle 或 per-request migration；联合控制 prefill ratio、admission、placement 和 budget；把 execution-conformance source localization 扩成另一条并行方法主线。

Native A/C/D transfer 可以作为 instrumentation qualification，但不能替代 pressure-capacity 实验，也不应吞掉论文主线。

## 直接新颖性碰撞矩阵

| 对比对象 | 已覆盖的信号/现象 | 已覆盖的动作 | 碰撞强度 | 仅剩的可能 residual |
|---|---|---|---|---|
| Gimbal | expert-token pressure + KV/prefill/queue | DP-engine request dispatch，并联合 placement | **最高、直接碰撞** | 固定 placement/router 下，within-engine cap 是否存在独立 action-conditioned knee；仅“控制位置不同”不够 |
| SCORPIO | batch/sequence/output-length 到 latency/SLO | reject/admit、queue、batch selection | **高动作碰撞** | MoE pressure 是否在同一 action space 上提供普通状态之外的因果增量 |
| SLOs-Serve | GPU profile、token allocation、SLO | soft admission、batch size、chunked prefill | **高动作碰撞** | 固定 route/placement 后的 expert-pressure conditional boundary |
| arXiv:2608.13962 | active-expert-union / saturation phenomenon | 现象侧碰撞，未必是相同 Controller | **高现象碰撞** | active expert count 不能作为 novelty；最多验证其何时失去信息并转向 rank load |
| Generic Token/KV controller | Token、KV、queue、recent latency | active token/sequence cap | **精确 baseline 碰撞** | pressure 必须改善 dangerous underprediction，并在相同 SLO 下带来 action-conditioned Goodput 增益 |

因此，最强新颖性表述不能是“expert-aware concurrency controller”，而只能是：

> 发现并因果验证一个不能被 Token/KV/queue state 吸收的 expert-pressure–budget interaction，并证明它定义了新的 request-level feasible-capacity boundary。

## 最弱因果环节

```text
pressure 对未来风险有预测 residual
!=
pressure 改变 budget 的边际 SLO 代价
```

需要识别的量近似为：

```text
Delta(x,p)
= Risk(b_high | x,p) - Risk(b_low | x,p)
```

关键不是 `p` 能否预测 latency，而是 `Delta(x,p_high)` 与 `Delta(x,p_low)` 是否稳定不同。当前 cost-model 描述还没有给出获得该量所需的随机化、共同支持域和候选请求集合定义。

## 唯一决定性下一实验

执行一个 **native-runtime、matched-prefix、action-conditioned budget branch experiment**：

1. 冻结模型、router、placement、scheduler 顺序、arrival episodes 和 SLO。
2. 将 action 精确定义为下一固定 epoch 的最大 active decode sequence 数，取围绕 Token/KV knee 的三个 budget。
3. 每个 branch 从相同 replay prefix 重建同一 pre-action state；branch 后分别真实执行各自 budget，并独立演化 request set、KV、batch、route、pressure 和 completion，禁止共享 future trace。
4. 预动作信号只用已完成 epoch 的 `max expert/rank token load`；为每个候选 budget 精确计算可提前知道的 Token/KV/queue state。
5. 同时覆盖 steady、bursty，采用隔离 repeat；主指标是 request-level SLO-goodput、TPOT/TTFT 和 dangerous underprediction。
6. 用 pressure-permutation 或 matched-cell decorrelation 作一个负控。
7. 预注册停止条件：pressure–budget interaction 不稳定、dangerous underprediction 不改善，或 action Oracle 的 SLO-goodput headroom 小于探索性 3% 时，停止 Controller。

这一个实验同时回答 residual 是否 action-relevant、Oracle 是否 material，并避免先写 Controller。仓库要求的两个 native conformance source events可嵌入 harness qualification，但不能计作容量证据。

## 最终结论

**REVISE，而不是 RETHINK。**

锚定问题值得做，三个月硕士论文也具备可控的 negative-result 出口；但当前方法尚未把“历史 pressure predictor”升级成“action-conditioned capacity variable”。只有上述交互效应和 request-level Oracle 同时成立，才进入最小 Controller。即使后续得到正结果，本次 Codex 评审仍是 same-family provisional，不能充当跨模型家族的新颖性裁决。

</details>
