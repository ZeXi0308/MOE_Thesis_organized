# PhaseMap-MILP Phase 1 终审

状态：**CONDITIONAL GO TO L2 ORACLE EXISTENCE TEST / NO SCIENTIFIC RESULT**  
日期：2026-07-23

## 结论

`B0/Q/J/R` 信息格只是识别方法，不是论文创新。Gimbal 已覆盖 backend
queue/remaining-work/MoE pressure，SIRD 已覆盖 receiver credit 和 SRPT policy，
coflow 已覆盖 all-or-nothing completion 与 deadline scheduling。PhaseMap 唯一可辩护
的增量是：

> 在同一 MoE combine-return 决策点，keyed receiver queue state `Q` 与 causal
> per-join sibling phase `J` 产生不能被任一单轴或简单规则恢复的交互信息价值，
> 并且该价值后续有可能压缩为 receiver-local credit stamp。

当前 necessity **未验证**：旧 FJRC 只增加 q-map，科学对象错位；CRQM 在
固定路由/全完成目标下 gap 和 first-action flip 都为 0。因此只允许进入一次
便宜、严格的 L2 oracle 判死门，不允许直接写 controller 或宣称物理拥塞收益。

## 信息格

| Arm | 额外可见信息 | Nonanticipativity |
|---|---|---|
| `B0` | 只见 queue/phase 多重集，不见 keyed mappings | 四世界共用一个 action |
| `Q` | `receiver -> backlog/availability` | 同 q-bit 跨 j-world 共用 action |
| `J` | `join -> committed/queued sibling bitmap` | 同 j-bit 跨 q-world 共用 action |
| `R` | `Q + J` | 可按 `(q,j)` 联合 observation 动作 |
| `C` | full future | 只作 ceiling，不进 gate |

四世界必须是 `(Q0,J0),(Q0,J1),(Q1,J0),(Q1,J1)` 的完整交叉，不得仅用两个
world 冒充交互识别。

## 入选门

两模型都必须满足：

1. `R` 相对 exact best of `{Q,J}` 的 miss relative reduction >=10% 且 absolute >=2pp；
2. aggregate CVaR90 normalized tardiness reduction >=5%；
3. actionable pairs >=50%；
4. 同 q 换 j 与同 j 换 q 的严格 singleton first-action interaction flip 均 >=25%；
5. 最强 causal simple baseline 捕获的 `best-single -> R` gain <90%；
6. equal-Q/equal-J/fanout-1/no-conflict/shuffled-key 对照必须为 0；
7. enumeration/MILP/event replay 一致，solver gap <=1e-7。

任一失败：`NO_GO_PHASEMAP_QUEUE_JOIN_INTERACTION`。通过也只能写
`PROMISING_L2_QUEUE_JOIN_INTERACTION_HEADROOM`。

## 边界与停止规则

5090 只提供 native route identity 和 pack/unpack/combine primitive LUT；receiver 队列、
incast 和 cut 仍是 causal L2 replay/analytic proxy。不得外推 RDMA/NCCL、TTFT/TPOT/P99
或 production benefit。

判死后不增大 synthetic qdepth/fanout/ready skew，不在 holdout 调 deadline，不把
128 world rows 当独立样本，不替换 invalid pair，不降低门槛，不用 predictor/bandit/
shadow-price 改名抢救，不合并 Energy-SLO。

