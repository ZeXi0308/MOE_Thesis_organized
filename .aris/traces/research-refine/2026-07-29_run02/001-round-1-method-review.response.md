## C01 Round‑1 Method Review

**CALIBRATION: none**  
**Evidence tier: design-only**。所有 pilot、coverage、speedup、跨模型现象和系统结果均为 `UNVERIFIED`。  
**Verdict: `KILL_CCFB_METHOD_CURRENT_FORM`**；可以保留 Phase −1 纸笔证伪和 Gate‑2 工程基础设施，但当前不能作为独立 CCF‑B 方法贡献。

### 七维评分

| 维度 | 权重 | 分数 | 加权 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 8 | 1.20 |
| Method Specificity | 25% | 4 | 1.00 |
| Contribution Quality | 25% | 3 | 0.75 |
| Frontier Leverage | 15% | 6 | 0.90 |
| Feasibility | 10% | 4 | 0.40 |
| Validation Focus | 5% | 6 | 0.30 |
| Venue Readiness | 5% | 3 | 0.15 |
| **Overall** | **100%** |  | **4.70/10** |

## 核心身份判断

当前三项所谓特殊机制仍是通用 product-system verification：

- common-event cancellation = differential simulation / partial-order reduction；
- top-k barrier absorption = 通用 `max` fork-join dominance；
- exact queue-state re-coupling = deterministic Markov property；
- `[L_ab,U_ab]` = 通用 abstract-interpretation enclosure。

把 “MoE top-k” 换成任意 fork-join barrier，把 continuous batching 换成任意确定性队列系统，现有 soundness theorem 基本不变。[proposal](/Users/leandrozhao/Desktop/毕设论文资料/refine-logs/c01-causal-closure/round-0-initial-proposal.md:206) 而 proposal 自己已经规定：若去掉 MoE 结构后 theorem 仍成立，应 KILL。[proposal](/Users/leandrozhao/Desktop/毕设论文资料/refine-logs/c01-causal-closure/round-0-initial-proposal.md:42)

此外存在一个尚未解决的二选一：

1. 若 arrival、service、tie-break 和 action 全部确定，每个 action 只有一条轨迹，CausalRank 是优化过的 paired simulator；
2. 若 service/order 是 interval-valued，则它变成通用 robust timed model checking，并立即面对 correlation loss、`2^b` 分支和 always-`AMBIGUOUS`。

文档同时使用确定性 action-specific DAG、所有合法 continuation 和独立 completion intervals，却没有给出 joint uncertainty world 或 paired abstract semantics。[proposal](/Users/leandrozhao/Desktop/毕设论文资料/refine-logs/c01-causal-closure/round-0-initial-proposal.md:54) [proposal](/Users/leandrozhao/Desktop/毕设论文资料/refine-logs/c01-causal-closure/round-0-initial-proposal.md:109) [proposal](/Users/leandrozhao/Desktop/毕设论文资料/refine-logs/c01-causal-closure/round-0-initial-proposal.md:173)

## Fatal counterexample

考虑两个 top‑2 请求共享资源 `R1/R2`：

- 请求 `q` 的 expert 分支为 `x@R1`、`y@R2`；其下一层工作 `q2` 使用 `R2`。
- 另一请求先有 `z@R1`，随后 `w@R2`。
- `y` 在两动作下均占用 `R2:[0,10]`。
- 动作 `a`：`x=[0,2]`、`z=[2,4]`，所以 `w` 在 `t=4` 已排入 `R2`。
- 动作 `b`：`x=[0,6]`、`z=[6,11]`，所以 `w` 到 `t=11` 才到达 `R2`。
- 两动作都有 `T_x≤6≤10=T_y`，因此 proposal 的 barrier 条件成立，`q` 均在 `t=10` combine。
- 动作 `a` 下，已排队的 `w` 先执行 `[10,11]`，`q2=[11,12]`。
- 动作 `b` 下，`q2=[10,11]`，然后才轮到 `w`。
- 若 `q` deadline 为 `11.5`，则 `a` miss、`b` meet。

所以，combine release time 相同，不代表 action 扰动被系统吸收：受影响 expert 对共享 queue 的写效应可以经另一请求重新进入原请求的下游路径。

结论是：

- barrier lemma 最多删除该 combine 的一条**语义传播边**；
- 它不能关闭系统 divergence cone；
- 所有 resource/batch side effects 必须一直保留到完整 queue-state re-coupling。

若实现保留这些边，算法可以 sound，但最困难情形下 MoE barrier shortcut 几乎没有全局削减作用，贡献重新退化为通用 paired state exploration。这直接反驳了“差异不向下一 layer/step 传播”的当前表述。[proposal](/Users/leandrozhao/Desktop/毕设论文资料/refine-logs/c01-causal-closure/round-0-initial-proposal.md:136)

## 各维度的关键修复

- **Method Specificity — CRITICAL**：当前 sufficient state 只是字段清单，不是 transition system；缺少 pending timers、future-input cursor、batch-dependent service、admission、KV/memory、communication/collective 和 controller state 的形式语义。[proposal](/Users/leandrozhao/Desktop/毕设论文资料/refine-logs/c01-causal-closure/round-0-initial-proposal.md:119) 当前 BCRD 资产也明确缺这些 formal state/surface。[BCRD audit](/Users/leandrozhao/Desktop/毕设论文资料/docs/current/bcrd_simulator_correction_2026-07-29.md:59)

- **Contribution Quality — CRITICAL**：必须证明一个 generic product checker 无法获得、但利用 route/top-k/replica 结构可以获得的更强 cut 或更小 asymptotic cone；否则只能降为基础设施。

- **Feasibility — CRITICAL**：文档承认最坏为 `O(2^b)`，而现有更小的 top‑16/six-hold exact space 已经可能超预算。[proposal](/Users/leandrozhao/Desktop/毕设论文资料/refine-logs/c01-causal-closure/round-0-initial-proposal.md:212) [BCRD audit](/Users/leandrozhao/Desktop/毕设论文资料/docs/current/bcrd_simulator_correction_2026-07-29.md:74)

- **Validation Focus — IMPORTANT**：Phase −1 只应判断 soundness、non-genericity 和是否存在非平凡 sublinear family；跨模型 prevalence、2× speedup、multi-GPU 系统指标必须留到后续，且目前全部未验证。

- **Venue Readiness — CRITICAL**：proposal 本身仍是 `METHOD_NOVELTY_UNPROVEN / NOT CCF-B READY`。[proposal](/Users/leandrozhao/Desktop/毕设论文资料/refine-logs/c01-causal-closure/round-0-initial-proposal.md:287)

## Phase −1 必须完成的 proof obligations

1. **Operational semantics**：为 assignment + bounded hold 定义完整 event alphabet、paired transition relation、read/write set。
2. **Markov sufficiency**：证明 canonical cut state 相同且 future input 相同时，后缀 completion-score vector 必定相同。
3. **Dynamic dependency closure**：`E_queue^a/E_batch^a` 随 action 改变，必须以 may-influence least fixed point 找后继，不能依赖 observed static descendants。
4. **Common-event cancellation theorem**：每个取消事件必须满足 paired-state identity，或与全部 retained divergent events 可交换。
5. **Coupled uncertainty semantics**：二选一：
   - 先只做 deterministic point service；或
   - 定义两 action 共享的 joint world `Ω` 与 sound abstraction/concretization。
6. **Resource-aware barrier lemma**：只允许剪 semantic release edge；resource/batch side effects 在 exact re-coupling 前不得删除。
7. **Queue re-coupling theorem**：state equality 必须包含 timers、running work、input cursor、open/sealed membership 和 terminal ledger；若用更弱 observational equivalence，必须证明 objective bisimulation。
8. **Deadline-bound theorem**：证明 completion intervals 覆盖所有 jointly reachable completions、deadline equality 的 tie 语义以及 fixed-denominator 聚合界。
9. **Budget safety**：branch/state budget 耗尽只能输出 `AMBIGUOUS/UNSOLVED`。
10. **Always-AMBIGUOUS family**：构造 `n` 个 seal-boundary arrival 产生 `2^n` paired states，明确方法是完整展开还是 abstain。
11. **Nontrivial positive family**：构造 `n`-event MoE family，使 divergence cone 为 `o(n)` 且无需 terminal replay即可确定排序。
12. **MoE separation theorem**：证明 generic paired fork-join/POR checker不能得到相同 cut；失败即 KILL 论文身份。
13. **Mechanical exhaustive check**：穷举 batching、queue-order、deadline-tie 和共享资源反例；false certification 必须为 0。它只能证明实现正确，不能作为科学 pilot。

## 最小修订路线

如果目标是保留工程资产：

- 删除 interval uncertainty、migration、admission 和 reconfiguration；
- 只做 deterministic assignment+hold paired exact replay；
- barrier absorption 仅作为 semantic-edge pruning；
- exact state equality只作为 memoization；
- 明确定位为 Gate‑2 exactness accelerator，不申报独立论文方法。

如果仍要抢救科学贡献：

- 用保留 shared jitter、batch membership 和 queue-order correlation 的 paired-difference relational domain，替代两个独立 completion interval；
- 首先证明 resource-aware MoE cut 相对 generic checker 的严格 separation；
- 在该 theorem 成立前，不实现通用 interval executor，不跑性能 benchmark；
- 不增加 learned predictor、RL 或 controller，`Modernization Opportunities: NONE`。

## Drift warning

Problem Anchor 本身保持了 current authority；真正的 drift 是从“MoE-specific causal-closure science”滑向“generic simulator verification accelerator”。

纸笔 Phase −1 仍在边界内；但当前 authority 只授权补齐 Gate 0 后运行共同 Gate 1，full request-DAG 实现必须等待 Gate 1 PASS。[README.md](/Users/leandrozhao/Desktop/毕设论文资料/docs/current/README.md:166) [BCRD audit](/Users/leandrozhao/Desktop/毕设论文资料/docs/current/bcrd_simulator_correction_2026-07-29.md:81)

因此最终裁决是：

> **KILL 当前 CCF‑B 方法身份；保留 Phase −1 作为一次最后的形式 separation probe。只有第 11、12 项同时证明成功，才允许以全新方法身份重新进入 `RETHINK/REVISE`。**

`review_independence=same-family`  
`acceptance_status=provisional`
