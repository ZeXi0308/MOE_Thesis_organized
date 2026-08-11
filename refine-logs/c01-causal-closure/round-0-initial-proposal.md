# Research Proposal: When Local MoE Oracles Lie

> 中文题目：**局部 MoE Oracle 何时会撒谎：完整请求因果闭包下的动作排序证书**  
> 方法工作名：**CausalRank — Pairwise Barrier-Coupled Differential Certification**  
> 状态：`EXPLORATORY / NOT_CURRENT_MAINLINE / PHASE_MINUS_1_ONLY / NO_EMPIRICAL_RESULT`  
> 日期：2026-07-29

## Problem Anchor

- **Bottom-line problem**：在固定 natural continuous-arrival、request/token/route identity、exact model semantics、service contract 和 action set 的条件下，判断 single-layer、local-window 或 truncated-horizon MoE Oracle 给出的动作排序，是否在 assignment/hold/reconfiguration 对后续 layer、decode step、dynamic batching、resource queue、admission 和 request completion 的影响全部闭合后仍然成立；若不能直接成立，给出 sound 的 `IDENTIFIED / AMBIGUOUS / INVALID` 证据边界。
- **Must-solve bottleneck**：贡献不能只是补一个 full-request-DAG simulator，也不能只是“区间不重叠即可排序”的通用 tautology；必须利用 MoE top-k fork→combine barrier absorption、continuous-batching queue-state re-coupling 和 deadline slack，证明何时可以不完整展开 DAG 就安全保持 action order，并在两个模型的自然 workload 上验证 local/full action reversal 具有系统显著性。
- **Non-goals**：不提出新的 scheduler、placement optimizer、migration policy、router predictor、RL/OPE controller 或 route/weight/precision 修改；不把 simulator fidelity、代码 exactness 或负载曲线本身当论文贡献；不复活已停止的 receiver/precision/prefetch/controller formulation。
- **Constraints**：保持 `docs/current/README.md` 的共同现象 Gate 为唯一正式执行线；本候选只作为隔离的 Phase −1 方法审计。所有 route、output、arrival、deadline、denominator、tie-break、service interval 和 action contract 必须冻结；evaluation 后不得改 horizon、workload、阈值或 action family 抢救。RTX 5090 只支持 route/service qualification 与 CPU exact replay；EP migration、TPOT/P99、SLO-goodput 的正式主张需要 optimized multi-GPU serving。
- **Success condition**：全文查新后不存在等价的 pairwise causal-cut action-ranking certificate；充分状态、barrier absorption、queue coalescence 和 soundness theorem 均闭合；两个模型自然 common cells 的 local/full top-1 action disagreement `>=10%`、净收益符号反转 `>=5%`、至少一个 common cell full-path effect `>=5%`；certificate 在不超过25% full DAG 或两个 downstream combine barriers时 coverage `>=50%`、certified error为0、相对两次 full replay至少2×；fixed longer horizon/简单方法不能捕获90%增量；最终由真实多卡 EP 复核且 fresh CCF-B review 无 fatal objection。任一硬门失败即 KILL。

## Why This Problem Exists in the Current Assets

当前 BCRD 资产已有 route-v3 因果身份、合法 assignment、queue hold、deterministic event engine、service catalog 和 symmetry-reduced local exact Oracle；但其正式 Gate 2 被硬编码为：

```text
INVALID_REQUEST_DAG_REPLAY_NOT_IMPLEMENTED
```

原因不是代码 polish，而是现有 `single_layer_window` 把 observed `layer_ready_us` 当固定输入，assignment/hold 改变的完成时间没有传播到下一 layer、下一 decode step、后续 batch membership 和 request SLO。因此当前 local Oracle 的“headroom”不是 full-path upper bound。

这个缺口有两种可能结果：

1. local ranking 在自然 workloads 中几乎总与 full-DAG 一致：补齐 simulator 是基础设施，不形成独立 idea；
2. local ranking 经常反转，且可以用结构证书在较浅 causal cone 内识别：才可能形成方法+测量型 CCF-B 候选。

当前两者都是 `UNVERIFIED`。

## Prior-Art Boundary

- [Vidur](https://arxiv.org/abs/2405.05465) 已做高保真事件驱动 LLM serving simulation，并明确指出小的 iteration 误差会通过 dynamic batching 级联；因此“完整仿真很重要”不新。
- [LLMServingSim 2.0](https://arxiv.org/abs/2511.07229) 已覆盖 heterogeneous hardware、MoE/EP、routing dynamics、网络与显存争用；因此“做 MoE full-system simulator”不新。
- [APEX](https://arxiv.org/abs/2411.17651) 已做 iteration-level dynamism-aware simulation 和 execution-plan search，并经验性评估 relative ranking；因此只报告 rank correlation 不新。
- [AMoE](https://arxiv.org/abs/2505.08944) 已跟踪 request/token/layer dependency、异步 expert queues 与 re-batching；因此 token dependency tracking 不新。
- [FATE](https://arxiv.org/abs/2605.07238) 已用有限 downstream horizon 评价 current-frontier action；因此“加一点 lookahead”不新。
- 一般 discrete-event simulation、timed automata、partial-order reduction、abstract interpretation 和 ranking-and-selection 已覆盖 event branching、状态削减、区间与统计正确选择。

剩余的 provisional gap 不是 full replay，而是：对相同 frozen trace 上的**两个 action**，只展开它们的 union divergence cone；利用 MoE fork-join criticality 与 serving queue re-coupling消除共同后继，并在无法证明 action order 时明确 abstain。若删掉这两个结构后算法和 theorem 基本不变，candidate KILL。

## Dominant Contribution

> 一个针对 fixed-ledger MoE serving counterfactual 的 pairwise causal-closure certificate：它不分别估计两个 action 的大而松的 tail interval，而是同步执行两条轨迹、取消共同事件，只扩展 action 差异的因果锥；当 top-k combine barrier 吸收扰动、resource queue state 重新耦合或 pairwise SLO interval 排除0时，soundly 证明 local/truncated action order 与 full-DAG order 一致；否则返回 `AMBIGUOUS`。

Supporting claim 只有一个：自然 continuous-decode 中，local-window action ranking 是否存在跨模型、系统显著的反转。若没有，方法即使正确也只是一项验证工具。

## Formal Object

### Frozen event graph

对 identity-complete trace `xi` 与动作 `a`，确定性 event engine 实现：

```math
G^a=(V,E_sem union E_queue^a union E_batch^a union E_mig^a).
```

- `E_sem`：arrival → route → dispatch → expert → top-k combine；layer `l+1` 等待 layer `l` combine；decode step `s+1` 等待 `s` 的最终 combine。
- `E_queue^a`：action 决定的合法 replica/resource 顺序。
- `E_batch^a`：contribution ready、open/seal、launch、finish 与 batch membership。
- `E_mig^a`：若包含 reconfiguration，copy、barrier、availability 必须作为真实事件进入 DAG；资格 Phase 先只支持 assignment+hold。

在一条已实现的 action-specific DAG 上：

```math
T_v^a = p_v^a + max_{u:(u,v) in E^a} T_u^a.
```

`E_queue^a` 和 `E_batch^a` 可以随 action 改变，不能把所有 action 预设成同一个静态 DAG。

### Action contract

资格阶段动作限定为：

```math
a = do{ target(i), h_{r,e} }.
```

- 每个 contribution 只能选择 `legal_replica_set` 中的 target；
- `h_{r,e}` 是 per-replica/expert queue 的 bounded hold；
- top-k、expert identity、gate weight、route、token、output 与 all-arrival denominator 不变；
- 任何 semantic hash 改变均为 `INVALID`。

### SLO objective

primary objective 固定为 all-arrival SLO-goodput：

```math
G(a)=1/N * sum_q 1{C_q^a <= d_q}.
```

不用可调 penalty 把 migration tax 另加到 objective；action tax 必须作为 event 进入 DAG，并通过完成时间影响同一 denominator。

## Rejected Route A: Per-Action Tail Intervals

对每个 action 独立运行到 horizon `H`，以确定 on-time、确定 miss、未决 requests 构造：

```math
lower_G_H(a)=|D_a^+|/N,
upper_G_H(a)=(|D_a^+|+|U_a|)/N.
```

这条路线被淘汰：interval separation 本身是 tautology；两个 action 分别把所有 downstream uncertainty 计入，丢失相同 trace 上的共同事件抵消；batch/order 一变就容易把全部后继放入 `U_a`，退化为 always-ambiguous。它只能作为安全 first-pass pruning，不能是论文方法。

## Selected Route B: Pairwise Barrier-Coupled Differential Replay

### 1. Pairwise seeds and union divergence cone

对动作 `a,b`，`S(a,b)` 是第一次 action/state 差异的 event seeds。两套 engine 使用完全相同的 arrivals、routes、service intervals 与 tie-breaking，同步前进：

1. state/event 完全相同且不受 seed 后继影响时直接 cancel；
2. 只物化两轨迹的 union divergence cone；
3. 遇到不确定 event order 或 batch membership 时覆盖所有合法 continuation；
4. 分支数超过冻结 budget 时返回 `AMBIGUOUS/UNSOLVED`，不得任选一种顺序；
5. 达到 full terminal state 时退化成 exact paired full replay。

### 2. Sufficient cut state

任一 horizon cut 的状态至少包含：

```text
per-request next semantic event + ready interval
per-resource running batch + remaining service
open/sealed queue order and membership
seal rule / capacity / deadline state
resource availability
placement/migration availability
frozen future arrivals/routes/service contract
accumulated completion verdicts
```

route-v3 CSV 不是充分状态。若未来转移还依赖未记录的 batch、KV、admission 或 resource state，certificate 为 `INVALID`。

### 3. MoE top-k barrier absorption

对 combine event `c`，受 action 影响的 expert predecessors 为 `A_c`，未受影响且两动作完成时间相同的 predecessors 为 `U_c`。若：

```math
max_{u in A_c} upper_T_u <= max_{v in U_c} lower_T_v,
```

则受影响分支不可能成为 barrier critical predecessor，combine time 在两 action 下相同：

```math
T_c^a = T_c^b.
```

该 token 的差异被 barrier 吸收，不向下一 layer/step 传播。这是区别于普通 descendants traversal 的第一个必需 lemma。

### 4. Event-order safety

对同一 mutable resource 的候选 event `e,f`，若：

```math
upper_tau_e < lower_tau_f,
```

或相等时 frozen tie priority 唯一决定 `e`，则所有合法 continuation 中顺序不变。区间重叠时必须分支或 abstain；不能用 observed order 代替 counterfactual order。

### 5. Queue-state re-coupling

若某个 cut 上两动作的 sufficient state 完全相同，且未来输入相同，则由 deterministic transition induction，后续执行完全一致。此 divergence cone 可以关闭，prefix 已累积的 goodput difference 就是 full-DAG difference。

典型 re-coupling 可能来自：

- action delay 被 idle gap 吸收；
- 两队列 drain 后 running/open/sealed state 重新一致；
- 非 critical expert 分支在 combine 前被吸收；
- request completion 仍在相同 deadline slack 一侧，且其后不再竞争共享资源。

### 6. Pairwise completion and action-order bounds

对未闭合 request `q`，在所有合法 continuation 上计算 completion interval：

```math
I_q^a=[lower_C_q^a, upper_C_q^a],
I_q^b=[lower_C_q^b, upper_C_q^b].
```

earliest completion 忽略 contention 的 semantic longest path；latest completion 使用 frozen finite-trace interference envelope，并由 order separation、barrier absorption、state re-coupling 迭代收紧。无法安全缩小时纳入所有可能排在该 request 前的剩余 resource work。

对 deadline score `s_q(t)=1{t<=d_q}`：

```math
l_q^{ab}=min_{x in I_q^a,y in I_q^b}(s_q(x)-s_q(y)),
u_q^{ab}=max_{x in I_q^a,y in I_q^b}(s_q(x)-s_q(y)).
```

于是：

```math
L_ab = 1/N * sum_q l_q^{ab}
     <= G(a)-G(b)
     <= 1/N * sum_q u_q^{ab} = U_ab.
```

输出：

- `L_ab > 0`：`IDENTIFIED_A_BETTER`；
- `U_ab < 0`：`IDENTIFIED_B_BETTER`；
- 0 在区间内：`AMBIGUOUS`；
- semantics/state/service/denominator 不闭合：`INVALID`。

### 7. Theorem obligation

**Pairwise Action-Order Soundness Theorem**：若 fixed trace 保持 exact semantic identity；event engine 和 tie-break deterministic；action seeds 完整；cut state 对未来转移充分；interval/branch executor over-approximate 全部合法 continuation；service duration 位于冻结 interval；则真实 `G(a)-G(b)` 必在 `[L_ab,U_ab]` 内。因此区间排除0时，不会输出错误 action order。

仅证明这个通用 enclosure 不够。论文必须同时证明 barrier absorption 与 queue-state re-coupling 对自然 MoE traces 显著缩小 divergence cone；否则 candidate 降为 generic model checking 并 KILL。

## Complexity and Fail-Closed Behavior

设 divergence cone 有 `k` 个 events、`e` 条 edges、`R` 个 resources、`b` 个无法由 interval separation 决定的 boundary branches：

```math
time = O(2^b * (k+e) * log R),
memory = O(2^b * k).
```

无歧义常见路径近线性；最坏 `k=|V|` 且 `b` 线性，退化为指数。方法不宣称 polynomial worst case；超过 state budget 必须 abstain。

对多 action，先用 Route A 的粗区间安全淘汰，再按 local rank 让 survivor 与 incumbent 做 pairwise tournament。它不解决 action-space 组合爆炸；现有 symmetry reduction 是前置。

## Phase −1: Paper-and-Pencil Gates

任何 full-DAG implementation 前必须完成：

1. **Sufficient-state proof**：列出 event alphabet 与 transition，证明 cut state Markov sufficient；若需要未记录的 future hidden state，KILL。
2. **Three lemmas**：semantic common-event cancellation、fork-join barrier absorption、queue-state re-coupling；若只能在 queue order 预先固定时成立，KILL。
3. **Hand counterexamples**：
   - 非 critical expert delay 被 combine 吸收；
   - queue drain 后两 action 重耦合；
   - arrival 跨过 seal boundary 导致 batch split 与 decode cascade，算法必须 `AMBIGUOUS`。
4. **Nontrivial family**：构造 `n`-event request DAG 而 divergence cone 为 `o(n)`，且 certificate 不跑 full DAG 就识别排序；简单实例也必须 terminal replay 则 KILL。
5. **Claim matrix**：对 FATE、APEX、AMoE、Vidur、LLMServingSim2、DES/timed automata/partial-order reduction/robust ranking 写 input、output、abstention、theorem、MoE semantics；发现等价 certificate 即 KILL。

Phase −1 最多授予 `METHOD_FORMALLY_PLAUSIBLE`，不是 scientific GO。

## Qualification Plan and Kill Gates

### CPU exhaustive correctness

冻结小规模枚举：2 replicas、2–4 requests、2–4 MoE layers、2 decode steps、top-2、hold `{0,1,2}`、全部合法 assignment，覆盖 deadline tie、same-time arrival、max-batch seal、queue reorder。

- exhaustive full replay 是 qualification ground truth；
- certified pair 错序数必须为0；
- 任一反例被错误 certificate 化，KILL。

### Two-model trace pilot

OLMoE 与 LLM-jp 的 natural, document-disjoint, identity-complete traces 上比较 local horizon、fixed longer horizon、full replay 与 CausalRank。资格结果不声明物理 EP。

同时必须满足：

- local/full top-1 disagreement `>=10%`；
- sign reversal `>=5%`；
- 至少一个 common natural cell full-path SLO-goodput difference `>=5%`；
- 不超过2个 downstream combine barriers或不超过25% full DAG时，`IDENTIFIED coverage >=50%`；
- certified ground-truth agreement `100%`；
- median divergence cone `<=50%` full events，P95 `<=80%`；
- CPU wall-time相对两次 full replay `>=2x`；
- fixed longer horizon 或简单 conservative rule 捕获 `<90%` 的增量。

任一失败即停止独立 CCF-B idea。只在 synthetic、单模型、单 action family、zero-cost 或 future-unbounded 条件下成立也 KILL。

### Formal systems boundary

资格通过后仍需：

- 至少两个模型，其中一个 serving-scale open MoE；
- optimized multi-GPU EP；
- natural continuous arrivals；
- identity-complete live ledger；
- 真实 assignment/hold，若声称 reconfiguration 则必须有真实 migration events；
- counterfactual replay 对 held-out live episodes 的 phase/P99 error `<=10%`、action ranking agreement `>=95%`；
- request-level TPOT/P99/SLO-goodput 与置信区间。

单5090不能建立 EP、NCCL/RDMA、migration 或生产 SLO 结论。

## Asset Reuse and Missing Pieces

可复用：`Contribution` identity、route-v3 stage ledger、`legal_replica_set`、deterministic tie semantics、open/seal/launch/finish queue lifecycle、per-queue hold、`ServiceCatalog`、symmetry-reduced action enumeration、all-arrival SLO objective、document-disjoint split 与 hash provenance。

不能冒充 full-DAG：observed `layer_ready_us`、当前 single-layer engine/local Oracle、缺失 attention/KV/next-layer/next-step/admission/migration/communication state 的 trace、未绑定 `(model,layer,expert,dtype,rows)` 的完整 service surface。

## Current Verdict

`PROCEED_TO_PHASE_MINUS_1 / METHOD_NOVELTY_UNPROVEN / NOT CCF-B READY`。

这只是当前最值得继续严审的候选，不是“已经达到 CCF B 的 idea”。下一步必须先完成纯纸笔形式检查和 fresh reviewer；若 reviewer 认为 theorem 仍是 generic interval verification，或 MoE-specific absorption/coalescence 无法产生 sublinear divergence cone，立即 KILL，不进入实现。
