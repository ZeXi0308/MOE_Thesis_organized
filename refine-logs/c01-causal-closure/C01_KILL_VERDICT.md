# C01 Final Verdict: KILL

> Candidate: CausalRank — Pairwise Barrier-Coupled Differential Certification  
> Status: `KILLED_AS_INDEPENDENT_CCF_B_METHOD`  
> Evidence tier: proposal, formal counterexample, and prior-art review only; `0` scientific pilot results  
> Review: `4.70/10 / KILL_CCFB_METHOD_CURRENT_FORM`  
> Acceptance: same-family provisional  
> Date: 2026-07-29

## Verdict

C01 当前形式不再作为独立 CCF-B paper route 继续。不实现通用 interval executor，不跑 coverage/speedup pilot，不用 learned predictor、RL、controller 或更大 action space 抢救。

这不是“local Oracle 与 full-DAG 排序已被实验证伪”。准确结论是：在进入 Phase −1 实现前，当前方法身份已与通用 differential DES / relational verification / timed POR 发生直接碰撞，且 MoE barrier 剪枝的主张存在反例。所有自然负载下的 reversal、coverage 和 speedup 仍为 `UNVERIFIED`。

## Terminal reasons

1. **Direct prior-art collision.** Hanai et al. 的 *Exact-Differential Simulation* 已提出只重放 altered events 及其 subsequent causal effects，并保持与完整 DES 重跑完全一致。这直接占据 C01 的 seed → divergence cone → common suffix reuse → 按受影响事件数加速的主体。
2. **Generic relational machinery.** Product programs、shadow symbolic execution、dynamic slicing、DPOR 和 timed-automata zone reachability 分别覆盖双轨同步、仅在 divergence 处分支、因果 slice、等价执行剪枝与时间区间可达性。
3. **No MoE-specific theorem.** common-event cancellation 是 differential simulation / POR；top-k barrier absorption 是通用 `max` fork-join dominance；exact queue-state re-coupling 是确定性 Markov/bisimulation；`[L,U]` 是通用 interval enclosure。删掉 MoE 名词后，soundness theorem 基本不变。
4. **Fatal resource-side-effect counterexample.** 某 token 的受影响 expert branch 可被该 token 的慢 sibling 在 combine 时间上吸收，但仍可占用共享 resource queue，延迟另一 request，再经 downstream queue order 回流影响原 request 的 deadline。因此 barrier lemma 最多剪一条 semantic release edge，不能关闭 system divergence cone。
5. **Unsound-or-generic fork.** 剪掉 resource descendants 会 unsound；完整保留它们则 MoE-specific 缩锥优势消失，方法退化为通用 paired state exploration。
6. **Deterministic-or-model-checking fork.** 若 arrival/service/tie/action 确定，每个 action 只有一条轨迹，CausalRank 是优化的 paired simulator；若使用 interval-valued service/order，它变成通用 robust timed model checking，面临 correlation loss、`2^b` 分支与 always-`AMBIGUOUS`。

## Preserved engineering value

C01 可保留为 Gate-2 exactness 基础设施的设计警示：

- full request-DAG 必须保留 action-dependent queue/batch/resource edges；
- barrier absorption 只能剪 semantic edge，不能在 resource state 重耦合前宣称后缀相同；
- 确定性 assignment+hold paired replay 可使用 exact-state memoization，但不宣称独立方法贡献；
- Gate 1 未通过前仍不授权实现 formal full-DAG Oracle。

## Only admissible successor

唯一可以独立重开的后继不是 CausalRank 小修，而是 paper-only 的 `Route-Join Quotient` 形式分离探针：

1. 证明两个尚未 exact recouple 的不同 queue states，可仅凭 MoE route-hypergraph / top-k 语义得到更粗的 SLO-observational bisimulation；
2. 对 open/seal/dynamic batching/resource contention 依然 sound；
3. 构造 generic exact-diff/DPOR 必须处理 `Theta(n)` changed events，而 quotient 只需 `o(n)` 的严格 separation family；
4. 若同一证明适用于普通 fork-join job shop，继续 KILL。

在这个 separation theorem 成立前，不实现、不跑数，也不把它标成 `REVISE`。
