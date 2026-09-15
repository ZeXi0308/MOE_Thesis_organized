# Fresh jury verdict

**Reviewer route:** `same-family / provisional`. 本席是 root 新派生的 fresh reviewer；`novelty-check` 要求的额外 xhigh reviewer 因线程上限未能启动。  
**Evidence tier:** `LITERATURE + DESIGN ONLY / NO PILOT RESULT`。packet 中所有 pilot、阈值、开销、攻击增量均是计划，**不是证据**。

**结论：15 KILL，1 PHASE0_ONLY，0 DEFER_INFRA，0 PROCEED_TO_REFINE。**

唯一 survivor 是 **N05 的冻结现象/攻击主张**：

> 对抗者能否仅通过 spectator 的 MoE route shape，控制 victim 所在 expert 的实际 kernel execution regime，并以显著高于 matched-random spectator 的概率造成 victim hidden-state → downstream route/output flip。

N05 的“margin/risk boundary 上切 stable/canonical path”方法本身不新，直接撞 MarginGate/LLM-42。survivor 资格只属于上述现象与因果链，canonical path 只能作机制消融，不得作为贡献。

## Scores and ranking

Novelty 写作 `总分（现象新/方法新）`；总评是七维算术平均。

| Rank | ID | Novelty | Problem evidence | MoE specificity | Method specificity | Impact | Feasibility | Venue readiness | Mean | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | N05 | 7.2 (8.0/5.0) | 6.5 | 8.5 | 7.5 | 7.0 | 8.5 | 6.0 | **7.31** | **PHASE0_ONLY** |
| 2 | R04 | 4.0 (5.0/3.0) | 6.0 | 8.0 | 6.5 | 6.0 | 8.0 | 4.5 | 6.14 | KILL |
| 3 | R01 | 4.5 (5.0/4.0) | 6.0 | 7.5 | 6.5 | 6.0 | 7.5 | 4.5 | 6.07 | KILL |
| 4 | R03 | 3.5 (4.0/3.0) | 6.0 | 6.5 | 7.0 | 6.0 | 8.5 | 4.5 | 6.00 | KILL |
| 5 | N01 | 2.5 (3.0/2.0) | 7.5 | 6.5 | 7.0 | 6.5 | 8.0 | 3.5 | 5.93 | KILL |
| 6 | S02 | 3.5 (4.0/3.0) | 5.0 | 7.5 | 7.0 | 6.5 | 7.0 | 4.5 | 5.86 | KILL |
| 7 | S01 | 2.0 (3.0/1.0) | 7.0 | 9.0 | 8.0 | 7.0 | 4.0 | 3.5 | 5.79 | KILL |
| 8 | R02 | 5.0 (6.0/4.0) | 6.5 | 7.5 | 6.5 | 6.5 | 3.0 | 4.5 | 5.64 | KILL |
| 9 | X01 | 5.5 (8.0/3.0) | 8.0 | 9.0 | 3.0 | 8.0 | 2.5 | 3.5 | 5.64 | KILL |
| 10 | N02 | 3.5 (5.0/2.0) | 4.0 | 8.0 | 6.5 | 4.5 | 8.5 | 3.5 | 5.50 | KILL |
| 11 | N03 | 4.0 (4.0/4.0) | 7.0 | 5.5 | 6.5 | 6.5 | 4.5 | 4.5 | 5.50 | KILL |
| 12 | S05 | 3.0 (3.0/3.0) | 3.0 | 8.0 | 8.0 | 7.0 | 5.5 | 4.0 | 5.50 | KILL |
| 13 | R05 | 5.0 (5.0/5.0) | 5.0 | 7.0 | 6.5 | 6.5 | 3.0 | 4.0 | 5.29 | KILL |
| 14 | S04 | 2.5 (4.0/1.0) | 7.0 | 8.0 | 7.0 | 6.0 | 3.0 | 3.5 | 5.29 | KILL |
| 15 | N04 | 1.5 (2.0/1.0) | 6.5 | 7.0 | 7.5 | 6.0 | 3.5 | 2.0 | 4.86 | KILL |
| 16 | S03 | 2.0 (1.0/3.0) | 2.0 | 6.5 | 7.5 | 6.0 | 3.0 | 2.5 | 4.21 | KILL |

没有给 `DEFER_INFRA`：这些候选的首要死因是直接碰撞、通用化或缺少真实机制，不是“只差几张 GPU”。

## Candidate-by-candidate adversarial findings

### N01 — MarginLock-MoE — KILL

Closest: [MarginGate](https://arxiv.org/abs/2605.30218), [LLM-42](https://arxiv.org/abs/2601.17768).

- 方法新：低。把 output-logit margin-triggered verification 换成 internal router kth/(k+1) margin，仍是同一“margin 高走快路、margin 低验证/回退”结构。
- Remove-MoE：把 router 换成任意 top-k classifier，方法完整保留，失败。
- Fatal objection：即使 router route 被证明不变，expert/attention/logit 的数值差异仍可改变最终 token；若要全链 verifier，就退回 LLM-42/MarginGate。若只保证 route，则影响不足以支撑 deterministic-inference claim。

### N02 — PermuteExact — KILL

Closest: [UniEP](https://arxiv.org/abs/2604.19241), which already uses deterministic token ordering for numerical consistency.

- 现象可能是工程 bug；方法只是 permutation audit + canonical order/fallback。
- Remove-MoE：任意 batched MLP/grouped GEMM 都有 row permutation contract，失败。
- Fatal counterexample：标准 GEMM 每一 row 的 K reduction 独立，同 expert 内交换 rows 只交换输出 rows；unpermute 后完全等价。若发现某 backend 不等价，更像 kernel bug report/test suite，不是 MLSys method。

### N03 — ReplicaSem — KILL

Closest: [LayerCast](https://arxiv.org/abs/2506.09501), [FloatDoor](https://arxiv.org/abs/2606.19535).

- 跨硬件/kernel 的平台数值指纹及其可传播性已经建立；“兼容类内调度”是通用 heterogeneous serving policy。
- Remove-MoE：换成普通 DNN/model replicas 仍成立，失败。
- Fatal objection：相同 logical expert 若使用不同量化/不同 kernel，已不再是 byte-identical replica；若完全同栈同设备，预计没有足以支撑调度机制的差异。单卡模拟不能证明真实 replica interchangeability failure。

### N04 — BiTree-MoE — KILL

Closest: [TBIK](https://arxiv.org/abs/2511.17826).

- TBIK 已明确通过统一 hierarchical binary tree 对齐 intra/inter-GPU reduction order，并报告跨 TP bitwise identity。
- Remove-MoE：任意两级 reduction/fork-join 都可使用同一 tree，失败。
- Fatal objection：若 expert combine 与 TP reduction 可融合，几乎就是 TBIK 的层级树扩展；若不能融合，则 method 无法执行。两边都没有独立空间。

### N05 — SpectatorRoute — PHASE0_ONLY

Closest collision/evidence triangle:

- [RaMP](https://arxiv.org/abs/2604.26039) establishes that MoE runtime expert histograms change performance regions and optimal tile/kernel choices.
- [Batch-conditioned refusal protocol, arXiv:2605.27763](https://arxiv.org/abs/2605.27763) finds real but low-rate batch flips, a continuous-composition aggregate null at available sensitivity, and 0/55 reproduced flips under a batch-invariant kernel.
- [MarginGate](https://arxiv.org/abs/2605.30218) already owns sparse margin-triggered verification/fallback.

Independent claim verdict:

- **现象新：有条件成立。** Existing work does not directly show a prompt-only adversary deliberately shaping per-expert group geometry to induce a chosen victim’s kernel-regime transition and downstream semantic flip.
- **方法新：不成立。** stable-shape/canonical fallback at a risk boundary is MarginGate/LLM-42-shaped.
- Remove-MoE：dense FFN has no spectator-controlled sparse expert histogram, so the frozen phenomenon disappears—pass. The fallback method itself remains generic.
- Fatal counterexample：RaMP shows a performance-choice dependency, not a numerical-reduction-order dependency. Existing kernels may hold one static configuration and perform independent per-row K reductions; route shape then changes occupancy/padding only, never victim arithmetic. In that world the security channel is exactly zero.

### R01 — MarginWitness — KILL

Closest: [2025 Max-Plus state-estimation work](https://doi.org/10.1016/j.ifacol.2025.09.577).

- Winner/runner-up gap is ordinary sensitivity/slack of a `max` operator。
- Remove-MoE：任何 fork-join barrier 保留完整方法，失败。
- Fatal counterexample：runner-up branch can degrade while another unaffected branch remains dominant forever; gap shrinks but never affects completion or fault identity. “Early warning” then reports harmless slack consumption, not a culprit certificate. Without an MoE-only identifiability theorem this is max-plus diagnosis instrumentation.

### R02 — ReplicaFlip — KILL

Closest: [DSN 2024 interventional causal fault localization](https://research.ibm.com/publications/fault-localization-using-interventional-causal-learning-for-cloud-native-applications), [Legolas/NSDI’24](https://www.usenix.org/conference/nsdi24/presentation/wu-haoze).

- Remove-MoE：replicated services、storage shards or RPC workers admit the same crossover intervention，失败。
- Fatal objection：crossover simultaneously changes queue position, destination rank, communication path and interference. “输出冻结”不能冻结这些 timing variables，因此 intervention 并不 isolate logical-expert vs rank vs path。若额外复制网络/queue 状态来消混杂，成本接近 full controlled replay。

### R03 — CensorTrop — KILL

Closest: [2025 Max-Plus state estimation](https://doi.org/10.1016/j.ifacol.2025.09.577).

- Right-censored completion as lower-bound inequality is standard censored constraint modeling.
- Remove-MoE：任意 timed fork-join/queueing network 保留方法，失败。
- Fatal counterexample：timeout 只给 `T > τ`；两个 worlds——slow expert 与 slow upstream queue——可满足完全相同 inequalities。规则只能返回 `unobservable`。若没有证明 route hypergraph 产生新的 unique-identifiability class，贡献就是 censored max-plus tomography。

### R04 — RouteSyndrome — KILL

Closest: [Monitor Placement for Fault Localization](https://arxiv.org/abs/2311.16594), with recent partial-trace diagnosis represented by [Spectrum-based fault diagnosis with partial traces](https://doi.org/10.1016/j.jss.2025.112689).

- 方法是 signature-separating checkpoint selection/set cover。
- Remove-MoE：任意 DAG 的 sparse monitor placement 都成立，失败。
- Fatal counterexample：两个 experts 在所有选中层具有相同 route incidence，或者 fault 位于两个 checkpoints 之间并产生相同 barrier shift，则 signatures 不可分。增加 checkpoints 直到可分会退化成 all-layer tracing。没有优于通用 set cover 的 theorem。

### R05 — BarrierSpectroscopy — KILL

Closest: [DSN 2024 interventional causal fault localization](https://research.ibm.com/publications/fault-localization-using-interventional-causal-learning-for-cloud-native-applications).

- Positive-delay probing 是通用 active diagnosis/system identification。
- Remove-MoE：任意 max-join barrier 完整保留，失败。
- Fatal counterexample：对隐藏 branch 加 delay，在未超过当前 slack 前观察为零；第一次出现 barrier response 只给 slack 边界，且 concurrent queue jitter 可产生相同响应。要编码区分所有 branches 需要多轮、可能影响 SLO 的 probes；“harmless slack”又必须预先知道，形成循环。

### S01 — EpochSeal-MoE — KILL

Closest/direct collision:

- [vLLM Elastic EP official design](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism) prepares standby groups, performs a synchronized switch, resets captured state and coordinates ranks at a common engine boundary.
- [EEP](https://arxiv.org/abs/2605.10670) already defines live EP validity in terms of peer reachability, expert coverage and routing state matching membership.

- Remove-MoE：generic versioned shard remapping/fork-join keeps the mechanism，失败。
- Fatal objection：real runtime can switch only at a global step/barrier, so mixed-epoch activation is unreachable. If a runtime does permit it, epoch-tagged atomic mapping is standard version consistency, while EEP already claims routing/membership validity.

### S02 — VersionFence-MoE — KILL

Closest: [FluxMoE](https://arxiv.org/abs/2604.02715), which exposes expert paging and stable logical identity over transient physical residency.

- 现象前提是同一 serving instance 同时进行 async paging、in-place model/adapter update 和 cross-version join；packet 没有真实 runtime evidence。
- Remove-MoE：generic sharded model/cache snapshot fencing 完整保留，失败。
- Fatal counterexample：blue-green deployment or per-instance immutable snapshot makes mixed-version join unreachable. If concurrent versions are supported, snapshot/digest capability is ordinary content-addressed consistency rather than MoE research.

### S03 — FanoutLease — KILL

Closest runtime evidence: [DeepEP official API](https://github.com/deepseek-ai/DeepEP), [EEP](https://arxiv.org/abs/2605.10670).

Direct premise verdict:

- **没有找到主流真实 EP runtime 中的“per-request、per-rank partial reservation”。**
- DeepEP exposes whole-operation `dispatch(topk_idx)` → `EPHandle` → `combine(handle)` all-to-all semantics and explicit event synchronization. EEP likewise states each decoding step depends on dispatch/combine across the active EP ranks.
- Remove-MoE：若换成需要多个资源的 distributed job，FanoutLease 就是 gang scheduling/atomic multi-resource reservation，失败。
- Fatal objection：没有 partial reservation，就没有 attacker 可占住的 hold-and-wait 边。软件队列模拟会人为制造攻击前提。若未来某 disaggregated RPC runtime 真有该语义，atomic lease 又直接落入 gang/coflow admission。

### S04 — ReplicaColor-MoE — KILL

Closest: [CRAFT](https://arxiv.org/abs/2603.28768), [UltraEP](https://arxiv.org/abs/2606.04101).

- CRAFT 已做 memory-budgeted fine-grained per-layer expert replication；UltraEP 做 post-gating exact-load expert replication/placement。
- “安全域着色”执行上就是：为部分 experts 做副本，然后按 tenant 固定映射到 disjoint ranks。
- Remove-MoE：普通服务副本的 tenant partition 保留完整机制，失败。
- Fatal objection：若 rank-disjoint，收益完全来自 physical partition；若不 disjoint，安全隔离不成立。所谓最小 hypergraph cut 只是减少 partition 成本，没有新的隔离 primitive。**S04 就是 selective partition + hot-expert replication。**

### S05 — JoinLedger-MoE — KILL

Closest runtime contract: [DeepEP official dispatch/combine API](https://github.com/deepseek-ai/DeepEP).

- DeepEP already binds combine to the dispatch-produced `EPHandle` and requires stream/event synchronization.官方接口没有建立 prompt-side cancellation造成独立 expert branch retry、late completion 跨 request reuse 的语义。
- Remove-MoE：generic async scatter-gather 的 generation ID、exactly-once 和 stale-result rejection 全部保留，失败。
- Fatal objection：若 runtime 使用 step-scoped collective buffers，late branch 根本不可达；若通过任意 stale DMA/错误 slot 注入才能触发，那是内存安全/implementation bug，不是 route-triggerable attack class。Ledger 是标准 lifecycle correctness hardening。

### X01 — RouteShield-P — KILL

Closest:

- Attack phenomenon: [RepetitionCurse](https://arxiv.org/abs/2512.23995).
- Route/expert-pressure scheduling: [Gimbal](https://arxiv.org/abs/2606.15177), [ExpertPlex](https://arxiv.org/abs/2607.18002).
- Fairness: [VTC](https://arxiv.org/abs/2401.00588), [FairServe](https://arxiv.org/abs/2411.15997).

Executable-action verdict:

- **没有。** “只允许能删除 dependency 的 action”是 action eligibility predicate，不是 actuator。
- Gimbal 可以把请求送离 overloaded DP engine；ExpertPlex 提供 tile-level expert execution；但 packet 没有给出哪个现有 runtime API 能按 tenant route footprint 将 victim 与 attacker 放进物理不相交的 expert queue/lane，并证明 exact semantics。
- Remove-MoE：route-footprint attack 会消失，问题是 MoE-specific；但依赖删除 selector 与 isolation 动作本身是 generic。
- Fatal counterexample：唯一真正删除 shared completion edge 的动作若是 full instance/rank partition，它就是已列强 baseline；若仍共享 expert rank/queue，就没有删除依赖。故 Oracle 不是可执行 policy，当前 action set 为空。

## Frozen Phase-0 obligations for N05

N05 不得直接进入 refine。Phase-0 必须冻结为一次可证伪的 mechanism proof：

1. **冻结 stack。** 预注册一个 model、精度、vLLM/kernel backend、GPU、victim 集和 spectator generator；不得按结果换 model/kernel/victim。
2. **Matched controls。** benign、random、adversarial spectators 必须相同 batch size、token 数、top-k 数、总 expert work 与 arrival pattern；唯一 treatment 是 per-expert route histogram/group shape。
3. **Prompt-only threat model。** attacker 只能控制自己的 spectator text/requests，不能调用 kernel selector、修改 model、读取 victim hidden state 或手工指定 routes。
4. **四段因果链必须全部可观测：**  
   `spectator route histogram` → `victim expert group geometry` → `actual kernel/config/reduction regime` → `victim expert output delta` → `next-router or emitted-token flip`。
5. **Actual regime，不是 performance proxy。** 只看到 occupancy、padding、latency 或 RaMP cost-region 变化不够；必须记录实际执行 config/tile/reduction path。若 config 没变，KILL。
6. **Victim-local counterfactual。** 对同一 victim row，在 canonical/stable-shape replay 下输出应复原；若差异来自 spectator 自身语义、scheduler arrival 或 unmatched work，KILL。
7. **Adversarial增量。** 固定攻击生成器必须在未用于生成它的 frozen victims 上显著强于 matched-random/benign spectators；只重现一般 batch flip 不算独立贡献。
8. **Batch-invariant消融。** 同一 positive set 在 batch-invariant/canonical kernel 下必须归零；若不归零，所声称 route-shape→kernel mechanism 被否。
9. **最低 Phase-0 positive gate。** 至少 8 个互异 frozen victims，各自 standard kernel 10/10 重现完整因果链、batch-invariant 0/10；同时 adversarial flip-rate 的 Wilson lower bound 必须高于 matched-random upper bound。任一不满足即 **KILL**，不得调 gate、换 denominator 或重新挑 victims。
10. **Phase-0 成功仍不是论文证据。** 它只允许进入 multi-GPU/真实 continuous scheduler 的下一阶段；不能声称多租户 P99、可利用性或生产安全影响。

## Conditional next-search boundary if N05 dies

若 N05 Phase-0 失败，则下一轮应排除：

- margin-triggered verification/fallback；
- generic max-plus/tomography/monitor placement；
- epoch/version/exactly-once consistency；
- gang/coflow reservation；
- tenant partition + expert replication；
- 没有 runtime actuator 的“dependency-deleting” selector。

新的搜索入口应先从 **2026 实际 runtime 的可变状态/API** 出发：真实 routing-aware kernel selection、tenant-visible expert queue、cancel/retry lifecycle、online mapping switch、persistent-kernel lane ownership。只有先证明“这个可由 workload 触发的物理边真实存在”，再生成方法候选。
