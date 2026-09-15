# StableBatch 后续候选池（MaxGate-v1 NO-GO 后）

**生成时间**：2026-08-10 17:59 +08:00  
**范围**：10 个机制候选；只做机械去重，不在本文件内排名。  
**生成方式**：execution/kernel、oracle/budget 两个独立 agent lens 加主 agent 的 scheduler lens。精确 `dedup_key` 无重复，因此 10 个候选全部进入 jury。  
**统一边界**：all-M1 或 deterministic arm 都只是 operational reference，不是模型质量 ground truth；单 RTX 5090 prompt-forward 不能外推 continuous decode、EP/NCCL/RDMA 或生产收益。

## D01 — Fixed-M Expert Capsules

- **Observation → Insight**：相同 contribution 在 M=1/M=64 下出现 raw delta；如果根因是 shape-dependent GEMM plan，把每个 expert 的执行形状固定为 C 可能直接阻断变化，无需预测危险 rank。
- **Mechanism/System**：路由后按稳定 token identity 排序，把每个 expert 的 rows 切成固定 C 的 capsule，尾块补 dummy row 并在 combine 前 mask；由 capsule builder、grouped-GEMM launcher、unpermute/combine 组成。
- **最小实验**：新文档输入；冻结 C=8 或 16；同一真实 row 在不同 companion、row slot 和外部 M 下进入固定 capsule，比较 raw hash、combine hash、下游 route、padding FLOPs 和 kernel latency。
- **成功信号**：固定 capsule 内同一真实 row 跨 context bitwise invariant，消除对应 route divergence，且明显快于逐 row M1。
- **失败边界**：失败只否定“固定 M 足够”；成功仍需和 vLLM 全局 batch invariance 比成本。
- `dedup_key=fixed-m-expert-capsules`

## D02 — K-Tree Batch-Invariant Expert GEMM

- **Observation → Insight**：若 shape delta 来自 K-split/tile/reduction order，固定 K 归约树可在动态 M 下保持 row 结果。
- **Mechanism/System**：BF16 输入、FP32 accumulator、固定 K-block 和二叉归约树，只让 M 控制 row mask；替换 expert gate/up/down projections。
- **最小实验**：同一 row 扫 M=1..64 和 row positions，比 native、K-tree、M1 reference 的 projection/expert/route hashes 与 latency。
- **成功信号**：跨 M/position bitwise identical 且保留 M>1 并行度。
- **失败边界**：数值成功但过慢只证明可行性；且与 vLLM deterministic kernels 强碰撞。
- `dedup_key=k-tree-batch-invariant-expert-gemm`

## D03 — Execution-Signature Plan Fence

- **Observation → Insight**：不稳定可能集中在少数 plan transition，而非所有 M；对实际 algorithm/workspace/tile/split-K signature 做 allowlist，可能保留多数快路径。
- **Mechanism/System**：离线 calibration 生成 signature manifest；运行时仅放行与固定 reference 等价的 plan，未知或漂移 signature 回退到 fixed capsule / invariant kernel；记录 provenance ledger。
- **最小实验**：独立 calibration 扫 M=1..64 并记录 plan；冻结 allowlist 后用新文档、stream 顺序和并发背景验证 false accept、coverage、fallback 与 latency。
- **成功信号**：held-out 零 false accept、非零且可复现 fast-path coverage、版本漂移 fail-closed。
- **失败边界**：同一 signature 的等价性依赖输入，则 plan identity 不充分；全回退没有系统价值。
- `dedup_key=execution-signature-plan-fence`

## D04 — Next-Router Verified Replay

- **Observation → Insight**：MaxGate 看当前 gate weight，与真正的下游 route boundary 错位；可在下一 router 处验证而不是预判 rank。
- **Mechanism/System**：目标 MoE 前保存 checkpoint；fast path 后在下一 router 以 margin 触发 fixed-shape suffix replay；比较 route membership 后 commit/rollback。
- **最小实验**：新文档，native、always-fixed、triggered replay 三臂，报告 misses、trigger rate、replicated FLOPs、latency。
- **成功信号**：相对 always-fixed 零漏检且 replay 比例显著低。
- **失败边界**：只管 route membership，不覆盖 value-only token flip；与 LLM-42/MarginGate 高度相邻。
- `dedup_key=next-router-verified-replay`

## D05 — Counterfactual Oracle Frontier → Runtime Distillation

- **Observation → Insight**：MaxGate 失败不等于 action set 无价值；先枚举每个可保护 contribution，才能区分“没有上界”和“只是 selector 差”。
- **Mechanism/System**：新文档上枚举单 action（以及小规模 joint action），求预算 B 下的 oracle frontier；再只用 action-time fields 蒸馏 cost-sensitive abstaining policy，最后由 budget arbiter 调度。
- **最小实验**：在相同 work surface 下得到 native、1000 个冻结 balanced shuffles、oracle、document-disjoint surrogate；第一阶段只需跑 oracle 与 shuffle envelope。
- **成功信号**：Oracle 在 matched budget 下超过 shuffle 95th percentile，收益跨至少 4 个文档；后续 surrogate 在 untouched test 保留至少 30% oracle-over-shuffle lift。
- **失败边界**：Oracle 不胜 shuffle 就杀掉该 action granularity；Oracle 正而 surrogate 负只说明 observability 失败。
- `dedup_key=counterfactual-oracle-frontier-runtime-distillation`

## D06 — Residual-Divergence Risk Budget

- **Observation → Insight**：传播稀疏，固定 action count 不是唯一控制目标；可声明 residual route-divergence risk budget。
- **Mechanism/System**：document-disjoint risk model + calibration upper bound；每 request 同时维护风险与 action budget，不可覆盖/漂移 context fail-closed 到 canonical path。
- **最小实验**：always native、always canonical、balanced shuffle、risk controller；报告 untouched test 的 one-sided residual-risk bound 与 canonical action fraction。
- **成功信号**：满足预注册风险上界，同时比 always-canonical 少用稳定 action。
- **失败边界**：若几乎全回退则没有 runtime value；统计风险不能冒充 bitwise certificate，且与 MarginGate 邻近。
- `dedup_key=residual-divergence-risk-budget-shield`

## D07 — Shadow-Labeled Adaptive Stability Policy

- **Observation → Insight**：action value 可能随 layer、M、kernel signature 与 load regime 漂移；静态 gate-weight score 失配。
- **Mechanism/System**：对少量请求 off-critical-path canonical shadow replay，产生 delayed action-value label；constrained contextual bandit 只在 shadow 中探索，committed path 用 pessimistic value 和硬预算。
- **最小实验**：新文档 replay stream，强制两个 shape/kernel regime 转换；对比 shuffle、static scorer、adaptive policy 和 offline oracle regret。
- **成功信号**：warm-up 后 reward/action 为正、regret 增长慢于静态策略、在 hard budget 内完成 regime adaptation。
- **失败边界**：若 shadow 接近完整双跑或标签来不及跟踪漂移，则机制失去价值；generic bandit 不是贡献。
- `dedup_key=shadow-labeled-adaptive-stability-policy`

## D08 — Hierarchical Stability-Quota Allocator

- **Observation → Insight**：MaxGate 只否定 rank identity ranking，不直接否定跨 layer/shape strata 的预算分配。
- **Mechanism/System**：离线估计 layer/M/margin/kernel stratum 的 reward-cost curve；运行时 robust knapsack/primal-dual 给 strata 配额，stratum 内仍用 seed-frozen balanced random 选择，隔离“配额”与“rank selector”。
- **最小实验**：uniform shuffle、proportional quota、robust quota、oracle stratum allocator，同 global action count 和同 randomized identity ledger。
- **成功信号**：robust quota 超过预冻结 shuffle distribution，收益跨至少 4 个文档且可归因于 cross-stratum allocation。
- **失败边界**：stratum curve 不可迁移就停止；generic allocator 只有在 Oracle 显示 strata heterogeneity 后才有意义。
- `dedup_key=hierarchical-stability-quota-random-within-stratum`

## D09 — Route-Barrier Verify-and-Repair

- **Observation → Insight**：大多数局部 delta 不传播，可以在最早 route commit barrier 之后稀疏验证并修复。
- **Mechanism/System**：保存前一安全边界 state；native 执行后触发 canonical suffix replay；route 不同则原子替换 canonical state，并 shadow-sample untriggered misses。
- **最小实验**：always native、always canonical、always verify、triggered repair、random trigger；验证 repair closure、misses、checkpoint/replay cost。
- **成功信号**：所有 triggered mismatch 被正确修复且 replay 少于 always verify。
- **失败边界**：若需重放接近完整 prefix 或 margin 漏检，则无系统收益；与 D04、LLM-42、MarginGate 高度相邻但不做主观去重。
- `dedup_key=moe-route-barrier-verify-repair`

## D10 — Stability-Aware Expert Shape Lanes

- **Observation → Insight**：scheduler 决定每个 expert 的 live M；与其猜哪一行会伤害下游，不如让 scheduler 把 routed rows 组织进少量经过验证的 canonical shape lanes。
- **Mechanism/System**：不改 top-k/expert identity；按 expert 将 ready rows 排入 C1/C2... 固定形状 lane，deadline 到期时 padding flush，高压时 split；lane 选择同时考虑等待时间、padding cost 和 signature stability，但不读未来 route outcome。
- **最小实验**：使用冻结 routed trace 回放 native variable-M、always-M1、fixed-capsule、two-lane deadline policy；数值用 raw/route hash，系统用 padding、queue delay 与 kernel time，先做 offline exact trace + 5090 microbenchmark。
- **成功信号**：相对 canonical reference 消除 observed route divergence；相对 always-M1/全局 invariant path 显著减少 kernel calls 或 GPU time，并满足冻结 deadline。
- **失败边界**：若低-M decode 使 padding/等待成本吞没收益，杀掉 lane scheduler；若 fixed shape 不保证 row invariance，先杀 mechanism，不做 controller。
- `dedup_key=stability-aware-expert-shape-lanes`

## 机械去重结果

- 输入 10，精确 `dedup_key` 唯一 10，输出 10。
- D04 与 D09 在语义上相近，但 key 不同，按协议不做主观合并；交由 jury 处理。
- D01 与 D10 共用 fixed-shape primitive，但一个是 kernel/execution capsule，一个是 serving queue/lane policy，不做主观合并。

