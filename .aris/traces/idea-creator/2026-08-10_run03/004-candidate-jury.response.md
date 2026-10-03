# Jury verdict

MaxGate-v1 只排除了这个叶子假设：

> 在冻结的 16 个窗口、240 个 victim-layer cells、all-M1 自监督 reference 和“一项 M1 + 七项 M64”动作面上，用当前层最大 gate weight 选择唯一受保护 contribution，能够比该次预先冻结的 balanced-shuffle rank 获得更高的聚合 downstream route-membership reward。

证据是 `-3` 对 `+3`，且 MaxGate-v1 有 13 个正 cell、18 个 harm cell。它没有证明 MaxGate-v1 普遍劣于随机分布，更没有否定 propagation、action upper bound、multi-action、canonical shape、pack surgery、stable kernel、sparse stability budget、route-aware batching 或 numerical stability as a runtime resource。

## 机制池检查

通过。进一步合并后仍有至少 11 个机械不同的机制族。`C03` 并入 `C01`；`C07+C12` 合并为 ShapeABI/PadCap；`C05+C13` 合并为 RouteGuard/precision-island；C14 只保留 canonical combine arithmetic；C09/C10 若退化成 scalar score swap 则不成立；C02 是 C01 之上的 allocator。旧 `NOVELTY_CHECK_TOP3.md` 的 C09/C02/C04 编号不得套到当前候选。

## Survivor ranking

1. `C01` Stability-budget oracle/action frontier（吸收 C03）
2. `C07+C12` ShapeABI / VirtualShape + PadCap
3. `C06` RouteStress WitnessPatch
4. `C13+C05` RouteGuard stable-kernel mux / precision island
5. `C02` Sparse protection cover allocator
6. `C10` PairGraph compatibility matching
7. `C08` BucketLock canonical ladder
8. `C14` CanonicalCombine，仅保留 arithmetic mechanism
9. `C11` FrontierLock coupled queues
10. `C09` RouteCohort composition control
11. `C04` First-divergence route clamp

## Top 3

### Top 1 — C01 Stability-budget oracle/action frontier

**Hypothesis**：若单贡献稳定化动作空间本身有价值，hindsight oracle 应能用少量 cell-rank actions 恢复显著 downstream route agreement，并优于等动作预算的随机 cell-rank 选择。

**Smallest discriminating experiment**：在现有 240 cells 执行 `R/U/A0..A7`；比较 no-action、forced oracle、abstaining oracle、等预算随机 cell+rank，并复跑 oracle 选中的正 action。

**Support**：recovery fraction ≥25%、相对等预算 random 优势 ≥8、正收益覆盖 ≥4 victims。  
**Falsify**：oracle 正收益为零/可忽略，或只落在孤立 victim。  
**Scope of failure**：只削弱当前 synthetic same-cell、单贡献 M1-in-M64 action；不影响 multi-action、natural pack、canonical shape 或 stable kernel。  
**Cost**：8–15 GPU 分钟。

### Top 2 — C07+C12 ShapeABI / PadCap

**Hypothesis**：将 variable logical M 映射到 deterministic slot 和固定 physical template，可在保留批处理的同时消除大部分 spectator-composition-induced route variation。

**Smallest discriminating experiment**：在 6–10 个 confirmed unstable rows 上改变 spectator composition，三臂比较 native variable-M、固定模板 masked padding、serial M1；只测 raw output、下游 route 和 kernel time。

**Support**：fixed template 明显降低/消除 route variation，同时 per-real-row time 优于 serial M1。  
**Falsify**：固定模板仍随 composition 改变，或成本等同/劣于 serial M1。  
**Scope of failure**：只否定该 template/slot/masking 实现，不否定其他 canonical arithmetic。  
**Cost**：35–55 GPU 分钟。

### Top 3 — C06 RouteStress WitnessPatch

**Hypothesis**：自然 heterogeneous expert pack 的传播分歧由小型 co-row conflict subset 触发，保护/切开该 witness 会比等预算 shuffled surgery 阻止更多 downstream divergence。

**Smallest discriminating experiment**：exploration split 生成并最小化 natural positive packs，冻结 predicate；在 4–8 个 fresh victims 上执行 native、witness-patch、budget-matched shuffled-patch。

**Support**：avoided divergence/protected contribution ≥ shuffle 的 1.5×，且覆盖 ≥4 victims。  
**Falsify**：fresh split 无法触发，或与 shuffled 持平/更差。  
**Scope of failure**：只否定该 witness predicate/minimal surgery，不否定其他 composition action 或 canonicalization。  
**Cost**：60–90 GPU 分钟。

## Oracle runner P0/P1

没有 P0。只发现 1 个 P1：abstaining oracle 与 random 的动作预算不匹配。原代码把“240 cells 每个都随机保护一个 rank”的期望直接与只在正收益 cells 动作的 abstaining oracle 比较，会改变 `STRONG/WEAK/WEAKENS` 分类以及“oracle 优于 random”的陈述，方向在跑前未知。

最小修正：对每个实际 `actions_used=B` 同时报告：

1. uniform-random `B` 个 cell-rank actions 的精确期望；
2. 在 oracle 所选 cells 内 uniform-random rank 的条件期望。

strong verdict 使用两种等预算比较，无需增加 GPU 工作。

`reviewer_model=gpt-5.6-sol`  
`reviewer_reasoning=xhigh`  
`review_independence=same-family`  
`acceptance_status=provisional`
