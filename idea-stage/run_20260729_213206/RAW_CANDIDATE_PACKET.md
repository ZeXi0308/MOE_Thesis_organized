# Fresh MoE Idea Candidate Packet

> 时间：2026-07-29 21:32 +08:00  
> 状态：`EXPLORATORY / GENERATION_ONLY / NOT_CURRENT_MAINLINE / NO_EMPIRICAL_RESULT`  
> 规则：以下 15 个新候选加 1 个既有协议候选只做机械汇总；此处不排序、不按质量删除。相同 `dedup_key` 为零，故 16/16 全部进入 fresh jury。

## 不可变边界

- 当前没有已成立的新 MoE system mechanism；旧 C10、C01、RouteGuard-KV 已终止，C09 plain NNLS/max-plus tomography 和 C14 robust-static 只能作通用诊断/强 baseline。
- 单 RTX 5090 只能做每个候选不超过 2 GPU-hour 的资格性证伪；EP/NCCL/RDMA、真实多租户 P99、rank/link failure 或迁移结论必须标多 GPU 未验证。
- 禁止把 workload、阈值、denominator 或 action set 按 pilot 结果改写来救方向。
- 已知强碰撞必须正面裁决：LLM-42、MarginGate、batch-conditioned refusal、TBIK、Exact-Differential Simulation、general max-plus diagnosis、gang scheduling/coflow admission、VTC/FairServe、RepetitionCurse、Gimbal/UltraEP/ExpertPlex、EEP/Tarragon、MoEcho/RouteScan/FloatDoor。

## A. Numerical / semantic execution shard

### N01 — MarginLock-MoE

- **Method**：对 kth/(k+1) router margin 建立保守数值误差界；仅在界证实 route 不变时走快 kernel，否则走 canonical/fixed-shape kernel。
- **Hypothesis**：多数自然 token 的 route margin 足以证明快路径不会改变 top-k，可把确定性成本集中到脆弱 token。
- **Pilot**：两个 MoE、256 prompts × 128 decode，比较 route/output divergence、fallback rate、overhead，1–2 GPU-hour。
- **Risk / neighbors**：误差界可能过松；与 LLM-42、MarginGate、batch-invariant kernels 直接相撞。
- **dedup_key**：`router-margin-certified-numerical-fastpath`

### N02 — PermuteExact

- **Method**：审计 grouped-GEMM 中同一 expert 内 token permutation 是否保持逐 token 输出等价；对非等变 kernel 使用 canonical ordering/fallback。
- **Hypothesis**：部分 optimized MoE kernel 对 token packing/permutation 不具备足够稳定的语义等价性。
- **Pilot**：10k routed rows，在不同 group sizes、permutations、kernels 下做 bitwise/ULP/route-cascade 检查，0.5–1.5 GPU-hour。
- **Risk / neighbors**：kernel 很可能本来逐 row 等价；即使发现 bug 也可能只是 audit/engineering。
- **dedup_key**：`within-expert-permutation-semantic-contract`

### N03 — ReplicaSem

- **Method**：为相同逻辑 expert 的副本建立 kernel/device numerical fingerprint；只在兼容类内调度，或在边界 token 上 canonical replay。
- **Hypothesis**：不同副本/设备/量化路径会产生可传播到后续 route 的语义差异，破坏“副本可互换”假设。
- **Pilot**：单卡模拟不同 kernel fingerprints，比较相同 expert 的输出、后续 routes 和 tokens，1–2 GPU-hour；formal 需多 GPU。
- **Risk / neighbors**：可能退化为通用异构推理 nondeterminism；EEP/Tarragon/FloatDoor 是强邻居。
- **dedup_key**：`expert-replica-semantic-fingerprint`

### N04 — BiTree-MoE

- **Method**：跨 expert combine 与 TP reduction 使用统一 canonical binary tree，使不同 expert/rank/TP 配置保持 reduction order。
- **Hypothesis**：联合两级 reduction tree 可在较低开销下消除 MoE+TP 配置相关输出差异。
- **Pilot**：构造 counterexample 和 Triton prototype，1–2 GPU-hour；formal 需 4–8 GPU。
- **Risk / neighbors**：实际 stack 的两次 reduction 可能无法联合；TBIK 已极接近。
- **dedup_key**：`joint-expert-tp-canonical-reduction-tree`

### N05 — SpectatorRoute

- **Method**：测量 co-batched spectator tokens 改变 per-expert group shape/tile regime 后，victim hidden/output 和后续 routes 是否变化；仅在 route-shape 风险边界使用 stable-shape/canonical path。
- **Hypothesis**：MoE 中别人的 route 决定 victim expert kernel 的 group shape，数值扰动会经后续 router 放大；对抗性 route shaping 可能比普通 batch composition 更易触发 victim semantic flips。
- **Pilot**：64–256 frozen victims，matched benign/random/adversarial spectator sets，标准与 batch-invariant kernel 消融，1–2 GPU-hour。
- **Risk / neighbors**：MarginGate 与 batch-conditioned refusal 已发现低频 batch flips；如果 adversarial route shape 无显著增量或 kernel 逐 row invariant，则 KILL。
- **dedup_key**：`spectator-route-shape-semantic-interference`

## B. Reliability / observability shard

### R01 — MarginWitness

- **Method**：记录 top-k combine 的 winner/runner-up completion gap 与 queue epoch，形成 max-barrier 约束并返回 culprit/equivalence class/unobservable。
- **Hypothesis**：runner-up margin 能在非关键 branch 累积退化时，比 critical-path-only tracing 更早暴露故障。
- **Pilot**：两模型冻结 routes，注入 5/10/20% expert/rank slowdown，比较 lead time、false certificate、trace bytes，CPU replay + ≤2 GPU-hour。
- **Risk / neighbors**：采集成本可能等同 full tracing；可能只是 max-plus fault diagnosis。
- **dedup_key**：`moe-runner-up-margin-censored-criticality-witness`

### R02 — ReplicaFlip

- **Method**：在 byte-identical expert replicas 间做平衡 crossover，冻结 route/weights/output，借对称干预区分 logical expert、physical rank 和 network path。
- **Hypothesis**：副本对称性可打破被动 route hypergraph 的定位歧义。
- **Pilot**：formal 4-GPU replicated expert，分别注入 compute/rank/path fault；单 5090 只能验证 replay 和 identity controls。
- **Risk / neighbors**：reassignment 同时改变 queue 和 path，仍然混杂；active diagnosis/replica testing 很通用。
- **dedup_key**：`semantic-preserving-expert-replica-crossover-diagnosis`

### R03 — CensorTrop

- **Method**：把 completed observation 编码为 max-barrier bounded equality，把 timeout/cancel/window-end 编码为 lower-bound inequality；只在所有 feasible worlds 同意时给 unique culprit。
- **Hypothesis**：route-conditioned censoring 保留 completion-only 诊断丢掉的最坏请求证据。
- **Pilot**：小 top-2 hypergraph 穷举零误证，再做 trace replay；CPU + ≤2 GPU-hour。
- **Risk / neighbors**：若无 MoE-specific identifiability theorem，直接退化为 censored max-plus tomography。
- **dedup_key**：`right-censored-moe-max-plus-fault-certificate`

### R04 — RouteSyndrome

- **Method**：用跨层 route diversity 选择最小 barrier checkpoint 子集，使 expert/rank/link fault signatures 可分离；仅记录这些 timestamps。
- **Hypothesis**：自然 cross-layer routes 可作为 fault-separating code，以小 tracing volume 保留 isolation 信息。
- **Pilot**：两模型 routes + all single-fault replay，比较 selected/random/all-layer checkpoints，CPU + ≤2 GPU-hour。
- **Risk / neighbors**：可能只是 monitor placement/set cover；重复 placement 导致不可辨识。
- **dedup_key**：`cross-layer-route-syndrome-sparse-barrier-checkpoints`

### R05 — BarrierSpectroscopy

- **Method**：对低优先级 diagnostic requests 的 expert branches 注入 positive-only coded micro-delays，以 combine barrier 的非线性响应推断 hidden critical branches。
- **Hypothesis**：受控微延迟可揭示被 max barrier 隐藏的临界性。
- **Pilot**：先做 small-model exhaustive check，再做 4-GPU replicated-expert microbenchmark；单卡仅做 emulation。
- **Risk / neighbors**：active diagnosis 很通用，且“harmless slack”本身需先可见。
- **dedup_key**：`coded-microdelay-moe-barrier-active-diagnosis`

## C. Security / integrity shard

### S01 — EpochSeal-MoE

- **Method**：把 top-k route、logical→physical map 与 join 绑定到同一 map epoch；dual-map drain，混 epoch 时整 token replay。
- **Hypothesis**：route burst 推动 online replication/migration 时可使 victim branches 跨 mapping epoch；atomic snapshot 可消除混 epoch output。
- **Pilot**：单 5090 的 software-rank/CUDA-stream fault injection ≤1.5 GPU-hour；formal 需 ≥4 GPU 真实 online remap。
- **Risk / neighbors**：真实 runtime 可能只在 global barrier 切 map，攻击面不存在；可能只是版本一致性工程。
- **dedup_key**：`mapping_epoch_atomic_fork_join`

### S02 — VersionFence-MoE

- **Method**：把 model snapshot/layer/expert/weight digest/replica digest 作为 ExpertCapability；请求固定 snapshot，join 只接受同版本 top-k contributions。
- **Hypothesis**：route-shaped cache churn 加滚动更新可导致 expert paging/offload 的混版 join。
- **Pilot**：单 5090 16–32 small experts、host/GPU async paging 与双版本切换，≤2 GPU-hour。
- **Risk / neighbors**：blue-green deployment 可能天然排除；内容寻址版本 fencing 属通用一致性机制。
- **dedup_key**：`expert-cache-snapshot-consistency`

### S03 — FanoutLease

- **Method**：把 top-k physical-rank set 当超边；只有全部 rank 有额度才原子提交，避免 partial admission/hold-and-wait；按 rank union × service 收费。
- **Hypothesis**：wide-route attacker 可在 per-rank admission 下超线性占用部分资源；atomic lease 可删除 hold-and-wait 边。
- **Pilot**：8 software queues + multi-stream kernels ≤1.7 GPU-hour；formal 需 ≥4 GPU multi-tenant EP。
- **Risk / neighbors**：主流 MoE runtime 可能根本没有逐 rank partial reservation；与 gang scheduling/coflow admission 强碰撞。
- **dedup_key**：`atomic-hyperedge-rank-admission`

### S04 — ReplicaColor-MoE

- **Method**：按 tenant security domain 给 logical expert replicas 着色，用 route hypergraph 选择最小副本 cut，使 protected/untrusted traffic physical-rank disjoint。
- **Hypothesis**：只复制覆盖的 experts 可切断同 rank contention，成本低于整模型分区，并优于 route-blind fairness。
- **Pilot**：单卡只做 queue replay/memory estimate ≤2 GPU-hour；formal 需 4–8 GPU。
- **Risk / neighbors**：本质可能是 fixed partition/hot-expert replication；显存成本和 replica semantics 可能吞掉收益。
- **dedup_key**：`security-domain-colored-expert-replicas`

### S05 — JoinLedger-MoE

- **Method**：为每个 activation 建不可复用 generation capability；join 校验 exactly-once、完整 expert multiset、tenant/generation/target rank，拒绝 stale/duplicate/cross-tenant result。
- **Hypothesis**：取消/重试/slot reuse 可放大 late activation 污染其他 request 的风险。
- **Pilot**：单卡 multi-stream toy top-2，注入五类 lifecycle fault ≤1.5 GPU-hour；formal 需 ≥4 GPU async EP。
- **Risk / neighbors**：NCCL/runtime 的同步与 generation tags 可能早已排除；更像 runtime correctness bug class。
- **dedup_key**：`activation-generation-conservation-ledger`

## D. Existing protocol-only candidate

### X01 — RouteShield-P

- **Method**：在 exact semantics 下，以 tenant expert/physical-rank demand 为公平对象；只允许能删除 victim→attacker completion-DAG dependency 的 batch/lane action；与 quota、VTC、FairServe、DRF、partition、RepetitionCurse placement、Gimbal 等强 baseline 比较。
- **Hypothesis**：black-box adversarial text 的 route concentration 在 matched work 下仍造成 route-specific victim TTFT harm，且 route-aware legal isolation 相对最强 route-blind/simple baseline 有稳定增量。
- **Qualification**：5090 只做 route-footprint census 与 full-DAG replay qualification；正式 victim P99 和 action existence 需 8×A100。当前 `PROTOCOL_ONLY / BLOCKED_MISSING_FORMAL_EVIDENCE`。
- **Risk / neighbors**：RepetitionCurse 已建立 attack，VTC/FairServe 已建立 fairness，Gimbal/UltraEP/ExpertPlex 已建立 expert-pressure scheduling；若 backend 无依赖删除动作或 strong baseline 捕获 ≥90% Oracle，则 KILL。
- **dedup_key**：`route-aware-tenant-expert-footprint-isolation`

## Mechanical merge record

- 新 shard 原始候选：5 numerical/semantic + 5 reliability/observability + 5 security/integrity = 15。
- 既有 protocol-only 候选：RouteShield-P = 1。
- exact `dedup_key` collision：0；未做主观语义合并。
- fresh actuation shard 在超时后中断，未产生可用候选；没有伪造补齐。
- 进入 jury：16/16；所有 pilot 数字均为计划，不是结果。
