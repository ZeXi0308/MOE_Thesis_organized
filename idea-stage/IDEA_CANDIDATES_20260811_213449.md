# MoE 下一轮系统化候选池（机械合并，未排名）

**生成时间**：2026-08-11 21:34:49 +08:00
**候选来源**：causal-critical-path shard（4）+ verification-transactionality shard（4）+ recent-literature-residual shard（4）
**机械去重**：12 个输入，exact `dedup_key` 重复 0；jury 前未按质量筛选
**硬边界**：当前 formal system/method GO 为 0；本文只生成候选，不授权实现、实验或论文 claim。

## C01 — Route-Conditioned Delay Survival Map

- **Method**：在 identity-complete full-request DAG 上，对真实 MoE 节点注入工作量保持的 1%/5% delay pulse；测量局部延迟被 slack/barrier 吸收还是传到 token/request completion；在 holdout 上比较 route-conditioned 特征与 generic slack、max-load、queue-depth。
- **Hypothesis**：局部 expert 延迟到 request completion 的传播存在稳定的 route-conditioned survival structure。
- **Minimum experiment**：两个模型，各一个中负载和近饱和 natural cell；每 cell 至少 32 paired episodes。
- **Positive**：两个模型都存在 non-trivial high-survival nodes，且 route 信息在简单基线外有增量。
- **Negative**：绝大多数脉冲被吸收，或 generic topology/slack 已完全解释；停止从局部 expert-tail proxy 设计机制。
- **Risk / effort**：MEDIUM / 3–4 weeks，CPU DAG + 1×RTX 5090 trace/probes。
- **Closest work**：COZ causal profiling、critical-path/slack sensitivity、TELLER/CRISP 类 trace diagnosis。
- **dedup_key**：`route-conditioned-delay-survival-map`

## C02 — Intervention-Calibrated Request-DAG Trace

- **Method**：定义 request/step/layer/route/dependency/resource-order/service identity trace contract；插入低幅度 delay、resource-occupancy、barrier-marker probes；基线 trace 盲预测 held-out probe 的 completion delta、顺序与受影响请求，并把误差定位到缺失 edge class。
- **Hypothesis**：只有经过真实小干预校准，profiler trace 才足以支撑 full-request Oracle；错误集中在少数动态 batching/resource-order/backpressure edge 类。
- **Minimum experiment**：两个模型各 32 frozen natural episodes；expert completion、combine release、batch turnover 三类 paired probes，半数建图、半数盲验。
- **Positive**：同一 schema 跨模型正确预测 held-out completion-delta 符号和主要受影响 request set，幅度误差进入预注册噪声合同。
- **Negative**：某 edge 类系统性失败则返回 missing-causal-category，并阻止后续 Oracle。
- **Risk / effort**：HIGH / 4–6 weeks，runtime instrumentation + GPU paired runs + CPU replay verifier。
- **Closest work**：COZ、CRISP、TELLER、deterministic record/replay、Exact-Differential Simulation。
- **dedup_key**：`intervention-calibrated-request-dag-trace`

## C03 — MoE Critical-Path Motif Census

- **Method**：从 natural full-request DAG 的 realized critical subgraph 中提取 route fan-in join、expert queue carry-over、batch-turnover recoupling、跨-step tail inheritance 等 MoE motifs；用 document/load-disjoint holdout 比较 offered-load、max-queue、CV 等简单基线。
- **Hypothesis**：少数可复现 MoE route-linked motifs 主导 continuous-decode tail，且不退化为 generic max-load/queue-depth。
- **Minimum experiment**：两个模型×两个负载 cell×128 requests；calibration 最多发现四类 motif，evaluation 冻结验证。
- **Positive**：小型 motif 集跨模型覆盖多数 tail critical subgraphs，并在简单基线外保留解释力。
- **Negative**：motif 不稳定或 route 标签无增量；停止以 MoE-specific critical-path structure 为主线。
- **Risk / effort**：MEDIUM / 3–5 weeks。
- **Closest work**：critical-path motif mining、tail-at-scale diagnosis、TELLER、Gimbal workload characterization。
- **dedup_key**：`moe-critical-path-motif-census`

## C04 — Semantic–Temporal Feedback Closure Audit

- **Method**：同一 request ledger 同时记录 source-dtype router state、top-k identity、batch/shape、resource-order DAG、token 与 completion；配对执行 native dynamic batch、batch-invariant numerical control、route-clamped shadow；把 completion 差异分为 ordinary scheduling、numerical route/work feedback 与 downstream propagation。
- **Hypothesis**：execution/batching numerical divergence 会改写后续 route/work DAG，导致固定-route局部 Oracle 系统性高估、低估或反转 full-request effect。
- **Minimum experiment**：两个 fixed revisions、128 frozen natural requests；逐层逐 step 记录 first divergence、route/work DAG edit distance、token divergence、completion delta，matched-composition shuffle 作负控。
- **Positive**：两个模型均出现审计闭合的 route/work feedback，且 route-clamped 与 native full-request effect 有稳定、超噪声差距。
- **Negative**：batch-invariant control 下 route 全稳，或局部 flip 不传播到 route/token/completion；允许 fixed-work DAG，并把历史阳性限制为 local numerical proxy。
- **Risk / effort**：HIGH / 3–5 weeks，1×RTX 5090 + CPU full-DAG/tensor provenance。
- **Closest work**：vLLM batch invariance、numerical-state divergence、RaMP/DA-MoE、causal replay。
- **dedup_key**：`semantic-temporal-feedback-closure-audit`

## C05 — Routed-MoE Commit-Horizon Atlas

- **Method**：在 fresh SFV2-O2 shadow outcomes 上预注册 post-expert、post-combine、next-router、token-state commit 四个观测边界；记录 unsafe edge 最早可见点与最后安全回滚点；把 checkpoint/buffer/stall/verify/replay cost 传播到 full-request DAG。
- **Hypothesis**：execution-shape divergence 在不可逆 token commit 前存在稳定且足够早的 observability horizon，决策可后移而无需预测 action value。
- **Minimum experiment**：至少 4 个新 documents；逐 edge 计算 unsafe miss、verified pairs、row coverage、buffer/stall/checkpoint/replay GPU-time。
- **Positive**：至少一个冻结 horizon 对 eventual unsafe 零漏检，至少 16 pairs、5% rows，完整计费后净正。
- **Negative**：所有 horizon 漏检，或信号过晚，或成本吞噬收益；冻结 horizon set，不换信号抢救。
- **Risk / effort**：MEDIUM / 1–2 weeks；full-request conclusion 仍依赖 DAG closure。
- **Closest work**：SFV2-O2、LLM-42、MarginGate、local D04 Next-Router Verified Replay。
- **dedup_key**：`moe-commit-horizon-atlas`

## C06 — Row-Atomic Shadow Transaction Ledger

- **Method**：tentative M2 endpoint 进入按 request/step/layer/expert/row 标识的 escrow；冻结 verifier 后 PREPARE；unsafe endpoint 做 M1 rescue；在 combine/next-router 边界审计 read/write/dependency closure 后 atomic COMMIT，否则 ABORT two-M1。
- **Hypothesis**：若 unsafe 只污染 pair 的部分 endpoint，row-granular commit 比 whole-pair rollback 少重算且保持 zero unsafe commit。
- **Minimum experiment**：fresh SFV2-O2 ledger 上比较 whole-pair rollback、row-atomic rescue、always-M1、ordinary verify/rollback，实测 M2/M1/stitch/ledger/boundary cost。
- **Positive**：zero unsafe commit，至少 16 pairs/5% rows，row-atomic 比 whole-pair 少实测重算且总净 saving >0。
- **Negative**：任一 unsafe 越界、污染不可 row-isolate 或 overhead 吞噬收益。
- **Risk / effort**：HIGH / 1–2 weeks prototype；full-DAG correctness later。
- **Closest work**：SFV2-O2、LLM-42、asymmetric recomputation/partial commit。
- **dedup_key**：`row-atomic-shadow-transaction`

## C07 — Causal Repair Cones for Sparse MoE Divergence

- **Method**：verifier 失败后，从 identity-complete request DAG 标记受影响 row 的确切 dependents；生成 invalidation certificate，仅撤销尚未 commit 的 repair cone；M1 重算并沿 cone replay 至冻结边界重合，无法证明 closure 时扩为 full rollback。
- **Hypothesis**：MoE sparsity 与 row identity 可使多数 divergence 的 repair scope 远小于完整 token suffix。
- **Minimum experiment**：在 fresh continuous-decode traces 上注入所有 observed M1/M2 mismatch，逐一构造 cone，以 full-M1 run 验证 closure。
- **Positive**：zero closure mismatch，预注册比例 failure 的 cone work 显著小于 full suffix，完整计费后保留 full-request headroom。
- **Negative**：attention/residual 使 cone 普遍扩为完整 suffix，或依赖不可追踪，或 overhead 吞噬 headroom。
- **Risk / effort**：HIGH / 2–4 weeks；受 full-request-DAG evaluator P0 阻塞。
- **Closest work**：LLM-42、SFV2-O2、route-barrier verify-and-repair。
- **dedup_key**：`dependency-scoped-moe-repair-cone`

## C08 — Slack-Funded Verification Escrow

- **Method**：所有 tentative M2 都进入 escrow，冻结 SFV2-O2 verifier；只依据当前 expert-ready queue、实测 service surface 和 hold deadline，把 verification job 放入自然 non-critical slack；超时直接 M1 ABORT；完整计入 shadow、producer interference、queue、hold、repair。
- **Hypothesis**：verification 若能落入真实非关键 slack，可降低 post-action safety 的 critical-path cost，而不恢复失败的 pre-action selector。
- **Minimum experiment**：先在四类 frozen realistic-tail cells 测 verifier cost，再在 fresh natural continuous-decode trace 上跑 no-user-visible-commit prototype。
- **Positive**：zero unsafe commit、满足 O2 coverage，full-request completion/SLO-goodput 净正；overlap window 本身不算 positive。
- **Negative**：只有 window 无 completion gain、producer tax/queue 抵消或多数 deadline abort；不改 priority/stream/threshold 抢救。
- **Risk / effort**：HIGH / 2–3 weeks。
- **Closest work**：JoinStream、LLM-42、SFV2-O2、generic slack scheduling。
- **dedup_key**：`slack-funded-verification-escrow`

## C09 — Stack-Versioned Conformance-Constrained Kernel Polymorphism

- **Method**：离线对 natural routing/shape cells 的 fused-MoE kernel、tactic、reduction tree、driver/library stacks 同时测 full-request latency 与 hierarchical conformance；只把 empirical-stable configs 编入 stack-versioned compatibility classes；在线按 live routing histogram 只在 qualified class 内选最快 config；unknown signatures fail closed。
- **Hypothesis**：只有当 UniEP deterministic ordering 未覆盖 cross-kernel/cross-tactic/cross-stack reduction difference 时才有 residual；该 residual 中 constrained dispatch 能保留 polymorphism gain 而不产生 unsafe state commit。
- **Minimum experiment**：两个模型、至少三种 materially different config，在 train-stack/holdout-stack natural cells 上冻结 class 后盲验；比较 latency-only dispatch、single canonical、always-verify/rollback、UniEP 与 constrained dispatch。
- **Positive**：存在可复现 safety–performance tension；holdout zero state/route/token divergence；完整计费后保留 latency-only gain 的 material majority。
- **Negative**：UniEP 已覆盖同一 cross-config/cross-stack contract、所有 configs 自然等价、class 不迁移、或 fallback/qualification 吞噬收益；停止 standalone direction。
- **Risk / effort**：HIGH / 6–10 weeks，含 multi-stack GPU qualification。
- **Closest work / strongest collision**：RaMP、DA-MoE（routing-aware adaptive dispatch）；UniEP（adaptive EP megakernel + deterministic ordering/numerical consistency，训练）；From Expert Reduction to Behavioral Divergence（hierarchical conformance）；LLM-42/MarginGate（verify/rollback）。Novelty residual 必须是 inference + routing-conditioned multi-config equivalence classes + hierarchical conformance，不能泛称 consistency-aware kernel。
- **dedup_key**：`conformance-aware-moe-kernel-polymorphism`

## C10 — Canonical Segmented Epilogues for Shape-Independent Dynamic Batching

- **Method**：保留 dynamic packed expert GEMM；给 row 标记 request/token/expert/canonical contribution position；以 segmented epilogue 重建每个 request 的 canonical accumulation order，使结果独立于 companions 与 packing shape；supported kernels 上 unconditional，不做安全 row selector。
- **Hypothesis**：若 divergence 主要发生在 accumulation boundary，可不做 universal C8 padding 就消除 companion-dependent state，并保留 dynamic batching efficiency。
- **Minimum experiment**：两个模型的 fresh natural requests，在多个 companion sets/packing shapes 下比较 native dynamic、fixed-shape、canonical serial、UniEP、segmented epilogue；要求 post-combine、route/state/token conformance。
- **Positive**：消除 companion-induced divergence，且完整计费后显著优于 serial canonical 与 universal fixed-shape。
- **Negative**：divergence 源于 GEMM 内部、segmentation 成本等同 fixed padding、或 UniEP 已实现同一 shape-independent canonical accumulation。
- **Risk / effort**：HIGH / 5–8 weeks，CUDA prototype + full-request replay。
- **Closest work**：From Expert Reduction to Behavioral Divergence、UniEP、StableBatch、RaMP。
- **dedup_key**：`segmented-canonical-epilogue-without-fixed-shape-padding`

## C11 — Request-Scoped Numerical Epochs for Elastic Expert Handoffs

- **Method**：每个 decoding request 绑定 qualified replica/kernel/reduction/software-stack numerical epoch；scale-out/migration 时，新 replica 只有证明属于该 epoch 才能接管 in-flight request，否则保留/排空旧 replica 至 token boundary；request state 记录 epoch transition。
- **Hypothesis**：相同权重的 elastic expert handoff 也可能改变 numerical trajectory；request-scoped epoch 能保 continuity 并保留大部分 elasticity benefit。
- **Minimum experiment**：两个模型 natural continuous decoding 中注入 scale-out、replica replacement、software-stack change；比较 unconstrained、drain-only、epoch-constrained handoff 的 route/state/token divergence 与 TPOT/P99/goodput。
- **Positive**：unconstrained handoff 有可复现 divergence；epoch contract 清零 unsafe transition，并优于 drain-only 的 goodput/tail。
- **Negative**：realistic handoff 本来 conformance、无 consequential transition、epoch 退化为 drain-only、或 transition delay 吞噬 elasticity gain。
- **Risk / effort**：MEDIUM / 6–9 weeks，serving-level reconfiguration harness。
- **Closest work**：MoEless、Mixture-of-Experts Serving、Gimbal、UniEP、numerical-state divergence。
- **dedup_key**：`request-numerical-epoch-elastic-replica-handoff`

## C12 — Schedule-Invariant Identity-Ordered EP Reduction

- **Method**：distributed contribution 携带 request/token/layer/expert/canonical-position identity；允许 fine-grained compute/communication 任意到达，但用 deterministic identity-ordered tree 或 bounded reproducible accumulator reduce；canonical set closure 前不释放 consumer。
- **Hypothesis**：可把 EP performance schedule 与 numerical reduction order 分离，在不采用 early partial consumer 的前提下保留 scheduling/overlap benefit。
- **Minimum experiment**：multi-GPU EP 或 faithful distributed replay；两个模型 natural requests 下排列 arrival orders，比较 fixed-order communication、arrival-order fast reduction、UniEP、schedule-invariant reduction。
- **Positive**：跨 arrival permutation 保持 route/state/token conformance，且相对 fixed-order 有 material TPOT/P99/goodput benefit。
- **Negative**：optimized collectives/UniEP 已覆盖同一 contract、canonical closure 重建 full barrier、或 buffering/ordering cost 主导。
- **Risk / effort**：HIGH / 7–12 weeks，需要 multi-GPU EP instrumentation。
- **Closest work**：FinDEP、Aurora、UniEP、LLM-42、numerical-state divergence。
- **dedup_key**：`identity-ordered-schedule-invariant-distributed-ep-reduction`

## 机械合并说明

- C05 与 C06 分别研究“观测/提交边界”和“row-atomic executor”；结构相邻但 `dedup_key` 不同，jury 前不合并。
- C01 与 C02 分别研究“延迟传播规律”和“trace 因果保真度”；前者是 phenomenon，后者是 measurement substrate，jury 前不合并。
- C04 与 C09 分别研究“semantic-work feedback 对 Oracle 的影响”和“把 conformance 作为 kernel dispatch constraint”；一个审计系统假设，一个提出 actuator contract，jury 前不合并。
- C09、C10、C11、C12 分别作用于 config dispatch、local epilogue、replica handoff、distributed reduction；都使用 conformance contract，但 actuator surface 不同，jury 前不合并。
- 无候选因主观质量在 jury 前删除。
