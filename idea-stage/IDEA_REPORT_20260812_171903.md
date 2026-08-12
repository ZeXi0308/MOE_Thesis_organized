# MoE 下一轮系统化 Idea 报告

**方向**：从多轮 MoE scheduling、execution-shape、verification 与 full-request Oracle 失败中提炼可证伪的系统主线
**生成时间**：2026-08-12 17:19:03 +08:00
**候选流程**：12 个机械去重候选 → fresh jury Top 3 → 近期文献查新 → devil's-advocate 重排
**总裁决**：`NO_QUALIFIED_METHOD_CANDIDATE / FORMAL_METHOD_SYSTEM_GO=0`

## 直接结论

对“当前已有机制能不能直接做成论文系统”，判断偏悲观；对“这些负结果能不能收束成一个更好的研究问题”，判断谨慎乐观。

| 层次 | 判断 | 原因 |
|---|---|---|
| 现有方法组合 | 约 3/10，偏悲观 | formal GO 为 0；selector、critical-path conversion、fixed overhead 和 protocol closure 反复失败 |
| 新研究问题 | 约 6/10，谨慎乐观 | 多条负结果共同指向 execution conformance、source localization 与 full-request causality，而非另一个 heuristic scheduler |
| 当前论文 claim | 未验证 | 近期工作拥挤，UniEP、batch invariance、verify/rollback 与 causal tracing 已占据大块空间 |

现行权威不变：RCBA 仍是 `docs/current/README.md` 中登记的 unvalidated Primary，但 preparation 被 protocol ambiguity 阻塞；本报告只调整下一轮探索优先级，不把任何候选写成正式 GO。

## 多轮失败真正给出的系统规律

| 反复出现的现象 | 已有证据 | 对下一方案的约束 |
|---|---|---|
| local opportunity 不等于 request benefit | JoinStream 有自然 window 但 safe benefit `0/4`；RCBA 缺 full-DAG closure | 第一指标必须是 charged full-request，不再以局部窗口/投影作 headline |
| action space 有值不等于 outcome 前可选择 | StableBatch hindsight Oracle 正，但 static/online selector 均为 `-7`；SemanticFence witness 4/5 unsafe | 优先 by-construction 或 post-action fail-closed 机制，不再发明第三个 pre-action predictor |
| correctness primitive 不等于可用 actuator | fixed-C8 能稳定输出，但 cost Gate 为 `NO_GO_D10_HEADLINE_COST` | 正确性、成本和 serving closure 必须在同一 Gate 中闭合 |
| 系统因果底座缺失会制造假阳性 | 多个方案缺 identity-complete DAG、service surface、matched denominator | 先资格化 trace/evaluator，再接受 Oracle 或 repair claim |
| 简单基线已覆盖大量机制空间 | UniEP、vLLM batch invariance、LLM-42/MarginGate、RaMP/DA-MoE | novelty 必须落在精确 residual，不以“组合已有部件”作贡献 |

一句话模型：

> **不要先问“哪种 row 要特殊处理”，先问自然 inference 中差异从哪里产生、会不会进入 full request、现有 deterministic contract 是否已经消掉；只有 residual 存在，才选择执行器。**

## 推荐 Idea 1：Inference-Specific MoE Execution Conformance Pipeline

**角色**：Primary research program
**状态**：`UNVALIDATED / PROCEED_TO_FROZEN_GATE / NOT_METHOD_GO`

- **Method（实际做什么）**：
  1. 在两个固定模型的 natural continuous decoding 中，为 request、step、layer、route、expert row、kernel/config 和资源顺序建立 identity-complete trace；
  2. 用 native、canonical 与 UniEP-equivalent controls，并交换冻结 raw/epilogue buffer，定位 first divergence 和 full-request propagation；
  3. 将通过盲验的配置写入 stack-versioned conformance contract，未知 signature fail closed；
  4. 只按已证明的 source 选择 actuator，并与 batch-invariant、single canonical、UniEP-like ordering 和 verify/rollback 比 charged full-request Pareto。
- **Hypothesis**：natural MoE inference 中仍存在 UniEP-style ordering contract 未覆盖的 consequential execution residual，并且至少一个安全 non-canonical execution path 能保留 full-request 性能优势。
- **Minimum experiment**：唯一先做 `IECP-G0 — UNIEP_RESIDUAL_SOURCE_LOCALIZATION`，详见后文。
- **Expected outcome**：正结果必须同时证明双模型 incidence、稳定 source、UniEP residual、安全配置多样性和 full-request headroom；负结果会一次性终止整类 method search，但仍留下可信 evaluator。
- **Novelty**：约 5/10；最大风险是 pipeline 只是 causal profiling + UniEP contract + existing actuator + standard evaluation 的拼接。
- **Feasibility**：1×RTX 5090 可做 source-localization 与单机 continuous-decode Gate；先用 1–2 周 source/collision slice 快速 kill，完整 Gate 约 4–6 周。
- **Risk**：HIGH。
- **Contribution type**：先是 systems measurement/qualification；只有 residual 和 conditional actuator 同时成立才升级为 method。
- **Pilot result**：`SKIPPED / NOT_RUN`；本轮没有新实验。
- **Reviewer's likely objection**：每个部件都已有近邻，框架本身不是新颖性；若 UniEP-equivalent control 清空差异，论文主线只剩基础设施。
- **Why do it**：它能用一个 Gate 解释并终止多条失败机制，而不是再加一个无法闭合的 heuristic；无论正负都能显著减少后续试错空间。

## 推荐 Idea 2：Causal Repair Cones for Sparse MoE Divergence

**角色**：Conditional backup method
**状态**：`CAUTION / BLOCKED_BY_QUALIFIED_DAG`

- **Method（实际做什么）**：
  1. verifier 发现 mismatch 后，从合格 request DAG 标记该 row 的全部 downstream dependents；
  2. 生成 invalidation certificate，只撤销尚未 commit 的精确 cone；
  3. 用 canonical path 重算并沿 cone replay，到冻结 boundary 与 all-canonical reference 重合；
  4. closure 无法证明时无条件 full rollback。
- **Hypothesis**：MoE sparsity 和 row/request identity 可让多数 per-request repair scope 严格小于完整 token/request suffix。
- **Minimum experiment**：`C07-G0 CONE_TIGHTNESS_AND_CLOSURE`；对全部 observed mismatch 构造 cone，逐边界对照 all-canonical run，完整计入 verifier、ledger、replay、stall。
- **Expected outcome**：zero closure mismatch，同时预注册比例的 cone work 显著小于 per-request suffix 且 full-request headroom 为正。
- **Novelty**：约 5/10；residual 是 exact MoE dependency cone + certificate + closure proof，不是 selective repair。
- **Feasibility**：C02 通过后约 2–4 周，CPU verifier + 1×RTX 5090 replay。
- **Risk**：HIGH；attention/residual/KV 很可能把 cone 扩成完整 suffix。
- **Contribution type**：conditional method + causal systems diagnostic。
- **Pilot result**：`SKIPPED / BLOCKED`。
- **Reviewer's likely objection**：只是 LLM-42/MarginGate/partial repair 的 full-DAG implementation refinement。
- **Why do it**：它避开已失败的 pre-action selector；但只应在 by-construction path 太贵、且 dependency cone 已被证明足够窄时启动。

## 推荐 Idea 3：Intervention-Calibrated Request-DAG Trace

**角色**：Foundation / infrastructure only
**状态**：`PROCEED_TO_FROZEN_GATE / NOT_HEADLINE_METHOD`

- **Method（实际做什么）**：
  1. 记录 request/step/layer/route/expert/resource-order/service identity；
  2. 在 calibration cells 注入低幅 delay、resource occupancy、combine release 与 batch-turnover probes；
  3. 用基线 DAG 盲预测 held-out intervention 的 completion delta、affected request set 和 order；
  4. 预测错时输出 missing-edge certificate 并阻止下游 Oracle/repair claim。
- **Hypothesis**：generic request trace 在 dynamic MoE batching 下会系统性漏掉少数 causal edge class；route/resource identity 与真实干预资格化可纠正。
- **Minimum experiment**：两个模型各 32 natural episodes，calibration/evaluation 各半；与 generic DAG、无 route identity 和普通 critical-path/slack baseline 比较。
- **Expected outcome**：跨模型正确预测 sign、主要 affected-set 和 order；任何 edge-class 系统性失败都 fail closed。
- **Novelty**：约 4/10；COZ/CRISP/TELLER 强碰撞，只有 blind qualification + missing-edge certificate + MoE identity residual 可保留。
- **Feasibility**：4–6 周，runtime instrumentation + CPU replay verifier + 1×RTX 5090。
- **Risk**：MEDIUM-HIGH。
- **Contribution type**：infrastructure / measurement；只有发现 generic trace 的系统性错误并修正，才可能单独成文。
- **Pilot result**：`SKIPPED / NOT_RUN`。
- **Reviewer's likely objection**：trace schema 与 causal slice 不是新贡献。
- **Why do it**：这是多个历史假阳性和 protocol block 的共同底座；即便不成 headline，也能防止下一轮再用错误 denominator 或局部 proxy 做决策。

## C10 的最终位置

`Canonical Segmented Epilogues` 不再是 standalone Primary。UniEP 已强覆盖其 deterministic identity/order/reduction contract，而本地 StableBatch 的主要差异已经在 raw expert output 出现，早于 epilogue。它只作为 IECP source Gate 后的条件 actuator：只有新证据证明 residual 首次发生于 epilogue、UniEP-equivalent control 未覆盖且 full-request cost 有优势，才以新实验卡重开。

## 唯一立即执行的 Gate

### `IECP-G0 — UNIEP_RESIDUAL_SOURCE_LOCALIZATION`

**冻结输入**：两个模型 revision；每模型两个 natural continuous-decode load cells；每 cell 至少 32 paired episodes；相同 requests、arrival trace、seed；canonical sequential、native dynamic、UniEP-equivalent control；至少三种 materially different kernel/tactic/packing configs。

**必须记录的链**：

`raw expert output → epilogue/combine → next-router → persistent state → token → request completion`

**通过条件**：双模型均有 natural performance–conformance tension；UniEP-equivalent control 后仍有 consequential divergence；source 稳定；至少一个安全 non-canonical config 优于 canonical；收益落在 charged completion/TPOT/P99/goodput。

**Kill 条件**：UniEP control 清空差异；差异仅在 raw GEMM 且无局部 actuator；source 不稳定；安全类退化为 singleton；局部差异不传播；或完整成本净非正。任一成立即停止 method program，不得用 selector、fixed-C8、epilogue 调参或扩大 repair scope 抢救。

## 未进入推荐队列的候选

| 候选 | 裁决理由 |
|---|---|
| C01 Route-Conditioned Delay Survival Map | 与 COZ/TELLER 强重叠，被 intervention-qualified C02 更严格覆盖 |
| C03 Critical-Path Motif Census | 偏描述性，route label 很可能不超过 generic topology/slack |
| C04 Semantic–Temporal Feedback Closure Audit | 不单独成方法，吸收到 IECP source-localization Gate |
| C05 Commit-Horizon Atlas | 与 SFV2-O2、LLM-42、MarginGate 重叠，缺独立 actuator |
| C06 Row-Atomic Transaction Ledger | 在 dependency closure 未成立前弱于 C07，且 partial commit/repair 拥挤 |
| C08 Slack-Funded Verification | 近似复活 JoinStream；window 不等于 completion gain |
| C09 Conformance-Constrained Kernel Polymorphism | 同时强撞 UniEP 与 RaMP/DA-MoE，并再次引入 selector |
| C10 Canonical Segmented Epilogues | `STOP_STANDALONE`；UniEP collision + raw-GEMM upstream divergence |
| C11 Request-Scoped Numerical Epochs | 需要真实 serving reconfiguration，单卡无法闭合 elasticity claim |
| C12 Schedule-Invariant EP Reduction | 依赖 multi-GPU EP，且强撞 UniEP/FinDEP |

## 建议执行顺序

1. 只冻结 `IECP-G0`；先做 1–2 周的 UniEP exact-collision 与 source-localization slice。
2. Gate 通过后再完成 C02 intervention blind-validation；在此之前不接受 full-DAG Oracle 或 repair claim。
3. 只有 source 和安全性能空间都成立，才选择 actuator：GEMM/config source 走 qualified configuration class；epilogue source 才看 C10；by-construction 过贵且 cone 足够窄才看 C07。
4. Gate 失败则停止 method search，保留 evaluator/negative result，不切换阈值、workload、denominator 或新 selector 抢救。

## Pilot Experiment Results

| Idea | GPU | Time | Key metric | Signal |
|---|---|---|---|---|
| IECP | NOT_RUN | 0 | Gate 尚未冻结 | SKIPPED |
| C07 | NOT_RUN | 0 | 被 qualified DAG 阻塞 | SKIPPED |
| C02 | NOT_RUN | 0 | 本轮只做查新与设计 | SKIPPED |

本轮没有把计划、审稿意见或既有 proxy 结果升级成新实验结果。
