# RouteFidelity-EP 两周生死实验计划

> **执行结果（2026-07-18）**：P0-B 已完成并触发 `KILL_CCFC_MAINLINE`。两个 primary cells 均为 0/20 seeds regret>=5%，因此按计划不进入 GPU P1。完整 sealed 证据见 `RouteFidelity_EP_严谨实验路径与P0B验证_2026-07-18.md` 与 `experiments/idea_a_mac/outputs/route_fidelity_p0_2026-07-18/p0b_sealed_eval_v1/`。

> 本计划只完成 CCF-C 候选的 problem/method gate。当前 `dev_v3` 为探索性数据，不得复用于 confirmatory claim。

## 0. 预注册假设

### Primary problem hypothesis

> 在至少两个 `model × backend-reference-contract` 中，exact-degree abstraction 相对 full ordered route 的配置排序 `Kendall tau <= 0.8` 或 selected-config regret `>=5%`。

### Primary method hypothesis

> windowed-hyperedge representation 在同一 sealed holdout 上达到 `tau >= 0.95`、regret `<=2%`，且 artifact size 或 capture/synthesis 成本低于 full ordered trace 至少 30%。

### Hard verdict

- problem 和 method 同时通过：`PROMOTE_TO_GPU_P1`；
- problem 通过、method 失败：`EXACT_REPLAY_ONLY / KILL_NOVELTY`；
- problem 失败：`KILL_CCFC_MAINLINE`；
- 只有 architecture-only uniform 失败：`BASELINE_TOO_WEAK`，不算 problem gate。

## 1. 数据治理

- [ ] 在任何新试验前冻结 `protocol.yaml`：model、dataset/document ids、seed、primary cells、configuration pool、metrics、thresholds。
- [ ] 重新采集未在 `dev_v1/v2/v3` 中出现的 request/article；按 request 划分 calibration/test，不按 token 随机拆分。
- [ ] 对 OLMoE top-8 与 LLM-jp top-16 分别固定至少 64 个 request；如资源允许，加一个 top-2/top-4 结构对照。
- [ ] 记录 model revision、tokenizer revision、dataset hash、route hash、OS/Python/dependency manifest。
- [ ] 所有主结论使用 request/article-level cluster bootstrap；不在数百 cells 上取 maximum 作通过依据。

## 2. Reference contracts

- [ ] `C0 expanded expert-major`：每个 routed pair 一条 record，用作不受 co-activation 影响的 negative control。
- [ ] `C1 rank-major unique-owner`：同 token 命中同 owner 的多 expert 只产生一条 owner record。
- [ ] `C2 hierarchical domain-partial`：显式定义 source rank、NVLink domain、remote domain 与 local aggregation boundary。
- [ ] 为每个 contract 写手工 toy cases，验证 record count、receiver occupancy、cross-domain count 与 placement dependency。
- [ ] 去掉当前 `token_position // B` 的 serving 含义；它只保留为 prefill-like logical-window control。

## 3. Route representation ladder

- [ ] `S0` architecture-only uniform unique route。
- [x] `S1` exact-degree double-edge rewiring；已实现并通过基本单测。
- [ ] `S1+` 报告 per-layer degree TV=0 的 machine-checkable certificate。
- [ ] `S2a` exact pair/coactivation histogram 合成；保持 top expert pairs 与 owner-set distribution。
- [ ] `S2b` compact count-min/pair sketch，测量容量–决策保真 Pareto。
- [ ] `S3` windowed hyperedge prototypes，保持 phase/request 级 burst，不保存 token text/logits。
- [ ] `S4` full ordered route oracle。
- [ ] 每个 stochastic representation 至少 20 seeds。

## 4. Configuration pool

- [ ] contiguous、round-robin、frequency-balanced 和 coactivation-aware 四类可解释 placement。
- [ ] 预先生成至少 128 个 balanced random placement，冻结 hash 后再跑 test。
- [ ] 增加 hierarchical placements：保持每 domain 的 expert 数一致，改变跨域 co-location。
- [ ] buffer capacity 不再等于 synthetic P99 的零余量阈值；仅做 `+1 record`、`1/2/5/10% headroom` sensitivity，且名称为 hypothetical logical-capacity exceedance。
- [ ] CPU 主结论只使用 ranking/regret，不使用“overflow”作通过 gate。

## 5. 统计输出

- [ ] 每个 `model × contract × representation` 输出 Kendall tau、Spearman rho、top-1/top-5 agreement、regret 和 95% CI。
- [ ] 将 request/article 作为统计单位，bootstrap `>=10,000` 次；不将 token 当独立样本。
- [ ] 预注册两个 primary model-contract cells，其余标为 secondary/exploratory。
- [ ] 对多个 confirmatory comparisons 使用 Holm correction。
- [ ] 输出 representation size、synthesis time、lowering time 和峰值内存。

## 6. GPU P1：仅在 CPU gate 通过后执行

- [ ] 选择一个可获得 backend（优先 DeepEP；若只有 NVLink 单机，可选 TensorRT one-sided）。
- [ ] adapter 使 reference record/frame count 与native application bytes 误差 `<=1%`。
- [ ] 追踪开销 median `<=2%`、P99 `<=5%`。
- [ ] 使用真实 scheduler 生成 prefill/decode batches，固定 clocks、warm-up、streams、CUDA graph 与 backend revision。
- [ ] 主结果必须是 actual operator latency/config ranking；logical records 仅作机制解释。
- [ ] 第一 backend 未出现 `>=5%` latency regret 时，暂停第二 backend，先重审 problem value。

## 7. 两周时间表

| 时间 | 任务 | 产物 / 硬门 |
|---|---|---|
| Day 1–2 | 冻结 protocol、fresh split、primary cells、placement hashes | `protocol.yaml`、manifest；未冻结不开跑 |
| Day 3–4 | 完成 C0/C1/C2 reference lowerer 与toy tests | 手算结果完全一致 |
| Day 5–7 | 完成 S2/S3 与 20-seed synthesis | invariants certificate、size/cost |
| Day 8–10 | calibration，冻结 representation hyperparameters | 不读 sealed test 结果 |
| Day 11–12 | sealed holdout + cluster bootstrap | H1/H2 PASS/FAIL 一次性报告 |
| Day 13 | 独立红队：检查泄漏、max mining、discrete capacity 伪阳性 | audit log |
| Day 14 | 做唯一判决 | promote GPU / kill CCF-C / benchmark-only |

## 8. 当前已完成

- [x] 完成 MLSynth、Chakra、AICB、Megatron router trace、activation-aware placement、DeepEP/NCCL EP/TensorRT 的文献碰撞审计。
- [x] 完成 architecture-only、exact-degree、hyperedge-order 与 exact route 的第一版 reference comparison。
- [x] 修复旧 `marginal_unique` 不精确保留 marginal 的方法缺陷，改为 degree-preserving double-edge swaps。
- [x] 将过度的 `PASS_P0_A` 降级为 `EXPLORATORY_SIGNAL`，并明确 hypothetical capacity 不是 backend overflow。
- [x] 完成 two reference tests，验证 degree preservation 和 rank-deduplicated lowering。

## 9. 本计划不做的事

- 不在 P0-B 前重新包装 receiver-aware、QuotaEP-H、CreditReduce 或 temporal DPCM；
- 不用 MILP 替代系统 observation；
- 不用 bytes/BW 宣称 TTFT、TPOT 或 P99；
- 不把 Megatron exact route capture/replay 当创新；
- 不为了“保住题目”而改动已预注册 gate。
