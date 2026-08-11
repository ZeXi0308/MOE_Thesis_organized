## 1. Negative-result insight

旧 SemanticFence exact admission 失败的直接原因不是“所有 `M>1` 都不安全”，而是 coarse `(layer, expert, M, kernel signature, stack)` entry 内同时混有 exact 与 mismatch rows，导致 4,237 个 contract entries 在零误接收规则下全部 abstain，正式 D 路径退化为 `M=1`。这排除了当前 coarse exact allowlist，也连同 C09-v1 的 0 TP / 14 FP 排除了当前 cheap input-only linear selector；它没有排除 row-level susceptibility、版本绑定 witness 或 downstream semantic safety。

D10 的 fixed-C8 correctness 仍成立，但冻结成本门为 `NO_GO_D10_HEADLINE_COST`：fixed-C8 / serial-M1 expert GPU-time ratio 为 0.8491007（要求 <=0.8），fixed-C8 / native token-step p99 ratio 为 1.4693813（要求 <=1.05），padding fraction 为 0.7740559。因此不能把 universal fixed-C8 lane 当 headline actuator，也不能改 `C`、阈值、workload 或指标救结果。

## 2. Mechanism classification

裁决为 **`MIXED = ROW_INTRINSIC within fixed M2/small-M ABI + SHAPE_KERNEL_CONDITIONED across M`**。

- H1 `ROW_INTRINSIC`：固定 `M=2` 时，512 个 focal 各换 4 个 companion，共 2,048 calls，标签 2,048/2,048 保持；slot permutation 也为 0 flip。补充 metric replay 的 64 个 focal 均 4/4 一致。新 semantic shadow 中重复出现在两条边上的 32 个 focal 也全部保持相同 route/top-k safety 状态（27 个两次安全、5 个两次不安全）。
- H2 `PAIRWISE_COMPATIBILITY`：当前不受支持；exact companion flip 为 0/2,048，semantic focal flip 也为 0/32。故不应把 CompatibilityGraph 当主抽象。
- H3 `SHAPE_KERNEL`：M2/M4/M8/M16 使用同一 small-M kernel signature，safe sets 嵌套且 safe rows 从 2,768 降至 2,153；M32/M64 切换签名后 safe rows 均为 0。shape/cardinality 是跨 M 的必要条件变量。
- H4 `ACCIDENTAL`：被重复稳定、可分解的 row labels 明显削弱，但 calibration-only 证据不能升级为 fresh-row 泛化。

## 3. Evidence

121 个 exact `M=2` packs 的精确定义是：自然 arrival-order pair 的两个 endpoint 都分别逐 bit 等于各自 `M=1` reference；不是两端误差相互抵消。它们占 121/16,117 = 0.7507601%，与 8.5872061% row-safe rate 下的独立期望 118.85 packs 接近（observed/expected = 1.0181），因此自然计数本身没有显示 pair affinity。

Exact matrix 与 Oracle：

- row-label consistency：512/512 = 100%；call-level 2,048/2,048 = 100%。
- companion flip rate：0/2,048 calls；any-flip focal rate 0/512；slot flip 0/512。
- balanced binary cohort 的 within-row variance = 0、between-row variance = 0.25、total = 0.25；0.25 是人工平衡设计内数值，不是总体效应量。
- sampled whole-pair safe-edge density 为 512/2,048 = 25%，仅是 2x2 分层设计；自然 arrival-order density 为 0.7508%，同 `(layer, expert)` complete candidate graph 为 6,160/786,281 = 0.7834%。
- Exact `M=2` maximum matching 得到 1,132 pairs，覆盖 2,264/32,234 rows = 7.0236%，使用 2,264/2,768 = 81.7919% exact-safe rows；总 calls 从 32,234 降至 31,102，减少 3.5118%。
- 同表 microcost 给出 `c2/c1 = 1.03088`；线性 expert-stage projection 从 1,591.597 ms 降至 1,537.429 ms，节省 54.168 ms = 3.4034%。不含 packing、queue、scheduler 或 serving tax。

Semantic shadow Oracle：在 64 条刻意富集的 raw-unsafe M2 edges 上，对 96 个唯一 endpoints 的原始 16-token continuation 分别注入 fresh M1 output 与 paired M2 output；native self-replacement、hidden/expert/rank、非目标 contribution 与 2/2/10/10 repeat gates 全部闭合。128 个 endpoint observations 中 24 个发生 downstream ordered-top-k change，0 个 greedy-token change，42 个 final-logit bitwise change；41/64 pairs 两端 route/top-k 都稳定。观测图 exact maximum matching 为 26 pairs，覆盖 52/96 rows = 54.1667%；同表 microcost projection 从 4.7850 ms 降至 3.5312 ms，节省 26.2038%。这是 enriched reused-calibration graph 的 shadow upper bound，不是自然 workload Semantic Oracle、fresh generalization 或 serving speedup。

## 4. System implication

在 safe-packing 机制族内，推荐主方向改为 **SemanticFence-v2 / Semantic Stability Budget**：保护 downstream ordered top-k，而不是把 raw BF16 bitwise exactness 当唯一 admission contract。runtime 由版本绑定 profiler、semantic witness/cache、同 `(layer, expert, ABI, deadline)` safe pool、risk-aware M2 matcher、shadow verifier 和 `M=1` fail-closed lane 组成。

Exact RowFence 保留为可核对 baseline/fallback，但 7.02% coverage 与 3.40% projected expert-stage saving 太窄，且当前 online selector 已被证伪；CompatibilityGraph 因 0 companion flips 降级；fixed-C8 ShapeABI headline 已被 D10 成本门否决。shape/cardinality 仍作为 certificate 的 ABI 条件，而不是独立 universal lane。

## 5. OR formulation

对同一 `(layer, expert, ABI)` 等待窗构图 `G=(V,E)`。令 `x_ij in {0,1}` 表示 rows `i,j` 组成 M2，`f_i in {0,1}` 表示 row `i` 回退 M1；`a_ij` 是执行前 certificate，`r_ij` 是 semantic risk 上界，`Delta_ij = c1_i + c1_j - c2_ij - h_ij` 是扣除 packing/decision overhead 后的收益。

```text
maximize  sum_(i,j in E) (Delta_ij - lambda*r_ij) x_ij

subject to
  f_i + sum_(j:(i,j) in E) x_ij = 1                 for every row i
  x_ij <= a_ij                                      certified edges only
  sum_(i,j in E) r_ij x_ij <= R                     window risk budget
  x_ij = 0 unless ABI-compatible and deadline-feasible
  x_ij, f_i in {0,1}
```

未匹配或不确定 row 由 `f_i=1` 强制走 M1。H1 下 exact certificate 可分解为 `a_ij=a_i(M2)a_j(M2)`；过滤后是 maximum-weight matching，可用 rolling-horizon augmenting-path/greedy 实现，不需要大型 MILP。只有 M2 action space 经 fresh Gate 成立后才考虑 M>2 set packing。

## 6. Top 3 system directions

已筛过 RowFence、CompatibilityGraph、Robust Safe Packer、WitnessCache、SemanticFence-v2、Dual-Lane Runtime、Stability Budget、Adaptive Batch Cardinality、Route-Margin Admission、Profile-once/Serve-many 十个机制族；合并同构项后的 Top 3 为：

1. **SemanticFence-v2 / Semantic Stability Budget**：action 是 M1/M2 与 pair selection；机制是 route/top-k certificate + risk-aware matching；组件是 semantic profiler、packer、shadow verifier、M1 fallback；CCF C/B 潜力最高，因为它把 numerical compatibility 提升为 runtime contract；最大风险是自然 workload 的 Semantic Oracle 和执行前 certificate 尚未验证。
2. **Shape-conditioned RowFence / Adaptive Cardinality**：action 是按 `safe(i,M,ABI)` 选择 M1/M2，后续才考虑 M4/M8；机制是 row susceptibility + maximum-weight matching；组件是 expert dispatcher 与 shape-bound profile table；机制证据最直接；最大风险是 C09-v1 已表明当前 cheap input-only signal 为 0 TP / 14 FP，exact upside 仅 3.40%。
3. **WitnessCache / Profile-once Serve-many**：action 是只让命中过版本绑定 witness 的 rows 进入 M2；机制是不泛化的 memoized safety；组件是 witness cache、safe pool 与失配 M1 fallback；安全边界最清楚、实现成本最低；最大风险是 row/signature reuse rate、cache hit rate 和论文新颖性未验证。

## 7. Next minimal experiment

本轮只选择并已执行一个新实验：[`semantic_oracle_shadow_20260810_run01`](../docs/ideas/semanticfence/experiments/outputs/semantic_oracle_shadow_20260810_run01/)。冻结 plan 为 16 layers、32 focals、64 M2 edges、96 unique endpoints；plan SHA256 为 `990213db3e2e7e8a6cb2abd1060eebc834b575e1d959d1b7c61e9fa52f3998a3`，runner/test 为 `966839ffd2259c13df0b9d4101146b769e74a0d832a0f2bc4fec5bf48f00928c` / `87620f9b67467b7811fd4f978dd9cef67619f0c04fdcf82a494a133eafbbb33b`，本地与远端 7/7 tests PASS，6 个 COMPLETE-bound artifacts 哈希复核 PASS。

机械结果是 `41/64 semantic-safe edges -> 26-edge maximum matching -> 54.17% sampled row coverage -> 26.20% additive expert-stage projection`，因此当前 batching formulation 不停止，主候选转为 SemanticFence-v2；但该实验复用 calibration rows，下一研究门只能是 fresh、pre-outcome、document-disjoint 的 online-observability Gate。本轮未启动第二个实验，也没有重跑或改写 SF-P0/run03。
