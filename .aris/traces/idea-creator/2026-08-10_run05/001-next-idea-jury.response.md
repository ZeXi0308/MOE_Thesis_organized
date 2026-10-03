# JOINSTREAM_FORMAL_FREEZE_AND_NEXT_IDEA_RESET

**裁决：`PRIMARY_NEXT_CANDIDATE = Route-Conditioned Barrier Amplification Boundary`**

它是 Oracle-first 研究问题：先确认自然 MoE barrier 是否真实占据请求级关键路径；在结果为正以前，不设计 scheduler、stream、priority、polling 或通知机制。

## 1. Route-Conditioned Barrier Amplification Boundary — Primary

- **精确定义**：固定到达、工作量和自然 top-k routes，在 identity-complete 请求 DAG 上比较真实 barrier、删除 barrier 的 Oracle 和 route-decorrelated 对照，只归因未被 overlap 隐藏的请求完成/SLO 差异。
- **当前证据**：`未验证`；现有 authority 明确缺少正式 Gate-1 现象、完整 denominator 和 full-request-DAG，单卡局部 expert 数据不能证明 barrier 暴露。
- **最大 novelty 风险**：可能退化为已有 load-imbalance/barrier profiling；必须证明跨模型边界不能被 max-load、CV 或简单 rank-tail 指标解释。
- **Minimum Oracle Gate**：两个模型、自然 continuous-decode arrivals、完整 request/layer/step identity、实测 service surface；以完全相同 work 比较 real-barrier、no-barrier 和 decorrelated replay。
- **Positive 含义**：共同自然 regime 保留 `>=10%` charged full-request Oracle headroom，且 MoE route/barrier 变量在简单 max-load/CV 之外仍有稳定解释力；这只授权随后寻找最小合法动作。
- **Negative 立即冻结**：所有共同自然 cells `<5%`，或 actionable mass `<20%`，或简单 max-load/CV 在 holdout 上达到同等边界预测能力；立即冻结 standalone direction，不换 synthetic skew、模型或 denominator。
- **资源**：CPU full-DAG replay；1×RTX 5090 采集 fresh routes/service surfaces；只有正式 EP/通信结论才需要 8×A100。
- **评分**：`(8, 9, 8, 5, 7, 5, 7, 9)`；**均分 `7.25`**。

## 2. SemanticFence-v2 End-to-End Qualification

- **精确定义**：不训练 selector，先在 fresh、document-disjoint 自然请求上计算 future-known、语义保持的 M1/M2 assignment 相对 all-M1/native 的 charged full-request Oracle。
- **当前证据**：自然 exact-M2 仅有 `3.4034%` expert-stage projection；`26.2038%` 来自 enriched reused-calibration semantic shadow，不是自然 prevalence、fresh generalization 或 serving speedup；C09-v1 selector 已冻结失败，fixed-C8 headline cost 已 NO-GO。
- **最大 novelty 风险**：safe batching、numerical verification、verify/rollback 和 selective repair 均有强近邻；若不存在执行前 certificate，会退化为昂贵的 shadow/replay。
- **Minimum Oracle Gate**：fresh pre-outcome 两模型连续解码 trace，完整请求 DAG；比较 all-M1/native 与 future-known semantics-preserving M1/M2 assignment，并计入等待、packing、dispatch、fallback 和语义检查成本。
- **Positive 含义**：在 sealed route/top-k 或更强语义契约下仍有 `>=10%` full-request headroom；仅证明 opportunity，不证明 online packer。
- **Negative 立即冻结**：净 full-request headroom `<5%`、无自然 eligible rows，或收益只在 enriched/reused calibration 中出现；冻结 SemanticFence-v2 论文机制，不再换 selector、阈值或 M。
- **资源**：CPU full-DAG replay + 1×RTX 5090 fresh route/output/service qualification；多卡仅用于后续 serving validation。
- **评分**：`(5, 6, 8, 3, 8, 7, 7, 9)`；**均分 `6.63`**。

## 3. Exact Action-Surface Existence Certificate

- **精确定义**：把 assignment、bounded seal、admission、release 编译为“可见状态—可改对象—删除依赖—下游传播—完整成本”，合并等价动作后在完整请求 DAG 上证明是否存在合法净空间。
- **当前证据**：`未验证`；只有 fail-closed harness/CPU 正确性资产，BCRD formal Gate 2 仍为 `INVALID_REQUEST_DAG_REPLAY_NOT_IMPLEMENTED`，不构成 headroom 证据。
- **最大 novelty 风险**：容易与 scheduling DSL、causal simulator、action-equivalence 工具重合，并可能只形成基础设施论文；因此不能排第一。
- **Minimum Oracle Gate**：在一个 near-real identity-complete full DAG 上精确枚举合法动作，冻结相同 arrivals/routes/service surfaces，输出依赖删除证书、replay closure、非预知约束和全部成本。
- **Positive 含义**：至少一个非预知、语义保持的动作等价类保留 `>=10%` net full-DAG headroom；随后只能选择该最小类继续。
- **Negative 立即冻结**：所有动作类 `<10%`，或任何正收益依赖 future leakage、zero cost、未闭合传播或语义改变；返回 `ACTION_SPACE_ABSENT` 并停止 controller 设计。
- **资源**：CPU exact enumeration + 1×RTX 5090 service qualification；正式通信成本需 8×A100。
- **评分**：`(7, 9, 6, 4, 6, 5, 6, 10)`；**均分 `6.63`**。

## 排除说明

QuantumYield/OpenBatch 按 CriticalSplit 同构排除；JoinStream 的 gate/priority/polling/stream/notification 变体全部排除。ShapeLease、ShadowCommit 最多只能作为 StableBatch/ShapeLane 冻结结论后的 bounded qualification，不能以旧 Oracle opportunity 复活为机制。Optimized EP Return-Path 仅保留为 8×A100 qualification measurement，不是下一主候选。

## 可改变选择的未决项

- **P0**：当前没有已验证的 identity-complete full-request-DAG evaluator；若两周内不能在不引入 proxy denominator 的条件下闭合，应改判 `NO_QUALIFIED_NEXT_CANDIDATE`。
- **P1**：若 barrier Oracle 被 max-load/CV 完全解释，立即冻结 Primary，转入 Exact Action-Surface Certificate。
- **P1**：本轮按要求未联网核验最新 prior art；若 Route-Conditioned Boundary 已有同 action/counterfactual/claim 的直接碰撞，应取消 Primary。

`review_independence=same-family`  
`acceptance_status=provisional`
