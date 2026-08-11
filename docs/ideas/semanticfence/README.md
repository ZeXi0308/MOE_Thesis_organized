# SemanticFence-MoE

> 当前状态：`GPU_PILOT_COMPLETE / COARSE_CONTRACT_WEAKENED / C09_V1_KILLED / CROSS_COMPANION_MIXED / SEMANTIC_SHADOW_POSITIVE / D10_HEADLINE_COST_NO_GO / SFV2_O1_PIVOT_TO_SHADOW_VERIFY`
> 证据截止：2026-08-11
> SF-P0 账本：[实验跟踪器](../../../refine-logs/EXPERIMENT_TRACKER_20260810_162110.md)
> SFV2-O1 账本：[Online-observability Gate tracker](../../../refine-logs/EXPERIMENT_TRACKER_SFV2_O1_20260811_000054.md)

SemanticFence-MoE 的完整方向保持为：先密封 route ledger，再只在版本绑定、可观测且经过 calibration 认证的安全 operating region 内跨请求合并 expert rows；未知状态 fail closed 到 `M=1`，目标是在保持选定语义边界时恢复 batching efficiency。

## 第一轮正式 Pilot 裁决

唯一权威运行是 [`semanticfence_pilot_20260810_run03`](experiments/outputs/semanticfence_pilot_20260810_run03/)。它在单张 RTX 5090、OLMoE、BF16、decode-style expert-stage 条件下完整结束并写出 [`COMPLETE.json`](experiments/outputs/semanticfence_pilot_20260810_run03/COMPLETE.json)。机器裁决为：

- `decision=WEAKEN`
- `paper_result=false`
- A reference 全部稳定；B 在 64/64 victims 上出现稳定 raw-BF16 mismatch
- calibration contract 共 4,237 entries，0 allowed
- D mismatch=0，但 coverage=0、自然 `M>1`=0，D/A latency reduction=-0.2364%
- raw、trace、signature 与 worker/parent recompute closure 全部通过

因此，被削弱的是 coarse `(layer, expert, M, kernel signature, stack)` calibration allowlist，而不是整个 batch-invariance / safe rebatching 问题。D 的 exactness 来自全量回退 `M=1`，不能作为机制成功。

## 当前机制更新

对封存 artifact 的 row-shape census 显示：32,234 个 `M=2` rows 中 2,768 safe（8.5872%）。自然 arrival order 中 121/16,117 packs（0.75076%）两端都是 exact row，与独立估计的 118.85 packs 接近（observed/expected=1.0181）；这表明 121 不是两端误差碰巧抵消，也没显示明显 pair affinity。

跨 companion 证据将机制定位为 `MIXED`：512 个 focal x 4 个替代 companion 共 2,048 次 M2 calls，focal label flip 为 0；slot permutation 也为 0 flip；补充 metric replay 的 64 个 focal 均为 4/4 一致，balanced design 下 within-focal variance=0、between-focal variance=0.25。因此当前最窄分类是 `fixed-M2/small-M ABI 内 row-intrinsic + 跨 M 的 shape/cardinality-conditioned`；H2 pairwise 主因不受支持，H4 accidental artifact 被削弱。该结论仍是 reused-calibration，不是 fresh generalization。

在同 `(layer, expert)` 的 complete exact-M2 safe graph 中，maximum matching 得到 1,132 pairs，覆盖 2,264/32,234 rows（7.0236%），相对 all-M1 减少 1,132 calls（3.5118%）；同表 microcost 的 additive expert-stage projection 为 3.4034% saving。此前 4.7621% 是允许 M2/M4/M8/M16 的 variable-M perfect-Oracle call-count 上界，两者不能混写。

随后冻结并运行的 C09-v1 input-only predictor 使用 8-fold document-disjoint split、train-only standardization 和 validation-only zero-error threshold。shape control 只有 1 TP / 5 FP；加入 BF16 hidden summaries、exponent bins 与固定投影后为 0 TP / 14 FP、零 admission，裁决为 `KILL_C09_V1_ZERO_ERROR_ADMISSION`。权威结果见 [`row_safety_predictability_20260810_run02`](experiments/outputs/row_safety_predictability_20260810_run02/)。

新的 [Semantic Oracle shadow replay](experiments/outputs/semantic_oracle_shadow_20260810_run01/) 对 64 条 raw-unsafe M2 edges 的两个 endpoint 分别回填 fresh-M1 与 paired-M2 contribution：41/64 pairs 两端 downstream ordered top-k 都稳定，观测图 maximum matching 为 26 pairs，覆盖 52/96 rows（54.1667%），同表 additive microcost projection 节省 26.2038%。这是 enriched calibration shadow upper bound，不是自然 workload prevalence 或 serving speedup，但足以将 safe-packing 家族的首选方向转为 SemanticFence-v2 / Semantic Stability Budget。

因此当前不再换 cheap row-local model、阈值或 seed 救活 C09。D10 fixed-C8 cost Gate 已因 expert-time 和 p99 两门失败而裁决 `NO_GO_D10_HEADLINE_COST`；exact RowFence 只保留为 baseline/fallback。

## SFV2-O1 fresh online-observability Gate

正式结果为 [`semantic_online_observability_20260810_run01`](experiments/outputs/semantic_online_observability_20260810_run01/)：12 个全新 documents 按 6/2/4 做 train/validation/test document-disjoint split，test 上 outcome-blind 冻结 128 条自然 candidate edges。Natural Semantic Oracle 有 77/128 downstream ordered-top-k-stable proxy edges，maximum matching 覆盖 154/255 rows（60.3922%），4/4 test documents 有正动作，additive expert-stage projection 节省 29.2714%；因此冻结规则中的 natural semantic action floor 成立。

冻结的 witness-v1 只执行 5 个 pairs，其中 4 个 unsafe；matching row coverage 为 10/255（3.9216%）。其 gross saving 仅 1.9007%，measured D2H + Python prototype certificate/greedy overhead 为 19.435709 ms，net projected saving 为 -183.3504%。机械裁决是 `PIVOT_TO_SHADOW_VERIFY`，不是 `GO_SEMANTIC_WITNESS_GATE`：natural action space 保留，但 pre-execution witness-v1 在 safety、coverage 和 net cost 上同时失败；不得在同一 test split 上换 feature、threshold 或 classifier 救活，也不得据此授权 online M2 admission。

当前 SemanticFence-v2 只保留为 route/top-k-stability proxy action-space / observability 问题主线；下一机制必须是一个冻结的 shadow verifier / selective-repair Gate，production path 在此前继续全量 `M=1` fail closed。此前 [`SEMANTICFENCE_COMPOSITION_VERDICT_20260810_204445.md`](../../../idea-stage/SEMANTICFENCE_COMPOSITION_VERDICT_20260810_204445.md) 是 pre-gate 历史裁决，当前结论以新的 SFV2-O1 verdict 与本页为准。

## 证据边界

SFV2-O1 是 fresh document-disjoint forward replay，但仍只覆盖 single-RTX-5090 pretrained-OLMoE BF16 的 expert-stage；此前 semantic shadow 则是重用 calibration continuation。两者都不是 serving、request-level latency、EP/NCCL、多卡、租户隔离、跨模型泛化或形式化正确性结果。SF-P0 的 `run01` 与 `run02` 因检测到 foreign GPU process 且没有 `COMPLETE.json`，只保留为 [`forensics`](experiments/outputs/semanticfence_forensics_20260810/)，不参与科学裁决。
