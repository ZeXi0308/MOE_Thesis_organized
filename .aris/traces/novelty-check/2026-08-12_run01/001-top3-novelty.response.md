# 查新裁决

本裁决为 same-family / provisional；基于指定输入中的一手文献与本地证据，未在 reviewer turn 独立联网复核。

| 候选 | Novelty 风险 | 裁决 | 核心判断 |
|---|---|---|---|
| C10 Canonical Segmented Epilogues | HIGH | STOP standalone | UniEP 已覆盖 deterministic identity mapping、完整 contribution-set wait、baseline-order reduction 与 numerical consistency；本地 physical-M divergence 已在 raw expert output 出现，位于 epilogue 上游 |
| C07 Causal Repair Cones | MEDIUM-HIGH | CAUTION | residual 仅为 MoE identity 精确 dependency cone、invalidation certificate 与 closure proof；很可能因 attention/residual/KV 扩成完整 suffix |
| C02 Intervention-Calibrated Request-DAG Trace | MEDIUM-HIGH | PROCEED_TO_FROZEN_GATE / infrastructure only | residual 仅为 MoE-specific identity-complete DAG 的 held-out real-intervention blind validation、missing-edge certificate 与 fail-closed Oracle qualification |

C10 不应投入 standalone 实现。只有新 natural inference evidence 证明主要 first divergence 位于 epilogue、UniEP contract 未覆盖且 charged full-request cost 有 headroom，才可重开为条件 actuator。

C07 必须等 qualified C02 trace；其 Gate 同时要求 zero closure mismatch、cone 严格小于 per-request suffix 和 full-request net positive。C02 值得先做但不是 headline method；只有证明 generic trace 在 dynamic MoE batching 下系统性预测错，而 qualified identity DAG 能纠正，才可能形成 measurement residual。

建议把主线调整为 Inference-Specific MoE Execution Conformance Pipeline：source localization → stack-versioned contract → conditional actuator → full-request Pareto。它只是 research program / PROCEED_TO_FROZEN_GATE，当前 method/system GO 仍为 0。
