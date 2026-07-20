# experiments/idea_a_mac/outputs · 实验产物索引

> 数据目录保留在此，便于脚本复现。结论文档的人类可读副本在  
> `docs/08_run_conclusions/`。

## 维护约定

| 前缀 / 目录 | 含义 |
|---|---|
| `_smoke/` | 冒烟、seedcheck、CPU/GPU 对照 |
| `_legacy_profiles/` | 早期 layer/lut/delta 剖面与 figures |
| `_loose_artifacts/` | 顶层散落的 csv/json/log |
| `paper_validation/` | 2026-07 中期 paper validation 大批次 |
| `thesis_evidence/` | 给导师看的早期 Idea A 证据包 |
| `<topic>_*_YYYY-MM-DD/` | 正式 dated run（优先看这里） |

顶层若再出现散落 `.md`，请移到 `docs/08_run_conclusions/` 并留 stub。

## 按路线索引（正式 run）

### Receiver-aware

| 目录 | 内容 |
|---|---|
| `receiver_isolation_rerun_2026-07-19/` | v1 confound 隔离重跑 |
| `receiver_aware_v2_2026-07-20/` | 陈旧度三级分层 |
| `receiver_aware_v3_2026-07-20/` | adaptive（含因果错误版） |
| `receiver_aware_v3_causal_{olmoe,llmjp}_2026-07-20/` | 因果修正审计 |
| `receiver_progressive_quality_{olmoe,llmjp}_2026-07-20/` | progressive 质量 gate |
| `receiver_codec_break_even_2026-07-20/` | Triton codec break-even |

### Quality Isolation

| 目录 | 内容 |
|---|---|
| `per_request_quality_isolation_p0_2026-07-20/` | 旧 P0（有 leakage） |
| `quality_isolation_proxy_strict_{olmoe,llmjp}_2026-07-20/` | 严格 split |
| `quality_isolation_proxy_frozen_replication_llmjp_2026-07-20/` | 冻结复现 |
| `decode_fragility_strict_llmjp_2026-07-20/` | prefill→decode |
| `quality_routing_synergy_test45_2026-07-20/` | 与路由协同检查 |

### Expert Prefetch / 路由可预测性

| 目录 | 内容 |
|---|---|
| `routing_predictability_p0{,b,c}_2026-07-20/` | 离线 P0 系列 |
| `expert_prefetch_prototype_2026-07-20/` | 系统原型 v1 |
| `expert_prefetch_v2_fused_capped_2026-07-20/` | fused + capped |
| `expert_prefetch_validity_{olmoe,llmjp}_2026-07-20/` | 有效性审计 |

### TokenRace / CreditReduce / MassCover / RouteFidelity

| 目录 | 内容 |
|---|---|
| `tokenrace_ep_p0*_2026-07-19*/` | Mac 仿真 P0 |
| `tokenrace_gpu_p{0,1}_2026-07-19/` | GPU 开销证伪 |
| `tokenrace_adaptive_p2_2026-07-20/` | 自适应触发复查 |
| `creditreduce_p0_2026-07-17/` | CreditReduce sealed |
| `masscover_ep_p0_2026-07-19/` | MassCover CVaR gate |
| `route_fidelity_p0_2026-07-18/` | RouteFidelity |

### PLTB / additive-KL / Shadow-Residual / Energy

| 目录 | 内容 |
|---|---|
| `llmjp_layer_budget_{mxfp4,nvfp4}_cal16_n32_2026-07-20/` | PLTB LLM-jp 补齐 |
| `additive_kl_audit_2026-07-20/` | 可加性审计 |
| `tf_residual_svd_2026-07-20/` | Shadow/Residual SVD-P0 |
| `temporal_residual_ep_p0_2026-07-18/` | 时序 residual 探索 |
| `energy_slo_p0_2026-07-20_olmoe/` | Energy-SLO 探针 |

### 早期 Idea A / paper_validation

| 目录 | 内容 |
|---|---|
| `main_experiments/` | 早期主实验 |
| `paper_validation/` | Graceful/QTree/QuotaEP/R-layout/信号比较等 |
| `thesis_evidence/` | 导师汇总包 |

## 人类可读结论副本

见仓库根下：

`docs/08_run_conclusions/<receiver|quality_isolation|prefetch|tokenrace|...>/`
