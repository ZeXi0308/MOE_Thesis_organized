# TriageAudit-MoE Phase 2 冻结实验协议

状态：**PHASE2_FROZEN_FOR_IMPLEMENTATION / NO_SCIENTIFIC_RESULT / GPU_NOT_APPROVED**  
冻结日期：2026-07-23

## 1. Scientific question 与 claim 边界

已有证据不支持用 prefill 特征直接预测 future-decode KL 或直接选择低精度。
本实验只检验一个更弱、可证伪的问题：

> 模型内、action-specific 的 post-prefill 风险排序，能否只作为 audit priority，
> 在安全性由 same-state shadow verification、强制最长未审计间隔和 BF16 lockout
> 提供的前提下，相对与风险无关的同预算验证策略减少验证成本？

本轮最高 claim 为：`SINGLE_GPU_TEACHER_FORCED_MECHANISM_PROBE`。
INT4 路径是 per-output-channel RTN quantize-dequantize W4A16 quality proxy，
不是 native INT4 kernel。不得从本轮报告真实 INT4 speedup、J/token、TPOT/P99、
continuous serving、EP/RDMA 或生产收益。

只有机制 Gate M 通过后，才允许设计 native low-precision backend 与系统 Gate S；
只有 Gate S 再通过，才允许探索网络拓扑、receiver 或调度扩展。

## 2. 与旧 H2 的不可继承边界

旧 `run_expert_precision_persistence_shadow_verify_p0.py` 的 all-BF16/all-INT4
轨迹和 `simulate_policies()` 只能作为错误实现对照，不得进入主结果：

1. 两条 KV 独立演化，不是同一 canonical KV 的分叉；
2. post-hoc mask 没有执行 mixed action history；
3. verify step 没有支付 low+high 双执行、KV clone 与废弃分支成本；
4. escalation 后仍读取 all-INT4 的未来轨迹。

新实现必须逐 policy 独立执行，每个 policy 只保留一个 served canonical KV。

## 3. 模型、action 与数据

模型分别校准，不做跨模型 predictor transfer：

- `allenai/OLMoE-1B-7B-0924`；
- `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`。

固定 action：decode expert FFN 全部线性层做 INT4 weight-only RTN
quantize-dequantize；attention、router、LM head 与 prefill 保持 BF16。
该 action 必须命中 OLMoE `16*64*3=3072`、LLM-jp `16*32*3=1536`
个 expert linear；数量不符 hard-fail。

数据为 `wikitext-103-raw-v1:train` 的 article unit：

- prompt 64 tokens；teacher-forced decode 32 steps；
- calibration 32 documents；sealed 64 documents；
- canonical text 只统一 CRLF/CR 为 LF，不 trim、不 Unicode normalize；
- 选择键为 `sha256(seed || canonical_text_sha256)`；
- seed `2026072302`；
- calibration、sealed 与仓库可发现的历史 text hash 必须零交集；
- 两个 tokenizer 均至少 97 tokens；两模型必须消费同一 document manifest；
- sealed manifest 在 Phase 4 `GPU Run Approved` 且 calibration lock 写入前不得打开。

数据不足、cache 缺失或历史排除无法闭合均为 `BLOCKED_DATA`，不是 No-Go。

## 4. 唯一 canonical KV 与 same-state audit

每个 policy、每篇文档：

1. fresh BF16 prefill；
2. reference policy 独立维护 always-BF16 KV，仅用于离线指标；
3. candidate policy 独立维护唯一 canonical served KV；
4. 非 audit step 只执行冻结状态机选择的一个 action；
5. audit step 从 audit 前同一 canonical KV clone 出 `K_high`、`K_low`；
6. 相同 input token 分别执行 BF16 与 INT4；
7. 计算 `KL(high_logits || low_logits)`；
8. 按冻结阈值选择唯一 served logits/KV，另一分支立即废弃；
9. 后续 step 只能从被选择的 KV 继续。

必须证明 audit 前 cache tensor 零存储别名、两分支输入 identity 相同、
branch cache 长度均增加1、废弃分支没有进入下一步。

## 5. Predictor、strata 与控制器

在线 predictor 只在 prefill 后运行一次。固定特征：

- `full_route_top1_weight_mean/std`；
- `full_route_top1_top2_margin_mean`；
- `full_route_tail_mass_mean`；
- `full_route_routing_entropy_mean`；
- `full_route_rank1_hhi_mean`；
- `full_route_active_expert_fraction_mean`；
- `full_route_same_id_adjacent_layer_rate`；
- `full_mean_nll`。

Calibration label 为 always-low canonical trajectory 上每一步 same-state
low/high discrepancy 的 document CVaR90。固定 ridge alpha=1.0，目标为
`log10(CVaR90+1e-12)`；不搜索 feature group、alpha 或模型结构。

按 calibration predicted score 的 1/3、2/3 quantile 冻结三档：

- high-risk：period=2；
- medium-risk：period=4；
- low-risk：period=8。

首次 audit phase 由 `sha256(policy || document_hash)` 决定；最长连续未 audit
不得超过8步。audit discrepancy 超过 calibration pooled step KL 的 P90
阈值时，本步服务 BF16，并令随后3步 BF16 lockout；否则服务 INT4。

Predictor 不得直接选择 BF16/INT4，不得读取当前 decode step route、未来 token、
reference logits 或真实 KL。

## 6. Arms 与公平基线

每个 arm 使用相同文档、token、模型、action、threshold、lockout 与独立 fresh KV：

- `always_bf16`；
- `always_low`；
- `triage_2_4_8`；
- `hash_control_2_4_8`：同样的 period palette，但 stratum 仅由文档 hash 决定；
- `fixed_2`、`fixed_4`、`fixed_8`；
- `full_shadow`：每个非 lockout step audit，只作昂贵上界。

主比较不是挑一个 fixed period，而是比较 `triage_2_4_8` 是否被
`hash_control_2_4_8` 或 fixed-period quality-cost Pareto envelope 覆盖。

## 7. 指标与统计

独立样本为 document。所有 step 先在 document 内聚合，再做 paired
document bootstrap；不得把 step 或重复 forward 当独立样本。

主质量指标：

- document mean token KL(reference || served)；
- document CVaR90 token KL；
- document P95 token KL；
- threshold violation fraction。

主成本指标（本轮仅计算调用/状态机成本，不作硬件收益 claim）：

- `high_forward_calls`、`low_forward_calls`；
- `audit_events`、`dual_branch_calls=2*audit_events`；
- `total_candidate_forward_calls`；
- `cache_clone_events`；
- served high/low steps、lockout steps；
- wall-clock 仅作诊断，明确 fake-INT4 proxy。

主统计均以相同文档 paired bootstrap 5000次，seed `2026072302`。
多个主比较使用 Holm correction。

## 8. Gate M：机制 Go / No-Go

两模型分别判决，不要求同一个 predictor 跨模型迁移。

模型内 GO 必须同时满足：

1. 相对 `hash_control_2_4_8`，document-CVaR90 KL 不劣：paired ratio
   95% CI 上界 `<=1.05`；
2. 相对 `hash_control_2_4_8`，audit events 降低的 95% CI 下界 `>=20%`；
3. 相对 fixed-period Pareto envelope，在相同或更低 document-CVaR90 下，
   total candidate forward calls 降低的95% CI下界 `>=10%`；
4. dangerous-step recall（以 calibration P90 阈值定义）不得低于
   `hash_control_2_4_8 - 5 percentage points`；
5. 无 cache identity、counter、hook 或数据隔离失败。

主论文继续条件：LLM-jp GO，且 OLMoE 至少安全退化——质量不劣门通过，
但允许成本增益为0。若 OLMoE 出现质量劣化，整体 No-Go。

若 predictor 与 hash control 等价、fixed-period 覆盖、或收益只来自漏审危险 step，
立即停止；不得换神经网络、改split、降低门槛或在sealed上重选特征。

## 9. Gate S 与后续系统扩展

Gate M GO 后才能另立 Phase 2：接入真实 native low-precision kernel，完整计入
kernel、cache fork/switch、废弃分支、HBM、GPU board energy 与真实 decode latency。
建议 Gate S 为相同 tail-risk 下 J/token 或 TPOT 改善LCB>=8%，P99 ratio UCB<=1.03。

只有 Gate S 通过，才探索：

- receiver/link-aware audit placement；
- 将 shadow work 放入网络/compute slack；
- batch-level audit budget 与 VTC/debt 调度；
- topology-aware verification placement。

在单卡 Gate M/S 之前，不得声称网络或多GPU收益。

## 10. Phase 4 Code Review 必查项

1. 唯一 canonical KV、same-state branch 与无别名证明；
2. 每个 arm fresh prefill/独立 KV，禁止 route/reference replay；
3. predictor 只读 prefill，sealed 前完全冻结；
4. INT4 hook 命中数量、实际调用 counter、BF16 arm零调用；
5. audit step 确实执行两路并支付两路成本；
6. cache length、input token、served branch identity闭合；
7. calibration/sealed/historical hash 零交集；
8. document bootstrap、paired比较和Holm correction正确；
9. no-overwrite output、完整 source/config/environment manifest；
10. CPU fake-backend negative tests 与 tiny-model CUDA smoke均通过。

只有 Code Review 明确写出 `GPU Run Approved: MECHANISM PROBE ONLY`，
才允许运行 calibration GPU；系统收益仍未获批。
