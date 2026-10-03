# StableBatch 单贡献因果实验审计

**日期**：2026-08-10  
**审计者**：GPT-5.6-Sol ultra，fresh same-family 只读审计  
**整体结论**：`PASS`  
**完整性状态**：`pass`  
**接纳状态**：`provisional_same_family`  
**冻结实验判定复算**：`SUPPORT`  
**原因码**：`FROZEN_CAUSAL_TRACE_RECOMPUTES_SUPPORT`

## 结论

本轮证据足以暂定支持一条窄因果链：在冻结的单 RTX 5090、OLMoE、BF16、eager prompt-forward 条件下，M=1 与 repeated-row M=64 执行形状产生的单个 raw expert contribution 差异，能够穿过 gate-weight/combine 边界，并在部分冻结、经过 local-delta/next-layer-margin 富集的目标上稳定改变后续 victim top-k membership。

这不是总体发生率估计，也不是 StableBatch 在线策略、自然 batching、serving、EP、延迟、吞吐、质量或特定 cuBLASLt kernel 机制的证据。

## A–F 审计结果

| 检查 | 状态 | 核心结论 |
|---|---|---|
| A. Ground-truth / proxy provenance | PASS | 无外部标签或伪造 GT；selection 使用 local M 差异和 native next-layer margin 做预冻结富集，没有使用 intervention route、final logits 或 greedy-token outcome。 |
| B. Score / decision integrity | PASS | 独立复算 1,920 个候选的公式、32 个目标的确定性选择和最终门槛，零不一致。 |
| C. Causal isolation | PASS | full input、attention mask、target hidden/router/top-k/weight、upstream routes、native target raw、non-target contributions 均闭合；唯一 target contribution 在 gate weight 前恰好替换一次；native self-replacement 为 bitwise no-op。 |
| D. Provenance consistency | PASS | V2 只修复 acceptance run03 在 candidate replay 缺少 `inference_mode` 的执行错误；正式 runner/config/test/workload、模型、driver、cuBLASLt、OLMoE source 和输出 manifest 均 hash-bound。 |
| E. Scope / dead code | PASS | 核心 treatment 与判定路径实际执行；证据类型仅为 real-GPU mechanistic single-contribution proxy。 |
| F. Alternative explanations | PASS | fixed arm order、cuBLAS heuristic state 和 margin enrichment 不会推翻已记录的 raw → combine → downstream 链，但会限制鲁棒性、机制解释和发生率外推。 |

## 独立复算的决定性数字

| 指标 | 复算值 |
|---|---:|
| Workloads | 16 |
| Candidate cells | 1,920 |
| Selected targets | 32 |
| 每个 layer band 的目标数 | 8 / 8 / 8 / 8 |
| 每个 victim 的目标数 | 2 |
| 每个 arm 的重复数 | 3 |
| Raw local change | 32 / 32 |
| Post-combine target MoE change | 32 / 32 |
| Reproducible downstream membership change | 12 / 32 |
| 对应 distinct victims | 8 / 16 |
| Reproducible greedy-token flip | 1 / 32 |
| Integrity PASS | 32 / 32 |

冻结支持门槛为 `route targets >= 4` 且 `distinct victims >= 2`。复算值为 `12` 和 `8`，因此冻结 verdict 为 `SUPPORT`。其中 `12/32` 是富集目标集中的命中数，不能写成自然样本发生率。

## Claim impact

- **C1 — artifact 与完整性存在**：supported。
- **C2 — execution-shape M → 单个 raw contribution → combine → downstream victim top-k propagation**：在冻结条件和富集限定下 `supported_provisionally`。
- **C3 — StableBatch controller、serving、自然 batch invariance、EP、latency、kernel algorithm 或一般化收益**：unsupported / unverified。

使用 C2 不需要补跑本轮实验；表述必须保留单 RTX 5090、固定 OLMoE revision、BF16、prompt forward、M1/M64 和 enriched-target 边界。只有希望把 `provisional_same_family` 升级为非暂定接纳时，才需要独立或 cross-family 审计。

## 非阻塞后续项

- reversed/interleaved arm order 与 fresh-process 复测可增强外部鲁棒性；
- raw vector retention 可增强数值复算能力；
- 新 GPU、model revision、自然 heterogeneous filler、continuous-decode/serving/EP 和在线 policy 必须作为后续独立实验，不能附会到本轮证据。

完整逐项审计响应与请求记录位于 [审计 trace](/Users/leandrozhao/Desktop/毕设论文资料/.aris/traces/experiment-audit/2026-08-10_run01/001-stablebatch-integrity.response.md)。本文件和 `EXPERIMENT_AUDIT.json` 是正式 run 完成后的派生审计文件，因此有意不写回实验期 `MANIFEST.json`。

