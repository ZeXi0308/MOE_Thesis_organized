# Idea A：Rank 长尾 + FP8-first Pareto

## 主张

MoE combine 中，**matched-byte 下 tail rank 的 INT4 远比 head 安全**，该 Claim 1 跨模型 GO。`SUPERSEDED`：「FP8-first frontier / 62.5% 已严格 GO」；按冻结全扫描最差点，Claim 2 跨模型 NO-GO，62.5% 仅是探索性逻辑 payload 点。

## 关键证据与边界

- GPU + bootstrap CI：OLMoE / LLM-jp 尾部相对头部安全边际可达数十倍量级（见 Idea A LUT GPU verify 报告）。
- 边界：字节口径不含 scale metadata/header/padding/alignment，**不是**真实 wire 或多卡 RDMA P99。

## 脚本与产物（本目录）

- 脚本：[`experiments/`](experiments/)（`run_fp8_*` / `lut_optimizer.py` / `plot_fp8_*` 等）
- 产物：[`outputs/`](outputs/)（含 `idea_a_rank_lut_gpu_verify_*`、`main_experiments/`）
- 共享捕获/量化：[`../../../experiments/shared/`](../../../experiments/shared/)（`capture_moe.py` / `fake_quant.py` / `policies.py`）

结论文档：本目录 [`原文/`](原文/) · [`设计说明.md`](设计说明.md)；当前状态见 [`../../current/README.md`](../../current/README.md)，历史跨方向 GPU 汇编见 [`../../archive/research_summaries/`](../../archive/research_summaries/)。
