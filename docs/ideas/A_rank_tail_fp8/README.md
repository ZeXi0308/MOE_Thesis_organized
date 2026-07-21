# Idea A：Rank 长尾 + FP8-first Pareto

## 主张

MoE combine 通信中，**尾部 rank 的低比特（INT4）远比头部安全**；在 FP8 为默认高精的前提下，存在清晰的 **FP8-first + tail-INT4** 质量–字节 Pareto（约 62.5% 低比特饱和点在原主张上 GO）。

## 关键证据与边界

- GPU + bootstrap CI：OLMoE / LLM-jp 尾部相对头部安全边际可达数十倍量级（见 Idea A LUT GPU verify 报告）。
- 边界：质量/字节证据，**不是**真实多卡 RDMA P99；在线自适应不在本 idea 主张内。

## 脚本与产物（本目录）

- 脚本：[`experiments/`](experiments/)（`run_fp8_*` / `lut_optimizer.py` / `plot_fp8_*` 等）
- 产物：[`outputs/`](outputs/)（含 `idea_a_rank_lut_gpu_verify_*`、`main_experiments/`）
- 共享捕获/量化：[`../../../experiments/shared/`](../../../experiments/shared/)（`capture_moe.py` / `fake_quant.py` / `policies.py`）

结论文档：本目录 [`原文/`](原文/) · [`设计说明.md`](设计说明.md)；GPU 第五辑见 [`../../02_gpu_audits/`](../../02_gpu_audits/)。
