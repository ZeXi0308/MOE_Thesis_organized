# Receiver 多 MoE 层 inference-time 5090 验证

## 目标

补齐此前 Receiver 资产没有真实 inference-time 测量的问题。在单张 RTX 5090 上测量：

1. 完整 prefill CUDA 时间；
2. 使用真实 KV cache 的逐步 decode CUDA 时间；
3. 每次 forward 中所有 MoE block 的累计 CUDA 时间和端到端占比；
4. 单独 untimed route census 的 expert-load imbalance；
5. per-layer observer 对完整 inference time 的计时税。

## 证据边界

该实验不包含 EP ranks、NCCL、NVLink/RDMA return all-to-all、receiver ingress queue、
continuous arrival 或 RankLane。多个 MoE 层在单卡上顺序累积时间，不等于 receiver congestion。

因此，本实验只能回答：

> 多个 MoE block 的本地累计执行时间是否已经构成完整 prefill/decode 的显著部分？

它不能回答：

> 多 GPU EP return path 是否产生 receiver congestion，或者 RankLane 能否改善 TPOT/P99？

后一个问题仍必须执行 [`../cpr_ranklane/EP_Return_Path_8xA100存在性Gate.md`](../cpr_ranklane/EP_Return_Path_8xA100存在性Gate.md)。

## 2026-07-27 已执行矩阵

- M1 主表征：`llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`，16 层、32 experts、top-k=16
- M0 脚本交叉检查：`jamesdborin/tiny-mixtral`，2 层、top-k=2
- `allenai/OLMoE-1B-7B-0924`：本次未运行，不把 M0/M1 数据写成 OLMoE 结果
- dtype：BF16
- M1 batch：1、4、8、16、32；M0 batch：1、4、8、16
- prompt length：128
- decode steps：32
- 独立输入/计时 repeats：5
- warmup decode steps：4
- route census steps：16
- 主 inference-time：没有 per-layer hooks 的 `unprofiled` arm
- 分解数据：带 CUDA Event hooks 的 `profiled` arm
- arm 顺序：每个 repeat 交替 AB/BA

路由输入为冻结的合成有效 token 序列，只用于控制 shape 和得到可复跑的 route variation；不是自然
serving arrival。每个 decode step 后同步 CUDA 以读取 event，因此结果是 isolated KV-decode forward
时间，不是 continuous-batching TPOT。

## 远端命令

```bash
python3 profile_multi_moe_inference.py \
  --model llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M \
  --cache-dir /root/autodl-tmp/hf-cache \
  --local-files-only \
  --output-dir /root/autodl-tmp/receiver_multi_moe_llmjp_formal_20260727 \
  --batch-sizes 1,4,8,16,32 \
  --prompt-len 128 \
  --decode-steps 32 \
  --warmup-decode-steps 4 \
  --repeats 5 \
  --route-census-steps 16 \
  --seed 20260727

python3 analyze_multi_moe_inference.py \
  --input-dir /root/autodl-tmp/receiver_multi_moe_llmjp_formal_20260727
```

输出目录拒绝覆盖。正式结果至少包含：

- `timings_raw.csv`
- `moe_layers_raw.csv`
- `route_census_untimed.csv`
- `run_manifest.json`
- `summary.csv`
- `route_latency_correlations.csv`
- `decision.json`
- `report.md`

## 解释规则

- `unprofiled` 是完整 inference-time 主口径。
- `profiled` 只用于分解 MoE block 时间；必须同时报告 observer tax。
- 如果多个 batch cell 的 MoE 累计占比很高，只能说明 MoE block 值得进一步分解。
- 如果路由不均衡与单卡 decode latency 相关，只能称为 local route-pressure correlation。
- 任何结果都不能改变 fixed RankLane 的现有 NO-GO，也不能替代 8xA100 Receiver Gate。

## 2026-07-27 实测结果

- 人读结论：[`RESULT_2026-07-27.md`](RESULT_2026-07-27.md)
- 单卡扩展结论：[`RESULT_EXTENSIONS_2026-07-27.md`](RESULT_EXTENSIONS_2026-07-27.md)
- 16 层 LLM-jp 主证据：[`outputs/receiver_multi_moe_llmjp_formal_20260727/`](outputs/receiver_multi_moe_llmjp_formal_20260727/)
- 2 层 tiny-Mixtral 交叉检查：[`outputs/receiver_multi_moe_tiny_formal_20260727/`](outputs/receiver_multi_moe_tiny_formal_20260727/)
- coarse 组件分解：[`outputs/receiver_moe_breakdown_coarse_llmjp_formal_20260728/`](outputs/receiver_moe_breakdown_coarse_llmjp_formal_20260728/)
- context 128/512/2048 汇总与自然/合成配对 A/B：[`outputs/receiver_single_gpu_extensions_v2_20260728/`](outputs/receiver_single_gpu_extensions_v2_20260728/)

当前机器结论为 `SINGLE_GPU_MULTI_LAYER_MOE_COST_CHARACTERIZED`；Receiver congestion 仍为
`NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`。

扩展实验进一步得到：coarse `expert_loop` 占 profiled decode 约 74.9%–85.7%；prompt
128/512/2048 的同 batch decode median 最大值/最小值不超过 1.0191；自然/合成输入的
interleaved decode 差异为 -0.052% 到 +1.130%。这些仍是单卡本地执行结论。

vLLM continuous-arrival serving 与 OLMoE 跨模型复现因 5090 实例关闭停在
`BLOCKED_REMOTE_INSTANCE_CLOSED`，没有产生正式数据。

## 后续分解与稳健性复核

- [`outputs/receiver_moe_breakdown_coarse_llmjp_formal_20260728/`](outputs/receiver_moe_breakdown_coarse_llmjp_formal_20260728/)：
  粗分解显示 local `expert_loop` 占 profiled decode 约 74.9%–85.7%；其包含
  gather、expert compute、weighting 和 `index_add`，不是纯 GEMM，但说明高 MoE 占比
  主要不是已观测到的通信成本。
- [`outputs/receiver_natural_llmjp_p128_20260728/`](outputs/receiver_natural_llmjp_p128_20260728/)：
  冻结连续自然文本 span 下复现约 82.9%–90.0% 的本地 MoE 累计占比，排除 token 重排
  输入是唯一原因；仍不是 continuous serving。
- [`outputs/receiver_context_llmjp_p128_20260728/`](outputs/receiver_context_llmjp_p128_20260728/)、
  [`p512`](outputs/receiver_context_llmjp_p512_20260728/)、
  [`p2048`](outputs/receiver_context_llmjp_p2048_20260728/)：不同 context length 下保持相同量级；
  仍仅适用于 LLM-jp + Transformers eager 单卡实现。

完整综合结论以 [`RESULT_2026-07-27.md`](RESULT_2026-07-27.md) 为准。
