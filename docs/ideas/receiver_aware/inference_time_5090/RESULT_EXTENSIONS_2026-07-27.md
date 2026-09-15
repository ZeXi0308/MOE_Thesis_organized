# Receiver 单卡扩展实验结论（2026-07-27）

## 1. 直接裁决

1. **本地 MoE 成本可以继续细分，但没有暴露 Receiver 现象。** 在 10% observer-tax
   fail-closed 门内，LLM-jp 的 coarse `expert_loop` 占 profiled decode 的约
   74.9%–85.7%；`routing_setup` 约 3.2%–5.7%，gate 低于 0.8%。`expert_loop`
   包含 gather、expert compute、routing-weight multiplication 和本地 `index_add_`，
   不是纯 GEMM，更不是 return all-to-all。
2. **prompt 128/512/2048 的 KV-decode 没有随上下文显著恶化。** 同一 batch 下三档
   decode median 的最大值/最小值最多为 1.0191；prefill 则按输入 token 数正常增长。
   该结果不支持“多个本地 MoE 层会随上下文自然形成拥塞”的猜测。
3. **自然文本没有形成稳定的额外单卡压力。** 在同一次模型加载中按 AB/BA 交替比较
   冻结自然连续文本与合成 token 序列，batch 1/4/8 的 decode median 差异分别为
   -0.052%、+0.058%、+1.130%。先前两个独立 run 的 6%–7% 表面差异被配对实验判定为
   运行间漂移，不能归因给 workload 内容。
4. **Receiver congestion 仍为 `NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`。** 单卡实验没有
   EP ranks、NCCL/NVLink/RDMA return collective、receiver ingress queue 或 RankLane。
5. vLLM continuous-arrival serving 与 OLMoE 跨模型复现已启动准备，但远端实例在依赖安装
   与权重续传期间关闭，状态为 `BLOCKED_REMOTE_INSTANCE_CLOSED`，不计为已完成实验。

机器汇总结论：

```text
SINGLE_GPU_EXTENSIONS_COMPLETE_NOT_RECEIVER_GATE
RECEIVER_CONGESTION_NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP
VLLM_SERVING_BLOCKED_REMOTE_INSTANCE_CLOSED
OLMOE_CROSS_MODEL_BLOCKED_REMOTE_INSTANCE_CLOSED
```

## 2. Coarse 本地 MoE 组件分解

### 协议

- 模型：`llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- revision：`1d5983076dfc67aee4a77ec06a27027f5bab6055`
- 结构：16 个 MoE block，32 experts，top-k=16
- batch：1、4、8、16、32；prompt=128；decode=16；repeats=3
- `unprofiled` 与 `breakdown` arm 按 repeat 交替 AB/BA
- 定量门：任一 decode cell 的 median observer tax 超过 10% 即停止解释分解值

逐 expert hook 的首次 smoke 将端到端 latency 抬高约 28%–31%，因此被否决。正式版本不再
为每次 expert invocation 安装 hook，而是在 Mixtral-style eager MoE block 内使用 coarse CUDA
event region，将每层划分为 gate、routing setup、expert loop 和未归属尾部。正式 run 的最大
decode observer tax 为 6.55%，通过预设门。

| Batch | 完整 decode median | MoE / profiled forward | Gate | Routing setup | Expert loop | 未归属尾部 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43.094 ms | 82.67% | 0.80% | 5.69% | 74.90% | 1.29% |
| 4 | 73.936 ms | 88.91% | 0.52% | 3.56% | 84.00% | 0.81% |
| 8 | 80.906 ms | 89.75% | 0.49% | 3.30% | 85.22% | 0.75% |
| 16 | 83.297 ms | 90.05% | 0.47% | 3.22% | 85.63% | 0.73% |
| 32 | 83.223 ms | 90.07% | 0.47% | 3.21% | 85.67% | 0.72% |

该表最直接的含义是：在当前 HF eager 单卡实现中，大头已经在本地 expert execution loop；
它没有给通信优化留下可量化的端到端 headroom。只有真实 EP timeline 才能测 exposed return
fraction。

## 3. Context length × batch

主口径为无 per-layer hook 的完整 CUDA forward；三档使用相同 batch、decode steps、repeats 和
seed protocol。

| Prompt | Batch | Decode median | Decode P95 | Prefill median | MoE / profiled decode |
|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 42.931 ms | 45.088 ms | 83.181 ms | 82.75% |
| 512 | 1 | 43.135 ms | 44.752 ms | 86.945 ms | 82.91% |
| 2048 | 1 | 43.029 ms | 47.870 ms | 90.259 ms | 82.85% |
| 128 | 4 | 73.750 ms | 77.479 ms | 87.387 ms | 89.00% |
| 512 | 4 | 72.507 ms | 76.513 ms | 90.490 ms | 88.88% |
| 2048 | 4 | 73.036 ms | 77.946 ms | 134.887 ms | 88.95% |
| 128 | 8 | 80.262 ms | 86.295 ms | 88.358 ms | 89.86% |
| 512 | 8 | 78.758 ms | 87.193 ms | 106.383 ms | 89.78% |
| 2048 | 8 | 79.145 ms | 80.811 ms | 191.538 ms | 89.74% |

这个 sweep 的 decode 使用 KV cache，每步只输入一个新 token；prompt length 主要增加 attention
KV 长度，而该 512-hidden 小模型的 decode 仍由本地 MoE loop 主导，所以三档差异很小。它不是
continuous-arrival TPOT 或 queueing 结果。

## 4. 自然文本与合成输入的配对 A/B

自然 workload 为 Project Gutenberg `Alice's Adventures in Wonderland` 的冻结连续 token span：

- source：`https://www.gutenberg.org/files/11/11-0.txt`
- SHA256：`a3a27f8edbf7fcd9b8ba8435494440e24952deaa3e2f2d65192d4cb7ca403754`
- tokenizer 后 41,232 tokens
- 每个 cell 五个独立 sequence pair；arm 顺序按 repeat 交替 AB/BA
- ratio 先在每条 sequence 内取 decode forward median，再跨五个 pair 取 median

| Phase | Batch | Synthetic median | Natural median | Natural delta median |
|---|---:|---:|---:|---:|
| decode | 1 | 43.658 ms | 43.617 ms | -0.052% |
| decode | 4 | 73.647 ms | 73.689 ms | +0.058% |
| decode | 8 | 79.329 ms | 80.487 ms | +1.130% |
| prefill | 1 | 83.302 ms | 83.840 ms | +0.214% |
| prefill | 4 | 86.458 ms | 86.844 ms | +0.309% |
| prefill | 8 | 87.045 ms | 88.223 ms | +0.087% |

五个 sequence pair 只支持描述性结论，但足以否决“独立 run 相差 6%–7% 就是自然 workload
效应”的解释。两种输入的 untimed route census 也非常接近，没有出现自然文本专属的强路由
失衡。

## 5. 未完成项与恢复点

### vLLM continuous-arrival serving

已在独立 venv 中开始安装 `vllm==0.26.0`，准备使用官方 `vllm bench serve` 测 TTFT、TPOT、
E2EL、request throughput 与 overload knee。远端关闭时仍在安装 Torch/Triton 依赖，未启动
server，未产生 serving 数据。因此当前不能写 TPOT/P99 或 continuous batching 结论。

恢复后应：

1. 验证 vLLM、Torch、CUDA 与 RTX 5090 可加载同一 LLM-jp revision；
2. 先用 `request-rate=inf` 测 saturation throughput；
3. 冻结 30%、60%、90%、120% saturation 的 arrival-rate sweep；
4. 每档报告 TTFT/TPOT/E2EL P50/P95/P99、goodput、失败数和实际请求率；
5. 仍只称单卡 serving queueing，不称 Receiver congestion。

### OLMoE 跨模型

`allenai/OLMoE-1B-7B-0924` 三个约 5 GB 权重 shard 中，第一个已完成，第二个接近完成后实例
关闭；Hugging Face incomplete cache 原设计可续传，但未能在断线后复核。没有生成 OLMoE
inference-time 或组件结果，不得拿 tiny-Mixtral/LLM-jp 代替。

## 6. 证据入口

- 合并报告：[`outputs/receiver_single_gpu_extensions_v2_20260728/report.md`](outputs/receiver_single_gpu_extensions_v2_20260728/report.md)
- 合并机器裁决：[`outputs/receiver_single_gpu_extensions_v2_20260728/decision.json`](outputs/receiver_single_gpu_extensions_v2_20260728/decision.json)
- coarse 组件分解：[`outputs/receiver_moe_breakdown_coarse_llmjp_formal_20260728/`](outputs/receiver_moe_breakdown_coarse_llmjp_formal_20260728/)
- context 128/512/2048：[`outputs/receiver_context_llmjp_p128_20260728/`](outputs/receiver_context_llmjp_p128_20260728/)、[`outputs/receiver_context_llmjp_p512_20260728/`](outputs/receiver_context_llmjp_p512_20260728/)、[`outputs/receiver_context_llmjp_p2048_20260728/`](outputs/receiver_context_llmjp_p2048_20260728/)
- 自然/合成配对 A/B：[`outputs/receiver_input_ab_llmjp_p128_20260728/`](outputs/receiver_input_ab_llmjp_p128_20260728/)

最终研究判断：这些单卡扩展进一步缩小了 Receiver 的可辩护空间。当前看到的是本地 expert
execution 主导，而不是 receiver return path 暴露；只有 8×A100 optimized EP Gate 能决定该
方向是否值得 reopen。
