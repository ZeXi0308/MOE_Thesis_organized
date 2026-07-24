# Route-row FP8/BF16 快速 GPU 探索（2026-07-23）

## 结论

**当前 Route-row 动态 FP8 机制 NO-GO，不进入 route-mass、controller 或 serving 集成。**

在 RTX 5090 上，使用两个真实模型的 BF16 expert 权重、一次性常驻 FP8 weight、真实三投影 expert path，rows=1..4096：

- 原始三次动态 activation quantization：6 个 expert 全部无 FP8 快区；OLMoE rows=4096 仍慢 24%–28%，LLM-jp 仍慢 40%–79%。
- gate/up 共享一次 input quantization：OLMoE 仅在两个 expert 的 rows=4096 出现约 6.4%–6.5% 加速；rows<=2048 全部 BF16 更快。LLM-jp rows=4096 仍慢约 32%–34%。
- `torch.compile(fullgraph=True, dynamic=True)`：OLMoE rows=4096 仍慢 9.4%，rows<=256 慢 93%–124%；LLM-jp 无快区。
- 本地 relative MSE 约 0.0036–0.0045，仅是 expert-local 数值诊断，不是 end-to-end quality 证明。

4096 rows/expert 至少要求 4096 active decode tokens，因为一个 token 不会在 top-k 中重复选择同一 expert；对单张 32GB 卡不构成可用的 decode operating region。

## 成本分解

单 projection 的预量化 FP8 GEMM 在大 row 上可比 BF16 快约 36%–48%，但动态 activation quantization 占当前 FP8 linear 时间约 56%–78%。

- 只假设 down quantization 完全免费，rows<=256 仍慢 62%–124%。
- 假设 input/down quantization 都完全免费，rows=256 的理论上限约可快 20%–21%。
- 因此剩余研究机会不是 route-row controller，而是另一个问题：能否让 dispatch 直接产生可复用的 FP8 token activation，并把 SwiGLU/down-input quantization 融入 producer kernel。
- 该上限目前没有真实 fused kernel 支持，不得作为系统收益引用。

## 验证与边界

- `[Observed]`：RTX 5090、PyTorch 2.8+cu128、原生 E4M3FN `torch._scaled_mm`、真实模型 expert weights/shapes。
- 每个 eager broad cell 20 个 paired blocks；原始 Gate 每 cell 30 blocks、1000 bootstrap；BF16/FP8 arm 顺序交替、row 顺序随机。
- OLMoE 与 LLM-jp 各抽前/中/后层 expert；13 个 row 点。
- 计数闭合示例：324 次 FP8 expert call = 972 次 activation cast = 972 次 scaled GEMM。
- `[Controlled synthetic]`：activation 为固定随机 BF16；这是 latency mechanism screen，不是 continuous decode、energy、route mass 或 end-to-end quality 结果。

## Artifact

- [`run_route_row_quick_gate.py`](run_route_row_quick_gate.py)
- [`run_fp8_cost_decomposition.py`](run_fp8_cost_decomposition.py)
- [`全部原始结果`](outputs/route_row_quick_20260723/)
- 下载包 SHA-256：`0b81cf5001bf3445486bd86383c092b2034e3109e15137e9e437f41a04bb4a3d`

