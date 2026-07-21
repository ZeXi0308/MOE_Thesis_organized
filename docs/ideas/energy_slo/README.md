# Energy-SLO Precision EP

## 主张

在 SLO 约束下联合选择 **batch 积压** 与 **compute 精度（FP8）**，用真实能耗/吞吐，而非仅 fake-quant。

## 关键证据与边界

- Batch 1→64：约 17× J/token 改善（nvidia-smi）。
- FP8 vs BF16 GEMM：约 2× 吞吐、~34% 单 matmul 能耗下降；expert FFN 路径 KL 小。
- 边界：联合 arrival×batch×precision Pareto 与多 GPU 通信能耗仍待完整评估；单卡杠杆已真。

## 脚本与产物（本目录）

- [`experiments/`](experiments/) · [`outputs/`](outputs/)
- `run_energy_slo_power_probe.py` / `run_energy_slo_fp8_compute_quality_gate.py`
- 文档：本目录 [`原文/`](原文/)
