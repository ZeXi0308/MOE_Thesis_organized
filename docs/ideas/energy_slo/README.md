# Energy-SLO Precision EP

## 主张

待检验的系统主张是在 SLO 约束下联合选择 **batch accumulation** 与 **compute 精度（FP8）**。当前只有两个独立硬件微基准，系统主张未验证。

## 关键证据与边界

- 固定 seq=64 full-forward 的 batch 1→64：measured J/counted-token 约 17× 改善；不是 KV decode/arrival SLO。
- 预 cast FP8 vs BF16 GEMM-core：约 2× 吞吐、~34% 单 matmul 能耗下降；不含 activation quantize/scale/cast。
- `SUPERSEDED`：batch64 在延迟和能耗两维严格支配；代理延迟实际从 142.2ms 增至 209.5ms，且无 queueing/P99。

## 脚本与产物（本目录）

- [`experiments/`](experiments/) · [`outputs/`](outputs/)
- `run_energy_slo_power_probe.py` / `run_energy_slo_fp8_compute_quality_gate.py`
- 文档：本目录 [`原文/`](原文/)
