# 统一 host-call 计时边界

本补充按主研究会话要求统一跨诊断比较，不新增复核。原 `summary.json` 与 `analyze.py` 保持不变，其中 `S` 取最早 host load-submit / recovery-call 边界，不能称为统一 host-call 起点。

对每段取 `min(x.time_s for x in recovery_starts if x.boundary == "host_engine_call")`，然后分别计算 `S_host_call − L` 和 `F − S_host_call` 的中位数。旧组 44 段、当前 eager 90 段均有此边界：

| 口径 | 旧组 diag-on 描述参考 | 当前 diagnostic-eager |
|---|---:|---:|
| 统一 L→S_host_call | 1.345704026 s | 0.689846126 s |
| 统一 S_host_call→F | 0.020532745 s | 0.020386009 s |
| 原 summary L→最早 host 边界 | 1.305452296 s | 0.649542873 s |
| 原 summary 最早 host 边界→F | 0.059612325 s | 0.059422791 s |

load-submit 事件及时间完整保留。两种边界都不是物理 DMA/kernel 起点；旧诊断属于不同实验组，仍只作描述性参考。
