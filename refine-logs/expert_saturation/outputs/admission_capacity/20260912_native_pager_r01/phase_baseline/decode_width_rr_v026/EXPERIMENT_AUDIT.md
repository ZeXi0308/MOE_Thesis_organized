# 新decode轮转动作定向审计

2026-09-13；GPT-5.6-Sol ultra，fresh read-only，same-family/provisional。

**WARN；P0=0，P1=0。** 当前MEASUREMENT_ONLY / NO_ADVANTAGE_DEMONSTRATED_FOR_RR2边界成立，不阻止后续实验。

- A: **PASS** — Native timing and actual pager counters; no GT or accuracy claim. Evidence: [protocol.json:124](protocol.json), [readback_completion.json:9](readback_completion.json).
- B: **PASS** — Independent request/wall/bytes/groups recomputation matched; mutually exclusive phase reconstruction residual0, no self-max normalization. Evidence: [analyze_decode_width_rr.py:26](analyze_decode_width_rr.py), [source/native_capture.py:281](source/native_capture.py), [phase_cost_diagnostic.json:10868](phase_cost_diagnostic.json).
- C: **PASS** — 8 cells,24 request executions,320 outputs;18,272 total pager calls,2,624 measurement calls,856392990720 measurement payload bytes; r01 GPU_BUSY retained. Evidence: [readback_completion.json:2](readback_completion.json), [readback_r02/results/pager_summary.json:926](readback_r02/results/pager_summary.json).
- D: **PASS** — Actual native budget/selection/held-state/rotation; every held request served next nonempty step; policy-specific state and no future routing. Evidence: [decode_width_rr.py:18](decode_width_rr.py), [decode_width_rr.py:34](decode_width_rr.py), [readback_r02/results/repeat_3_rr32/raw.json:1013](readback_r02/results/repeat_3_rr32/raw.json).
- E: **WARN** — One triplet,two counterbalanced observations per arm,22.83% RR wall drift,physical KV IDs differ and tensor bytes untested,admission cap2 baseline missing at review. Current report already bounds claims. Evidence: [summary.json:34425](summary.json), [REPORT.md:24](REPORT.md).
- F: **PASS** — Real single-GPU native vLLM eager pager integration measurement without GT; artificial cap16, not simulation, quality or true-over-HBM validation. Evidence: [readback_r02/launch/run.log:1](readback_r02/launch/run.log).

独立核查：static32与RR的调度/路由/cache结构到step7相同，首次调度改变在step8。两格新TTFT与旧最大ITL均在该动作前确定；RR旧maxITL实际来自生成token7→8的间隔，不能归因于轮转。新请求首token/首动作开始分别1.601720/1.602004s与1.802153/1.802687s。

第一处输出分叉为旧请求0000507第14个生成token：static32=281，RR=20453；在动作后，属于独立自然执行，不自动使实验INVALID，但不支持相同token轨迹/质量等价。

只读复算与summary/phase账本一致。各臂独立演进且全部请求等待计入完整时钟。没有把样本最大差当总体噪声界；没有稳定效应量、SLO、质量、family或问题级NO-GO。无需再扩历史审计。
