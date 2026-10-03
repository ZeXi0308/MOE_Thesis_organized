# 负控解释的独立定向复核

2026-09-13。审阅者：GPT-5.6-Sol ultra，fresh same-family，provisional。
Agent：`/root/null_drift_review`。范围仅为已有 waiting/per-request/host-timing 证据的负控、分母、独立性与结论强度；不是新 GPU 实验或方法资格审计。

**Overall WARN。** 保留原报告 MEASUREMENT_ONLY 合理。E 项 FAIL 指“把有限相关对照差当作可迁移噪声界”的推断，不指原始 GPU 测量无效。

| 检查 | 状态 | 独立核实证据 |
|---|---|---|
| A 零目标动作与完整路径 | WARN | waiting `analyze_results.py:114` 检查无重排/无bypass；六个all_short raw逐步请求/token路径不同。forward steps306/301/304，reverse309/305/313；首次入场顺序不变不足以证明相同batch路径。`runtime/waiting_order.py:16`与`:82`仍有排序、deque及ledger工作 |
| B 分母与最大差 | WARN | waiting `analyze_results.py:146`、per-request `analyze_results.py:188`正确使用对应baseline分母；样本最大差没有总体覆盖率/置信界保证 |
| C 原值可复现 | PASS | 各报告主表在各自summary可复现；per-request all_short完成均值+3.3886%/+0.8154%、wall+2.6368%/+0.1686%。host115/512、host/native分类差0/1024亦可复算 |
| D 分析路径调用 | PASS | waiting `analyze_results.py:176`、per-request `analyze_results.py:220`及host `analyze_host_timing.py:147`实际读取cell、构造对照并写分析结果 |
| E 跨域噪声界与3×max推断 | FAIL | 同GPU/同文档/固定到达两顺序块，每臂每块一个episode。共享臂差值非独立重复；all_short不校准mixed；3×max不能支持显著性或方法GO。per-request `DECISIONS.md:54`只提出新输入确认的探索条件 |
| F 性能、SLO及质量边界 | PASS | 原始数据是真实native请求性能测量，无任务质量/显著性结果。全通过时goodput等于完成吞吐，不能由此推出安全容量；定义见两实验 `runtime/metrics.py:137` |

## 审阅者配对集合澄清

首次反馈 E 项引用了“目标策略对主参考基线”的两个差值，但未明确该集合与用户的六个全两两差值不同。根代理指出集合差异后，审阅者仅澄清该口径，没有扩展审计；两版结果均保留如下。

| 集合 | 完成均值最大绝对差 | wall最大绝对差 | TTFT均值最大绝对差 |
|---|---:|---:|---:|
| waiting：bounded vs FCFS，2对 | 1.0827% | 0.6146% | 2.8241% |
| waiting：三臂全两两，6对 | 1.3675% | 1.4697% | 2.8241% |
| per-request：per_request512 vs native1024，2对 | 3.3886% | 2.6368% | 5.0910% |
| per-request：三臂全两两，6对 | 3.3886% | 2.6368% | 5.9043% |

两个集合都只是已观察漂移，没有一个是总体噪声界。根报告及 observations.json 明确使用第二类全两两集合。

host `REPORT.md:3`、`:15`明确115/512来自其他进程paired-r01的独立重分析，初始新增GPU为0；512为16个steady cell×32次请求执行，只有32篇重复文本，不能记作新的独立GPU复现。

## Claim impact与停止点

- 支持：真实动作/原值、两块差值方向、参考SLO通过数、观察到的运行路径差异。
- 需限定：探索继续门槛没有正式误判校准；没有显著性、非劣性、质量等价或新颖性通过结论。
- 不支持：固定noise floor、跨all_short/mixed迁移、3×max推出GO、首次顺序相等推出全路径相等、重复请求执行冒充独立样本、共享raw重分析冒充新实跑。
- 本次定向复核结束；下一条科学证据应来自同负载A/A与策略的独立执行，不追加同一批raw的全面审计。

路径根：waiting=`refine-logs/independent_ideas_20260910/waiting_bypass_limit_r01`；per-request=`refine-logs/independent_ideas_20260911/per_request_prefill_share_r01`；host=`refine-logs/independent_ideas_20260908/host_timing_boundary_r01`。原始文件未改写，hash见EXPERIMENT_AUDIT.json。
