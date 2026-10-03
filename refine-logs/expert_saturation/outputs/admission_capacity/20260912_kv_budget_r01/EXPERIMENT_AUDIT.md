# 四项KV配置对照：针对性独立复核

2026-09-12。Reviewer：GPT-5.6-Sol ultra，fresh same-family，provisional；只读。遵照本轮轻量要求，没有额外`.aris`、授权层或GPU重跑。

**`PASS_FOR_QUALIFIED_MEASUREMENT`，P0=0，P1=0。**

- 来源/执行：固定95→90→90→95、cap32、4个新进程，全部退出0、各32/32完成；五份执行源码与冻结包相符。
- 身份/工作量：32个唯一request/internal/external ID、文章/prompt hash/arrival、3072输入/1024输出一致；捕获保留failed/unfinished路径；本组全部输出序列也一致。
- 指标/分母：reviewer在新临时目录独立重跑分析，`analysis.json`和`report.md`与留存产物逐字节一致。host receipt计时、成功调用区间重算守恒，无重复累加暂停。
- 结果：吞吐+3.9928%/+4.6985%、wall−3.8395%/−4.4876%、最长暂停小于0.1s；同时mean completion恶化1.0160%/0.1525%，已保留。
- 范围：P2为额外KV资源和未连续监控clock/power/temperature。实际GPU UUID一致；四项均单token chunk，ITL无chunk内分辨率缺口。只能作本cohort/configuration的`NATIVE_SERVING`测量。

核心证据：[execution](execution.json)、[完整重算](analysis/report.md)、[均值](cpu_analysis/completion_means.json)、[返回粒度与GPU记录](cpu_analysis/integrity_observations.json)。实际调用与会计：[run_campaign](frozen/run_campaign.py)、[native_capture](frozen/native_capture.py)、[metrics](frozen/metrics.py)、[重算分析](../20260908_native_preemption_r01/analyze_native_preemption.py)。

允许结论：普通KV增配消除了此运行域观测到的原生抢占/秒级恢复暂停，改善整批吞吐与wall，但平均请求完成延迟未改善。它不是同预算调度、expert paging、质量、第二模型、EP或生产SLO结果。

本次复核完成，后续应获取pager新数据；不追加同类审计。
