# 单请求 prefill 份额：两块已完成

状态：**MEASUREMENT_ONLY；当前per_request512参数扫描停止。** 两个fresh-engine block均完成6次正式测量和6次预热，共12次正式、12次预热；两类各192次请求执行。原始数据及运行文件已全部回传并通过字节核验，远端原件保留。

[独立审计](EXPERIMENT_AUDIT.md)为WARN，无P0/P1完整性缺陷；P2来源端点问题已单独说明，正向继续条件未满足。

本轮比较FCFS下的native1024=(global1024, threshold0)、global512=(512,0)和per_request512=(1024,512)，均为cap8、编译容量1024。mixed两块中，per_request512相对native1024的平均完成延迟为+1.57% / +0.51%，wall为+1.13% / +0.64%，没有稳定整体收益。每块24次实际限额分为调度后无等待者15次、名额满且仍有等待者6次、同一步首次服务短请求3次。局部ITL尾部改善和实际服务机会均保留，但没有达到完整请求净收益的继续条件。

- [冻结问题与配置](DECISIONS.md)
- [结果报告](REPORT.md) 与 [完整指标表](analysis/TABLE.md)
- [逐请求及全部比较](analysis/summary.json)、[24次限额分布](analysis/prefill_budget_partition.json)
- [forward原始数据](readback/results/forward/)、[reverse原始数据](readback/results/reverse/)
- [forward回传核验](TRANSFER-forward.json)、[reverse回传核验](TRANSFER-reverse.json)、[执行状态](STATUS.json)
- [文章来源边界补充](ADDENDUM_SOURCE_BOUNDARY.md)：第4条来源结束边界错误，16/16实际输入前缀不变，原始metadata保留。

唯一下一步尚未运行：按固定源顺序、排除本轮文档，选择16篇全文不超过3968 tokens的完整文章，只测native1024/FCFS及固定128输出，判断小prefill尾块是否受固定2048长度对齐影响。这是自然长度基线诊断，不复活512策略，不扫描新阈值。
