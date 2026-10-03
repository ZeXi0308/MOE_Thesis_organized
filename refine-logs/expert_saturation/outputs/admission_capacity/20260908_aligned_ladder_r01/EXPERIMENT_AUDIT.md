# 单轮实验完整性检查

2026-09-08；GPT-5.6-Sol ultra，fresh same-family，provisional。

**Overall WARN，具体 P0=0 / P1=0。**

| 项目 | 审查结论 |
|---|---|
| A 数据/参考来源 | PASS |
| B 指标归一化及分母 | PASS |
| C 结果存在及数字一致性 | PASS |
| D 指标代码实际执行 | PASS |
| E 范围与限制 | WARN |
| F 证据类型 | PASS |

审查者从24个raw cell再次运行分析，除生成时间外与已有analysis/report一致。
768次请求执行、四组相对最好静态点变化均核对一致；goodput按完整episode时长计算，
平均TPOT按首末token时间差除以127计算，未发现自归一化或伪造参考。
源码摘要、引擎配置和完成状态匹配。

WARN保留已披露的成功warmup完整raw缺失、仅有cell边界GPU隔离检查，以及单模型、
单卡、有限cohort范围。原始24个测量cell本地完整；该检查不提供新的GPU数据，
不将历史旧/新ladder差值升级为因果作用。

证据入口：[结果表](REPORT.md)、[原始重算结果](analysis/analysis.json)、
[归档缺口与解释边界](ADDENDUM.md)、[传输校验](TRANSFER.json)。
只做本目录的一轮针对性检查；未生成.aris，未扩展历史审计。同期准备的paired实验
不在本次审查范围内。

审查者返回的最终摘要：

> Overall WARN（same-family/provisional）。A/B/C/D/F PASS，E WARN。24/24、768及四组变化（+3.3992%、+49.5405%、−4.0804%、−4.1828%）与 raw 复算一致。warmup raw 缺失、仅边界隔离、单模型单卡有限范围均已披露。无具体 P0/P1。
