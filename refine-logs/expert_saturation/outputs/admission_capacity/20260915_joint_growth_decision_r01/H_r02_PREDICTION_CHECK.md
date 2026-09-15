# H 事前 late64 预测核验

**事前方向预测获得本次两对数据支持；晚到达组的收益幅度明显较小，不能称为统计稳定。** 成本预算通过是另一项结论，不用于替代持续性判定。

仅使用已完成的 [H r02 主分析](../20260915_natural_cadence_holdout_r02/execution_weste_26862/analysis.json) 中每请求 arrival 和最大输出间隔，按 [事前约定](H_PREDICTION_PREDECLARED.md) 固定分为 early（arrival<12.8 s）与 late（arrival≥12.8 s）。每个 current/eager 臂的两组均为 64 请求，均有 64 个已定义 gap、0 个未定义 gap；没有填零或替换请求。

| 配对 | early64 最大 gap 变化 | late64 current→eager | late64 变化 / 方向判定 |
|---|---:|---:|---:|
| block0 | −45.876% | 3.397820→2.935854 s | −13.596% / 通过 |
| block1 | −20.215% | 3.880885→3.759627 s | −3.124% / 通过 |

两对 late 的 eager/current 比值分别为 0.864041、0.968755，均满足事前 `<1` 的方向要求；没有在看到第二对小差后更换阈值。四个 current/eager 运行的全局最大 gap 恰好均来自 late64，因此它证明结果没有仅靠早到达请求改善支撑，**不能把全局结果和 late 结果算作两套独立重复证据**。

late 的改善从约 13.6% 缩小至约 3.1%，且两对都弱于对应 early 组。支持的是“本次独立输入中，晚到达组也保留改善方向”，不支持固定收益幅度、统计稳定或所有后续持续到达场景。输入和到达批次状态共同变化，不能将幅度差直接归因于 episode 变长、host turnover 或某一种内部等待。

完整服务预算直接沿用主分析已有判定：两对均满足 tokens/s 损失≤3%、平均完成增幅≤5%。这确认的是已声明预算下的 H 简单取舍；与上面的 late 持续性方向支持分开成立。H 各臂实际输出量并不完全相同，不能把服务指标称为等工作量加速；EOS、host 和系统参照的详细计数由原执行方报告，本核验不重复。

**研究选择：维持 selected/eager 为 G/H 已测域的停顿优先简单基线，停止 cadence/window/predictor 加法。** 保留晚到达收益减弱及小差边界，不因预算通过而扩张持续性主张，也不为维持“host-history turnover”解释追加分析或埋点。若扩大使用域，应继续通过未用于选择的事实决定，而不是按本次晚组结果调参。

复算：[check_h_prediction.py](check_h_prediction.py)；仅含分组派生值和复用预算结论的小结果：[H_r02_prediction.json](H_r02_prediction.json)。未读取 raw，未重跑主分析或 GPU 主表。
