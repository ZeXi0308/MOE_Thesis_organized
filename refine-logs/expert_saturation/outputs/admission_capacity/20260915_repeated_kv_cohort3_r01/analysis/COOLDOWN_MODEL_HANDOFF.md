# 资源模型的决策预警交接

问题：模型是否只会复算轨迹，还是能阻止一次错误的代理指标选优？

复用主方已完成的start_timing_screen及cooldown五格主分析，不重算raw、不重跑模型、不拟合时间。现有结构筛选中，取消global cooldown让调用1312→1270，同时平均完成步1167.125→1177.15625、load jobs43→89。模型因此没有支持“调用少就整体更快”的选择。

新GPU主分析两对：平均完成+2.0417%/+1.4104%，吞吐−0.5021%/+0.1244%，最大生成gap−9.7975%/−10.4133%。它支持的是模型对完成代价的**方向性预警**：减少调用不足以决定选择提前恢复。不是模型准确预测了秒级代价，也不证明代价全部来自传输。吞吐变化异号，不宣称稳定加速。

模型的suffix gap与实测全episode最大gap窗口不同，不算预测误差/拟合度。此次没有新增样本、统计显著性或独立负载；新调度机制是否值得采用仍由主会话固定停顿/效率目标及边界实验决定。

改变的决策：保留模型作为筛除错误代理推理的工具，停止仅为增加轨迹匹配精度而扩模；不再扫描固定cohort的cooldown。唯一下一科学动作沿用主方安排的异构/EOS/持续到达/压力边界，本方只在出现新的资源决策差异时介入。

来源和提取结果： [cooldown_decision_check.json](cooldown_decision_check.json)。仓库根目录复算：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_repeated_kv_cohort3_r01/analysis/check_cooldown_decision.py
```

该检查仅读取原方两份分析，不将原主表重复计为本线实验。cohort3保存迁移结论与本页cooldown比较是不同实验，不合并样本。
