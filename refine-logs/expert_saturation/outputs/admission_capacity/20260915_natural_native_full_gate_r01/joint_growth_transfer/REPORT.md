# A：共同增长信号跨保存底座检验

**结论：容量诊断可保留，不能直接用作短恢复预测或服务保证。** 原生完整保存的新25个prepare前态仍由最大可释放块规则取得全部最优条件容量余量（含全部同分候选），没有复杂容量排序residual。完整保存下全部25前态都有某个合法候选使条件余量非负；这不等于那些候选实际已执行或收益已验证。

模型函数与上一轮字节相同，SHA256 acf293f5cbbb0d53c66d258ac4b50dcc5a86ac582e9df5473230e170afcff20f。h=k=16不变，只让检查脚本可关闭旧D事件step903/1161的诊断提取，避免把旧step身份套到E。两份原方分析分别按target内部ID、preempt、prepare、first-execution及next-preempt边界关联，均25/25唯一匹配，无缺失；输出结果从未输入容量评分。

## 与当前free/释放块基线的区别

基线为free+victim可释放块−target当前历史需求（不预支未来释放）；共同增长则再核算16位置/16新输出的条件空间。前者两组各25次均非负，是已经准入的prepare集合，其全通过不能外推所有waiting状态。

| 既有episode | 共同增长余量负 | 其中随后≤2输出再抢占 | 余量非负但随后≤2输出再抢占 |
|---|---:|---:|---:|
| selected保存 D | 4/25 | 2 | 0 |
| native完整保存 E | 2/25 | 0 | 0 |

D的4个负余量前态随后分别输出5/2/1/8个token；E的2个负余量前态（prepare851/1014）随后输出9/4个token。故负余量不能当作≤2输出的充分条件。这不反驳容量公式：公式本就假设所有peer同步推进16位置，目标短服务是不同且更强的命题。上述只是检验这个容易误用的代理解释，不训练分类器、不报告显著性或总体准确率。

正余量也不是16输出服务保证：D prepare635 margin2只输出14后再抢占；E prepare667 margin6输出15、prepare1215 margin0输出13。真实执行包含prepare/load延迟、peer非同步推进、准入与原生受害选择；当前包络没有表示这些转换。不由三个反例直接指定是哪一项导致，不用实际未来EOS补模型。

## 边界与决定

D/E为同输入资源策略下两次各自真实诊断，保存范围和诊断负担不同，状态及输出量也不同。不能将两组看成匹配时间线、随机对照、独立50样本，不能计算诊断墙钟性能增量。所有数字均复用原主方raw派生分析，不增加运行次数。只评价25次forced prepare，不覆盖全部原生恢复。

已排除：仅凭即时free/释放量可描述后续共同进展；负共同余量自动意味着1—2token短恢复；非负共同余量保证16token服务；本容量子目标需要复杂victim predictor。没有排除：结合真实prepare/load与执行约束的短期模型有决策价值，或最大释放量与most_output有完整服务取舍。

**机制调整：** 不落地“margin<0就延后恢复/延长保护”控制器。最大可释放量仍是容量评分的强简单基线，服务价值必须另计。两种保存底座轻量对照由既定C执行方完成，本线不重复启动。

**唯一下一项模型工作：** 用当前前态检验prepare期间一拍peer增长如何改变可资助余量，给出延后恢复的条件反例；不使用未来完成释放，不预设等待会释放容量。只有能改变候选动作判断，才接入后续runtime比较。

证据层级：STRUCTURAL + 原NATIVE诊断结果关联；无新运行、时间预测、quality或方法GO。输出 full_screen.json/full_outcomes.json/selected_outcomes.json，脚本与冻结模型同目录。复算从仓库根执行，使用新输出路径：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/joint_growth_transfer/check_joint_growth.py --selective refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/execution_weste_26862/readback/results/diagnostic-native-full/selective-store.json --output /private/tmp/full-screen-new.json --event-steps
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/joint_growth_transfer/compare_growth_transfer.py --screen /private/tmp/full-screen-new.json --analysis refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/execution_weste_26862/analysis.json --output /private/tmp/full-outcomes-new.json
```

D关联复算同脚本，screen换原D joint_growth_model/analysis.json、analysis换原D execution_weste_26862/analysis.json；输出新路径，全部脚本拒绝覆盖。
