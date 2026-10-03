# 后续轻量运行的直接抢占事件

上一分解使用诊断位置标注轻量输出；可见输出完全吻合仍不能证明隐藏抢占位置相同。此次实现关闭的是未来数据的这个测量缺口，旧五格和原保存组没有因此获得新的直接事件证据。

共享[request_measurement.py](../../../../experiments/admission_capacity/request_measurement.py)增加显式参数 `record_preemptions=True`，默认False。启用后，只链式包装当前实例的 `_preempt_request`；现有staged强制轮转与native容量回退都调用该方法。调用原方法一次并保留返回值/异常，退出恢复原实例hook或原类方法解析方式；不替换schedule、allocator或轮转规则。

每次方法调用记录请求身份、当前engine-call号、**此前实际返回**的输出数与最后新输出时间；原生内部输出计数单独记录。方法返回/失败的host时间和状态分列。它不重置输出等待，不声称GPU执行或异步flush已经完成，也不从一次方法调用猜测forced/native原因。失败保留事件与既有partial raw行为；原runner的finally落盘合同继续适用。

启用结果明确标记 `SPARSE_PREEMPTION_EVENTS`、`measurement_hooks_modified=["_preempt_request"]` 和 `policy_hooks_modified=True`（仅表示方法被包装），不能冒称没有observer。`actual_preemption_count`计原方法正常返回次数，attempt总数单列；没有正常返回不推断是否发生部分状态修改。默认关闭仍保留原 `None` 和 `DIAGNOSTICS_DISABLED`，不是把未测冒充0。

实现先在原独立worktree完成，通过后校验共享旧文件仍与已读版本一致，再同步两个源码文件。必要检查已实际运行：原3项计时/失败分母/自然EOS行为，加2项稀疏事件成功与失败行为；成功例同时覆盖原类方法和已有实例hook、内部计数99但实际返回仅1、重复输出不重置等待。5项全部通过：

```sh
python3 -m unittest discover -s refine-logs/expert_saturation/experiments/admission_capacity -p test_request_measurement.py -v
```

调用方式沿现有入口，仅显式增加一个测量参数：

```python
raw = measure_episode(engine, workload, config,
    regime="steady", arrival_scale=1.0, run_id="measured",
    record_preemptions=True)
```

状态为 **CPU_VERIFIED / GPU_UNRUN**，没有新的服务性能结论，也未测observer在GPU运行中的扰动。当前已执行包及raw保持原样。交接给唯一主执行方的下一动作是：若下一代表性负载组需要直接归因，在两臂共同启用并把该测量版本和全部运行成本纳入组合同；不另开观察器参数扫描或独立GPU组。此补丁不是调度方法或论文贡献。
