# 本地接续

从独立worktree `/private/tmp/moe-a-recovery-components-20260914` 使用 `/Users/leandrozhao/.brew/bin/python3 -B`。实现、CPU资格、打包和远端暂存已完成；GPU执行仍UNRUN。

本轮已完成 `event_contract.json` 和 `CPU_CHECKS.json`，随后实际运行了以下准备命令；当前目录不可再次覆盖：

```sh
/Users/leandrozhao/.brew/bin/python3 -B refine-logs/expert_saturation/experiments/admission_capacity/prepare_single_victim_runtime.py
```

`execute.py stage`只暂存新独占目录，不启动引擎。只有共享队列完整释放、现场GPU/锁/源码检查通过后，原执行方才能首次调用`execute.py run`。原目录有launch-once/results时禁止重启；保留原终态和所有失败。

当前22文件已暂存并逐文件验证，勿再次执行stage。唯一原执行目录是`/root/autodl-tmp/moe-single-victim-runtime-20260914-r01`。按最新GPU_COORDINATION确认B Qwen和A repeated-staged都已完整终态并释放后，原方可首次执行：

```sh
/Users/leandrozhao/.brew/bin/python3 -B refine-logs/expert_saturation/outputs/admission_capacity/20260914_single_victim_runtime_r01/execute.py run
```

实际结果回读后，用新输出路径重分析（没有GPU原件时输出明确UNRUN）：

```sh
/Users/leandrozhao/.brew/bin/python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_single_victim_runtime.py --bundle refine-logs/expert_saturation/outputs/admission_capacity/20260914_single_victim_runtime_r01 --funding-analyzer refine-logs/expert_saturation/experiments/admission_capacity/analyze_funding_filter_comparison.py --context-analyzer refine-logs/expert_saturation/experiments/admission_capacity/analyze_context_victim_calibration.py --analysis-library '/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/experiments/admission_capacity' --output /private/tmp/NEW-single-victim-analysis.json
```

CPU事件合同复跑仅写新根目录：

```sh
/Users/leandrozhao/.brew/bin/python3 -B refine-logs/expert_saturation/experiments/admission_capacity/qualify_single_victim_event.py --output-root /private/tmp/NEW-single-victim-qualification --scheduler-source /private/tmp/moe-native-v026-recovery-source/scheduler.py
```

上述CPU fixture使用已取回的vLLM scheduler原源码，哈希由原fixture验证；没有GPU、tensor或性能资格。原`event_contract.json`与包保持只读，`single-victim-event.json`的proposal替换标记仍需与native实际forced事件核对。
