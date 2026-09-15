# Commands

CPU测试：
```sh
cd refine-logs/expert_saturation/experiments/admission_capacity
python3 -m unittest test_native_capture_eos test_host_budget_observed test_analyze_streaming_recovery test_effective_recovery_service
```

准备源入口`prepare_streaming_recovery.py`；封包内全部命令和文件hash见`preparation/pkg/run.sh`及`SHA256SUMS`。open adapter测试需封包pkg优先于主experiments目录出现在PYTHONPATH，并复用verify_headroom_fast/verify_rotation_native及固定runtime源码。

远端暂存/执行统一入口（run只在前序组终态释放后执行一次）：
```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260914_streaming_recovery_r01/execute.py stage
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260914_streaming_recovery_r01/execute.py run
```

结果统一回读后：
```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_streaming_recovery.py --results-root refine-logs/expert_saturation/outputs/admission_capacity/20260914_streaming_recovery_r01/execution/readback/results --inputs refine-logs/expert_saturation/outputs/admission_capacity/20260914_streaming_inputs_r01/frozen --output-dir refine-logs/expert_saturation/outputs/admission_capacity/20260914_streaming_recovery_r01/analysis
```


完成后说明：原四格现已COMPLETE，以上stage/run命令为执行记录，不要再次执行。原件统一回读至execution/readback；首次分析输出analysis目录已存在，复算必须选择新的输出目录，保留原结果。

本次资源/EOS补充命令（输出已存在；脚本以exclusive create防止覆盖）：
```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260914_streaming_recovery_r01/summarize_resources.py
```
该命令读取host receipts、实际pool、decision和raw末尾EOS ID，写analysis/resource_summary.json；不运行GPU或修改raw。
