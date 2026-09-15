# 可重跑命令

工作目录：`/private/tmp/moe-window-main-20260914`。输出目录须不存在；原 raw 只读。

```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_effective_recovery_service.py --workspace '/Users/leandrozhao/Desktop/、++++++++' --output-dir refine-logs/expert_saturation/outputs/admission_capacity/20260914_service_window_r01/lifecycle
cd refine-logs/expert_saturation/experiments/admission_capacity
python3 -m unittest test_effective_recovery_service test_service_window_model -v
```

8 格来源为 `20260914_ltr_packing_r01`、`20260914_restore_completion_r01` 的 `execution/readback/results`。输出 JSON 保留输入 raw 的绝对路径与 SHA256；不复制大 raw、不更改原件。CPU 测试是人为夹具，仅验证计费逻辑，不作科学测量。

模型实例，从 worktree 根目录执行，输出文件同样不得已存在：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260914_service_window_model_r01/run_examples.py --source-root '/Users/leandrozhao/Desktop/、++++++++' --output /private/tmp/moe-window-examples-verification-20260914.json
```

合并验证实际结果：10/10 单元检查通过；实例支持 mixed-4，拒绝未摊销 mixed-1 与 peer 等待越界 solo-4；低 KV 反例的全部候选无合格项；两个真实 step406 证书复现。该实例输出与 model bundle 原 examples.json 完全一致。

新六格回传后的实际接续命令（不修改原8格输出）：

```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_effective_recovery_service.py --workspace '/Users/leandrozhao/Desktop/、++++++++' --campaigns 20260914_restore_token_reservation_r01 --expected-cells 6 --output-dir refine-logs/expert_saturation/outputs/admission_capacity/20260914_service_window_r01/lifecycle_reservation
```
