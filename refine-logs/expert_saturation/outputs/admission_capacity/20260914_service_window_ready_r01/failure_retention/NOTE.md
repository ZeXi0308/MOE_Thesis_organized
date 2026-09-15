# 重复保存runner：失败时保留请求原件

状态：**PATCH_CPU_VERIFIED / ORIGINAL_STAGED_PACKAGE_UNCHANGED / GPU_UNRUN**。

原A组`20260914_repeated_staged_probe_r01/pkg/run_probe.py`在capture返回后，先drain、卸载observer、检查至少两次动作，最后才写`raw.json`。如果前述操作抛错，外层异常处理只写status，全部请求与部分输出可能不落盘。这是后续失败证据保留问题；原已完成单事件两臂的raw均存在，其结果不因此失效。

`preserve_capture.patch`在独立worktree已实际应用：capture前`raw=None`，以外层finally在capture已经返回时写raw一次。成功路径仍先完成drain及策略检查再写大JSON，避免原件落盘耗时人为推迟drain。它不修改调度、保存/加载、安全检查、请求时间或错误传播；文件写失败和进程被终止不在该补丁的保证内。未改主树/远端已暂存21文件包，原包SHA6f412e39…f2c3保持不变。

两个CPU异常fixture执行实际runner原文片段：drain抛错、applied_rotations不足。原片段两例均不写raw；补丁片段两例均恰写一次，保留两个请求、既有token与observation_end。root另验证测试片段逐字属于原/实际应用补丁后的runner，完整源码能解析。此检查不代表完整native/GPU资格。

原runner SHA256：`38c6c7ff04a01871a0d12330f1550ee73d16fbb74330f61281d0587d460f8477`。

独立修正版SHA256：`cb40664607ce2248abc0779d599d5580e38f523bb5f14c3aca5ff9d31b744172`，路径为`/private/tmp/moe-window-measurement-20260914/refine-logs/expert_saturation/outputs/admission_capacity/20260914_repeated_staged_probe_r01/pkg/run_probe.py`。

复跑CPU检查时使用新目录，保留原输出：

```bash
retention_probe_dir=$(mktemp -d /tmp/recovery-retention.XXXXXX)
cp fixture_check.py runner_fragment_before.py runner_fragment_after.py source.sha256 "$retention_probe_dir/"
python3 "$retention_probe_dir/fixture_check.py"
```

下一步由原A执行方在接续启动前将此修复纳入明确的新包版本并保留旧包；不要原地改变已登记的冻结包后仍沿用旧SHA。root不创建另一个driver或抢占B Qwen窗口。本补丁没有升级重复保存机制结论，也没有新增GPU矩阵。
