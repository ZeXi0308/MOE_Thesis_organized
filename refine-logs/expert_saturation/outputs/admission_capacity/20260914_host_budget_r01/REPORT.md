# 主机预算：当前只有共享容器90GiB上限

`remote_readonly_probe_attempt03.json` 在远端时间1789331367.7476158成功读取：cgroup2挂载为只读，`cgroup.subtree_control`为空，当前 `/sys/fs/cgroup/memory.max=96636764160`（90GiB）。它约束整个容器，不能把当时memory.current或该父组峰值归因某一实验进程树。没有可写的memory子组委派；未尝试修改父组、remount或移动已有进程。

同次检查确认新cohort3八格group COMPLETE，原PID49026/49027不存在，完成时间1789331335.906304。GPU在该读取瞬间无计算进程，但后续已由其它登记组接续，不能把快照当作保留窗口。本分支没有启动GPU。

前两次只读尝试保留：sandbox中SSH socket不可访问；提权允许后，远端非交互PATH没有python3。第三次使用既有vLLM解释器成功。这些是访问/命令问题，不是科学负结果或自动审批拒绝。

[host_budget_envelope.py](../../../experiments/admission_capacity/host_budget_envelope.py)已完成。它在已有可写委派父组下，仅新建自有子组，显式写/读回host charge与swap预算，子启动器入组后exec，组外每秒记录charge/成员/RSS/失败及监控成本；不改变现有调度器、pager、冻结包或其它会话。无委派时返回NOT_ENFORCED并拒绝目标命令，不静默降级。8项本地测试通过；预算文件模拟测试和真实本地bootstrap顺序不等于Linux硬限制验证。

实现及边界详见[HOST_HELPER_RESULT.md](HOST_HELPER_RESULT.md)。本工具未上传或运行远端，真实Linux enforcement与监控扰动仍UNRUN；当前远端也不具备其所需委派。90GiB父组快照不是所有既有运行均受独立同host预算约束的证据。

既有native offload的16GiB是connector容量，累计store/load是传输量；旧反序trace的根进程observed VmHWM约20.116GiB(on)/6.005GiB(off)也不是完整进程树峰值。不能将这些量相加，不能用VmPin=0证明CUDA pinned=0。原有只读提取说明保留在HOST_BUDGET_ADDENDUM.md，其STAGED状态是较早检查快照，后续状态以本报告及原执行回读为准。

当前结论是明确的资源验证缺口：可以继续CPU模型和已有raw分析，但不能把现有结果升级为独立进程树硬主机预算下的确认性结论。下一实际GPU比较须由唯一执行入口落实预算范围、限额和采样；本分支不创建新的GPU队列项。
