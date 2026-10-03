# 外层主机预算 helper：本地实现完成，远端未运行

新增 [host_budget_envelope.py](../../../experiments/admission_capacity/host_budget_envelope.py)（193 行）和 [8 项必要测试](../../../experiments/admission_capacity/test_host_budget_envelope.py)，仅在本 worktree。没有上传、修改冻结包、修改 A observer 或启动 GPU。

实现行为：

- 预算与 swap 字节必须显式提供，没有 32 GiB 默认值。只接受已有 cgroup2 memory 委派父路径，只新建随机命名 owned 子组；不启用父 subtree_control，不接管现有组或 PID。
- 创建子组后写入并读回 `memory.max`、`memory.swap.max`；无委派、权限失败或读回不符都落 `NOT_ENFORCED` 收据，目标命令不启动；没有静默降级模式。
- 独立 Python bootstrap 只把自己放入新组，读回成员身份后记录 `membership.json`，然后 exec 原命令；目标及其随后创建的后代继承组。外层 sampler 留在组外。
- 每秒流式写 `samples.jsonl`：charge current/peak/events/stat/swap、owned 子树成员 PID/starttime、各 RSS 字段、sampler 自身 RSS/采样成本。RSS 求和与 cgroup charge 字段分开。日志及终态/失败收据保留。
- 超时/异常只清理本次 Popen 子进程及 exclusively-created cgroup；清理不支持/失败记录 `cleanup_error` 并留下路径，不递归删除文件树、不杀其它会话。

本地执行：`python3 -B -m unittest -v test_host_budget_envelope`，**8/8 PASS**。覆盖预算读回/父组不变、无委派与读回错误时不启动、启动失败留证、失败命令的流式数据与 returncode、先入组再 exec/创建后代、入组失败禁止执行，以及 populated/frozen 字段分离。预算文件是模拟文件树；bootstrap 顺序用本地真实 Python 子进程和模拟 cgroup.procs 检查。**这不是 Linux cgroup enforcement、CUDA 或内核继承实测。**

集成位置是外层唯一 driver 包装原 `bash pkg/run.sh <现有Python路径>`，不改变冻结包。`--parent`、`--budget-bytes`、`--swap-bytes`、`--output` 都必须明确给出；timeout 是完整 campaign 的外层超时，需要执行方按整组规模设置。CLI 示例只展示接口，不是授权运行命令：

```text
python host_budget_envelope.py --parent <已委派的专用父组> \
  --budget-bytes <已分配字节数> --swap-bytes <固定swap字节数> \
  --output <全新收据目录> --timeout-s <完整组超时> -- \
  bash <原冻结pkg/run.sh> <现有vLLM的Python>
```

该预算精确定义为 cgroup charge，不保证与全部进程 RSS 或共享页的唯一物理归属相等；文件缓存 first-charge、bootstrap 入组前的少量分配及驱动记账边界仍按内核机制处理。`memory.peak` 缺失时明确记录读错误，sampled 最大值不升级为全程峰值；VmPin=0 不视为 CUDA pinned 为零。尚未测量此 wrapper 的 CPU/墙钟扰动。

Root 本轮只读核验的当前远端环境：`/sys/fs/cgroup` 为只读 cgroup2 mount、W=False、subtree_control 为空，无可写 delegated 子组。父容器 `memory.max=96,636,764,160`（90 GiB），属于整个容器；其 current 约 31.7 GB 不能归因本实验树。本 helper 在该条件下应返回 NOT_ENFORCED，不尝试 remount 或修改父组。该环境判断由 root 实地读取，本子任务未访问远端。

Root 同时确认新 cohort3 八格已经 COMPLETE，原 PID 49026/49027 已退出；**不能给已完成原件补写“受此预算限制”**。当前可报告父容器 90 GiB ceiling，具体实验进程树成本仍未测；本代码仅作为下一次有可写委派环境时可审阅的复用工具。
