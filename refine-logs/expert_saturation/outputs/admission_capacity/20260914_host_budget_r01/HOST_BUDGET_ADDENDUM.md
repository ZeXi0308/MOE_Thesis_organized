# 当前八格缺少实际主机预算，适合在外层执行器补齐

2026-09-14；只读检查，无 GPU/远端修改、无共享代码修改。核对对象为 `20260914_recovery_holdout_comparison_r01` 的 cohort3 八格，包 SHA `b7557c2e6e65f3d7b2a316f2c16ac04990dc1f347425e26a63b0641fb018516d`；检查时 [execution.json](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_holdout_comparison_r01/execution/execution.json) 为 STAGED。原报告 CPU_PREPARING 是较早状态，尚无本组结果。

**该包没有进程树主机预算记录或执行限制。** [run.sh](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_holdout_comparison_r01/preparation/pkg/run.sh) 只设置整组 GPU flock、逐格 GPU 空闲检查与 600 秒 timeout；[run_probe.py](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_holdout_comparison_r01/preparation/pkg/run_probe.py:131) 固定 GPU KV、禁用引擎 multiprocessing，未配置 host offload。[memory_snapshot](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_holdout_comparison_r01/preparation/pkg/memory_telemetry.py:16) 的参数/KV存储和 CUDA allocated/reserved/peak 全是设备存储；`memory-*.json` 文件名不代表已记录主机 RSS。禁用引擎 multiprocessing 也不能证明整个 Python/CUDA 编译进程没有子进程。

## 已有 raw 能补出的唯一主机量

复用 [反序 offload 的 host samples](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_offload_decode_trace_reverse_r01/readback/results)，从完整 `/proc/<pid>/status` 提取；没有执行新实验。

| 原件 | 根 PID / 样本 | 最大 sampled VmRSS | 最大 observed VmHWM | VmLck / VmPin / VmSwap |
|---|---:|---:|---:|---:|
| trace-offload-on-host.json | 47496 / 98 | 21,298,806,784 B（19.8361 GiB） | 21,599,191,040 B（20.1158 GiB） | 全部观测为 0 |
| trace-offload-off-host.json | 47780 / 70 | 6,173,372,416 B（5.7494 GiB） | 6,447,669,248 B（6.0049 GiB） | 全部观测为 0 |

这些是该单根进程的原始观测：VmHWM 包含此前初始化/预热，而非只含 measurement；最后样本之后的峰值未知。没有后代 PID 或 cgroup memory 数据，不能给出整个进程树峰值。RSS 不能与 host KV 容量、pinned 字节相加；这些集合可能重叠。VmPin/VmLck 为零不能认证本 CUDA 后端没有 pinned allocation，当前 raw 未验证这些内核字段覆盖 CUDA 的全部主机分配路径。[/proc 字段定义](https://docs.kernel.org/filesystems/proc.html)

此前 native on 的 **16 GiB** 是 connector `cpu_bytes_to_use` 配置；**12 GiB store / 2.25 GiB load** 是累计传输量，既不是缓存驻留峰值，也不是进程树内存。当前 recompute 八格 host KV offload 容量为未启用，仍需给 Python、原始 trace、模型加载缓存、编译子进程及运行库计费，不能称 host 成本为零。

## 最小接入：冻结包不变，只扩外层 driver

主预算定义为 `host_charge_budget_bytes=H`，范围是**本次 campaign launcher 及其全部后代的 cgroup memory charge**；并列报告 sampled 进程树 RSS，名称不混用。H 由执行方按已分配环境在运行前写定，全部八格一致；不能由新结果事后反选。若需要一个预备数值，32 GiB 可作为待现场容量核验的提议，依据是旧 offload 根进程已观察约 20.12 GiB HWM；它不是现有主机已授予的容量，也不是由这些旧样本证明足够的上界。

在有已委派 memory controller 的 cgroup v2 子树内，为整组创建**仅包含本组进程**的专用子组。在启动 Python/模型与其子进程之前放入该子组，写入并读回 `memory.max=H`；显式记录统一的 `memory.swap.max`，如实验定义不允许使用系统 swap，则在该专用组设为 0。保持 CPU、GPU、KV、offload、prompt 保存范围、全部策略参数不变。现有整组 flock 不变；不移动运行中的其它会话，不更改共享父 cgroup。`memory.max` 不是纯 RSS，它计入该组承担的匿名、文件缓存及部分内核内存；短暂超限和资源分配失败仍需记录。[内核 cgroup v2 定义](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory)

执行前、约每秒、执行结束记录 `memory.current`、可用时的 `memory.peak`、`memory.events`、`memory.stat`、`memory.swap.current/max`、本组成员及读错误；外层监控留在组外，以便 OOM 后仍能落盘。`memory.peak` 存在时可避免把稀疏采样最大值冒称全程峰值。所有格使用相同封套，分别保留根 PID/开始时间和对应样本；整组 cgroup 的峰值必须标作整组峰值，不冒充每格峰值。结束后先保存计数再清理该空组。达到限制或失败时保留原件并停止本组，不自动降 host KV 或改策略重试。[内核 memory.current/max/peak/events 语义](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files)

如果当前容器没有可写的专用 cgroup，仍可只读采样现有组与进程树，状态须写 `HOST_BUDGET_OBSERVED_ONLY / NOT_ENFORCED`；若现有 cgroup 混有其它作业，其 charge 不能归因本实验。1 Hz RSS 检查只能发现已采到的越界，不能保证两样本间不超限；不能把它称为硬预算。该权限/现场状态本会话未查远端，不宣称可用或不可用。

## 标准库 helper 的准确边界

有必要做一个外层约 100 行的 `host_budget_envelope.py`，但当前不另写重复 observer。现有 [host_environment_samples.py](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/experiments/admission_capacity/host_environment_samples.py) 只采直接子 PID 的 status、全机 cpuinfo/CPU cgroup，未采进程树或 memory cgroup，可由执行方复用其 Popen 生命周期并缩小字段。

建议接口：`run_budgeted(command, cgroup_dir, budget_bytes, sample_period_s=1.0, output_jsonl)`。只用 `subprocess/os/pathlib/time/json`；子启动器先加入指定新组再 `exec` 原 `bash pkg/run.sh`，使随后产生的后代继承同组。监控按成员列表读取 PID 与 `/proc/<pid>/stat` 的 starttime，采 `VmRSS/RssAnon/RssFile/RssShmem/VmHWM/VmLck/VmPin/VmSwap`；记录新增/退出/读失败，防 PID 重用。若无 cgroup，仅用 PPid 遍历识别树，明确短命/重父化进程可漏采。RSS 求和可能重复共享页，不能叫唯一物理驻留；`smaps_rollup` 的 PSS 只在阶段边界诊断，不在每步热路径读取。

helper 不 import torch、不同步 GPU、不遍历 KV 或 request 对象，不新增每步事件。逐次记录 sampler 墙钟/CPU时间、实际采样间隔和错误，流式写 JSONL，不累积全程 samples 列表；监控自身的组外 RSS/CPU另报。采样成本现在 UNMEASURED，不能预称零开销。A 正在处理的低分配 `memory_telemetry` 是另一个范围；不把本任务塞进其 scheduler hook。

本轮可下的结论只有：现有 GPU 同预算证据成立；**实际主机进程树预算尚未被这些数据满足**。最小补齐是执行器层统一 cgroup 封套与稀疏外部采样，保留全部 KV 能力和冻结策略。是否启动该封套、H 的实际数值及现场委派权限由唯一 GPU 执行方落实，本会话未改包或启动任何进程。
