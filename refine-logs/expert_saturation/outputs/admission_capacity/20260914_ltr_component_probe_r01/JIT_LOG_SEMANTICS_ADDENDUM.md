# LTR component probe：JIT 日志语义勘误

日期：2026-09-14。性质：对既有日志解释的纠正；没有新增 GPU 运行、重算指标或更换 canonical 结果。

**纠正结论：旧日志不支持“fused-MoE JIT 发生于 measured episode 内、其成本属于 measured engine wall”的断言。** 四条警告只证明：引擎自身初始化/预热结束、监测器激活之后，当前进程遇到了 `fused_moe_kernel` specialization 的进程内缓存缺失路径。事件属于哪段应用预热或 measured episode、是否重新编译机器码、发生次数与耗时，旧日志均未确定。也不能反向断言 measured episode 内没有这类事件。

## 原日志顺序与测量边界

四个文件均位于 `execution/readback/results/`。以下为日志原样时间标签；行号按 LF 换行计数，与 `rg -n` 一致，CUDA graph 进度条中的回车不另计行。

| 日志 | 监测器激活，第 46 行 | 引擎初始化完成，第 47 行 | kernel 警告，第 49 行 |
|---|---|---|---|
| `block0-d6-boost-off.log` | 02:24:30 | 02:24:31 | 02:24:31 |
| `block0-d6-boost-on.log` | 02:25:43 | 02:25:43 | 02:25:43 |
| `block1-d6-boost-on.log` | 02:26:54 | 02:26:54 | 02:26:55 |
| `block1-d6-boost-off.log` | 02:28:06 | 02:28:07 | 02:28:07 |

第 49 行使用通用文案 `Triton kernel JIT compilation during inference: fused_moe_kernel`，没有应用阶段、request ID、scheduler step、specialization 或编译耗时。其紧邻初始化完成的顺序不足以把事件归入 measured episode，也不足以确定某一个具体 warmup。

本地封存源码 `preparation/pkg/run_probe.py:148–159` 在引擎创建后依次执行三个应用 warmup：short/32 requests/cap16、short/32 requests/cap32、long/2 requests/cap2，并分别保存原件。只有这些 warmup 完成后，`:169–171` 才以 `run_id='measured'` 执行压力 episode。因此监测器所说的 “during inference” 比本实验的 measured 区间更宽。`RESULTS_ADDENDUM.md:7` 所述三次预热在请求 episode wall 之外，与这一边界一致；“已预热”只表示执行了已列预热，不证明覆盖所有 specialization。

## 实际读取的源码与警告语义

源码来源分为两类，不能混作本地历史快照：

- 本地封存文件根目录为 `/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_ltr_component_probe_r01/`，上文的 `preparation/pkg/run_probe.py` 已直接读取。
- 本地已有安装源码缓存 `/private/tmp/moe_vllm026_installed/engine/arg_utils.py:655–656,1843–1844` 定义并传递 `jit_monitor_mode`、`jit_monitor_verbose`；`/private/tmp/moe_vllm026_installed/model_executor/layers/fused_moe/fused_moe.py:295–353,781–850` 显示同名 kernel 接受多项维度、stride 和 constexpr/config，kernel 名本身不是完整 specialization 身份。
- 监测器与 Triton 源码通过既有 SSH 连接直接读入标准输出，没有落成本地源码文件。远端为 `root@connect.westc.seetacloud.com:53036`；以下 `V` 指 `/root/autodl-tmp/expert-saturation/vllm-0.26/lib/python3.12/site-packages/vllm`，`T` 指同一 `site-packages` 下的 `triton`。这是本次只读源码核查，不是新运行的证据。

| 实际读取位置 | 支持的语义 |
|---|---|
| `V/utils/jit_monitor.py:51–79` | `activate()` 在 worker 的 `compile_or_warm_up_model` 结束时激活；没有识别本实验后续三次应用 warmup 与 measured episode。 |
| `V/utils/jit_monitor.py:131–135` | 非 verbose 模式使用 `logger.warning_once`。相同 kernel 的后续事件可被去重；四个进程各一条可见警告不是各一次事件的计数。 |
| `V/utils/jit_monitor.py:211–229,233–254` | 事件来自 `knobs.runtime.jit_post_compile_hook`；只有 verbose 才格式化 constexprs、signature、compile extras 和 key。旧环境记录为 `jit_monitor_verbose=False`。 |
| `T/runtime/jit.py:707–720` | 先查询当前进程/设备的 kernel cache；缺少该 specialization 时进入 `_do_compile`。 |
| `T/runtime/jit.py:826–852` | 同步路径调用 `self.compile(...)` 返回后更新进程缓存，再调用 `jit_post_compile_hook`；该层不区分磁盘命中与实际代码生成。 |
| `T/compiler/compiler.py:261–276` | 磁盘 metadata cache 命中时直接构造并返回 `CompiledKernel`，仍可回到上一行的 post-compile hook。因此警告不能单独证明重新生成 GPU 机器码。 |
| `T/compiler/compiler.py:227–229,268–275,355–358` | 编译 listener 才分别提供 `cache_hit=True/False` 和阶段时间；旧四条警告没有这些信息。 |

警告文案中的 latency spike 是监测器的通用提示，不是本实验测得的 spike 时长或因果归属。现有材料不能把 wall 波动、重复间变号或某个长输出 gap 全部归因于 JIT。

## 对保留审计的纠正范围

已只读检查 `audit/EXPERIMENT_AUDIT.md` 与 `audit/EXPERIMENT_AUDIT.json`。本 addendum 对以下解释具有纠正优先级，原文件保留不改：

1. `audit/EXPERIMENT_AUDIT.md:73` 的 “That event belongs to the measured engine wall” 未获支持，撤回该时间区间归属及据此作出的 JIT 原因解释。四条警告文本存在这一事实保留。
2. `audit/EXPERIMENT_AUDIT.md:84` 的 “No correction is required” 不适用于上述日志解释；其中 “in-episode JIT/runtime variance” 不能继续作为已定位事实。实际运行时间存在差异与其具体成因是两个问题。
3. `audit/EXPERIMENT_AUDIT.json:161`，`bounded_observations[3]` 中由 “during inference” 警告直接推出本次请求测量未预热的归因被本 addendum 限定。该字符串只可作为警告原文描述，不能当作 measured 区间的编译记录。

本次纠正不改写 raw/metrics/输出/动作轨迹、所有运行保留事实、实际 quantum 返回新 token 的观察、两次重复的符号变化，或当前 `MEASUREMENT_ONLY` claim ceiling。缺少统计显著性、稳定性能、质量与完整 LTR/Oracle/方法 GO 证据的边界继续成立，但不能再用未定位的 JIT 警告充当其已知原因。本文件不提出或执行新的实验。
