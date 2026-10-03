# 共同引擎下的原生接纳对照：准备完成

2026-09-06。HEAD `2a37765`，代码与实验产物未提交。

**状态：`UNRUN / MODIFIED_SOURCE_TRANSFER_APPROVAL_PENDING`。本轮没有新增 GPU 结果。**

本轮问题是：前序原生 cap 8 的优势，在共同引擎/编译配置下，仅改变接纳上限时
是否仍存在？上一轮的 max_num_seqs 同时改变了 CUDA graph 捕获计划，不能把其静态
配置收益直接写成接纳动作收益。

## 已完成的最小实现

- 原生入口新增 `--engine-max-seqs`，本轮固定为 8，两臂接纳 cap 为 6/8。
- `set_empty_admission_cap` 只修改 scheduler.max_num_running_reqs，并同时要求 frontend
  无未完成请求、native running/waiting 为零、requests 映射为空。保持 engine/worker/
  graph 配置不变；不在已有 8 running 时直接降为 6，避免触发 native scheduler 断言。
- 增加实际条件 warmup 及明确 `WARMUP_START/END`、`CELL_START/END` phase 标记，
  区分初始化后的 JIT 告警是否落在测量中。warmup summary 保留，不冒充测量 episode。
- 支持预先声明的主 SLO，同时从相同 raw 保留旧 SLO 的参考指标。配置记录两套阈值，
  不修改 prepared 输入或前序结果。
- 两个新增定向测试与原三个 capture 测试共 **5 项通过**；覆盖只在空引擎设置、拒绝
  活跃/排队/非法 cap 且不修改配置。入口及分析器 `--help` 通过；没有运行全仓审计。
- 分析器按四组 plans 总数判断完整性，兼容此次 16 个 episode；仍保留旧 32 次标记
  的原义。配对时核对共同 engine_max_num_seqs，不把不同编译容量当作相同配置。

## 冻结的执行内容

同一批 16 条真实文本，128 prompt / 16 output tokens，原有 pinned OLMoE BF16、
同步 native V1、compiled/FA2/Triton MoE。两个快速到达模式均为 scale .02、0–.03 秒。
四个新进程按 6/8/8/6 执行，每进程两种到达 × 两次反序重复，共 **16 个测量 episode**。
每策略独立执行后续 KV、请求队列、batch、route 和完成时间。

主 SLO 为 TTFT≤0.20 秒、mean TPOT≤0.009 秒，来自上一轮合并 p75 的向上取整校准；
同时记录原 5 秒 / 0.2 秒参考结果。这是探索阈值，不是生产承诺。
完整设计见 [DECISIONS.md](DECISIONS.md)，精确命令见 [run_campaign.py](run_campaign.py)。

## 实际阻塞与权限边界

启动前远端只读检查确认 GPU 空闲、无相关实验进程且本轮目标目录不存在。
本地实验包已准备；两次上传均被自动审批拒绝，后续启动命令没有执行。

第二次提交前已比较两个归档：文件路径完全相同，prepared config/workload 与 metrics.py
逐字节未变，没有新增研究输入或凭据文件；代码变化仅为空引擎接纳上限、SLO 参数及
warmup 阶段日志，另有本轮设计记录。尽管用户更新目标授予继续实验的总权限，自动
审批仍认为此前普通回复仅覆盖上一批代码，要求对更新代码的传输明确授权。
没有改用其他传输方式绕过拒绝。

计划主机仍为用户指定的 `root@connect.westd.seetacloud.com:37116`；新目录为
`/root/autodl-tmp/moe-native-fixed-engine-20260906-r01/`。本轮需要传输的仍只有实验
代码、设计/配置和已批准的 16 条研究请求，没有其它工作区材料。

## 已有数据上的执行路径定位

等待传输期间，仅阅读与本轮问题直接相关的 dispatcher 源码及前序 native raw。
本地 stock 与远端已安装的 vLLM 0.26 `v1/cudagraph_dispatcher.py` 内容一致。
该文件 209–231 行按 `max_num_seqs` 限制 FULL decode 捕获尺寸，132–155 行把实际
token 数补齐到下一捕获尺寸，307–318 行先匹配 FULL，再尝试 PIECEWISE。

在此前记录的 FULL_AND_PIECEWISE 配置、单 token uniform decode 下：

| 引擎 max_num_seqs | 已准备的 FULL decode 尺寸 | 实际 decode width | 补齐后 tokens | FULL key |
|---|---|---:|---:|---|
| 6 | 1/2/4 | 5 或 6 | 8 | 缺少；源码会继续查 PIECEWISE |
| 8 | 1/2/4/8 | 5–8 | 8 | 存在；允许 FULL 时优先匹配 |

这不是仅存在于配置中的边缘情况。前序八个快速 cap6 cells，各有 43 个纯 decode
step，其中 28 个为 width6（65.12%）；合计为 224/344。八个快速 cap8 cells 各有
29 个纯 decode step，其中 28 个为 width8；合计为 224/232。纯 decode 的筛选只要求
scheduled 非空，且每个请求 `scheduled_tokens=decode_tokens=1、prefill_tokens=0`。
这些是重复轨迹的调度步数，不是独立 workload 数、时间占比或可消除延迟上界。

因此已有证据支持 `STRUCTURAL_CONFIG_DIAGNOSTIC`：两个 cap 的比较触及了不同的
FULL graph 覆盖范围。由于原 raw 没有逐 step 的实际 graph mode，也没有开销隔离
对照，仍不能声称已经测到 graph 切换成本，或把 16%–59% 全部归因于这个差异。
固定引擎容量为 8 是下一次接纳对照的必要控制变量，本轮不新增 predictor 或测试。

## 结论与唯一下一步

当前仅完成了实际可执行的控制变量分离及其本地检查。真实引擎中的 graph 配置一致性、
实际并发与排队、新 SLO 达标分布、固定 cap 排名尚未测量；没有新的 U/C 方法证据，
没有动态 Oracle。这是传输审批阻塞，不是实验无效或科学 NO-GO。

唯一下一步：传输获准后执行已冻结的 16 个 episode，从实际记录核对共同引擎配置
及请求结果。如果结果变号，先保留并做一次同配置受控复测，不增加 predictor。
