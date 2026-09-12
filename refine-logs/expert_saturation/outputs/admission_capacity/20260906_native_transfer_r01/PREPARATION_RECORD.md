# 原生 vLLM 并发响应：执行准备

2026-09-06。HEAD `2a37765`，新增实验实现尚未提交。

**状态：`GPU_CAMPAIGN_STARTED / RESULT_PENDING`。用户已明确批准上传与运行，正在执行原生对照。**

本轮问题是：在原生 vLLM 中，cap 6/8 是否实际约束并发并产生可重复的请求级响应？
前序 custom runtime 的相同输入与 U/C 轨迹仍有排名反转，因此先验证运行时和容量边界，
不增加专家感知 predictor。

## 已完成

- 复用原始 16 条真实文本、128-token prompt、16-token 固定输出、模型 revision 和到达序列。
- 原生入口及采集 helper 已实现：所有到期请求进入 native FCFS 队列；记录真实 scheduler
  running/waiting、逐请求 prefill/decode token 数、内部 ID 映射与 preemption。
- 请求到达、提交及 host 输出接收使用统一时钟；core 时间单独保留。累计输出按真实
  chunk 接收时间记录，不插值生成 token 时间。各 episode 独立执行未来状态。
- 预定 cap 6/8/8/6 四个新进程；每个进程两个到达强度 × steady/bursty × 两次反序重复，
  共 32 个 episode。先保留原始 TTFT 5 秒、mean TPOT 0.2 秒；全通过时需要另行校准，
  不据此宣布控制收益。运行前设计见 [DECISIONS.md](DECISIONS.md)。
- `unittest` 三项定向检查通过，覆盖到期提交/实际调度计数/ID、累计 chunk 时间和
  metrics 快照、prefix 错误保留与采集 wrapper 恢复。没有运行全仓测试。
- [分析入口](analyze_native.py) 已完成，`--help` 通过；将从 raw 重算请求指标，汇总
  actual active/decode/wait、调度 token、preemption 与 chunk，并保留 ABBA 各次对照。
  client submission lag 与 native queue 时间分列；未生成任何不存在的测量值。
- SSH ControlMaster 已核实存活；远端只读检查时 GPU 没有计算进程，也没有已启动的
  native capacity campaign。该检查只代表检查时刻，不授权并发占用其它任务。
- 执行包已在本地生成：四个实验 Python 文件、设计记录及 prepared config/workload，
  共七个文件、24,106 字节压缩包；不包含 SSH 凭据、模型权重或其它仓库目录。

## 传输审批与首次启动历史

自动审批两次拒绝 SCP 上传，理由是未识别到可信用户对这一批代码/请求数据及目的
主机的具体授权。第二次提交前已核对完整文件范围并说明本轮 GPU 任务授权，仍被拒绝。
随后用户明确回复“我允许”，并再次逐项确认上传与运行。审批已通过，没有绕过限制。

计划目的目录是用户指定 GPU 主机上的
`/root/autodl-tmp/moe-native-capacity-20260906-r01/`，使用已有 vLLM 0.26 环境和模型缓存。
首次启动 queue PID 5872 / child 5873 已退出，0 个 serving episode。失败日志和状态保留在
[attempt 1](gpu_attempt01_results/)：FlashInfer 在初始化 dummy sampler 的 JIT 架构检查失败。
实际 GPU 为 SM12.0；Torch CUDA 为 13.0，但 FlashInfer 找到的系统 nvcc 为
`/usr/local/cuda/bin/nvcc` 12.8，低于其 SM12 所需的 12.9。

使用官方 `VLLM_USE_FLASHINFER_SAMPLER=0` 开关修复兼容性，保留编译、FA2 和 Triton MoE。
输入、SLO、cap 和重复顺序不变。第二次启动前再次确认 GPU 空闲，使用新目录
`/root/autodl-tmp/moe-native-capacity-20260906-r02/` 和 [顺序执行脚本](run_campaign.py)，
queue PID 6631。实际 temperature=0 请求仍走 greedy 分支；初始化失败不算科学负结果。

## 结论范围

在取得原生 episode 输出前，仅证明本地采集与执行准备完成；真实 scheduler capture、并发暴露、
计时开销、SLO 区间及 cap 排名均未验证。固定 cap 6/8 是待运行的简单对照；动态 Oracle、
U/C 增量和在线机制未运行。此次上传阻塞不是实验失败，也不是科学 NO-GO。

唯一下一步：完成已声明的 32 个原生 episode，先判断实际并发/排队和
SLO 是否有区分力，再决定是否需要单独的负载或 SLO 校准。
