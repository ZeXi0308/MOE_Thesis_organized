# 共同引擎下的原生接纳对照：16 个 episode 完成

2026-09-06。HEAD `2a37765`，分支 `agent/publish-current-moe-code`；本轮未提交、未 push。

**结论：固定相同引擎配置后，cap 8 在全部 8 个配对中仍有更高完成吞吐和 SLO-goodput。**
因此，前序不同图捕获配置不能单独解释 cap 8 的优势。本轮支持当前有限请求负载下的
静态接纳响应；没有证明 U/C、动态控制或论文方法成立。

| 裁决字段 | 本轮结论 |
|---|---|
| Verdict | `MEASUREMENT_ONLY / FIXED_ENGINE_STATIC_ADMISSION_RESPONSE_OBSERVED` |
| Evidence type | `NATIVE_SERVING`：单 RTX 5090、vLLM 0.26 同步进程内 host capture；不含 HTTP/network serving |
| What was measured | 固定引擎容量 8、接纳 cap 6/8，steady/bursty 两种快速到达，共 16 episodes、256 次请求执行 |
| What was not measured | U/C 增量、在线调节、动态 Oracle、长期稳态容量、任务质量、第二模型、多卡 EP |
| Strongest baseline | 已测动作集合 `{6,8}` 内为固定 cap 8；不是所有 cap 的全局最优 |
| Oracle/headroom status | `UNRUN`；静态对照差距不是超越最佳静态策略的 Oracle headroom |
| Claim ceiling | 当前模型、输入和有限到达 episode 中的静态接纳结果，不外推生产 SLO 或 CCF-B 方法 GO |
| Failure category | 运行失败 0；原配置混杂已在本轮分离，跨 workload 泛化与动态增量仍未测 |
| Resurrection condition | 未判死 problem family；若更小 cap 在新真实负载中形成稳定优势，才重开 6/8 自适应选择的存在性问题 |
| One next smallest experiment | 冻结引擎、到达、长度与 SLO，在一组未使用的 16 条真实请求上复跑同一 16-episode 对照 |

## 实际执行与比较口径

用户明确同意更新代码上传后，原 7 文件包（24,885 bytes）成功传至指定主机
`root@connect.westd.seetacloud.com:37116`。队列 PID 11053 顺序执行 `6/8/8/6` 四个新进程，
每个进程 2 种到达 × 2 次反序重复；全部以 exit 0 结束。完成后 GPU 计算进程及相关
实验进程均为空。远端目录为 `/root/autodl-tmp/moe-native-fixed-engine-20260906-r01/`。
此前的准备和传输阻塞记录原样保留在 [PREPARATION_RECORD.md](PREPARATION_RECORD.md)。

两臂的 `engine_args.json` 完全一致：引擎 `max_num_seqs=8`、token budget 1024、
max model length 256、BF16 OLMoE、FA2/Triton MoE、compiled、同步 FCFS、prefix cache 关闭。
只在完全排空时将 scheduler admission cap 设置为 6 或 8，不修改 worker/graph 配置。
这不是已有请求运行期间的在线降 cap 实验。

沿用 16 条真实文本，每条 128 prompt / 16 output tokens。两种到达均为 scale .02，
首尾跨度 30 ms；名称 steady/bursty 描述该有限 episode 内的到达形状，不代表长期稳态。
每个 episode 独立推进 KV、队列、batch 和输出；256 次执行仍只有 **16 个唯一输入**。

主 SLO 在本轮前冻结为 TTFT ≤200 ms、request mean TPOT ≤9 ms，来自前序所有快速
请求测量合并 p75 的向上取整。这是相同输入上的探索阈值，不是业务 SLO 或独立 workload
验证。同时保留原 TTFT ≤5 s / TPOT ≤0.2 s 的参考结果。

`goodput = 联合达标的完成请求数 / observation_end_s`。分母是含客户端提交、native
排队、实际执行、输出交付及内存 trace 开销的 episode host 墙钟时间；初始化、warmup、
episode 外 GPU 状态查询不计入。不将 native core 时钟与 host 相对时钟相减。

## 全部重复结果

下表每行均含 4 个 episode；范围是实际最小值到最大值，不是置信区间。TTFT/TPOT
列为每个 episode 的请求中位数；mean TPOT 达标不保证每个 ITL 达标。

| 到达 / cap | 完成吞吐 req/s | 主 goodput req/s | 每 episode 达标数 /16 | TTFT p50 ms | mean TPOT p50 ms |
|---|---:|---:|---:|---:|---:|
| steady / 6 | 37.98–43.90 | 23.74–32.93 | 10–12 | 136.16–193.03 | 7.349–7.568 |
| steady / 8 | 49.35–60.53 | 46.26–60.53 | 15–16 | 79.27–132.85 | 7.257–7.662 |
| bursty / 6 | 39.30–46.21 | 19.65–34.66 | 8–12 | 124.07–184.08 | 7.291–7.627 |
| bursty / 8 | 50.01–54.93 | 37.51–54.93 | 12–16 | 102.43–128.12 | 7.493–7.737 |

主 SLO 共 **206/256** 达标：cap 6 为 **89/128（69.53%）**，cap 8 为
**117/128（91.41%）**。50 次未达标中，33 次只违反 TTFT，17 次只违反 TPOT，无同时
违反；全部请求均正常完成，不能把 SLO 失败写成运行失败。原参考 SLO 为 256/256 达标，
其 goodput 等于完成吞吐。

按预定 `a0/b0`、`a1/b1` 的同 repeat 配对，cap 8 的完成吞吐和主 goodput 均为 **8/8 胜**。
steady 的 goodput 相对增加 43.29%–128.37%，bursty 为 8.22%–97.47%；该百分比受有限
请求数及离散 SLO 达标数影响，不应写成持续服务容量增益。对应完成吞吐增量分别为
14.63%–57.00%、8.22%–38.61%。

cap 8 的达标数为 7 胜 1 平，TTFT p50 为 7 胜 1 负；TPOT p50 则 cap 6 在 5 个配对
更低，cap 8 在 3 个更低。因此结论是 goodput 优序保持，不能说所有延迟指标一致改善。

## 已关闭的具体测量疑点

- 四份引擎参数逐字段一致；capture sizes 均为 `[1,2,4,8,16]`，FULL decode 均捕获
  4 个图。没有记录逐步实际 graph mode，不能量化图切换成本或归因前序收益份额。
- 所有 16 个 episode 实际 active/decode 都达到各自 cap；调度后等待峰值 cap 6
  均为 10，cap 8 均为 8。这里没有把刚提交、尚未调度的请求全部当作拥塞。
- 每个 episode 实际调度 2,048 prefill tokens 与 240 decode tokens；prefill 产生
  每请求的第一个输出，合计每 episode 256 输出 token。无 preemption、computed
  adjustment 或多 token chunk；全部结果保留，无 token 时间插值。
- 16 份主指标从 raw 重算与保存指标一致；旧 SLO 参考指标也逐项一致。全部 16 份 GPU
  episode 边界检查 PASS；这是边界检查，不保证连续隔离或锁定频率。
- 每进程 3 个 warmup 都完成。日志中监测到的 fused-MoE JIT 警告分别落在 cap 6 的
  warmup 0、cap 8 的 warmup 1；测量阶段未出现这类警告。不能由无警告断言无任何编译开销。
- 四进程使用相同已记录 runtime/source、环境变量和 104 个 Torch CPU threads。
  仍有默认 MoE config 的后端告警；这不构成优化后的所有 vLLM backend 的普遍结果。
- 16 次执行形成 4 种整组输出轨迹，4/16 个请求出现两个 token 序列变体。尚未定位
  first divergence，也未做任务质量评估；不声称两臂输出逐 token 相同。

## 一次定向复核

fresh GPT-5.6-Sol ultra 只读复核四类问题，结论为 **PASS / P0=0 / P1=0**，
`review_independence=same-family`、`acceptance_status=provisional`。检查范围为独立
策略状态/未来信息、request-step-host timing、指标会计、公平简单基线；独立重算
256 个请求及 16 份主/参考指标，并核对分析结果。上述证据边界保留，不因此升级为
方法 GO。没有追加测试、独立审计文件或 `.aris`；本轮审计到此停止。

## 数据与复跑

- [冻结设计](DECISIONS.md) 与 [四进程命令](run_campaign.py)
- [完整原始数据、环境及日志](gpu_results/)；包括全部 16 cells 与 12 个 warmup summaries
- [逐 episode 重算表](analysis/report.md) 与 [机器可读指标](analysis/analysis.json)

分析命令如下，输出目录需为新目录；不得覆盖本次或前序结果：

```bash
.venv/bin/python refine-logs/expert_saturation/outputs/admission_capacity/20260906_native_transfer_r01/analyze_native.py \
  --results-dir refine-logs/expert_saturation/outputs/admission_capacity/20260906_native_fixed_engine_r01/gpu_results \
  --output-dir /private/tmp/moe-fixed-engine-analysis-fresh
```

**本轮问题的回答是“优势仍存在”。** 当前最强简单基线应使用固定 cap 8；下一次只用
一个未使用的真实请求 cohort 检查这一优序是否跨输入保持。在证明可超过这个基线的
动作空间前，不增加 U/C predictor 或 Controller。
