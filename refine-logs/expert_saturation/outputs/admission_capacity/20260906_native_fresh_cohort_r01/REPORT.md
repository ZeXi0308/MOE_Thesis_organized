# 新输入上的固定引擎对照：16 个 episode 完成

2026-09-06，HEAD `2a37765`。全部 16 个 episode、256 次请求执行正常完成。

**cap 8 的完成吞吐和主 SLO-goodput 在新输入上仍为 8/8 配对胜。** 这加强了当前
有限请求负载下固定 cap 8 的简单基线地位；没有证明 U/C 或动态切换的额外价值。

| 裁决字段 | 本轮结果 |
|---|---|
| Verdict | `MEASUREMENT_ONLY / STATIC_CAP_ORDER_REPRODUCED_ON_NEW_INPUTS` |
| Evidence type | `NATIVE_SERVING`：单 RTX 5090 / OLMoE / vLLM 0.26，同步进程内 host capture |
| What was measured | 新 16 条输入、相同 128/16 token 长度与 30 ms 到达跨度、共同引擎 8 下静态 cap 6/8 |
| What was not measured | 长期容量、HTTP/异步 frontend、U/C、动态策略、Oracle、任务质量、第二模型或 EP |
| Strongest baseline | 已测集合 `{6,8}` 中固定 cap 8；不是所有 cap 的全局最优 |
| Oracle/headroom status | `UNRUN`；没有测到超越最佳静态策略的动态 headroom |
| Claim ceiling | 两组不重叠输入上的描述性静态优序；不是文章级独立验证或方法 GO |
| Failure category | 0 次运行失败、62 次主 SLO 未达标；不将 SLO 失败写成执行失败 |
| Resurrection condition | 若新运行域中出现可复现的吞吐—延迟取舍，再资格化最小动作；不凭换名称或阈值复活 selector |
| One next smallest experiment | 固定 cap 8、bursty 到达时刻及长度，交换首批和第二批请求身份，做匹配负控以区分到达阶段与文本身份的影响 |

## 输入与执行没有混在一起改变

原 manifest 的 128 条源行中，只有 48 条符合 128-token 长度条件；本容量研究线已经
使用了这 48 条。新输入从同一缓存 WikiText-103-raw-v1 test Arrow 的 row 205 起
顺序选择，不读 route、logits、时间或结果。实际行号为：

`205,206,210,211,215,219,223,227,231,232,237,238,245,250,252,256`。

16 条请求的文本和 token 哈希各自唯一，且与本线旧 48 条零重叠。它们仍来自同一数据集，
不声称文章级独立或全仓历史从未使用。Arrow SHA 与旧数据相同；当前派生 fingerprint
为 `439d5d8a78474efb`，旧 manifest 为 `1d21d2114e992cd6`，二者不作为原始字节漂移的证据。
Pinned tokenizer 的三个文件哈希均匹配，截断策略和 128-token 长度相同。

本轮发现共享入口已包含后续多 cap 扩展，因此从上一轮**实际执行归档**取出代码，
而没有把共享源码的新改动带入实验。[execution.tar.gz](execution.tar.gz) 保留实际
上传的 7 文件包（25,447 bytes）；三个执行模块的哈希与上一轮 GPU 环境记录完全一致，
campaign 也逐字节相同。离线 [prepare_inputs.py](prepare_inputs.py) 只生成新输入。

引擎参数、软件源、环境和 104 个 CPU threads 与上一轮一致：engine max_num_seqs=8、
FA2/Triton MoE、BF16、compiled、同步 FCFS、prefix cache 关闭。四个新进程仍按
6/8/8/6 执行；只在引擎排空时设置 admission cap。每进程 3 个 warmup 和 4 个测量 episode。
全部未来 KV、batch、队列和输出分别真实执行；没有共享未来 trace。

主 SLO 保持 TTFT ≤200 ms / request mean TPOT ≤9 ms，未用新结果重标定；参考 SLO
仍为 5 s / 0.2 s。goodput 分母保持含 host 提交、排队、执行、输出和 trace 开销的
完整 episode 时间，初始化和 warmup 在外。到达曲线与旧输入逐值、序列化字节相同。

## 新输入的全部重复

每行含 4 个 episode。范围为观测 min–max，不是置信区间；延迟为每 episode 的
请求中位数。256 次执行只有 16 个唯一输入。

| 到达 / cap | 完成吞吐 req/s | 主 goodput req/s | 每 episode 达标数 /16 | TTFT p50 ms | mean TPOT p50 ms |
|---|---:|---:|---:|---:|---:|
| steady / 6 | 40.43–44.27 | 30.32–33.20 | 12 | 136.56–169.99 | 7.348–7.592 |
| steady / 8 | 54.19–60.19 | 54.19–60.19 | 16 | 80.54–107.42 | 7.395–7.594 |
| bursty / 6 | 38.95–44.98 | 14.61–33.73 | 6–12 | 133.30–189.20 | 7.523–7.776 |
| bursty / 8 | 49.45–50.54 | 37.09–37.91 | 12 | 129.21–132.52 | 7.497–7.715 |

主 SLO 共 **194/256** 通过：cap 6 **82/128（64.06%）**，cap 8 **112/128（87.50%）**。
cap 6 有 34 次 TTFT 单项失败、12 次 TPOT 单项失败；cap 8 有 16 次 TPOT 单项失败，
均来自 bursty。无两项同时失败。旧 SLO 仍为 **256/256** 通过。

8 个预定配对中，cap 8 的完成吞吐、goodput、TTFT p50 均为 8 胜；达标数 7 胜 1 平。
TPOT p50 两 cap 各胜 4 次，不能写成所有延迟指标一致改善。

| 输入组 | cap 6 主 SLO 达标 | cap 8 主 SLO 达标 | cap 8 goodput 配对胜数 |
|---|---:|---:|---:|
| 前一组 16 条 | 89/128 | 117/128 | 8/8 |
| 本轮新 16 条 | 82/128 | 112/128 | 8/8 |

优序在两个 cohort 中保持。达标总数从 206 降至 194，不能直接归因为文本，因为两轮
同时跨越了运行时间和进程；这里没有执行新旧输入随机交错的内容因果实验。

## 利用现有 raw 定位 TPOT 失败

cap 8 的 16 次 TPOT 失败全部来自首批四个请求：row 205、206、210、211，均在
arrival_s=0 到达。每个 cell 内，四请求的下列指标相同；最长间隔都是 token 1→2。

| cap 8 cell | mean TPOT ms | 最长 ITL ms | 占 t16−t1 | 配对 cap 6 同四请求：mean TPOT / 最长 ITL ms |
|---|---:|---:|---:|---:|
| b0 / 001 | 9.245 | 37.128 | 26.77% | 9.265 / 37.136 |
| b0 / 002 | 9.341 | 37.189 | 26.54% | 9.152 / 36.109 |
| b1 / 001 | 9.241 | 36.785 | 26.54% | 9.355 / 35.669 |
| b1 / 002 | 9.221 | 37.104 | 26.83% | 7.882 / 14.078 |

其余 14 个间隔平均为 7.23–7.35 ms。四次长间隔内都出现 scheduler step 1：
4 decode tokens + 512 prefill tokens、actual active=8、waiting=8。
配对 cap 6 的同一步为 4 decode + 256 prefill、active=6，但同四请求仍三次 TPOT
超标、一次通过。因此 prefill token 数量本身不足以解释时延差异。

这是首批请求、长 host ITL 与混合 prefill/decode 步的**时间对齐观察**。scheduler
时间戳没有覆盖完整 GPU 执行；没有确认具体 kernel、权重读取或专家负载的因果作用，
也不能把最长间隔的占比当作可实现收益或 Oracle headroom。

本轮问题的回答是：**cap 8 的静态 goodput 优势在新输入上保持。** 唯一下一步先做
上表所指的到达组身份交换负控；如果失败随到达位置迁移，优先研究普通 prefill/decode
阶段的影响；如果稳定跟随文本身份，再定位内容相关执行差异。该负控本轮尚未运行，
当前不增加 U/C predictor 或 Controller。

## 核对与结果留存

输入由第二 agent 独立筛选并复核。主/参考指标又从 raw 独立重算，与保存值及分析结果
一致。复用上一轮已审查的执行代码，没有重复代码审计或追加测试、`.aris`。

每 cell 均为 2,048 prefill / 240 decode / 256 output tokens；零 preemption、
computed adjustment、多 token chunk。实际 active/decode 达到 cap；调度后等待峰值
cap 6 为 10、cap 8 为 8。全部 16 份 GPU 边界检查 PASS，仅证明边界观测。
四组图捕获尺寸相同，12 个 warmup 均完成；日志中的 monitored fused-MoE JIT 告警
落在 warmup，测量阶段未见该告警，不由此推断完全没有编译成本。

整组输出有 3 个变体，3/16 请求出现两个 token 序列；未做 first-divergence 或任务质量
验证。不能把正常完成等同于两策略 token 输出相同。

队列 PID 13713 与四子进程均正常退出；结束时 GPU 与相关实验进程为空。全部数据保留于
[gpu_results/](gpu_results/)，重算见 [analysis/report.md](analysis/report.md) 和
[analysis/analysis.json](analysis/analysis.json)，运行设计见 [DECISIONS.md](DECISIONS.md)。
远端目录为 `/root/autodl-tmp/moe-native-fresh-cohort-20260906-r01/`，本轮未覆盖前序数据。

复算可使用原 `native_transfer_r01/analyze_native.py`，`--results-dir` 指向上述
gpu_results，`--output-dir` 指向一个不存在的新目录。离线准备命令为：

```bash
.venv/bin/python refine-logs/expert_saturation/outputs/admission_capacity/20260906_native_fresh_cohort_r01/prepare_inputs.py \
  --dataset-arrow ~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow \
  --output-dir /private/tmp/moe-native-new-inputs-reproduced
```
