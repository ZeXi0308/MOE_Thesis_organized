# pro-dev-r01：完整请求结果与正常容量边界

本组是**固定新准入开发与原生压力探针**，fixed 臂确实改变新请求准入，运行请求与恢复规则保持原生：7 个 cell、共 1,216 个计划请求全部到达并完成；失败、拒绝、超时、未完成均为 0。所有 cell 的实际 preemption 和采样恢复积压均为 0，尚未检验恢复信号的增量决策价值。这里没有运行 `kv` 或 `recovery` 臂，不能将 `changed_by_recovery=0` 写成候选机制无效。

设备为单张 RTX PRO 6000 Blackwell Server Edition；OLMoE-1B-7B-0924、BF16、原生 vLLM、32,768 个可用 KV block（64 GiB，另有 null block）、16 GiB host KV、native max sequences 256、batched token budget 1,024。各臂硬件、模型及预算相同。基础提交 `5593b5f`，本次实际源文件散列见 [source.json](runs/pro-dev-r01/source.json)，预声明规则见 [protocol.json](runs/pro-dev-r01/protocol.json)。

前四臂使用相同 192 个开发请求和全部在 t=0 到达的轨迹。三个 probe 使用另一组文章：low 为 64 请求、间隔 0.5 s；near/high 为相同 192 请求、间隔分别为 0.1/0.02 s。probe 之间是压力表征，不能当作相同到达轨迹的策略对照。每臂只有一次运行，没有反序复现或统计显著性结论。

| Cell | 完成/到达 | TTFT mean / p95 (s) | 完整 flow mean / p95 (s) | 每请求最大生成间隔 p95 (s) | Token/s | 排空时间 (s) |
|---|---:|---:|---:|---:|---:|---:|
| dev-native256 | 192/192 | 5.878 / 13.478 | 57.466 / 64.584 | 0.122 | 2938.5 | 64.765 |
| dev-cap96 | 192/192 | 20.784 / 43.575 | 55.962 / 75.924 | 0.567 | 2525.0 | 76.096 |
| dev-cap128 | 192/192 | 15.699 / 46.477 | 52.726 / 72.300 | 0.183 | 2587.5 | 72.474 |
| dev-cap192 | 192/192 | 5.951 / 13.566 | 57.407 / 64.531 | 0.214 | 2941.0 | 64.711 |
| probe-low | 64/64 | 0.056 / 0.082 | 15.715 / 18.020 | 0.132 | 1427.1 | 13.713 |
| probe-near | 192/192 | 0.198 / 0.861 | 49.659 / 53.232 | 0.156 | 2841.2 | 49.045 |
| probe-high | 192/192 | 3.840 / 9.649 | 56.516 / 60.874 | 0.221 | 2993.2 | 60.868 |

TTFT 和 flow 均从外部到达起算；排空为最后完成减最后计划到达。吞吐分母为 `max(到达窗口, 最后完成, observation_end)`，各臂依次为 64.765、76.096、72.474、64.711、45.213、68.145、64.688 s。全请求原始分布与逐请求数据见 [analysis/pro-dev-r01.json](analysis/pro-dev-r01.json) 及其链接的 CSV/JSON。所有输出 chunk 大小为 1；生成间隔使用实际 host-return token 时间，没有插值。

固定基线按预声明的“完成运行中最小平均 flow”选择 **cap128**：52.726 s，低于 cap96 的 55.962 s 和 cap192 的 57.407 s。但这只说明它赢得该选择目标，不能称为最强完整服务基线。相对 native，cap128 平均 flow 降 8.25%，同时 TTFT p95 增 244.83%、flow p95 增 11.95%、排空延长 11.90%、token/s 降 11.94%。

没有应用 SLO；以下保留预声明全部 20 点（TTFT 2/5/10/20/40 s × gap 0.25/0.5/1/2 s，flow ≤120 s），不选最有利阈值。联合 goodput 相对 native 的胜/负/平为 cap96 **0/20/0**、cap128 **0/20/0**、cap192 **8/12/0**。cap192 的微小运行差异未复现，不能据此声称收益；native/cap192 必须保留为完整服务强对手。

自然 EOS 开启，输出上限 1,024、min_tokens=0，未预知真实输出长度。下表的 stop 是测量记录的自然停止，length 是触及上限；没有做任务质量评价。

| Cell | 总输出 token | 自然 stop / length | 相对 native 序列不同请求 | 相对 native 长度不同请求 |
|---|---:|---:|---:|---:|
| dev-native256 | 190,312 | 7 / 185 | — | — |
| dev-cap96 | 192,144 | 5 / 187 | 77/192 | 5/192 |
| dev-cap128 | 187,531 | 10 / 182 | 64/192 | 4/192 |
| dev-cap192 | 190,312 | 7 / 185 | 0/192 | 0/192 |
| probe-low | 64,523 | 1 / 63 | 不同到达/请求规模 | 不作等工作对比 |
| probe-near | 193,615 | 3 / 189 | 不同到达轨迹 | 不作等工作对比 |
| probe-high | 193,623 | 3 / 189 | 不同到达轨迹 | 不作等工作对比 |

cap128 比 native 少生成 2,781 token（−1.46%），cap96 多生成 1,832 token（+0.96%）。平均完成时间差异混有输出工作量和序列变化，不能当作等工作量提速，也不能把逐 token 一致性替代任务质量。各请求 token 序列保留在对应 `raw.json` 中。

| Cell | 观测最少空闲 block | 观测最大占用 KV (GiB) | 观测最大 active | cap 拒绝评估次数 | controller wall (s / 服务占比) | controller CPU (s) |
|---|---:|---:|---:|---:|---:|---:|
| dev-native256 | 661 | 62.709 | 188 | 0 | 0.111 / 0.17% | 0.109 |
| dev-cap96 | 15,416 | 33.891 | 96 | 76,887 | 2.175 / 2.86% | 2.215 |
| dev-cap128 | 10,030 | 44.410 | 128 | 42,921 | 1.239 / 1.71% | 1.263 |
| dev-cap192 | 651 | 62.729 | 188 | 0 | 0.104 / 0.16% | 0.103 |
| probe-low | 27,441 | 10.404 | 36 | 0 | 0.102 / 0.23% | 0.100 |
| probe-near | 3,510 | 57.145 | 189 | 0 | 0.108 / 0.16% | 0.107 |
| probe-high | 410 | 63.199 | 189 | 0 | 0.102 / 0.16% | 0.101 |

KV/active 极值合并 100 ms snapshot 与新请求 gate 评估点，仅是观测峰值，不是逐步精确峰值。占用 KV=`(32768−free_blocks)×2 MiB`，不等于重新分配 GPU 总预算。controller 计时包含状态观察、日志及 queue 操作，已处于端到端服务计时内，不能再次扣除。大量 cap 评估是同一请求的重复等待，并非独立拒绝请求。

四个开发臂首次许可请求顺序相同，但 cap96/128 的许可时间及其相对 native 的批次不同；native 与 cap192 的许可顺序、批次和所有输出序列均相同，最大首次许可时间差为 0.115 s。全部首次许可序列和外部到达等待见 [analysis/decisions-pro-dev-r01.json](analysis/decisions-pro-dev-r01.json) 及各臂 `.starts.csv`。cap96/128 的 prefill 等待均值分别 20.683/15.589 s，p95 为 43.484/46.335 s。年龄 ≥10 s 标记在这两臂涉及 93/60 个请求，但它不绕过 fixed cap；本组没有启用 KV/recovery gate，不能把这些标记解释成“恢复门控后超时放行”。运行超时上限为 240 s，实际请求均完成。

当前问题的答案：**正常容量、已测输入和到达压力下没有出现恢复积压，因此仍无法判断恢复状态相对 KV/并发是否有增量价值。** high 已观测到只剩 410/32768 个空闲 block，却没有实际 preemption，说明“接近 KV 容量”本身不足以保证可检验的恢复动作空间。下一最小实验应只改变一个负载压力因素，先获得真实恢复；随后在同一外部轨迹上执行 fixed/KV/recovery 及反序重复，并保留 native/cap192 强参照。若仍无恢复，记录正常容量边界，不扩大机制收益主张。

复算命令（在本目录执行；输出禁止覆盖，复算时换新文件名）：

```sh
python3 -B analyze.py runs/pro-dev-r01/{dev-native256,dev-cap96,dev-cap128,dev-cap192,probe-low,probe-near,probe-high} --output analysis/pro-dev-r01.json
python3 -B decision_summary.py runs/pro-dev-r01/{dev-native256,dev-cap96,dev-cap128,dev-cap192,probe-low,probe-near,probe-high} --output analysis/decisions-pro-dev-r01.json
```
