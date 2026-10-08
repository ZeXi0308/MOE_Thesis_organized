# pro-pressure-dev-r01：中断组的独立描述

**组状态：INTERRUPTED；有效证据仅为 dev-cap128 的一个完整运行。** 2026-10-07 恢复连接后的现场检查确认旧 C 进程已不在；原 [status.json](runs/pro-pressure-dev-r01/status.json) 的 `RUNNING / pid 15300` 是遗留状态，不能作为仍在执行的证据。这里保留原文件，不将缺少请求级记录的臂补写成失败或成功。

| 开发臂 | 计划请求 | 已有证据 | 可报告的请求结果 |
|---|---:|---|---|
| dev-cap128 | 384 | `raw.json` 为 COMPLETE；准入及完成轨迹齐全 | 384 到达、384 完成；失败/拒绝/超时/未完成均为 0 |
| dev-cap192 | 384 | 第二臂已启动；有配置、workload 和 warmup，缺 `raw.json` 与 admission 过程记录 | 实际到达、完成、失败、超时及未完成数量均未知；不能推断全部已到达 |
| dev-cap256 | 384 | 未启动、未运行 | UNRUN；没有可计入服务结果的请求级观察 |

没有 `selection.json`，因此没有完成三配置固定基线选择。不能把本组写成“384/1152 请求成功”，也不能将另外 768 个计划请求当作服务失败、未完成或成功；计划执行覆盖和已观察服务结果是不同口径。

唯一完整运行采用 384 个不同请求、0.1 s 外部到达间隔、38.3 s 到达窗口；OLMoE-1B-7B、BF16、64 GiB 可用 GPU KV、16 GiB host KV，native 上限 256，固定准入 cap128。自然 EOS 开启，最多 1,024 输出 token。计时均从外部到达开始，以下每个分布的分母均为 384。

| cap128 单次运行指标 | 值 |
|---|---:|
| TTFT mean / p95 | 32.049 / 71.355 s |
| 完整 flow mean / p95 | 75.387 / 108.498 s |
| 新 prefill 等待 mean / p95 | 31.937 / 71.224 s |
| 每请求最大生成间隔 p95 | 0.183 s |
| 观察/吞吐分母 | 144.624 s |
| 最后到达后的排空时间 | 106.324 s |
| 总输出 / token 吞吐 | 381,071 / 2,634.91 token/s |
| 自然 stop / length 上限完成 | 14 / 370 |
| 实际 preemption / 观测恢复积压最大值 | 0 / 0 |
| 观测最大 active / snapshot 最小 free block | 128 / 11,367 |
| controller wall / 服务时间占比 | 7.473 s / 5.17% |
| controller CPU / 每调度调用 | 7.606 s / 2,083.84 μs |

固定 cap 造成 193,133 次重复 defer 评估，但没有拒绝或丢弃请求。252 个请求曾出现年龄 ≥10 s 的 signal-bypass 标记；固定 cap 仍有效，这些标记不代表已获准开始 prefill，也不代表恢复 gate 触发。实际首次许可为 384/384，恢复信号改变决定为 0。本臂是 fixed 模式且没有恢复，不能据此评价 recovery 候选的收益。controller 观测、日志和队列操作开销已计入服务时间，不能再次扣除。

预声明探索性联合 SLO 的 flow 限制为 120 s；全部 20 点结果保留如下。表内为满足联合 SLO 的请求数，分母均为 384；goodput 对应除以同一 144.623787935 s 服务分母。这不是应用 SLO，也没有选择最有利阈值。

| TTFT 限制 | gap ≤0.25 s | gap ≤0.5 s | gap ≤1 s | gap ≤2 s |
|---|---:|---:|---:|---:|
| ≤2 s | 131 | 131 | 131 | 131 |
| ≤5 s | 132 | 132 | 132 | 132 |
| ≤10 s | 132 | 132 | 132 | 132 |
| ≤20 s | 134 | 134 | 134 | 134 |
| ≤40 s | 265 | 265 | 265 | 265 |

原始设备为 [gpu-before.json](runs/pro-pressure-dev-r01/gpu-before.json) 记录的 `GPU-bf3fc5ab-804d-b9d6-759b-4390899f15b9`。即将重跑的设备 UUID 前缀为 `94203fc3`，本次旧设备数据只保留为独立描述，**不与新设备运行合并成同组对照、运行重复或基线选择**。没有配对收益结论、输出质量结论或请求级独立重复的统计推断。

完整分布及逐请求 CSV/JSON：[pro-pressure-dev-r01-interrupted.json](analysis/pro-pressure-dev-r01-interrupted.json)。实际许可顺序和等待：[decisions-pro-pressure-dev-r01-interrupted.json](analysis/decisions-pro-pressure-dev-r01-interrupted.json)。来源、配置和计划保持在 [原始运行目录](runs/pro-pressure-dev-r01)；仅对唯一完整 cell 执行 CPU 分析，没有补造缺失结果。

复算命令（本目录执行；已存在输出不可覆盖，复算时使用新文件名）：

```sh
python3 -B analyze.py runs/pro-pressure-dev-r01/dev-cap128 --output analysis/pro-pressure-dev-r01-interrupted.json
python3 -B decision_summary.py runs/pro-pressure-dev-r01/dev-cap128 --output analysis/decisions-pro-pressure-dev-r01-interrupted.json
```

下一步是在新设备上从头执行完整开发组并保留失败，完成后再选择固定基线；本中断组不补齐合并。当前可回答的仅是：旧设备 cap128 的完整 384 请求运行没有恢复积压，且有显著新请求等待；恢复信号的增量价值仍未被这组数据检验。
