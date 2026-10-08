# westb-pressure-dev-r02：固定基线、恢复发生与准入机会

**三臂真实完成；按预声明平均 flow 规则选择 fixed256。** 每臂相同 384 请求、相同 0.1 s 外部到达轨迹（窗口 38.3 s）；合计 1,152 次请求到达全部完成，失败、拒绝、超时、未完成均为 0。各配置仅一次开发运行，不是反序重复或独立确认。本报告只使用新设备 `GPU-94203fc3-1021-3a9c-a367-cff792479616`，不与旧 GPU 的中断组或候锁超时组混合。

来源为部署提交 `4bfde70`、远端 `v2/pro-pressure-dev-r02`；实际核验 `run.py` SHA256 为 `484275bc507dfd54f1abc9d10a56ef0038daa2d0d1cda4c85dadc7bf0e46e422`。完整来源散列、原始文件索引、归档 SHA256 和独立解压目录见 [raw-index](analysis/westb-pressure-dev-r02.raw-index.json)。`protocol.json` 的 `base_commit=5593b5f` 是继承字段，执行版本以部署记录及实际文件散列为准。归档成员已检查路径、类型、重复和目标冲突，全部文件以 SHA256 再次核对，未覆盖现有内容。

三臂均为 RTX PRO 6000 / OLMoE-1B-7B / BF16 / 64 GiB 可用 GPU KV / 16 GiB host KV，native max sequences 256、batched token budget 1,024，仅固定新 prefill 并发 cap 分别为 128/192/256。计时包含外部到达后的所有等待；natural EOS 开启，max output 1,024。运行 PID 11853 已退出 0，运行级 `status.json` 为 COMPLETE。

| 固定 cap | 完成/到达 | TTFT mean / p95 (s) | 完整 flow mean / p95 (s) | 最大生成间隔 p95 (s) | Token/s | 排空 (s) | Preempt 次数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 384/384 | 34.010 / 74.991 | 78.924 / 112.971 | 0.189 | 2566.89 | 110.726 | 0 |
| 192 | 384/384 | 20.304 / 48.905 | 77.743 / 100.632 | 0.406 | 2824.88 | 97.017 | 0 |
| 256 | 384/384 | 15.967 / 45.906 | 74.371 / 96.484 | 12.402 | 2926.76 | 92.383 | 51 |

吞吐/联合 goodput 的完整服务分母分别为 149.026422、135.317390、130.682683 s；排空减去最后到达时刻。fixed256 的 51 次 preemption 涉及 31 个请求，其最大生成间隔 p99=27.983 s、最大值 30.876 s。它在平均 flow、TTFT 和吞吐上有开发优势，却有明显生成停顿代价，不能称为所有服务口径下最优。所有 token 均单个返回，间隔没有插值。

全部 20 点预声明探索性 SLO 为 TTFT 2/5/10/20/40 s × maxgap 0.25/0.5/1/2 s，flow≤120 s。以下每格是联合达标请求数，按 **cap128 / cap192 / cap256** 排列，各自请求分母均为 384；除以上述各自服务分母即为 joint goodput。不选择某个阈值充当应用 SLO。

| TTFT 限制 | gap≤0.25 s | gap≤0.5 s | gap≤1 s | gap≤2 s |
|---|---:|---:|---:|---:|
| ≤2 s | 130 / 8 / 7 | 130 / 198 / 214 | 130 / 198 / 214 | 130 / 198 / 214 |
| ≤5 s | 131 / 8 / 8 | 131 / 200 / 220 | 131 / 200 / 221 | 131 / 200 / 221 |
| ≤10 s | 131 / 8 / 8 | 131 / 200 / 220 | 131 / 200 / 221 | 131 / 200 / 221 |
| ≤20 s | 132 / 8 / 8 | 132 / 201 / 220 | 132 / 201 / 221 | 132 / 201 / 221 |
| ≤40 s | 264 / 8 / 9 | 264 / 251 / 249 | 264 / 251 / 250 | 264 / 251 / 250 |

| 相对比较 | 全20点 goodput 胜/负/平 | 联合达标率胜/负/平 | 输出序列不同请求 | 输出 token 总量差 |
|---|---:|---:|---:|---:|
| cap192 对 cap128 | 15 / 5 / 0 | 12 / 8 / 0 | 213/384 | −280（−0.0732%） |
| cap256 对 cap128 | 15 / 5 / 0 | 12 / 8 / 0 | 203/384 | −58（−0.0152%） |
| cap256 对 cap192 | 19 / 1 / 0 | 13 / 4 / 3 | 181/384 | +222（+0.0581%） |

三臂输出总量依次为 382,535 / 382,255 / 382,477，均有 12 个自然 stop、372 个 length 上限完成，但不同策略停止的请求身份和输出序列仍会变化。总 token 接近不能证明等工作或任务质量一致；没有任务质量结论。逐请求数量、首个序列差异与全部20点相对结果见 [fixed-comparisons](analysis/westb-pressure-dev-r02-fixed-comparisons.json)。

| cap | 最早实际 preempt (s) | 首次非零 backlog snapshot (s) | 首次 latch snapshot (s) | 最大 backlog | latch 下 gate 评估/请求 | age<10 且 active<cap 且 free≥3277 的评估/请求 |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 无 | 无 | 无 | 0 | 0 / 0 | 0 / 0 |
| 192 | 无 | 无 | 无 | 0 | 0 / 0 | 0 / 0 |
| 256 | 27.828 | 27.892 | 28.414 | 23 | 1 / 1 | 0 / 0 |

上述时刻均相对外部到达原点；snapshot 是观测边界，不是连续状态的精确起止。fixed256 的恢复在到达窗口内已出现，不能归因于“恢复只在所有请求变老以后出现”。然而，never-started 请求实际进入 `Gate.defer` 且 latch=true 的记录只有一次，时间 58.636 s：请求 age=34.336 s、bypass=true、active=199<256、free=132<3277。该次 recovery_count 已为 0，latch 尚未在下一采样清除。全部计划请求在 48.3 s 达到年龄10 s，唯一记录已超过这一边界。

因此，固定臂记录中没有观测到“latch 与未过年龄限制、KV/cap 允许的新请求 gate 决策”交集。**这不证明 native full-ISL allocation 的未运行反事实一定成功，也不覆盖被原生队首 break 挡住、未进入 gate 的请求。** 三臂均为 fixed 模式，latch 只是影子状态，`changed_by_recovery=0` 不是 recovery 候选的因果实验结果。三臂实际首次许可顺序与同次调度批次一致，许可时间不同；不是相同的接纳时序。详情见 [decision summary](analysis/westb-pressure-dev-r02-decisions.json) 和逐臂 `.starts.csv`。

| cap | 观测最少 free block | 观测最大 active | cap 重复 defer 评估 | controller wall / 服务占比 | controller CPU |
|---|---:|---:|---:|---:|---:|
| 128 | 11,314 | 128 | 197,520 | 9.320 s / 6.25% | 9.434 s |
| 192 | 1,137 | 192 | 78,825 | 4.549 s / 3.36% | 4.598 s |
| 256 | 0 | 234 | 0 | 0.260 s / 0.20% | 0.257 s |

极值合并 snapshot 与 gate 观测，不能解释成逐步精确峰值。控制开销包含观察、日志和队列操作，已包含在服务时间内；低 cap 大量重复评估的开销是本实现的实测负面影响。它没有被从主指标中扣掉。年龄≥10 s 标记涉及 253/184/141 个请求，不绕过固定 cap。

首个 `dev-cap128` 的 `CELL_BEGIN` 后，原始 launch log 第62行记录一次推理期 `fused_moe_kernel` JIT 警告。这里只记录存在性：日志没有足够信息量化该编译耗时，不估计时长、不减去开销，也不把固定配置差异全部归因于该警告。三个开发臂按128→192→256顺序各运行一次，仍存在顺序/首次编译影响的确认边界。

完整人口分布及逐请求数据：[analysis](analysis/westb-pressure-dev-r02.json)。原始数据在 `runs/westb-20261007/pro-pressure-dev-r02`，包 SHA256 为 `a90c12b51b09198762aa8f05e5eebade1d7369b35fa31da37f40f0b2ff8160db`。

当前结论是 **NATIVE_SERVING / 开发测量**：真实恢复及其完整请求停顿已出现；fixed256 是预声明 mean-flow 选择，但并不统治全部联合口径。恢复信号能否提供额外准入价值，要由同设备、同轨迹六臂实际干预及反序重复回答；本开发组不能替代正在进行的六臂结果，也不能据影子gate零机会判死整个问题。

复算命令（在本目录执行；已存在文件不会覆盖，复算需换输出名）：

```sh
python3 -B analyze.py runs/westb-20261007/pro-pressure-dev-r02/dev-cap128 runs/westb-20261007/pro-pressure-dev-r02/dev-cap192 runs/westb-20261007/pro-pressure-dev-r02/dev-cap256 --output analysis/westb-pressure-dev-r02.json
python3 -B decision_summary.py runs/westb-20261007/pro-pressure-dev-r02/dev-cap128 runs/westb-20261007/pro-pressure-dev-r02/dev-cap192 runs/westb-20261007/pro-pressure-dev-r02/dev-cap256 --output analysis/westb-pressure-dev-r02-decisions.json
```
