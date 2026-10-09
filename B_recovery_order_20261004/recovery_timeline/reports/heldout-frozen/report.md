# KV 恢复时间线与容量等待诊断

输入：`/Users/zhaozhenyu/Desktop/毕业设计/B_recovery_order_20261004/recovery_service_age_native/session-20261009-r01/cell-00-cap256-native/output`

10 个抢占 episode；3 个 LOAD/方向未知 job；7 个 episode 没有观察到关联 LOAD；0 个多 LOAD episode。

**这是既有轨迹的 CPU 案例分析。时间差不是因果分解；host 完成回报不是 GPU 实际完成时刻。**

所有重复事件保留；缺失、歧义、无法对齐的时钟返回未知。STORE/全局 flush 的完整记录仍在输入原件中。

## 全部恢复 episode

| episode | 外部请求 | 相邻输出间隔 | 抢占→下一输出 | LOAD job | 分配失败/尝试 |
|---|---|---:|---:|---|---:|
| episode-1 | b-normal-0054660-short | 6.482272 s | 6.478162 s | []（歧义 []） | 2/3 |
| episode-2 | b-normal-0054735-long | 6.193935 s | 6.189810 s | []（歧义 []） | 12/13 |
| episode-3 | b-normal-0054690-short | 4.874351 s | 4.869606 s | []（歧义 []） | 2/3 |
| episode-4 | b-normal-0054853-long | 4.581118 s | 4.576625 s | []（歧义 []） | 12/13 |
| episode-5 | b-normal-0054707-short | 3.157598 s | 3.150734 s | []（歧义 []） | 2/3 |
| episode-6 | b-normal-0055149-long | 3.090254 s | 3.085624 s | [9886]（歧义 []） | 19/21 |
| episode-7 | b-normal-0054804-short | 0.729286 s | 0.725124 s | []（歧义 []） | 2/3 |
| episode-8 | b-normal-0054804-short | 0.298086 s | 0.294710 s | []（歧义 []） | 2/3 |
| episode-9 | b-normal-0055287-long | 0.744197 s | 0.740397 s | [9631]（歧义 []） | 5/7 |
| episode-10 | b-normal-0055287-long | 0.511760 s | 0.504079 s | [9823]（歧义 []） | 3/5 |

同一请求可能在下一输出前再次抢占；这些输出窗口可重叠，不能合计为独立请求或可加等待。JSON 的 overlapping_episode_ids 和 job.episode_candidates 保留这种歧义。

## 全部 LOAD 的 host 阶段

| job | episode 关联 | ready→提交入口 | 提交入口→host 完成回报 | host 完成回报→ack |
|---|---|---:|---:|---:|
| 9631 | unique ['episode-9'] | 0.000005 s | 0.015153 s | 0.054829 s |
| 9823 | unique ['episode-10'] | 0.000008 s | 0.072138 s | 0.062180 s |
| 9886 | unique ['episode-6'] | 0.000006 s | 0.013484 s | 0.056606 s |

## 自动选择的历史案例

规则：选包含抢占的最大相邻输出间隔，不使用固定 request ID。选中 `episode-1` / `b-normal-0054660-short`。

| 观测阶段 | 时间或未知原因 |
|---|---|
| preempt_to_next_output | 6.478162 s |
| previous_to_next_output | 6.482272 s |
| preempt_to_first_lookup | 0.072526 s |
| preempt_to_first_schedule | 6.334634 s |
| first_schedule_to_next_output | 0.143527 s |
| first_failed_to_first_successful_allocation | 6.260490 s |

最长相邻 allocator 重试间隔：6.188688 s；前一失败快照缺 34 块。区间中没有该请求的 allocator 重试记录；持续缺容量与未重试原因均为**未知**。
证据：`['capacity-handoff.json#/events/2:end_host_perf_s']` → `['capacity-handoff.json#/events/92:begin_host_perf_s']`。

source handoff：**未知（该 episode 没有关联的 source handoff 观测）**。这不等于观察到了空 LOAD 列表；不能据此确认走了重算路径。

### LOAD 的独立链路

LOAD ready / submit / 完成回报 / ack：未知（没有观察到关联 LOAD）；不假设每次抢占必须有一次 LOAD。

### 尚不能归因的区间（最长五段）

| 时长 | 左端原始证据 | 右端原始证据 |
|---:|---|---|
| 6.188662 s | `['capacity-handoff.json#/events/2:end_host_perf_s']` | `['recovery-order.json#/events/60938']` |
| 0.143515 s | `['capacity-handoff.json#/events/93']` | `['raw.json#/requests/255/token_times_s/186']` |
| 0.072462 s | `['raw.json#/preemption_events/0:method_returned_s']` | `['recovery-order.json#/events/52703']` |
| 0.071769 s | `['capacity-handoff.json#/events/1:end_host_perf_s']` | `['capacity-handoff.json#/events/2:begin_host_perf_s']` |
| 0.001570 s | `['capacity-handoff.json#/events/92:end_host_perf_s']` | `['recovery-order.json#/events/60951']` |

### 自动关联的时间线

| 相对抢占秒 | kind | job | 原始证据 |
|---:|---|---|---|
| -0.004110 | output_receipt | None | `raw.json#/requests/255/token_times_s/185` |
| 0.000000 | preempt | None | `recovery-order.json#/events/52624` |
| 0.000056 | logical_free_after_preempt | None | `recovery-order.json#/events/52625` |
| 0.000057 | preempt_end | None | `capacity-handoff.json#/events/0:end_host_perf_s` |
| 0.000064 | raw_preempt_end | None | `raw.json#/preemption_events/0:method_returned_s` |
| 0.072526 | lookup | None | `recovery-order.json#/events/52703` |
| 0.072560 | allocate_begin | None | `capacity-handoff.json#/events/1:begin_host_perf_s` |
| 0.072573 | capacity_wait | None | `recovery-order.json#/events/52704` |
| 0.072574 | allocate_end | None | `capacity-handoff.json#/events/1:end_host_perf_s` |
| 0.144344 | allocate_begin | None | `capacity-handoff.json#/events/2:begin_host_perf_s` |
| 0.144354 | allocate_end | None | `capacity-handoff.json#/events/2:end_host_perf_s` |
| 6.333015 | lookup | None | `recovery-order.json#/events/60938` |
| 6.333041 | allocate_begin | None | `capacity-handoff.json#/events/92:begin_host_perf_s` |
| 6.333063 | allocation_ok | None | `recovery-order.json#/events/60939` |
| 6.333064 | allocate_end | None | `capacity-handoff.json#/events/92:end_host_perf_s` |
| 6.334634 | scheduled | None | `recovery-order.json#/events/60951` |
| 6.334646 | resumed_schedule | None | `capacity-handoff.json#/events/93` |
| 6.478162 | output_receipt | None | `raw.json#/requests/255/token_times_s/186` |

## 功能边界与价值

入口自动连接请求抢占、0..N 个 LOAD 的 host 生命周期、容量分配尝试、再调度计划与下一 host 输出。它补充本线原生 transfers 汇总所没有的请求/job 关联、失败快照与重试空档，并把无法归因的区间明确留下。

普通 profiler 的 kernel/copy 时间仍有用；本工具只整合这里已有的调度语义和证据位置，不声称 profiler 普遍不能实现同类关联。逻辑块范围与 scheduler step 未采集时保持 null；source-handoff 只覆盖一个选定请求。

当前定位：可复现的案例分析工具。自动时间线不能确定未重试原因或服务改进动作，没有独立工具论文或优化收益主张。
