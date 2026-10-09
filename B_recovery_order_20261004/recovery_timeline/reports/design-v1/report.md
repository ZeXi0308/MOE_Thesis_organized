# KV 恢复时间线与容量等待诊断

输入：`/Users/zhaozhenyu/Desktop/毕业设计/B_recovery_order_20261004/recovery_service_age/session-20261009-r01/cell-01-cap256-stall8/output`

15 个抢占 episode；12 个 LOAD/方向未知 job；3 个 episode 没有观察到关联 LOAD；0 个多 LOAD episode。

**这是既有轨迹的 CPU 案例分析。时间差不是因果分解；host 完成回报不是 GPU 实际完成时刻。**

所有重复事件保留；缺失、歧义、无法对齐的时钟返回未知。STORE/全局 flush 的完整记录仍在输入原件中。

## 全部恢复 episode

| episode | 外部请求 | 相邻输出间隔 | 抢占→下一输出 | LOAD job | 分配失败/尝试 |
|---|---|---:|---:|---|---:|
| episode-1 | b-normal-0071017-short | 0.438722 s | 0.434959 s | [8805]（歧义 []） | 2/4 |
| episode-2 | b-normal-0071017-short | 0.747934 s | 0.744164 s | [9061]（歧义 []） | 2/4 |
| episode-3 | b-normal-0071017-short | 0.731054 s | 0.727137 s | [9318]（歧义 []） | 2/4 |
| episode-4 | b-normal-0071017-short | 1.013772 s | 1.009716 s | [9582]（歧义 []） | 2/4 |
| episode-5 | b-normal-0070571-long | 6.318797 s | 6.315146 s | []（歧义 []） | 9/10 |
| episode-6 | b-normal-0070991-short | 0.528642 s | 0.523539 s | [9062]（歧义 []） | 2/4 |
| episode-7 | b-normal-0070991-short | 0.947681 s | 0.942811 s | [9317]（歧义 []） | 2/4 |
| episode-8 | b-normal-0070991-short | 2.256402 s | 2.252194 s | [9903]（歧义 []） | 4/6 |
| episode-9 | b-normal-0070525-long | 4.830565 s | 4.825646 s | []（歧义 []） | 6/7 |
| episode-10 | b-normal-0070969-short | 0.515198 s | 0.509366 s | [9319]（歧义 []） | 2/4 |
| episode-11 | b-normal-0070969-short | 1.230549 s | 1.226334 s | [9581]（歧义 []） | 2/4 |
| episode-12 | b-normal-0070473-long | 3.371421 s | 3.367125 s | []（歧义 []） | 6/7 |
| episode-13 | b-normal-0070835-short | 1.097719 s | 1.093150 s | [9647]（歧义 []） | 3/5 |
| episode-14 | b-normal-0070284-long | 1.750928 s | 1.744815 s | [9646]（歧义 [9902]） | 15/19 |
| episode-15 | b-normal-0070284-long | 1.750928 s | 1.083490 s | []（歧义 [9902]） | 11/13 |

## 自动选择的历史案例

规则：选包含抢占的最大相邻输出间隔，不使用固定 request ID。选中 `episode-5` / `b-normal-0070571-long`。

| 观测阶段 | 时间或未知原因 |
|---|---|
| preempt_to_next_output | 6.315146 s |
| previous_to_next_output | 6.318797 s |
| preempt_to_first_lookup | 0.073262 s |
| preempt_to_first_schedule | 5.940025 s |
| first_schedule_to_next_output | 0.375121 s |
| first_failed_to_first_successful_allocation | 5.864009 s |

最长相邻 allocator 重试间隔：5.284278 s；前一失败快照缺 189 块。区间中没有该请求的 allocator 重试记录；持续缺容量与未重试原因均为**未知**。
证据：`['capacity-handoff.json#/events/15:end_host_perf_s']` → `['capacity-handoff.json#/events/114:begin_host_perf_s']`。

source handoff 观测：`[{"evidence": "source-handoff.json#/events/10", "external_tokens": 0, "load_jobs": []}]`。
空 LOAD 列表只是该 handoff 的观测；不能将缺失的 LOAD 阶段填成零耗时。

### LOAD 的独立链路

LOAD ready / submit / 完成回报 / ack：未知（没有观察到关联 LOAD）；不假设每次抢占必须有一次 LOAD。

### 尚不能归因的区间（最长五段）

| 时长 | 左端原始证据 | 右端原始证据 |
|---:|---|---|
| 5.284163 s | `['source-handoff.json#/events/8']` | `['recovery-order.json#/events/60841']` |
| 0.375110 s | `['capacity-handoff.json#/events/115']` | `['raw.json#/requests/254/token_times_s/190']` |
| 0.077931 s | `['source-handoff.json#/events/1']` | `['capacity-handoff.json#/events/9:begin_host_perf_s']` |
| 0.073224 s | `['raw.json#/preemption_events/1:method_returned_s']` | `['recovery-order.json#/events/52980']` |
| 0.073210 s | `['source-handoff.json#/events/0']` | `['capacity-handoff.json#/events/7:begin_host_perf_s']` |

### 自动关联的时间线

| 相对抢占秒 | kind | job | 原始证据 |
|---:|---|---|---|
| -0.003650 | output_receipt | None | `raw.json#/requests/254/token_times_s/189` |
| 0.000000 | preempt | None | `recovery-order.json#/events/52895` |
| 0.000033 | logical_free_after_preempt | None | `recovery-order.json#/events/52896` |
| 0.000034 | preempt_end | None | `capacity-handoff.json#/events/3:end_host_perf_s` |
| 0.000038 | raw_preempt_end | None | `raw.json#/preemption_events/1:method_returned_s` |
| 0.073262 | lookup | None | `recovery-order.json#/events/52980` |
| 0.073284 | allocate_begin | None | `capacity-handoff.json#/events/5:begin_host_perf_s` |
| 0.073291 | capacity_wait | None | `recovery-order.json#/events/52981` |
| 0.073292 | allocate_end | None | `capacity-handoff.json#/events/5:end_host_perf_s` |
| 0.073437 | selected | None | `source-handoff.json#/events/0` |
| 0.146647 | allocate_begin | None | `capacity-handoff.json#/events/7:begin_host_perf_s` |
| 0.146654 | allocate_end | None | `capacity-handoff.json#/events/7:end_host_perf_s` |
| 0.146687 | allocation | None | `source-handoff.json#/events/1` |
| 0.224618 | allocate_begin | None | `capacity-handoff.json#/events/9:begin_host_perf_s` |
| 0.224627 | allocate_end | None | `capacity-handoff.json#/events/9:end_host_perf_s` |
| 0.224667 | allocation | None | `source-handoff.json#/events/2` |
| 0.296448 | allocate_begin | None | `capacity-handoff.json#/events/10:begin_host_perf_s` |
| 0.296458 | allocate_end | None | `capacity-handoff.json#/events/10:end_host_perf_s` |
| 0.296493 | allocation | None | `source-handoff.json#/events/3` |
| 0.367482 | allocate_begin | None | `capacity-handoff.json#/events/11:begin_host_perf_s` |
| 0.367491 | allocate_end | None | `capacity-handoff.json#/events/11:end_host_perf_s` |
| 0.367527 | allocation | None | `source-handoff.json#/events/4` |
| 0.438962 | allocate_begin | None | `capacity-handoff.json#/events/12:begin_host_perf_s` |
| 0.438971 | allocate_end | None | `capacity-handoff.json#/events/12:end_host_perf_s` |
| 0.439009 | allocation | None | `source-handoff.json#/events/5` |
| 0.510674 | allocate_begin | None | `capacity-handoff.json#/events/13:begin_host_perf_s` |
| 0.510683 | allocate_end | None | `capacity-handoff.json#/events/13:end_host_perf_s` |
| 0.510718 | allocation | None | `source-handoff.json#/events/6` |
| 0.581727 | allocate_begin | None | `capacity-handoff.json#/events/14:begin_host_perf_s` |
| 0.581736 | allocate_end | None | `capacity-handoff.json#/events/14:end_host_perf_s` |
| 0.581772 | allocation | None | `source-handoff.json#/events/7` |
| 0.652993 | allocate_begin | None | `capacity-handoff.json#/events/15:begin_host_perf_s` |
| 0.653002 | allocate_end | None | `capacity-handoff.json#/events/15:end_host_perf_s` |
| 0.653054 | allocation | None | `source-handoff.json#/events/8` |
| 5.937217 | lookup | None | `recovery-order.json#/events/60841` |
| 5.937280 | allocate_begin | None | `capacity-handoff.json#/events/114:begin_host_perf_s` |
| 5.937300 | allocation_ok | None | `recovery-order.json#/events/60842` |
| 5.937301 | allocate_end | None | `capacity-handoff.json#/events/114:end_host_perf_s` |
| 5.937336 | allocation | None | `source-handoff.json#/events/9` |
| 5.937366 | native_handoff | None | `source-handoff.json#/events/10` |
| 5.938412 | release | None | `source-handoff.json#/events/11` |
| 5.940025 | scheduled | None | `recovery-order.json#/events/60853` |
| 5.940036 | resumed_schedule | None | `capacity-handoff.json#/events/115` |
| 6.315146 | output_receipt | None | `raw.json#/requests/254/token_times_s/190` |

## 功能边界与价值

入口自动连接请求抢占、0..N 个 LOAD 的 host 生命周期、容量分配尝试、再调度计划与下一 host 输出。它补充本线原生 transfers 汇总所没有的请求/job 关联、失败快照与重试空档，并把无法归因的区间明确留下。

普通 profiler 的 kernel/copy 时间仍有用；本工具只整合这里已有的调度语义和证据位置，不声称 profiler 普遍不能实现同类关联。逻辑块范围与 scheduler step 未采集时保持 null；source-handoff 只覆盖一个选定请求。

当前定位：可复现的案例分析工具。自动时间线不能确定未重试原因或服务改进动作，没有独立工具论文或优化收益主张。
