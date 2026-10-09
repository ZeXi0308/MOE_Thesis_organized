# B 恢复时间线：本地案例分析工具

本工具把 B 既有采集记录中的请求、恢复 episode、多个 LOAD 与调度/容量观察串联起来，帮助检查一段恢复经历中哪些阶段有证据、哪些仍未知。它只支持本地 B 的既有采集 schema，不是公共 trace 框架；当前定位是跨层案例分析工具，尚非论文贡献。

## 使用

在 `B_recovery_order_20261004` 根目录执行，输出使用新报告目录，不覆盖历史输入：

```sh
python3 recovery_timeline/analyze.py --input <历史output目录> --output <新报告目录>
```

设计案例：

```sh
python3 recovery_timeline/analyze.py \
  --input recovery_service_age/session-20261009-r01/cell-01-cap256-stall8/output \
  --output recovery_timeline/reports/design-r01
```

留出案例（规则冻结前未读取内容）：

```sh
python3 recovery_timeline/analyze.py \
  --input recovery_service_age_native/session-20261009-r01/cell-00-cap256-native/output \
  --output recovery_timeline/reports/heldout-r01
```

输出为 `report.json` 与 `report.md`。JSON 保留事件、全部可识别 episode、每个 LOAD 的阶段及未知原因；Markdown 提供可读摘要。不完整 episode 也应保留，不能只展示可完整关联的成功恢复。已完成的冻结结果见下方；输入原件不变。

## 字段与关联

事件最小字段为 `request_id`、`job_id`、`kind`、`time_s`、`clock_domain`、`scheduler_step`、`logical_block_range`、`storage_layer`、`evidence`。字段允许为空；没有证据时不得补成零、猜测时刻或推定区间。`evidence` 用于说明原始记录出处及必要的推导依据。

- 一个请求或 episode 可关联零个、一个或多个 LOAD；不使用“每次恢复只有一个 LOAD”的假设。
- 所有重复记录保留，包括重复 ready、提交、回报和相同时间事件；不能按 request/job/kind 简单去重。
- job 关联缺失或不唯一、episode 边界不完整、时钟无法对齐时，输出未知原因，不用最近时间或首条记录强行唯一匹配。job ID 只在其历史运行上下文内解释。
- 没有真实 scheduler step 就留空，不能把 host 调用序号当作调度轮次。只有块数量、字节汇总或 key 文本时，不能据此补造 job 的逻辑块区间。

## 证据边界

`time.perf_counter()` 记录的是 host 观察时间。提交返回、job 完成回报、ack 退休、native handoff 与恢复调度各有不同含义；host 完成回报不是 GPU 的绝对完成时刻。原生 transfer elapsed 是耗时，不是可直接放入 host 时间轴的 GPU 起止时间。无法证明属于可对齐时钟域的记录保持分离；不把可能重叠的传输耗时相加成恢复墙钟时间。

已读采集代码给出的限制如下：

- `pkg/recovery_order.py` 的 lookup 和 allocation 状态仅在变化时记录。因此 capacity 变化日志不能证明每次重试，也不能用缺少事件证明没有重试。
- `capacity_handoff/observe_tail.py` 记录被观察恢复窗口内的分配尝试，直到首次重新排入正 token 调度；该调度是计划，不是 GPU 执行或输出完成。它记录 GPU 块数量和限定布局下的需求，不提供完整物理块映射。
- `source_handoff/source_handoff.py` 每次运行只跟踪首个合格请求。其有序 key 前缀及 Host block ID 不能冒充全体请求的来源轨迹，也没有完整 job→key 映射；Host block ID 不是 GPU block ID。
- `pkg/native_offload_observer.py` 的 transfer 统计是 LOAD/STORE 方向的 bytes/time/sizes 汇总，不带逐 job 的完整时间锚。可以与报告作有范围的统计核对，但不能据此还原缺失的逐 LOAD 阶段。此判断仅针对已读 B 采集代码，不泛称其他 profiler 无法提供这些信息。

存储层只按可辨识证据填写：例如已限定 CPU offloading 的 LOAD/STORE 方向、明确的 Host 或 GPU 块字段。未记录的其他存储层保持未知。时间线关联和局部等待只能支持案例解释，不能单独证明调度因果、端到端收益或方法新颖性。

## 已完成的 CPU 检查与功能判决

同一冻结实现（SHA256 `5e33539012e4dac7a32d598c56a049ec19ddf2657628f969a4659cd7d0818935`）自动生成：

- [历史案例报告](reports/design-frozen/report.md)：15 个 episode、12 个 LOAD，自动选回原人工案例；最大相邻输出间隔 6.318797 s，最长分配重试间隔 5.284278 s。重叠抢占使一个 job 的 episode 归属未知，完整保留。
- [留出报告](reports/heldout-frozen/report.md)：10 个 episode、3 个 LOAD；最大间隔 6.482272 s，最长分配重试间隔 6.188688 s。规则没有据此改动。该最大案例无 source handoff 覆盖，不能把“无观测”读成“确认没有 LOAD”。

两者均不能证明这些长间隔持续缺容量，亦不能解释为何没有重试。本轮 GPU/远端动作均为零，没有新服务性能实验。两份真实轨迹没有确定的单 episode 多 LOAD；多个 LOAD、重复事件、缺失和歧义的边界仅由小型合成测试覆盖：

```sh
python3 -m unittest recovery_timeline.test_analyze -v
```

原型已经能无人工 request ID 选择地回答具体的跨层观测问题；原因判断仍需其他证据。当前交付停在**案例分析工具**，不扩展采集协议，不宣称独立论文或优化收益。

**报告措辞修订。** 留出检查后仅修正 `render()` 对缺少 source handoff 观测的显示：明确返回“未知”，不再紧接空 LOAD 列表的解释。解析、事件关联和区间计算未变，未重新读取轨迹，两个冻结 `report.json` 原样保留；上述实现 SHA 对应原始分析版本。当前 Markdown 从这些 JSON 重新渲染。
