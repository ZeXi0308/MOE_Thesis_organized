# STORE 覆盖缺口诊断入口

**功能主张。** 这个独立 CPU 工具将已有调度区间、STORE 枚举游标、任务创建/完成回报和下一次连续 Host 命中关联起来，不依赖请求名称、固定事件序号或旧 target spec。它能自动发现“该块已经重算，但观察窗口内没有创建补写任务，后续前缀仍止于该块”的案例。它不恢复缺失的驻留历史，不自动判定上游 bug，不预测服务收益。

当前证据只支持可复用的案例诊断与工程材料；没有第二个独立覆盖缺口正例。完整驻留/驱逐协议不属于本轮实现范围，不因此追加采集、GPU 或策略搜索。

## 运行

在本目录运行；只需要 Python 3 标准库及已有 `analyze_host_history.py` 的两个纯 CPU 关联函数。无 vLLM、torch、GPU、远端或网络依赖。

```sh
python3 -B diagnose_store_coverage.py \
  execution/oct08_e246_store_replay53005_r01/00_same_engine/00_recompute/raw.json \
  execution/oct08_e246_store_replay53005_r01/00_same_engine/01_recompute/raw.json \
  execution/oct08_two_burst160_20s53005_reverse_r01/00_same_engine/00_recompute/raw.json \
  --compare --cpu-fixture cpu_preparation/store_replay_eviction_result.json \
  --out store_coverage_diagnostic.json
```

输出同名 JSON 与自动生成 Markdown。前两份是开发用的既有游标 off/on 干预；第三份在规则实现之前按结构盘点选定，未用于本诊断规则设计，但以前用于性能分析，**不是整个研究的盲测**。`--compare` 仅比较前两份，第二份减第一份。

新轨迹只需：

```sh
python3 -B diagnose_store_coverage.py /path/to/new/raw.json --out /path/to/report.json
python3 -B cpu_preparation/test_store_coverage_diagnostic.py
```

同目录 `resources.json` 给出块布局，`status.json` 给出运行状态。支持当前 E raw 格式的单 KV group，以 Host chunk 为逻辑块；只有 Host chunk 与 GPU block 等大且均匀字节布局已知时换算 payload 字节，否则字节为 null。非零本地 computed 的恢复、未知 chunk 布局或缺观察不作“未创建”判定。请求 ID 仅作关联键，不需要输入 E246 或任何目标 ID。不同运行的 request ID 不混连；可选对照仅适用于唯一且一致的 external_id 集合，不能仅靠对齐身份证明因果前态匹配。

对照现在强制检查全部到达均进入 work 表、两臂请求集合一致且 scheduler 观察存在；不满足时输出 `INCOMPARABLE` 并保留逐臂报告，不计算所谓全体收益。重复 request_id/external_id 直接报错，防止字典覆盖成本。有 scheduler 观察但某请求没有调度行时，保留其零**已记录**工作及未完成状态；缺少整个 scheduler 观察时工作为 null，不当作零。没有实际补写目标时，不虚构“同伴”分组。

## 证据语义

每条归一化事件带 `request_id`、`job_id`、`kind`、`time`、`clock_domain`、`scheduler_step`、`logical_block_interval`、`storage_tier`、`raw_evidence_location`，缺失为 null。非连续 STORE 使用附加 `logical_block_intervals`，不把两个区间之间的块虚构为已存储。游标与内容版本按需附加；历史观察器没有版本字段，保留 null。

- 时间采用单次 episode 的 CPU 单调时钟相对秒。不同 raw 的 clock_domain 不混用。scheduler_step 是从零开始的服务步；原生包含预热的累计步另记为 native_scheduler_step。
- STORE 回调的时间是**入口**，不是返回或数据完成时刻。严格早一服务步的已返回 ACK 才证明在后续 lookup 前已承认完成。同一步入口先后不作此证明。
- 重算是实际 scheduled token 位置与同请求此前执行位置的交集；不是 token 重新输出、FLOPs 或 GPU 执行时长。逻辑块区间为半开区间；边界处是否整块重算单独检查。
- cursor 是枚举游标，不是驻留水位。Host hit 是连续 ready 前缀，不是全部驻留块；第一个洞后方的块是否存在未知。
- 窗口从一次成功恢复到下一次抢占；其后第一次 lookup 作后继证据。每个被分析调度步都必须有对应正常返回的 builder，且分配链和 job 映射完整，才能由无任务记录判断“未创建”。观察器只跟踪已被抢占请求；更早任务历史并不完整。

| 输出 | 足够证据 | 不支持的推论 |
|---|---|---|
| NOT_CREATED | 首个非 ready 块完整重算；完整窗口无覆盖该块的新 STORE；下一次前缀仍止于它 | 全局从未存过、其他块都不存在、上游 bug |
| NOT_YET_COMPLETED | 同一创建生命周期的原生 job 在后续严格跨步回调入口仍明确 pending | CUDA 数据仍未完成 |
| COMPLETED_THEN_EVICTED | ACK 后、lookup 前，有同一逻辑块/STORE 生命周期的显式驱逐 | 仅凭 ACK 后 miss 猜测驱逐 |
| INSUFFICIENT_EVIDENCE | 缺日志、顺序、映射、后继 lookup，或先前洞遮挡目标块 | 缺日志等于无动作 |
| COVERED_AT_NEXT_LOOKUP | 下一次前缀包含该块；这不是故障分类 | 两次观察之间持续驻留 |

历史 GPU 输入**无驱逐字段**。可选 `--extra-events events.json` 接受单个 raw 的显式 EVICT 数组，使用上述最小字段，另外必须带被驱逐写入的 `creation_record_index`；job_id + 创建记录 + request_id + 逻辑区间标识生命周期。时钟必须与报告中的 clock_domain 一致；事件时间必须来自实际证据，内容版本已知时须一致。只有物理 slot 或后来 miss 不足以构造此输入。这个接口没有附带采集器或公共框架；目前正向驱逐分类仅经合成证据测试，未在真实 GPU 驱逐轨迹上验证。历史 CPU fixture 的 resident 集合变化单独报告，不伪装成完整 STORE 生命周期。

## 已完成的检查与代价

开发 off/on 对照自动发现一个 NOT_CREATED 窗口；规则未引用请求名。首个非 ready 逻辑块 `[73,74)` 完整重算，off 游标保持 197，四个 builder 返回无任务；后续 Host hit 1168。on 游标 197→73，创建 `[73,74)`、`[189,192)`、`[194,195)` 共五块，ACK 在下一 lookup 之前，Host hit 3152。**124 块连续前缀差不等于 124 块物理缺失。**

窗口新建 payload 为 0→10 MiB；全程 STORE 均为 135499087872 字节，LOAD 增加 262144000 字节。全部请求重复位置减少 1974，只发生于补写目标；其他请求全程重复位置差为零。GPU 同伴驱逐与其引起的重算无法归因，均为 null，不能从零净工作差推出无驱逐。完成均值增加 0.033349 秒，完成时刻增加 0.148412 秒；这些来自旧 GPU 运行，非本次 CPU 重放收益。两臂输出长度相同但内容不全相同，既有质量等价主张不成立。

每臂另外有 24 次恢复的 Host 前缀已覆盖全部完整已知块，不算待诊断洞；25 个窗口没有下一次可用 lookup，保留证据不足，不把它们写成新缺口。规则外轨迹有 38 次成功恢复，但 STORE 观察器关闭，全部无法关联创建/游标，正确返回证据不足；不能算第二正例或方法独立确认。

旧 native CPU LRU 反例实际新增一块并移除同伴 B0；A 前缀 2→8，B 8→0，而 B1–B7 仍驻留。无 payload 布局、后续 B 执行或真实传输，因此新增字节、后续重算和时延均未知。这个反例保留补写代价的可能性，不能补成 GPU 案例的驱逐归因。

**English case claim.** The tool reconstructs a window-local STORE coverage gap from existing native events without selecting a request by name. It separates absence of a newly created write, an explicitly pending native job, and an acknowledged write followed by an evidenced eviction; unsupported histories remain inconclusive. The observed cursor intervention creates five STORE chunks and reduces repeated scheduled positions, but does not establish a service-latency benefit. Transfer to a preselected existing trace demonstrates conservative handling of missing instrumentation, not independent confirmation of a second gap. This is a reproducible systems case and diagnostic artifact, not a validated general repair or a new recovery policy.
