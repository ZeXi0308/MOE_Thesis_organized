# JoinStream GPU 正式运行前的合同纠正

> 冻结时间：2026-08-10 20:18:33 +08:00  
> 状态：`FROZEN_BEFORE_FIRST_GPU_BINARY_EXECUTION / NOT_RUN`  
> 原因：远端第一次命令在 shell 查找 `/usr/bin/time` 时以 127 退出，benchmark binary 未执行，未生成 CSV/meta，也没有可见 GPU 结果。

只读合同 review 在真实数据产生前发现 1 个 P0、2 个 P1。以下是为保持用户原始“same work / pre-launched consumer / useful-work start”要求所做的最小纠正，不改变 8 cells、三 variants、consumer utility、warmups/repeats、noise、backfeed 或 verdict。

## 1. P0：冻结等量 producer work

原实现按绝对 deadline 停止 FMA；consumer interference 会让 C 在同一 deadline 内少做 FMA，从而低估 producer tax。纠正为：

- 正式 trials 前仅做设备预热与单-block FMA 速率校准；它不是新增实验 cell 或 variant。
- 将 `{0,5,15,30} us` 映射成四个冻结的 `tail_fma_chunks_per_thread`，每 chunk 固定 32 个 dependent FMA。
- 同一 GPU cell 的 A/B/C 每个 producer thread 执行完全相同 chunk 数，不按时间提前结束。
- raw CSV 记录 chunk 数，paired contract 要求 A/B/C 严格相等；meta 记录校准值与四个映射。
- `actual_tail_window` 仍只从 WholeBarrier 的真实 `%globaltimer` 时间戳得到；target gap 只用于 cell 标签与 calibration error，不替代实测 x。

## 2. P1：闭合 B/C consumer pre-entry

- B/C 仍各只有一次 consumer launch。
- Consumer kernel 进入后写 `consumer_entered` 并做 system visibility fence。
- Host 通过独立 nonblocking copy stream 和 pinned 4-byte buffer轮询该字段；确认 entry 后才 enqueue producer。
- A 不使用该 handshake；B/C 必须满足 `consumer_entry_ns <= producer_start_ns`，否则合同失败。
- 该 host handshake 位于 `producer_start` 之前，不进入跨 variant elapsed 指标。

## 3. P1：真实 useful-start 边界

`consumer_start_ns` 移到最后一个 CTA synchronization 之后；记录后不再执行 barrier，随即进入 hash 或 RMSNorm/projection。这样不会把尚未完成的 barrier 等待计入 overlap。若少数 utility threads 比 lane 0 更早离开 barrier，该时间戳只会保守地稍晚，不会制造假 overlap。

这些纠正在第一条 CUDA benchmark 指令真正执行前冻结；不得根据后续结果修改。
