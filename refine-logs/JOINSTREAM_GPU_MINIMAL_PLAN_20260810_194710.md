# JoinStream 最小单 GPU Action-Space Pilot 冻结计划

> 冻结时间：2026-08-10 19:47:10 +08:00  
> 状态：`FROZEN_BEFORE_IMPLEMENTATION / NOT_RUN`  
> 范围：独立 CUDA/C++ microbenchmark；1 GPU；不接 vLLM/PyTorch/NCCL；不改 CPU Oracle 或 CriticalSplit

## 1. 唯一问题

当 K-way critical-row join 已闭合、但 enclosing producer kernel 仍有 residual tail 时，一次预启动的 same-device consumer 能否在内存正确的前提下完成固定有用工作，且扣除 polling、notification 与 residency interference 后仍有净 flow 改善？

证据顺序冻结为：`Memory legality -> GPU schedulability -> Net utility -> CPU Oracle backfeed`。

## 2. 三个完全配对的 variants

| Variant | Producer | Consumer |
|---|---|---|
| A `WholeBarrier` | 单次 producer launch；不发布 row flag | 单次相同 consumer launch，由 producer-complete CUDA event 后启动 |
| B `AllDoneSham` | 与 C 同 kernel/grid/work，只在所有 producer blocks 完成后 release-store flag | 与 C 完全相同的高优先级、预驻留、poll/acquire 单次 launch |
| C `JoinStream` | 与 B 同 kernel/grid/work，critical row materialize 后立即 release-store flag，bulk CTAs 继续 tail | 与 B 完全相同 |

三者的 producer launch 数、consumer launch 数、input、consumer work、grid/block 与 tail target 均相同。不拆 producer，不为 row 新增 kernel。

## 3. Memory contract

- `K=4`。最先实际调度的 K 个 CTAs 通过 device-scope atomic role allocator 成为 contributors，避免 near-saturating grid 中固定 blockIdx 死锁。
- 每个 contributor 由其 thread 0 写独占 contribution row，再对 join counter 执行 `acq_rel fetch_add`。
- 最后 contributor acquire 完整 release sequence，由同一 thread 生成完整 critical row，记录 `join_close/row_materialized`。
- C 对 `consumer_ready` 执行 device-scope release store；B 在 producer blocks done counter 闭合后执行同样的 release store。Producer 不等 consumer。
- B/C consumer lane 0 acquire-poll，CTA barrier 后所有读 row 的 threads 再执行 acquire load，然后才记录 useful-work start。
- 任何 stale read、row hash / consumer output mismatch、launch/work 不等或时序非法，都机械判为 `INVALID_MEMORY_CONTRACT`。

## 4. Producer / consumer work

- Producer：K 个 CTA contributions -> dynamic join -> one critical row；其他 CTAs 与 contributor CTAs 在 row materialize 后执行到 `row_materialized + target_tail_ns` 的有限 FMA work loop。
- `tail-friendly`：少量 producer blocks，明确留出多个 SM。
- `near-saturating`：`SM_count x cudaOccupancyMaxActiveBlocksPerMultiprocessor`个 producer blocks，但 producer 永不依赖 consumer，因而不会调度死锁。
- Correctness mode：consumer 对 critical row 做确定性 bit-hash/reduction，A/B/C 必须逐 bit 一致。
- Utility mode：单个 consumer CTA 做固定 row-wise RMSNorm + 64-output projection + top-k surrogate，三者代码路径和 work hash 相同。

## 5. 冻结矩阵与运行

- Cells：`tail_gap={0,5,15,30} us x residency={tail-friendly,near-saturating}` = 8。
- 每 cell/mode：30 warmups；correctness 30 paired repeats；utility 200 paired repeats。
- repeat 内按固定 6-permutation 轮换 A/B/C 顺序。无 profiler 运行是正式结果。
- 时钟：内联 PTX `%globaltimer`；任何跨 variant 比较都先减该 trial 的 `producer_start`，再按 repeat 做 paired delta。不用 `clock64()`。
- 抖动门槛：`noise_guard=max(timer_resolution, 3 * 1.4826 * MAD(paired_metric))`；报 median/MAD/P10/P90，不做大规模统计检验。

## 6. 指标

```text
visibility_latency = C.observe - C.flag_publish
overlap_window = C.producer_end - C.consumer_start
critical_completion_gain = B.consumer_end_elapsed - C.consumer_end_elapsed
critical_gain_vs_whole = A.consumer_end_elapsed - C.consumer_end_elapsed
producer_tax = C.producer_end_elapsed - A.producer_end_elapsed
total_makespan_delta = C.total_end_elapsed - A.total_end_elapsed
tail_calibration_error = (producer_end - row_materialized) - target_tail
```

`critical_completion_gain` 用于附件的 sham 门槛；CPU 回灌使用已扣 sham structure tax 的 `critical_gain_vs_whole`。

## 7. 不修改 CPU Oracle 的保守回灌

- 只读 `artifacts/joinstream_pilot/20260810_184136/joinstream_results.json`。
- 对每个 CPU cell，取 baseline 与 expanded 间 per-request gain 最大的唯一 critical request（tie 按 request_id），不把单-row GPU 收益复制给全部 M rows。
- 对两个 residency 分别以实测 `A.producer_end-A.row_materialized` 为 x，对 `critical_gain_vs_whole`、正 producer tax 及 noise 做预冻结分段线性插值；不按 target gap，不外推。
- `backfed_gain = interpolated_critical_gain - (M-1)*max(0, interpolated_producer_tax)`。
- CPU cell 只在 `backfed_gain>0` 且 `backfed_gain>backfed_noise` 时为 positive；同时报两种 residency，不删负项。

## 8. 唯一机械 verdict

1. 无 CUDA GPU：`BLOCKED_NO_GPU`。
2. 任一 correctness / memory / work / launch / timestamp 合同失败：`INVALID_MEMORY_CONTRACT`。
3. 少于 2 个非零-tail cells 的 median `overlap_window>0`：`WEAKEN_GPU_SCHEDULABILITY`。
4. 任一成立：`WEAKEN_TAX_DOMINATES`：
   - 少于 2 个非零-tail cells 的 `critical_completion_gain>noise_guard`；
   - 任一 residency 在非零-tail cells 上的 median producer 或 total-duration regression `>5%`；
   - CPU 回灌后少于 2/8 cells positive；
   - 没有一个 positive CPU cell 同时 `gain>=1 us` 且 `gain>noise`。
5. 其余：`SUPPORT_GPU_ACTION_SPACE`。

Novelty positioning：`SUPPORT -> SUPPORTS`；两类 `WEAKEN -> WEAKENS`；`BLOCKED/INVALID -> DOES_NOT_ADDRESS`。

不因近门槛而改 tail、residency、repeats、CPU cells、公式或 verdict。
