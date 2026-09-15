# JoinStream GPU 冻结计划的运行前机械澄清

> 冻结时间：2026-08-10 19:52:32 +08:00  
> 状态：`FROZEN_BEFORE_IMPLEMENTATION_COMPLETE / NOT_RUN`  
> 作用：只消除原冻结计划中会改变 verdict 的两个歧义；不改变 8-cell 矩阵、workload、repeats、variants 或阈值。

## 1. CPU Oracle 插值 query

每个 CPU Oracle cell 的 query x 是：

```text
cpu_headroom_us = max_request(
    baseline_atomic_exact.request_completion_us[request]
    - expanded_joinstream_exact.request_completion_us[request]
)
```

并列时按 `request_id` 字典序选择唯一 critical request。该 gain 只作用于这一条 request，不复制到其余 `M-1` 条 request。

## 2. Backfeed noise 与 residency collapse

对每个 residency 独立插值，且不外推：

```text
backfed_gain
  = interpolated_critical_gain_vs_whole
  - (M - 1) * max(0, interpolated_producer_tax)

backfed_noise
  = interpolated_critical_gain_noise_guard
  + (M - 1) * interpolated_producer_tax_noise_guard
```

即使插值后的 producer tax 被截断为 0，仍保留其 noise guard。单个 residency 只有在 `backfed_gain > 0` 且 `backfed_gain > backfed_noise` 时为 positive。

最终 `2/8` gate 是 action-space existence gate：一个 CPU cell 只要至少一个 residency positive 就计为 positive；两种 residency 的结果必须全部原样报告，不得删除负项或 unavailable 项。

## 3. 其余机械细节

- `clear gain` 使用 paired `AllDoneSham.consumer_end_elapsed - JoinStream.consumer_end_elapsed` 的 median，并要求大于该 paired metric 自己的 frozen noise guard。
- schedulability 与 clear-gain 计数都只看 6 个非零-tail GPU cells（两个 residency x 三个非零 tail）。
- `>5%` systematic regression：对每个 residency，先计算三个非零-tail cell 的 median producer-duration regression 百分比与 total-duration regression 百分比；任一个 median 超过 5% 即触发。
- 若某 CPU query x 落在某 residency 的实测 actual-tail 支撑域之外，该 residency 标记 `unavailable_no_extrapolation`，不计 positive。

以上规则在 CUDA 编译和任何 GPU 结果产生前冻结；不得根据结果修改。
