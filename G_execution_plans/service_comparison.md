# G 完整服务对照状态

**G 完整服务主对照未执行，不能报告相对收益。** 重审后已执行CPU容量包络与一个compact真实启动，dense因锁忙未运行。三层假设分开评估；不再以旧A/B/C或10%线机械决定一般问题。NA 是未测，不是零。

## 本轮有区分力的结果

| 指标 | compact 已测一次 | dense |
|---|---:|---|
| 原生实际 graph 数 | 19（10 PIECEWISE + 9 FULL） | 未运行，exit75 |
| 总 / 可用 KV blocks | 36828 / 36827 | NA |
| high 全请求最大容量包络 | 34430 blocks | 同一冻结输入 |
| 相对包络余量 | 2397 blocks / 4.682 GiB | NA |
| graph 原生估计 / capture delta | 132 / 76 MiB | NA |
| 最终共享 workspace / PyTorch graph pool reserved | 128 / 28 MiB | NA |
| 清cache后整个设备占用减KV storage | 14.037 GiB | NA |
| 显式清cache释放 / KV blocks变化 | 106 MiB / 0 | NA |
| graph估计 / 正式capture / 引擎启动 | 2.291 / 3.316 / 35.932 s | NA |
| 完整吞吐、尾延迟、输出正确性比较 | 未运行 | 未运行 |

来源：`evidence/review_20261008/compact.json`、`dense.json`、`startup_summary.json`，机器可读表 `startup_results.csv`。compact进程45.472秒（含导入、建引擎和退出），dense在0.115秒内锁忙退出，未初始化CUDA。顺序为compact→dense尝试，只有一个独立启动，无性能重复或噪声估计，不能估组合间效应。

**范围明确的结论：**当前compact运行时的单KV group、16-token blocks、零lookahead/watermark、无prefix/spec/connector、同步设置已核验。冻结high所有请求同时达到上限仍放得下，所以在该端点运行这批有限请求时，KV不可能触发分配失败或准入等待；这不是实测服务吞吐。相对旧峰值余量论据，容量包络允许执行时间与batch轨迹改变。dense尚未测，不能推广到所有组合；即使两端点都覆盖，也不能无条件借桶包含关系推出任意subset的内存单调性。

**成本解释：**14.037 GiB包含模型、context/driver、workspace、allocator及graph，绝非独立graph驻留；扣除init_device_after已有占用后为13.383 GiB。132 MiB是原生估计，76 MiB是capture阶段差额，28 MiB是持有的PyTorch graph池，它们不相加也不能互相替代。清cache可释放的106 MiB没有增加已分配KV。GPU utilization=0.9是相同原生预算策略，不是精确90%整设备硬上限。所有分项及字节数保留于JSON。

原始compact结果仍含旧静态限制字符串“Prototype untested on GPU”；保留原件不改。以`STARTUP_COMPLETE_NO_SERVICE_RUN`及阶段数据判读：启动路径已运行，输入正确性与服务性能比较仍未运行。日志含FlashInfer capability警告，实际使用TritonExperts和FlashAttention2，未把CUTLASS算作已验证。

## 服务主对照仍未运行

| 对照角色 | 预定负载 | SLO goodput / 完整吞吐 | P99 TTFT / 完成 / 请求最大 gap | 实际 KV / batch 分布 | 执行 / 计划驻留 / 启动与 capture | 状态及副作用 |
|---|---|---|---|---|---|---|
| 调好的紧凑执行基线 | low、high | NA | NA | NA | NA | compact启动诊断不是调优后的强服务基线 |
| 按局部执行速度选择 | low、high | NA | NA | NA | NA | 需先证明至少两个竞争配置 |
| 简单预算规则固定组合 | low、high | NA | NA | NA | NA | 真实集合成本未测，不能声称最强简单基线已知 |
| 候选一 | low、high | NA | NA | NA | NA | 门槛前未实现选择器，无收益或退化结论 |
| 最强简单基线与候选的固定 KV 辅助 | low、high | NA | NA | NA | NA | 仅在主对照成立后运行，不替代主实验 |

机器可读完整列位于 `service_comparison.csv`。本表没有将历史 D 单臂重命名为 G 基线。

## 既有完整请求数据的可行性背景

以下复算只用事前选定的 D fixed2048 单臂，旧 GPU UUID `GPU-bf3fc5ab-804d-b9d6-759b-4390899f15b9`，不同于当前 GPU；D 修改了 prefill 调度，因此不能用于 G 方法胜负比较。固定输出长度、无网络交付测量，无重复噪声估计。

| 历史负载 | 完成 / SLO 通过 | Goodput req/s | 完整吞吐 req/s | P99 TTFT / 完成 s | P99 请求最大 gap ms | KV 峰值 / 可用 blocks | running batch P50 / P95 / max |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 12/12，12 通过 | 2.0301 | 2.0301 | 0.1204 / 3.1209 | 25.0887 | 914 / 36764 | 3 / 6 / 7 |
| high | 160/160，24 通过 | 0.6894 | 4.5958 | 5.3779 / 30.5486 | 72.6276 | 32031 / 36764 | 113 / 160 / 160 |

SLO 同时要求 TTFT ≤ 4 s、最大 token gap ≤ 100 ms、原定到达至完成 ≤ 20 s。high 中 16 请求超过 TTFT、136 超过完成时间，两类可重叠；无 gap 失败，无抢占。由此不能把 high 的 goodput 损失归因于 KV 不够。

历史 scheduled-token batch P50/P95/max：low 为 3/6/2053，high 为 136/2184/2207。high 超过 2048 是 D 的 prefill-only 预算外另计 decode，不能与未来 G 的原生总 token 预算混淆。完整分布及定义见 `evidence/historical_service_metrics.json`。

同一历史引擎记录的全集 graph capture 差约 0.17 GiB，估计 0.14 GiB，capture 6 s；均为日志舍入值，不是图独立稳态驻留。未保存可与 G 对比的局部 kernel 执行时间或完整启动规划时间，记 NA。原始 SHA 与内存余量计算见 `evidence/historical_feasibility.json`。
