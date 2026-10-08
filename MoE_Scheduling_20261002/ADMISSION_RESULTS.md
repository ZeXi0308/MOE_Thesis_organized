**停止当前 unequal-horizon 准入启发式。** 在本组实际执行中，model 相对 native16 的 episode wall 从 8.910 s 增至 9.509 s，慢 **6.7%**；mean flow 从 7.084 s 增至 7.543 s，慢 **6.5%**。这个决定只针对当前将“一步等待成本”与“新 prompt 全程附加成本”比较的启发式，不否定 MoE 调度这一研究方向。

六臂分别使用独立 fresh engine，按 native16 / static2 / static4 / static8 / age_gate / model 顺序各执行一次。共同配置为 OLMoE、24-expert cache、1 GiB KV、512 token budget；旧组 GSM8K 0–15 完整 warmup，测量组为源序 16–31 的 16 条完整 prompt、同时到达，每请求强制生成 32 tokens。此次输入选择未使用这 16 条的既往输出或质量；当前结果仍属探索，不能作为未来调参后独立验证集。

表中 flow 与 TTFT 从请求到达计时；TTFT/ITL 均列均值。wall 是原始测量 episode 时间，不含初始化与 warmup，未作人工扣时。copy bytes 为测量期间实际专家拷贝量，以 GiB（2³⁰ bytes）显示。

| 实际执行臂 | Wall (s) | Mean flow (s) | Mean TTFT (s) | Mean ITL (ms) | Copy bytes (GiB) | Groups |
|---|---:|---:|---:|---:|---:|---:|
| native16 | 8.910 | 7.084 | 3.106 | 128.318 | 294.035 | 2417 |
| static2 | 16.273 | 9.303 | 7.717 | 51.158 | 398.426 | 5008 |
| static4 | 11.665 | 7.433 | 5.058 | 76.616 | 351.949 | 3231 |
| static8 | 9.941 | 7.419 | 3.530 | 125.449 | 307.441 | 2682 |
| age_gate | 9.919 | 7.794 | 3.996 | 122.499 | 303.762 | 2792 |
| model | 9.509 | 7.543 | 3.408 | 133.398 | 299.449 | 2476 |

六臂全部 COMPLETE，均完成 16/16 请求、512 tokens，preemption 和重计算 token 均为 0。model 相对 native 的完整输出序列一致率为 **10/16**；age_gate 同为 10/16，static2/4/8 分别为 11/16、9/16、11/16。因此不能声称保持逐请求 token 完全一致，也未评估自然 EOS 或答案质量。

model 的 81 次决策包含 **2 次 myopic admit、11 次 myopic hold、35 次 age guard**，另有 1 次无运行请求时必须推进、32 次无等待请求；fallback 为 0。11 次 hold 中有 1 次仍存在 running prefill。共记录 81 次完成步骤观察，其中 **79 次实际用于 EWMA 并改变估计值**（mixed 21 次、pure decode 58 次），所以不是模型未更新造成的空执行；这些计数是决策次数，不等于实际新准入请求数。

model 相对 age_gate 的 wall/mean flow 分别减少 4.1%/3.2%，但仍输给 native16。在 static2/4/8 中，按 wall 和 mean flow 事后选择的最优者均为 static8；model 比它 wall 少 4.3%，mean flow 多 1.7%。这个最优 static 是 **hindsight 强参考**，不是在线选择基线。每臂只有一次执行，本报告不作显著性或普适收益结论。

下一唯一实验：六个独立 engine 使用相同 **4096** 编译容量，采用共同 **512** budget warmup，再实际执行 static token budget **512 / 2048 / 4096** 的两次重复，测量传输摊销随预算变化的曲面，记录 wall、flow、TTFT/ITL、实际拷贝字节与 groups。继续使用现有输入作为探索数据；该实验尚无结果，不据本组轨迹推演策略收益。

数值来源：[六臂 metrics.json](results_admission_r01/metrics.json)；原始各臂记录保留在 [results_admission_r01](results_admission_r01/)。本报告未修改 raw 或 metrics。
