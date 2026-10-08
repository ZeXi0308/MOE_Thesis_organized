# LongBench150：子树工作密度排序的单格开发结果

按[冻结计划](C_LONG_DOCUMENT_QA_DENSITY_PLAN_20261002.md)，在已知全部 150 条 prompt token ID 的同一批次内，保持 16-token 前缀子树连续，按子树独有 prompt 工作量/请求数升序安排兄弟子树。对照是已完成的作者默认 DFS 单格；模型、输入、到达时刻 0、原生 APC、128 seq、1024 batch token、4096 可用 KV 块及 greedy/EOS-only/64-token cap 相同。此行动是已知 Smith 规则的树排序启发式开发参照，不是新颖性或原生连续批处理最优性的证明。两次是先后运行的**单格描述性比较**，不是匹配重复。

| 实际观察 | 作者默认 DFS | 子树密度候选 |
|---|---:|---:|
| 完成请求 | 150/150 | 150/150 |
| 扣除 decode 后的 prompt 调度 token | **386,498** | **386,498** |
| 总调度 token / 输出 token（含 EOS） | 390,288 / 3,940 | 390,299 / 3,951 |
| 自然 EOS / 长度上限 | 127 / 23 | 126 / 24 |
| 全输出 English QA F1 / 归一化 exact match | 37.13733% / 17 | 37.08429% / 17 |
| 原生 KV 分配返回 `None` / 抢占 | 0 / 0 | 0 / 0 |
| after-schedule 活跃块峰值 | 3,163/4,096 | 3,269/4,096 |
| host 平均 TTFT | 4.535271 s | 3.645126 s |
| host 平均到达至完成 | 5.104493 s | 4.241056 s |
| host p95 到达至完成 | 9.068325 s | **9.583250 s** |
| host 全批观察时间 | 9.412255 s | **9.903573 s** |
| 每请求最大不同 host 返回间隔的最大值 | 0.039531 s | **0.656449 s** |

相对 DFS，平均 TTFT 下降 **19.627%**，平均 flow 下降 **16.915%**；两项均有 90 条请求改善、60 条恶化。p95 flow 和全批完成时间却上升。最大不同 host 返回间隔达 0.656449 s；调度调用 380→381 的 host 时刻为 8.832650→9.489040 s，前者调度 911 token、11 running、0 waiting，后者 11 token。它位于最后 prefill 转 decode 附近，**原因未由日志确定**；不能删去这次实际长间隔或归因于 JIT。host 返回间隔也不能分解同一次返回中的多个 token。候选重排 CPU 耗时 0.075014 s，已计入本格测量起点。

两格 prompt 调度工作同为此前受限前缀模型的 386,498-token 理想下界，但输出相差 11 token，所以不构成严格等工作量速度比较。150 条中有 6 条输出 IDs/文本改变、2 条长度改变、1 条 `stop→length`，5 条 F1 改变（2 升、3 降）。候选的 F1 下降且 126 EOS/24 cap；无论均值如何，不能称完整服务质量或全请求体验优势。

**原件与运行状态。** [质量 JSON](long_document_qa_density_quality_v1.json) SHA-256 `2bf4842e94faf31aad70e5d2b124aee164ccba1fa927db80f53d47ad48701a93`；[全请求对照 JSON](long_document_qa_density_comparison_v1.json) SHA-256 `fd249a4035c35930c54a07446e73219409d601095abe2730f0aed58d4383917a`；[三臂静态图](long_document_qa_order_plot_v1/long_document_qa_order_development_v1.png)同时显示源序、DFS 和候选的 prompt 工作与 flow 分布。完整原始归档 `c-instruct-longbench-density150-dev-v1-complete.tar.gz` 为 4,031,125 字节，SHA-256 `faf5fe3394e452a6b79c7f9abbb71d15e70622fb91c58f3b909d9692a4eb324f`。SSH31124 对应 launcher 于 2026-10-01 17:28:21.723566–17:29:40.495115 UTC 完成；child 30812 reaped0、GPU 2 MiB 且无 compute 进程、私有 stage inode `128:333` 已移除、锁释放。

冻结 launcher receipt 的 `scientific_scope` 描述误沿用了 “author default DFS” 字样，原件保持不改。实际冻结 cell、`density-reorder-source.json`、提交顺序及 `status.subtree_density_reorder_s=0.0750139039` 均标识并执行的是 density 排序。当前单格是开发信号，不支持密度排序优于作者 DFS 的一般结论或 C 新方法主张。同机固定 **DFS / density / density / DFS** 配对块正在准备，**尚未运行**；不能把它的结果预先写入本格。
