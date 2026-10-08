# LongBench150：作者默认 DFS 重排的单格开发参照

本格按[冻结计划](C_LONG_DOCUMENT_QA_DFS_PLAN_20261001.md)使用 PEEK 作者提交 `3aca72b63c1dd6433193c0857b965acecdbfbb8a` 的 `reorder_for_prefix_sharing` 默认 guard 与 128-token trie 深度，在同一批 MultiFieldQA-en 源序 150 条输入进入原生引擎前重排。仅执行作者离线第一阶段的纯 DFS 入队顺序；未启用 shadow LRU、queue-aware eviction 或在线 PEEK。两格都保持原生 APC、4096 可用 KV 块、128 seq、1024 batch token、相同模型、提示、EOS-only greedy、64 输出上限与全体到达时刻 0。默认 guard 通过，150 条全部保留，149 个置换位置变化。结果是**先后两个开发单格的描述性对照**，不是匹配重复或等工作量速度实验。

| 实际观察 | 旧源序原生格 | 作者 DFS 格 |
|---|---:|---:|
| 完成请求 | 150/150 | 150/150 |
| 调度调用 | 512 | 441 |
| 调度 token 总数 | 462,946 | 390,288 |
| 扣除 decode 后的 prompt 调度 token | 459,154 | **386,498** |
| 输出 token（含 EOS） | 3,942 | 3,940 |
| 自然 EOS / 64-token cap | 127 / 23 | 127 / 23 |
| 全输出 English QA F1（全 150 分母） | 37.10656% | 37.13733% |
| 归一化 exact match | 17 | 17 |
| KV 分配返回 `None` / 原生抢占 | 0 / 0 | 0 / 0 |
| after-schedule 活跃块峰值 | 3,055/4,096 | 3,163/4,096 |
| host 生成观察区间 | 10.713917 s | 9.412255 s |
| host 平均 TTFT / 到达至完成 | 5.191331 / 5.739873 s | 4.535271 / 5.104493 s |
| 每请求最大不同 host 返回间隔均值 | 0.024505 s | 0.030753 s |

DFS 的 **386,498 prompt 调度 token** 恰好等于[先前 CPU 工作核算](long_document_qa_cache_headroom_v1.json)在“仅共享完整 16-token prompt 前缀、每请求至少重算末 token”下的理想 prompt 下界；在这个**受限模型及本批 prompt** 内已没有剩余 prompt 工作余量。源序到 DFS 的 prompt 差额为 72,656 token；调度总数差 72,658，其中另 2 token 是实际生成输出工作差异。这个核算不等于真实逐块命中/逐出计数，也不把理想界推广为其他轨迹的可达性能。

全 150 条中，6 条输出 token IDs/文本改变、3 条长度改变、5 条 F1 改变（2 升、3 降），结束原因均不变。host 到达至完成有 83 条变短、**67 条变长**；TTFT 有 85 条变短、65 条变长；每请求最大不同 host 返回间隔有 22 条变短、**128 条变长**。该间隔只反映不同批次的 host 返回时刻，无法细分同一次返回中的多个 token。重排本身耗时 0.015748 s，计入 DFS 格的 host 观察起点。单个较晚运行的格及生成工作差异不足以宣称匹配加速、质量因果提升或完整 PEEK 的收益。

原始运行根目录：`C_research_artifacts/20261001/c-instruct-longbench-dfs150-dev-v1/native/`。`launcher-receipt.json` 记录 2026-10-01 16:58:09.363988–16:59:27.930202 UTC，child 21031 以 0 退出并被回收；GPU 结束时 2 MiB、无 compute 进程，私有 stage 已移除，锁已释放。冻结包 SHA-256 为 `f9b78a1e200f38459af10d49e5e567998e2eb92b387396e47c72f4c5f6cf3a0d`。[质量 JSON](long_document_qa_dfs_quality_v1.json) SHA-256 `50e9e54b8eab4b154cf4a0cd1e7fa333c61ab7786444c2f6d0b7603e4a2ec971`；[全请求对照 JSON](long_document_qa_dfs_comparison_v1.json) SHA-256 `9d95257472517caf5a46eab5305dded9d535fb0e8f07828d4c7d6d12a5349a3d`。

这验证了已有作者默认重排在本批次能回收受限模型中的全部 prompt 工作余量；不构成 C 的新方法。拟议的 subtree 密度排序（每请求独有 prompt 工作量与子树请求数）是下一项**未运行**的直接开发测试，当前没有结果或收益结论。
