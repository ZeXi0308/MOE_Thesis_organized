# H1 B1：零直接提交时的解释边界

证据：`A_H1_PERFORMANCE_PAIR_B1_AUDIT_R01_20261001.json`，SHA-256 `cde40968cf2fa0fc2080ff34b2fd393a247d367830a2665d577f0186790fb962`；两格均 128/128 完成、归档哈希核验通过。以下只解释这个先看过的 H128 输入上的 B1 配对，不把两次执行当成同状态重放。

| 观察项 | off → 第 1 格 | on → 第 2 格 |
| --- | ---: | ---: |
| `direct_commits` / 原生预留检查次数 | 0 / 0 | 0 / 0 |
| READY 后的 `commit_recheck` 结果 | 216 次 `KEEP_RECHECK_OFF` | 254 次 `KEEP_INSUFFICIENT_FREE_BLOCKS` |
| 强制旋转 | 216 | 254 |
| 完成流时均值（秒） | 39.311 | 35.805 |
| 每请求最大生成间隔的最大值（秒） | 3.699 | 3.176 |
| 实际输出 token / 观测时长（秒） | 123736 / 87.322 | 123248 / 80.236 |

**代码事实。** 同一冻结 runner 的 on 仅额外启用 `--commit-recheck`。在 `staged_store_rotation.py:217–235`，on 的资格谓词每次都在完整空闲块检查处返回 `KEEP_INSUFFICIENT_FREE_BLOCKS`；`_native_reservation_disposition` 对该结果原样返回，未调用原生 inflight 预留探针。off 直接给 `KEEP_RECHECK_OFF`。两者在各自 READY 提交时随后均执行计划 victim 的原生抢占、计数及 target 保护；on 没有进入 `direct_resume`。on 增加了只读谓词求值和不同的原因记录。冻结源码 `staged_store_rotation.py:41–94,217–235,322–340` 支持此分支结论；归档事件与零直接提交相符。

**测量与解释。** B1 的预设配对指标通过：on/off 输出率 1.0840、完成流时 0.9108、最大间隔更低。但 128 个请求中 on 的流时较短 125 个、TTFT 较短 121 个，最大间隔较短 76 个、较长 52 个；两格有 60 个输出序列及 4 个停止原因不同。on 的输出 token 总数少 488，输出率差主要伴随观测时长缩短 7.086 秒。两格的旋转次数和运行轨迹不同，额外只读检查及异步运行时序也可能影响后续状态；现有归档不能把 B1 的性能差归因于 H1 的直接提交机制。B2 是预先冻结的反序配对，应独立报告，不能用 B1 通过来选择或取消它。
