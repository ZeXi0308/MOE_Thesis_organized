# KV 恢复等待最小诊断（单格 S16）

0044 的额外停顿已定位到**数据就绪后的 GPU KV 块分配阻塞**。step52 已观察到 `finished_recving=True`，但 free blocks 为 0；剩余 3 个位置在 step52–55 连续申请失败，step57 才成功增加 1 块。这格仅观测原策略，不计作 R02 性能重复，也不证明任何改进策略的收益。

`capture_alignment.verified=true`，wrapper/capture 均为 179 步。16/16 请求完成，共 1416 个输出 token；与 R02 `02_swap16` 的 16 条完整输出及 prompt 均完全一致。排除时间和随机内部 ID 后，全部 179 步调度字段完全一致；抢占同为 49→0044、56→0045、72→0047。

| step | 0044 状态（前→后） | free blocks（前→后） | 0044 持有块（前→后） | 观测 |
|---|---|---:|---:|---|
| 49 | RUNNING→PREEMPTED | 1→44 | 45→0 | 0044 被抢占，computed 706→0 |
| 50 | PREEMPTED | 44→44 | 0→0 | 外部 704 位置、0 新位置的 allocation 失败 |
| 51 | PREEMPTED→WAITING_FOR_REMOTE_KVS | 90→0 | 0→44 | 704 位置恢复分配成功；同一步新 0045 prefill 726 位置 |
| 52 | WAITING_FOR_REMOTE_KVS→PREEMPTED | 0→0 | 44→44 | 开始时已收到 KV；更新后清除 ready 标记；3 位置申请失败 |
| 53 | PREEMPTED | 0→0 | 44→44 | 3 位置申请失败 |
| 54 | PREEMPTED | 0→0 | 44→44 | 3 位置申请失败 |
| 55 | PREEMPTED | 0→0 | 44→44 | 3 位置申请失败 |
| 56 | PREEMPTED | 0→44 | 44→44 | 0045 被抢占；未记录 0044 allocation |
| 57 | PREEMPTED→RUNNING | 44→42 | 44→45 | 0044 调用时 free=43，成功分配 1 块并调度 [704,707) |

每块 16 个位置，44 块覆盖 704 位置，恢复执行到 707 需要第 45 块。step51 的目标分配使 free 90→46，而整步结束为 0，且同一步启动了 0045；该 helper 未单独记录 0045 的分配调用，不能把所有中间块变动细分到它。所有目标 allocation 的 `full_sequence_must_fit=True`、`reserved_blocks=0`；step52 的 `_update_waiting_for_remote_kv` 返回 `None`，这是记录到的原函数返回值。

首次观察到 ready 为测量时钟 6.693400 s；step57 开始调度为 7.307443 s，**观察到 ready 后仍等待 614.0 ms**，至首个新 token 为 738.7 ms。ready 是 host 首次观察，不是 DMA 完成时间；上述等待是实际 data-ready 等待的下界，不能把抢占后的全部停顿归于 DMA。

0044 的完整 token gap 为 1.161711 s，缺席区间 1.036623 s，重新调度至新 token 0.124696 s；gap 内其它请求输出 88 token，缺席期间 79 token。原 R02 对应 gap 1.157883 s、缺席 1.032521 s；两格均恢复计算 3 位置，其中 2 个原 prompt 位置。时间仅用于机制定位，不作性能比较。

证据：[压缩结果](results_kv_restore_diag_r01/diagnostic_summary.json)、[原始观测](results_kv_restore_diag_r01/00_swap16/restore_wait_diagnostic.json)、[原始轨迹](results_kv_restore_diag_r01/00_swap16/raw.json)。后续最小动作应检验恢复请求所需最后一块的容量保障；本格尚未执行该动作，也未证明它优于现有简单 headroom 基线。
