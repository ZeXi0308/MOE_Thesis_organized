# Fit-first on 格前三个最大生成间隔：事件链定位

依据：已完成的探索配对结果 `A_FIT_FIRST_PAIR_RESULT_R01_20261001.json`（SHA-256 `37613561779a464d44de5ccfb06c5415892101712923d8448f98796b4d410d87`），以及 on/off 原始 `raw.json`、on 的 `selective-store.json`。时间均为各格测量起点后的主机秒；两个格子的时钟不共享同一执行状态。

| on 请求 | on 最大间隔：前后累计 token、秒 | 间隔内实际抢占与恢复 | fit-first 绕过原目标 | 同源 off 最大间隔、完成流时 |
| --- | --- | --- | ---: | --- |
| `0017914` | 187→188；63.013477→75.199650，**12.186174 s** | step 4000 强制抢占本请求；step 4739/4740 常规 prepare/commit 后恢复，call 4742 返回新 token | **20 次** | 2.311171 s；63.987198 s（on 64.996584 s） |
| `0020662` | 179→180；65.026831→70.695971，5.669140 s | step 4125 原生抢占；step 4467 自身被 fit-first 接纳，call 4469 返回新 token | 0 次 | 0.826890 s；56.910128 s（on 59.523130 s） |
| `0020531` | 563→564；70.599492→75.344721，4.745228 s | step 4466 原生抢占；step 4743/4744 常规 prepare/commit 后恢复，call 4747 返回新 token | 0 次 | 1.309584 s；58.893281 s（on 57.749253 s） |

`0017914` 的上一个 token 在 call 3999 返回；随后 `prepare` step 3999 将它列为计划 victim，`commit_check=READY` 与 `commit_recheck=KEEP_RECHECK_OFF` step 4000 后，`raw.preemption_events` 在 63.014209 s 记录了对它的真实 `_preempt_request`。它在 gap 内没有再次被抢占。65.043567–74.878211 s 的 20 条 `fit_first_choice` 将它列为 `original_target`，分别选择 7 个不同的替代请求；20 次都原生接纳、随后有新输出。每次原目标完整历史需 193 块，现场空闲 77–192 块，替代者需 57–187 块。这 20 条是**等待目标被跳过**，并非对它的 20 次抢占。

最终 `prepare` step 4739 以 `0017914` 为 target，step 4740 的常规提交实际抢占计划 victim `0018254`（75.142023 s）；`0017914` 在 call 4742 的 75.199650 s 返回第 188 个 token，step 4743 才记录 `target_new_output`。这定位了等待结束的观察路径；不证明若早些停止改选，原目标会更快恢复。

在 `0017914` 的 gap 对应 step 4000–4742，日志另有 662 次 `prepare_rejected`，原因均为 `victim cannot fund target full history`；只有两次 `commit_check`，均为 `READY`（step 4000 和 4740），没有该区间的 commit 取消。`prepare_rejected` **没有 target ID**，不能把 662 次逐条归给 `0017914`，也不能断定替代者消耗空闲块导致其单 victim 融资失败。另两个 top-gap 各有一次真实抢占，但没有被 fit-first 反复作为原目标绕过：`0020662` 后来自己成为 fit-first target，`0020531` 则由常规轮换恢复。三者均以 `length` 完成 1024 token；on/off 是不同轨迹，其中前两者输出序列也不同。
