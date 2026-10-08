# B：commit 当刻资源重检的证据边界

**结论（2026-09-29，HEAD `76d6d888de42081c63cd440a8a67623161d8181f`）：** 旧 D859/E1399 是已经入账的两个 *KV 块余额* 候选。隔离检出的旧 CPU 派生 JSON 记录两次 commit 分别有 `26/22` 个 running，在共同上限 `32` 下有 `6/10` 个空槽。恢复出的 G 诊断原始快照让其 **54 次 READY commit 的 H1 即时 KV 资助机会精确收紧为 0**；G 轻量与 H 轻量仍无逐次状态，机会数分别只能界定为 `0..126`、`0..690`。D/E 原始逐步数据仍缺席，也没有 H1 的原生执行或因果收益。GPU 对照 **UNRUN**；没有新 GPU 样本、控制器或原始数据改动。

## 可读取材料与复算

从仓库根目录执行：

```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/b/recompute_checkout.py
```

该只读脚本从 `RESULT_LEDGER.md` 取 D/E 已接受数值，验证并计算余额；从 G 的 `RESULTS.md` 取轮转数；从 H 已物化的 `execution_weste_26862/report_tables.json` 取轮转数；用 `GIT_NO_LAZY_FETCH=1 git cat-file -e` 查关键 blob 的本地可用性。实跑成功，输出 D/E 余额 `98/171`，G selected 共 `180` 次（诊断 `54`、轻量 `126`），H selected 共 `690` 次。传入隔离检出可追加核对旧 CPU 派生 JSON：

```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/b/recompute_checkout.py \
  --historical-checkout /private/tmp/moe-recovery-source-20260929-r01
```

此命令实跑验证 D/E 各 `25` 个 commit、各 `1` 个 direct 候选，running `26/22`，协议上限 `32`，空槽 `6/10`，当前 peers 一步增长 `1/0` 块，`pending_load_count=0`。它也核对恢复出的 G `diagnostic-eager/selective-store.json`：`2650` 个 step 与快照一一对应，`54` 次 READY commit 均 `free<need`，最小/最大缺口 `2/193` 块。G 四个轻量格原始事件合计 `128` 次 commit 检查，其中 `126` 次 READY 并实际轮转、`2` 次取消；其资格快照记录为关闭。脚本不写文件、不连接机器，也不会把未物化 Git blob 按需抓取。

`HEAD` 的树列有旧 `commit_recheck/{REPORT.md,selected.json,full.json}`、`commit_recheck_cost/analysis.json` 及 G/H `readback` 和 `analysis.json`；主 sparse/partial checkout 没有这些文件，且本地缺少其 blob。实际 `git show HEAD:.../commit_recheck/REPORT.md` 尝试 promisor 网络抓取后失败。随后 root 提供只读隔离检出 `/private/tmp/moe-recovery-source-20260929-r01`，本轮已直接读取其中旧 `REPORT.md`、`selected.json`、`full.json`、成本报告/分析，以及 **G 诊断和四个轻量格的原始 `selective-store.json` 与对应包内源码**。D/E 原始 `selective-store.json` 和 H 轻量格逐次 readback 仍未取得；G 轻量原件有 commit 事件但未开启资格快照。H 的 `report_tables.json`、G 的 `RESULTS.md` 与共享源码在主工作树。旧 D/E 派生结果可核对，但不能称从 raw 重新生成；G 诊断则可核对逐次状态，不等于执行 H1。

## 旧 D/E：CPU 快照有块与槽位，直接恢复尚未原生执行

以下 D/E 状态和成本均为 [共享台账](../../RESULT_LEDGER.md) 第 608、673 行的**既有结论**；本轮核对了隔离检出的旧 CPU 报告和派生 JSON，并非新增 GPU 样本或从原始事件重跑。`free−need` 是可复算的算术；“保留 peer 下一步后”来自旧派生模型。

| 历史 commit | prepare | free / need / 余额（块） | 原 victim KV（块） | running / cap / 空槽 | peer 下一步后余额（块） | 派生 CPU 状态 |
|---|---:|---:|---:|---:|---:|---|
| D859 | 858 | 193 / 95 / **98** | 233 | **26 / 32 / 6** | 97 | pure decode；peer 增长1块；pending load 0 |
| E1399 | 1398 | 277 / 106 / **171** | 184 | **22 / 32 / 10** | 171 | pure decode；peer 增长0块；pending load 0 |

旧 `commit_reason(...)=READY` 在现存 [staged_save_contract.py](../../staged_save_contract.py) 中只验 `free + victim.blocks >= target.remaining_blocks`、目标/受害者状态与版本、保存登记等；它不验 `free >= need`，也不主动验序列槽。旧 `selected/full.json` 的 `running_count` 和两个已接受实验 README 的 `Running cap32` 在这两例组合成 `6/10` 个空槽，故**旧 CPU 快照的槽位约束通过**。`free>=need` 仍只是当前 KV 资金条件。直接分支还需要目标仍 `PREEMPTED` 且未部分加载、目标与受害者身份/版本未变、队列无冲突、pool 所有权和 native load/flush 屏障成立；真实原生调度要再次即时验证，不能把 `READY` 解释为“host 必然可加载”。

旧派生 JSON 的 `pending_load_count=0` 实际取自当时 `skipped_ids` 长度，不是对所有 connector 队列的完整快照；尤其 D 的 prepare858 **job122 已登记但尚未提交**，所以不能把它写成“没有保存义务”。E 的 prepare1398 没有新增 store，之前 **60 job / 183 块**已完成。当前 H1 只改变 victim 动作，不撤销 D 已登记的延期 store，也不能追回 E 已完成的保存。原生 `jobs_to_flush`、load 完成通知、块可复用与数据可覆盖屏障是不同状态；仅看 free 数和空槽不能跨过它们。

旧路径随后 D victim0748 有 `464 MiB` 下一次 load、`8` 个重算位置、`0.112243 s` gap；E victim3790 为 `366 MiB`、`5` 个位置、`0.260893 s`。两者都不是该轨迹最大 gap。它们是旧路径的实际后续工作，不是 H1 可删除的净成本；部分工作可能重叠在其他请求执行中，H1 保留 victim 的 KV 也可能增加更晚 peer 压力。即时条件余额 `97/171` 仅说明旧模型中当前 peers 的**下一步块增长**不会立刻耗尽预算；空槽 `6/10` 也只属于历史 CPU 快照。两者都不证明原生 target 首新输出、未来 EOS 或完整请求结果。

## G/H：G 诊断精确为零，轻量格仍只有识别界

| 轨迹与来源 | 已实际执行的 selected 强制轮转 | H1 同类 commit 的可识别数 |
|---|---:|---:|
| G diagnostic-eager（逐次原始快照） | 54 | **0/54**：即时 `free>=need` 均不成立 |
| G 轻量 current/eager/current/eager 四格 | 16 + 41 + 50 + 19 = 126 | `0..126` |
| H 轻量 current/eager/eager/current 四格 | 120 + 221 + 242 + 107 = 690 | `0..690` |
| G/H selected 历史执行合计 | **870** | **`0..816`，不是发生率估计** |

G 轮转总数取 [G RESULTS](../../../../outputs/admission_capacity/20260915_natural_recovery_cadence_r01/RESULTS.md)；G 诊断的逐次判断使用隔离检出的 `execution_weste_26862/readback/results/diagnostic-eager/selective-store.json`；H 数取 [H report_tables.json](../../../../outputs/admission_capacity/20260915_natural_cadence_holdout_r02/execution_weste_26862/report_tables.json)。G 诊断与轻量格复用相同 64 篇，H 四格复用相同 128 篇；这些是**跨不同实际运行轨迹的 commit 数**，不是 870 个独立文档、独立重复或可转移概率。H 原生 full/no-extra-rotation 两格的各 `41` 次自然抢占不属于该 selected `prepare→commit` H1 动作空间。

G 诊断核对细节：原件有 `2650` 次 schedule 与 `2650` 个不同 step 的资格快照，`54` 个 `prepare` 都在紧接的 `step+1` 对应一个 `READY commit_check`，并与快照的 `plan_target/plan_victim` 身份一致；实际应用轮转也是 `54`。每个目标都在 waiting、状态 `PREEMPTED`，不在 running/skipped；原 victim 在 running。包内 `RequestState.remaining_blocks` 定义为 `max(0, ceil((prompt+output)/16)−held_blocks)`，块大小资格固定为 `16`；快照中的 `free_blocks` 来自 `pool.get_num_free_blocks()`，在同次 `begin` 的 commit 检查之前记录，中间无显式 pool 修改。54 个目标 `held_blocks=0`，按同 step 快照算出的 need 为 `91..245` 块，free 为 `6..200` 块，**每次 need−free 为正，范围 `2..193` 块**。因此不论其他槽位或 connector 条件，H1 在这 54 个历史 commit 上都没有可直接资助目标的必要 KV 条件。此为原生诊断执行留下的状态的离线计算，**不是** H1 的 GPU direct 动作或效果。

G 四个轻量格的原始事件显示 `128` 次 commit 检查、`126` 次 READY/应用、`2` 次取消，但均关闭了资格快照记录；H 只有汇总表。这些轻量数据没有每次 READY commit 的 pool free、目标 need、槽位、在途 store/load/flush 或块所有权，因此其机会下界仍为零，上界分别为历史实际轮转的 `126/690`。原 selector 在 prepare 看见 `free >= target need` 会 `noop`；H1 机会必须发生在随后一个准备步中，不能从“轮转很多”推出发生率。旧 D/E 的 `2/50` 是开发事件，不能外推为 G/H 轻量格的 4% 或 H1 确认集。G/H 的请求级 current/eager 损益只能说明 peer 代价需要完整计入，不能归因到尚未运行的 commit 重检。

## H1 的判别与反例

唯一预测：在同一已接受 target、同一 selected 保存与 native 服务规则下，若 commit 即时 `free>=native_need`、有空槽、无队列/在途冲突且所有权与屏障合法，H1 置 `V=∅` 应少一次该 victim 抢占，并避免**由这次抢占引起**的下一次 load/重算机会；该 target 的首个新输出不得被推迟。是否减少实际传输、请求停顿和改善 goodput 要由两条独立演进的完整请求轨迹决定。

反例/停止条件：该域没有合法直接事件；或有动作但 slot/connector 使 target 未及时恢复；或保留 victim 导致其他 peers 等待、后续抢占/保存/加载增加，抵消局部节省；或检查本身进入关键路径，使同底座 on/off 的完整请求指标无增量。输出长度、EOS 与调度轨迹可能随动作改变，不允许离线删除旧 victim gap 充当反事实。原生 full 和合理校准的兼容 LTR-style 仍须保留为系统与最近邻强基线。

证据上限是：旧 D/E 的**已保存 CPU 块余额、6/10 空槽与后续成本记录**，G 诊断 **0/54** 次即时 KV 资助机会，G/H 轻量格 **`0..816`** 次尚未识别的可能机会，以及当前源码中直接动作所需的状态约束。没有 H1 的独立执行，就不能说原生 direct 恢复已证明、H1 因果收益或 MoE 特有性成立；G 诊断的零机会也不能外推到 G/H 轻量格。`b/decision_model.py`、`b/test_decision_model.py` 是其他进程在本次工作中写入，未由 B 打开、修改、测试，也不属于本交付。
