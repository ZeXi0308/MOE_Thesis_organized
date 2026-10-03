# 2026-09-29 单主线交接：恢复提交时重检资源

## 接续事实与本轮唯一假说

- 仓库 `agent/publish-current-moe-code`，HEAD `76d6d888de42081c63cd440a8a67623161d8181f`，启动时工作区干净。过程规则见 `AGENTS.md`；科学事实以 `SERVING_RESOURCE_STUDY.md`、`PAPER_ARGUMENT.md`、`CURRENT_EXPERIMENT.json` 和共享 `RESULT_LEDGER.md` 的 2026-09-15 结果为准，`docs/current/README.md` 的 8 月主攻已被较新实测覆盖。
- G/H 的 selected/eager 改善全局最大生成间隔，但原生 full/no-extra-rotation 在吞吐、平均完成和多数请求自身间隔上仍有优势。H 两对 eager/current 的最大间隔变化为 −13.60%/−3.12%，后者只少 0.121 秒；这不是独立新方法或 MoE 专属证据。
- cooldown/window/headroom/growth-predictor 已停止。恢复后强制服务窗口在 cohort3 被 native/most 覆盖；自然 EOS 档没有反复短恢复。保存到 host 的有效历史在再次抢占后大多复用，不能按永久丢失计费。
- 2026-09-30 接续状态：G64/T30/Q10 [原生生命周期诊断](A_G64_NATIVE_LIFECYCLE_RESULT_20260930.md)已观察到真实 store/load/后续服务；同机 [完整服务三格首组](A_G64_PERF_PILOT_RESULT_20260930.md)已完成 native full、selected/eager、LTR-style T30/Q10，但四点开发网格尚余三点。LTR-style 是兼容恢复组件，不是完整 LTR；合理校准后的性能比较和新动作价值仍缺席，不能宣称超越最近邻。下面早期 `UNRUN` 记载是当时的历史计划。
- 旧 D859/E1399 的保存 CPU 快照在下一步提交时分别有 free193/need95、free277/need106，运行中请求 26/22 个、上限 32，故有 6/10 个空闲序列槽；旧路径仍驱逐持有 233/184 块的 victim。两例无 pending load，保留现有 peer 下一步后仍有 KV 余量。旧台账还记录了 default-off CPU adapter 及随后 victim gap 仅 0.112/0.261 秒的成本分析。旧报告和快照已在同一 HEAD 的隔离检出中恢复；原始 selective-store/raw 未恢复，物理动作和完整请求收益仍未证实。这是历史候选，不能写成新发现或预告最大停顿收益。
- G 详细诊断的原始 `selective-store.json` 现已在隔离检出中恢复：54 次 READY commit 对齐真实记录的即时快照，全部 `free<need`，故这条历史轨迹的 H1 直接资助机会为 0。G/H 其余轻量格无逐 commit 资源快照，识别界是 `0..816`；这不表示机制已在 GPU 上执行。

**唯一可证伪假说 H1：** 对一个已经按共同策略选中并完成合法保存准备的恢复目标，在 `commit` 当刻重新读取实际可用 GPU KV 块和序列槽；如果二者都足够且 native connector 状态允许目标直接恢复，则将本次资源让渡者置空。这个动作可能避免不必要的 victim 抢占、后续加载或重算及其请求停顿；已登记的 store 仍须实际计费。物理准入不保证目标首个新输出不晚于旧路径：调度 token 预算的 [CPU 条件反例](H1_TOKEN_BUDGET_ADDENDUM.md)已推翻这一无条件预测。若自然事件没有完整可行状态，或目标延期、额外检查、未来 KV 增长和其他请求代价抵消完整服务收益，就接受简单规则无增量的结果。此机制改变 **commit 时实际 victim 动作**，使用准备之后的新物理状态；它不是提前预留 headroom、调整 cooldown 或延长服务量子。

本轮只实现/评价这一个 `commit` 重检，target 选择、victim 原排序、selected 保存、恢复后的运行规则保持同一强简单底座。`keep_existing_commit` 是合法回退。旧 D/E 事件用于开发，不充当独立确认。若 LTR 诊断失败或其同后端语义尚不合法，先修共同基线，暂停 H1 的性能组；不并行推进第二个 controller。

## 统一观察与动作接口（B/C 均按此输出）

决策点为 scheduler 的 `prepare` 已接受、但本次 `commit` 尚未抢占 victim 之前。`StateSnapshot` 为只读快照，带 `schedule_step`、`observed_at`、`target_id`、`planned_victim_id` 和下面字段。字段未知必须标 `UNKNOWN` 并回退旧路径，不能填未来 EOS、剩余生成长度、未来 route 或未来到达。

| 状态字段 | 实际来源和更新边界 | 用途与未知处理 |
|---|---|---|
| 请求 phase、原队列序、prompt/已生成/已计算位置、EOS/已结束状态 | native `Request`、`scheduler.running`/waiting/preempted；每次 schedule 前后刷新 | 确认原目标/受害者仍有效；身份或版本变化则取消本 intent |
| `last_new_output_at`、逐请求生成间隔 | `native_capture.py` 的实际新输出事件；每次交付刷新 | 只用于停顿评估及后续扩展，不用 output cap 代替剩余工作；缺失则不作停顿预测 |
| GPU 有效 KV、块所有权、空闲块、共享引用、序列槽 | `KVCacheManager`/`block_pool` 和 native 分配器；commit 前即时刷新。槽位已用数包含 `running` 与 `num_waiting_for_streaming_input` | 用**物理可回收量**验资；共享/未知所有权或布局不符则回退 |
| host 有效 KV、store/load/flush 在途与完成通知 | native offload connector/worker 通知；事件到达时刷新 | 未完成 store/load 不算可恢复；状态不明则回退 |
| peer 下一步 KV 增长、近期 step 时间、传输排队 | `Request` 的 computed/owned 块与 scheduler 近期调用及 connector 队列；每步刷新 | 记录 H1 的后续代价与暴露关键路径，不以全体未来增长总预留阻断合法动作；缺项记未知 |

统一动作表示 `a=(恢复目标 r，资源让渡者 V，保存/恢复方式 m，后续服务额度 q)`。H1 仅允许在同一个已接受目标和 `m=selected-native` 下，将 `V={原计划victim}` 改为 `V=∅`；`q=native` 不变。`no-op` 表示保留原 commit，绝不凭 CPU 预测跳过 native 的 store/flush/load/ownership 屏障。门禁与效用评价分开：`free_blocks >= native_need(target)`、含 streaming 保留的空闲槽、目标和 peer 状态、pending 队列、块唯一所有权等仅作**原生准入前置筛选**；native 分配和 connector lookup 仍须实际兑现，否则该资格运行 `INCOMPLETE`。未来实际输出、TTFT、gap、完成时间、吞吐及 host/GPU 过程占用决定是否值得执行。计时只按暴露关键路径记账，不把混合 decode 调用全算成可删除重算税。

建议纯函数输出 `Proposal(decision_id, state_version, target_id, victim_ids, reason)`；root 在同一 scheduler 生命周期内 `observe -> propose -> validate/commit -> feedback`。任一版本不符、目标 EOS、资源不足、部分加载、队列忙或所有权不清，回退原生路径并记录原因。不得让离线评价字段进入在线 `propose`。

## B/C 所有权与交付

| 负责人 | 独占可写范围 | 有界任务 | 交付 |
|---|---|---|---|
| B：状态/结果分析 | 本目录 `b/` | 只读重算 D859/E1399 及 G/H 中同类自然事件的频率、可见字段、peer 代价；辨别 `free>=need` 与 load/flush/slot 真正可执行的差异。最多列 3 个候选缺口，但只对 H1 写预测和反例。不启动控制器/GPU，不改 raw。 | `b/EVIDENCE.md`、必要的只读分析脚本和复算命令；明确旧数据无法识别的因果。 |
| C：后端接口与强基线 | 本目录 `c/` | 核查已接受 LTR-style adapter 与 selected/eager、native-full 的共同后端语义；先按索引与台账核对旧 default-off CPU 组件，若缺失 blob 不可复用，再在独立补丁文件中重建最小 commit 修改及状态过期/块归属/部分加载/EOS/回退的定向 CPU 测试。不可修改 `rotation_native.py`、任何已接受包或共享 controller。 | `c/INTEGRATION.md`、适用于清洁共同源码的 `c/commit_recheck.patch`（若需重建）、测试脚本或测试补丁、复算命令。 |
| root：主线与唯一 GPU 执行 | 本目录根、共享源码、`CURRENT_EXPERIMENT.json`、`RESULT_LEDGER.md`、新实验 raw | 选定唯一动作、逐个审阅并集成 B/C 补丁、维护台账和状态；串行安排原生、selected/eager、兼容 LTR-style、H1 on/off；现场验机器/锁/预算/源码后独占执行。 | 可运行共同代码、一次代表性完整对照、失败/未完成也入账；未经明确 GPU 资源与预算授权，GPU 状态写 `UNRUN`。 |

提交方式：B/C 只在各自目录新增文件，向 root 报文件清单、相对路径、`git diff --no-index`/unified patch、测试命令和结果。共享源码修改只交 `.patch`，由 root 顺序审阅后应用；不切分支、不 `git reset/checkout`、不提交或推送、不覆盖其他进程未合并的文件。已有原始结果和已接受包保持只读。任一文件出现外部新修改时先停在该文件，root 再决定合并。B/C 不修改主结果台账或当前实验状态，也不连接 GPU。

## 串行比较及停止条件

先核准已接受 LTR 诊断的实际原生生命周期，再在同一资源/输入/后端上固定 native full、selected/eager、校准后的兼容 LTR-style 与 H1-off/on。开发时先选一档自然压力，不重跑已完成 G/H 资格；新机器上的结果另列，不与旧硬件直接配对。运行前固定到达 cohort、EOS/cap、host 与 GPU 物理预算、warmup/服务/drain 和超时。**开发阶段预选的无业务含义阈值前沿**为 `D_F∈{10,20,30,40}s`、`D_G∈{1,2,4,8,12}s`；完整请求且有新输出、TTFT 和最大生成间隔均达阈值才计 goodput，除以同一 capture 时长。没有两个不同 host-return 时间戳的请求，其 gap 分布记未定义；若成功结束且有输出，goodput 的生成间隔条件真空成立，不插值同批输出。主评价是这张完整 cohort 的请求级 goodput 前沿和相对 native 的实际 token 吞吐/平均 flow 代价，不事后挑一格改为唯一主指标。同时报告 TTFT、完成、每请求最大 gap 分布与逐请求损益。拒绝/失败/未完成留在分母；未完成的 flow 按 `max(180s, capture_end−arrival)` 罚时，另报完成者真实 flow。输出长度和终止变化并列报告。计算入口为本目录 `evaluate_goodput.py`。

最小消融为同一 selected/eager 底座 `commit_recheck=off/on`。若 on 从未改变 victim，停于无动作空间；若只修共同后端错误，则吸收进所有兼容基线，不作独立方法；若 H1 被合理 LTR 或更简单的直接验资覆盖，保留其强基线地位。任何动作收益先按实际物理 store/load、首新输出、peer 停顿和完整请求核验，不能从少几次驱逐推吞吐收益。旧 G/H 不是 H1 的盲测集；确认须冻结策略后用新输入/到达序列。机器/资源/预算未授权时，仅做 CPU 接口、补丁和可执行命令，GPU 结果 `UNRUN`。


## GitHub 原始数据恢复（2026-10-04）

本次推送保留全部未忽略的代码、报告、失败和实验结果。122 个大型 JSON 以相邻 `.json.gz` 无损保存；原始总大小 8,034,228,861 字节，压缩后 364,900,511 字节，全部完成解压往返 SHA-256 校验。本地原件保持不变。可重建的 GPU 编译缓存不纳入 Git。

从仓库根目录执行下列命令，再运行需要 `raw.json` 或其他原始 JSON 的分析、冻结包检查。恢复约需额外 8.04 GB 空间；脚本校验已有文件且不覆盖它们。清单也包含压缩文件哈希。

```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/restore_large_data_20261004.py
```

[完整压缩清单](GITHUB_LARGE_DATA_20261004.json)与[恢复脚本](restore_large_data_20261004.py)仅改变 Git 存储形式，不改变实验字节或科学结论。
