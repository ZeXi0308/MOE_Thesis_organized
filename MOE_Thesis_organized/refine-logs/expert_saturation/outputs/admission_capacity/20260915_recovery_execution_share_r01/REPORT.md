# 给定恢复对象后的执行份额

本会话承担恢复执行与运行域泛化（子会话C），与资源演进方共用状态和动作接口，自主研究给定恢复对象后的计算与KV分配。旧计算份额模块在原定长域保持暂停；不独立选择victim，不重复容量过滤六格或主方服务分析。本方代码在独立分支 `agent/a-recovery-components-20260914`，没有新GPU运行。

**最新结论见末节：补齐restore_first简单对照，从D/G全部6次冷恢复的首个resident前态分叉；preserve_calls与restore_first的目标首输出调用数全相同。** 三例余额规则多给peer 13/50/52次输出机会，其中50/52已被原ready_first覆盖；另两例只改变输出顺序、一例完全相同。不能把仅四个晚期冲突窗口的末态相同外推到完整resident阶段，也不能把多输出机会称作完整服务加速。默认关闭钩子仍为CPU组件、未装入G，无新GPU。主方优先独立输入与较长到达验证，本方不新增份额组；首输出后completion-bridge继续暂停。

## 现有模块的实际差异

| 策略 | 恢复/victim 决策 | KV 保护和终点 | 恢复与正常 decode 的计算份额 |
|---|---|---|---|
| 当前停顿优先基线 selected/eager | most_output victim、最长absence target、两阶段准备/提交；全局cooldown=0 | 当前完整历史保护，首个实际新输出或terminal解除；原生保存/加载与覆盖屏障 | 原序可行peer先用，目标使用余额；G已测。selected/current仅cooldown=20，保留为对照 |
| 本轮默认关闭 recovery_compute_share | 复用已选target/victim、原顺序、保存范围与全部启动规则 | 完全复用上述保护与终点，不延长到首输出之后 | 仅resident冷重算时保住本次R−(ceil(R/B)−1)B个目标位置，其余预算给原序可行peer；CPU钩子分支已测，GPU未接入 |
| restore_first计算组件参照 | 同一外部目标、首个resident前态开始，不另选victim | 同一完整历史保护和首输出终点 | 目标每次尽量取满，末次余量给可行peer；D/G共6个原生CPU分支已执行，非完整论文系统或GPU性能基线 |
| native_full_native系统参照 | 完整原生保存，无额外most轮转 | 原生KV/offload生命周期 | 原生计算规则；与selected/eager同时有保存和调度差异，G已测，不能归因一个开关 |
| F的 staged most_output＋selected / native_full | 相同most/最长absence与两阶段提交；4096 GPU块、16GiB host | 首个新输出解除；正常完成处理归还块池，覆盖旧块前可能须flush | 共同使用可行peer＋剩余预算；两者只改变保存范围与增量更新，沿用F原组比较 |
| 本方条件完成分配组件 | 外部给定target，不选victim；仅从resident选择声明cap可资助的finisher | 保留target和finisher到finisher声明cap所缺块；观察实际完成后重读池并重规划，保留原生覆盖屏障 | 每次给target/finisher各1位置，其余peer用余额；仅CPU条件路径，未安装完整策略 |
| 旧 most_output | `AbsenceRotation.decide`：缺席目标、guard 合格的最多已输出 victim，检查释放块能否资助目标 | `rotation_native` 只为主动交换目标登记保护；每调用留足当前完整历史，首个新输出被观察后解除 | 先按原生 running 顺序服务可行 peer，目标使用剩余预算；`hold` 只拦会侵占保护块的 peer |
| fit_scan | LTR 200/10 组件＋可行项扫描；较低优先级 victim 资助 | 无首输出义务 | 原生组件的贪心分配 |
| guard_all | 同 LTR 组件，rank-prefix＋恢复义务优先级 −2 | 实际恢复登记，保留历史；观察到首个新输出或完成才解除 | 优先给恢复，可能占满 1024；最后不足预算时仍可服务 peer |
| guard_residual | 与 guard_all 相同 | 同一保护 | 为后序 resident、pending=1 的候选预留一个 token，恢复使用余额 |

代码依据是已执行包中的 [rotation_native](../20260914_recovery_holdout_comparison_r01/execution/readback/pkg/rotation_native.py)、[ltr_recompute_native](../20260914_restore_token_reservation_r01/execution/readback/pkg/ltr_recompute_native.py) 和 [restore_obligation](../../../experiments/admission_capacity/restore_obligation.py)。容量过滤的 `least_feasible` 只改变资助前的候选过滤，执行层仍是 `rotation_native`。其六格和完整资源模型直接复用 [funding 报告](../20260914_funding_filter_comparison_r01/REPORT.md)，不另算原主表或重跑。

因此，“保护 KV 不必独占计算”是已测发现及已有 most 行为；不是本轮新机制。most 的恢复保护也不等于它会自动保护每个自然 resume。

## 保留的真实运行结果

旧 residual/guard_all 两 block 的目标 3640 均在调用 405–408 恢复，408 返回首个新输出，409 观察后解除。分配分别为 guard_all `[994,1024,1024,232]`、residual `[994,994,994,292]`。406、407 中 residual 各让 30 个 peer 实际返回新 token，guard_all 为 0；首输出调用没变，不能说首输出毫秒没变。见 [原定位](../20260914_restore_token_reservation_r01/analysis/localization/FINDINGS.md)。

全组 residual/guard_all 吞吐 +4.351%/+1.926%，但最长 ITL 增加 0.046/0.198 s；相对 fit 两次全部请求完成更晚。后续 cohort3 中 most 的完整 wall 25.327/26.579 s、最长间隔 2.814/2.942 s，均优于 residual 的 28.117/28.450 s、4.291/4.369 s。residual 的平均完成和多数请求仍有优势。原 [强基线结果](../20260914_service_window_holdout_r01/REPORT.md) 保持，不按单项指标宣布胜负。

## 可复用的最小执行模型

一句研究问题：**给定上层选定的恢复对象和已实际释放的 KV，能否用逐调用计算分配及时兑现新输出，在固定停顿要求下减少完整请求完成代价？**

实现：[recovery_execution_share.py](../../../experiments/admission_capacity/recovery_execution_share.py)。输入是目标及原顺序 resident peers 的当前历史 H、已计算 C、已持块 A、已返回输出、声明剩余 cap，以及实际空闲 F、块大小 b、计算预算 B。它不选择 victim，不读取距离真实 EOS 的长度，不接收未来轨迹或固定每请求毫秒成本。

1. 目标待执行 `R=H−C`；当前历史尚缺块 `E=max(0,ceil(H/b)−A)`。实际 F 不足 E 就返回资源层，执行层不能靠降准入上限或借未来完成释放补足。
2. 保留 E，按上层现有顺序服务 pending=1 的 peer。peer 本次增长只可使用 `F−E`；被延后者保持全部 KV。目标执行剩余计算份额。
3. 分配后必须有 `F_after ≥ E_after`，先分配和执行，返回后才计新输出及完成释放。内部 C 增加不解除保护；computed 已到 H 而尚无新输出时返回 `WAIT_OUTPUT_RETURN`。原生远端 KV 尚未 ready 时返回 `WAIT_NATIVE_READY`。
4. `Protection.released` 只接受已观察的新输出数或完成。caller 必须在该义务期间阻止别的准入消耗保留块；本函数是可组合的计划器，还不是 runtime hook。

`ready_first` 抽取已存在的 most 份额规则。另实现可解释的 `preserve_calls` 条件：`k=ceil(R/B)`，本次至少分给目标 `q_min=R−(k−1)B`，peer 只使用其余份额。若每调用 B 不变、目标合法持续执行且保留不被外部破坏，则剩余成功调用数上界每次减少一。它只约束到首输出输入的**计算调用数**；混合 batch 的时间和最终新输出通知仍需实测，不能据此保证毫秒停顿或多个请求的全局等待上界。

`restore_first` 是先完成本次恢复份额再使用余额的简单对照，末次有余额时仍服务 peer，不能使用故意空闲的弱基线。4 项针对性测试通过，覆盖整数边界、peer 损失、KV 守恒、不能借后续完成、原生 ready 及实际输出/完成解除边界。没有添加全面审计。

## 本轮实际完成的分析

当前结果为 [revision02/analysis.json](revision02/analysis.json)，脚本为 [probe_recovery_execution_share.py](../../../experiments/admission_capacity/probe_recovery_execution_share.py)。原六格 raw 只读，逐文件 SHA 留在结果中。

| 范围 | 已检查调用 | ready_first 与真实 token map 完全相同 | preserve_calls 改变动作 |
|---|---:|---:|---:|
| 两 most | 132 | 132 | 0 |
| 两 least | 120 | 120 | 0 |
| 两 least_feasible | 126 | 126 | 0 |
| 合计 | 378 | 378 | 0 |

这里检查的是已有保护建立后的调用；126 个首次强制交换调用因快照位于释放之前明确排除，自然无保护调用不在这 378 个中。不是完整 runtime 重放、独立重复或性能测试。新整数规则只在边界 fixture 中产生区别，当前六格没有支持为它单独跑 GPU 的动作空间，停止该规则在本负载上的性能扩展，不扫阈值寻找触发。

另复用已定位的 filtered `before1026`：目标 0020902、H3772/C0/A0、F242、27 resident peers。固定目标、无新 victim、无新准入、假设未知 EOS 不提前触发，各规则独立推进到首输出输入，最多 8 调用：

| 计算分配 | 四调用的目标位置 | 四调用的 peer 输出机会 | 目标首输出条件 | 延后 peer 机会 |
|---|---|---|---|---:|
| restore_first | 1024/1024/1024/700 | 0/0/0/27 | 第 4 调用 | 81 |
| ready_first＋持续 KV 保留 | 997/997/997/781 | 27/27/27/26 | 第 4 调用 | 1 |
| preserve_calls＋同保留 | 同上一行 | 同上一行 | 第 4 调用 | 1 |

逐调用保留只需在第 4 调用延后 source0019699 的一次输出机会；该调用分配峰值 F=0，目标 E=0。source0017190 达到已声明 cap 后才释放 256 块，未提前借用。原自然恢复实际在这一调用被 peer 增长挤掉、2991 部分恢复清零；事件原解释复用 [既有1026证书](../20260914_service_window_context_r01/funding_1026_prestate_r01/NOTE.md)。这里没有使用 1027 以后的 raw 生成替代路径。

新结果相对原固定 prefix22 证书的 20 个 peer 机会转移更小，是**逐调用可行分配**的结构结果；尚未执行这条替代路径，不能声称省时间、提高 goodput 或超过 most。是否把自然 resume 纳入义务由上层策略决定，不在本函数里偷偷选择恢复对象。

最初 `analysis.json` 的第三个 `exclusive` 参考禁止全程共批，末次也空置余额，不能当作 guard_all/强基线。已保留该初稿，并以 `revision02` 中真正使用末次余额的 restore_first 替换；378/378 和单 peer 机会结论未变。

## 旧未运行六格的评价预设及自然 EOS 边界

以下3s约束仅属于本方后来暂停、从未运行的重算六格设计，不更改共同主线的探索目标。合流后唯一当前主评价及执行版本以 [统一清单](../../../CURRENT_EXPERIMENT.json)和[一页论证](../../../PAPER_ARGUMENT.md) 为准；本方不另选SLO。旧设计预先固定：**所有已开始生成请求最大引擎返回 ITL ≤ 3.0 s 时，比较整组全部到达请求的平均完成延迟，越低越好。** 3.0 s 来自既有 most 校准域约 2.8–2.94 s 的能力范围，是新实验的研究约束，不是业务 SLO；不据此重判旧数据。两策略必须同一要求、同输入、同一物理卡、同 KV 池和相同恢复后端。若都不满足则无可行胜者；若仅一方满足只报可行性差异，不能又换吞吐宣布全面胜出。

不完成/失败/拒绝使该组主比较不能合格，保留全部到达而不对成功子集重算平均值。并报 TTFT、TPOT、每请求最大 ITL/完成差、受损人数、整组 wall/吞吐，及恢复前等待/实际恢复启动/下一新输出。分母 `wall=scheduler inclusive+engine其余+engine外`，正常 decode、重算、观察及调度均保留；混合调用不能整体扣成重算税。此模型没有已校准的动作时间曲线，因此当前不做毫秒排序。

自然 EOS 的既有 [四格结果](../20260914_streaming_recovery_r01/RESULTS.md)：64篇完整自然文章、0.5s持续到达、4096usable KV块、原阈值30、真实EOS允许；256请求全部完成，9恢复均持续到完成，零再丢弃，most 强制动作0，生成中恢复间隔35.92–379.08ms。每格只有6条EOS、58条达长度上限，覆盖仍有限。该冻结范围没有本执行层要修复的短恢复问题；不降低阈值、改输入或压低KV制造动作。没有动作空间也不否定其它预先规定的高压范围。

## 合流与唯一接续

上层负责恢复谁/何时开始/真实释放哪些 KV 及独立后续演进；本模块只返回同一目标的计算 map、保留块、held peer 及输出义务。新主机26862环境复制继续由共享协调记录的唯一原方负责，本方未另传模型或占卡。

唯一下一小实验是与资源演进方共用其确定的恢复对象，在同一后端验证**自然恢复是否承接现有逐调用 KV 保护**：关闭/开启仅此义务，计算规则仍是已有 ready_first，保留 most 完整强基线和全部请求结果。首先应在其现有状态模型接入该计划器，再由同一执行组验证原生实际分配与首输出；不另外搜索 victim，不为零动作的 preserve_calls 启动一组。若既有 most 在指定范围已覆盖完整目标，保留基线，停止这一执行扩展。

Verdict：`CPU_EXECUTION_MODEL_ONLY / MEASUREMENT_ONLY`；evidence 为实际旧运行的调用核对＋独立条件块/位置演进。新 GPU、端到端收益、质量、真实动作时间排序和全局 Oracle 均未测。最强基线仍是 most，完整 headroom 未证明。当前整数份额规则的失败类别为已有状态无增量动作；自然恢复义务属于尚未执行的接入差异。复开条件是预先定义的独立范围中出现同强基线后的实际执行缺口，不是修改阈值。

本轮直接答案：**先持续保留到新输出的 KV，再让可行正常 decode 使用计算份额；已有 most 已这样做。当前可复用的新资产是其纯执行模型，待验证差异在自然恢复是否获得同样的执行义务，而不是重新发明 victim 策略或宣称更复杂份额规则有收益。**

复跑（新输出目录）：

```sh
python3 -B -m unittest discover -s refine-logs/expert_saturation/experiments/admission_capacity -p test_recovery_execution_share.py
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/probe_recovery_execution_share.py --funding-bundle refine-logs/expert_saturation/outputs/admission_capacity/20260914_funding_filter_comparison_r01 --output-dir /private/tmp/NEW-execution-share-analysis
```

## 接续：原生恢复入口和资源模型已接通

本轮接续起点为 `2905fe7e`，前一轮归类为实质进展。新增 `rotation_native.install(..., protect_native_recovery=False)`，默认关闭；开启时只在原生 waiting 请求已成功分配、已登记本次 scheduled token、状态尚未变为 RUNNING 的位置接入义务。它不另选恢复对象或 victim，实际失败分配和首次 prefill 都不登记。登记后复用原有保留与余额机制，并阻止其它 waiting 准入消耗该义务的保留块。实现副本为 [native_integration/rotation_native.py](native_integration/rotation_native.py)，只读原共享入口没有被覆盖。

这是一项集成差异，不是新的抢占算法。支持域仍是同步原生 recompute、无 APC/connector/推测输出、单一不共享 KV 池、恢复期间其余 running 请求为 pending-one decode。尚未给保存/加载后端或开放到达引擎增加此开关。

执行模型和原生入口使用的保护终点仍是已观察的新输出；仅内部 computed 增加不解除。原生 waiting 选择实际开始后才登记保护，因此不把保护前的等待时间算成该计算份额可以直接消除的时间。

**完整原生 schedule 的 CPU 资格已经执行通过。** [verify_native_recovery_execution.py](../../../experiments/admission_capacity/verify_native_recovery_execution.py) 约260行，复用已有块池/队列夹具，直接编译封存原生 schedule、preempt、cached-request payload 和 post-schedule 方法。新增支撑代码是为了真正执行新 waiting 钩子；旧夹具使用自写 schedule loop，无法检查钩子位置。没有复制另一套调度策略或扩大审计层。块分配和同步输出返回仍是模拟对象，没有执行模型 forward、真实 token 语义或 GPU。

同 `before1026`、同截止此前的真实队列事件，off/on 各独立推进5调用。当前显式指定adapter源码的可移植命令也已通过，结果在 [native_fixture_cli.json](native_integration/native_fixture_cli.json)；先前默认模块入口结果保留。关键结果：

- 1026–1028：两者各997目标位置＋27 peers，分配相同。
- 1029：off 原生抢占目标，丢弃已计算2991位置；on 延后0019699一次decode，目标执行781位置，模拟返回第701个输出；分配后free0，批次返回后完成peer释放256块。
- 1030：on 观察到新输出解除目标保护。另一个声明cap夹具中目标先完成移除，随后仍正常解除。
- on前4调用的token map、空闲块和held逐项符合纯执行模型；off前4调用与原raw的token map/空闲块相同。1026后的raw仅用于验证，未驱动替代路径。
- 分配失败不登记、首次prefill不登记、computed增长不解除、新输出及完成解除均通过；旧四个rotation allocator兼容用例也通过。

**同时已与另一会话的当前资源模型合流，而不是重建模型。** [recovery_execution_model_adapter.py](../../../experiments/admission_capacity/recovery_execution_model_adapter.py) 只为原 `recovery_progress_model.py` 的自然恢复提交点增加默认关闭开关，并沿用原 selector/状态转移/完整成本位置账本。原模型没有被原地修改；本次读入版本保留在 [resource_model_source.py](native_integration/resource_model_source.py)。只读两份校准原件，在同一上层规则内做on/off，不枚举victim。

| 固定上层规则 | 默认off与真实原件完全匹配 | 自然义务次数 | 最后完成步 off→on | 重算位置 off→on | 请求完成步改善/损害 |
|---|---:|---:|---|---|---|
| most_output | 1134调用 | 4 | 1223→1223 | 93044→93044 | 0/0，32个均相同 |
| least_feasible | 1362调用 | 4 | 1451→1448 | 89761→86770 | 4/1，其余27个相同 |

[完整模型结果](native_integration/model.json) 保留全部32个请求的完成步差。filtered的4个改善为−4/−3/−4/−3步，受损0019699晚1步；不能把3调用或2991重算位置换算成可扣毫秒。most 首次分叉1077是延后另一waiting恢复的512位置，故执行路径并非零动作，但全部完成步及总重算不变；这只排除了当前结构指标上的增量，不是总执行时间等价证明。

本次没有证明相同3秒要求下的平均完成优势，该主评价仍须真实GPU请求结果。当前最强most没有结构完成收益，所以不为此另开独立性能路线；将自然恢复义务作为共享执行修正，待资源方出现需要它的已确定动作时复用同一个runtime入口。新26862环境已由原方完成复制并启动其原两格保存资格，本会话未占卡、未启动新组，也不把另一会话的保存结果冒充本修正结果。

当前状态更新为 **CPU_NATIVE_SCHEDULE_QUALIFIED / GPU_UNRUN**。未测部分仍包括新分支真实输出/时间、同停顿要求下的完整服务收益及自然EOS泛化。唯一接续是资源方将已确定的恢复动作与本默认关闭入口放进同一运行包，保留most和逐请求损益；不重新做容量过滤六格、不搜索新victim，也不以CPU正方向直接启动参数矩阵。

接续命令（新输出路径）：

```sh
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/verify_native_recovery_execution.py --cell refine-logs/expert_saturation/outputs/admission_capacity/20260914_funding_filter_comparison_r01/execution/readback/results/funding-block0-least_feasible --adapter-source refine-logs/expert_saturation/outputs/admission_capacity/20260915_recovery_execution_share_r01/native_integration/rotation_native.py --output /private/tmp/NEW-native-recovery-fixture.json
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/probe_native_recovery_model.py --resource-model refine-logs/expert_saturation/outputs/admission_capacity/20260915_recovery_execution_share_r01/native_integration/resource_model_source.py --funding-bundle refine-logs/expert_saturation/outputs/admission_capacity/20260914_funding_filter_comparison_r01 --output-dir /private/tmp/NEW-native-recovery-model
```


## 最新合流：保存后，计算份额还有什么可改变？

当前接续 HEAD `eed642a7`。先准备过 most/filtered/filtered+natural-guard 正反序六格，复用原容量过滤输入、同6656块和1024预算，主目标为每请求最大ITL≤3s后的全到达平均完成。原20文件包1.3MB已暂存新机，未解包、未传controller、未初始化GPU；在新保存强基线结果完成后主动保持 `HELD_GPU_UNRUN`，已在共享GPU协调记录让出队位。不是GPU失败或机制NO-GO，不自动重启。分析器和必要CPU旧轨迹检查保留供限定后备使用，没有为旧强基线重跑原六格。

新增问题是**给定最新保存路径实际选择并commit的目标，加载就绪后是否还存在计算份额争用，足以支持改变调用分配？** 直接复用 [新service完整服务结果](../20260915_repeated_kv_service_r01/analysis/REPORT.md)，不重算其性能表。该结果将两阶段most＋重复KV保存纳入有16GiB host预算的强基线；其收益仍是该封闭同长输入和后端下的小样本测量。

| 当前路径 | 上层与恢复后端 | 实际执行层区别 | 可比范围 |
|---|---|---|---|
| 旧即时most / filtered | 即时单victim，native recompute，无CPU KV | 主动目标全历史保留；可行peer优先，目标用余额 | 原异构容量过滤域；本轮沿用旧六格结果 |
| 最新 staged save-off | 两阶段most，OffloadingConnector，实际分配16GiB host但禁止测量store | prepare后下一步commit；主动目标保护到新输出；无自然waiting登记 | 与下一行同输入、GPU/host、后端、观测 |
| 最新 staged save-on | 相同两阶段most和16GiB host；按native有效前缀加载、缺失尾部重算 | 同样的KV保留和计算份额；另保留WAITING_FOR_REMOTE_KVS与加载完成依赖 | 当前此后端强基线，不能与无host旧组直接排名 |
| 本方 natural guard | 不改旧上层；已资格的native recompute路径 | waiting实际分配/提交后登记同一输出义务 | 未给connector后端接入，GPU未运行，不能据此宣称超越save-on |

新保存路径没有natural-waiting自动登记；但“没有登记”不等于本轨迹存在缺陷。其38次主动commit均兑现输出，已有原分析44段恢复均有新输出。下一比较必须尊重加载/失效/原生ready依赖，不能移除旧adapter的connector拒绝检查后强行套用。

实际完成的只读分析为 [staged_boundary/analysis.json](staged_boundary/analysis.json)，脚本 [probe_staged_execution_boundary.py](../../../experiments/admission_capacity/probe_staged_execution_boundary.py)。覆盖各诊断臂**全部38次主动commit**，没有按有利时刻选样；另外6段自然恢复没有被冒充成已验证保护分配。全请求损益/时间成本继续引用上述原表，不把诊断两格当作性能重复。

| 给定主动目标的执行测量 | save-off | save-on |
|---|---:|---:|
| commit / 实际新输出兑现 | 38 / 38 | 38 / 38 |
| 保护内调用 | 152 | 115 |
| 实际执行目标计算的调用 | 152 | 41 |
| 从首次实际计算到新输出只需1调用的恢复 | 0 | 37 |
| 不执行目标计算的调用 | 0 | 74 |
| resident前态可直接输入纯执行模型 | 114 | 3 |
| 这些前态的原分配逐项吻合 | 114 | 3 |
| preserve_calls改变上述分配 | 0 | 0 |
| 保护期间因KV保留延后peer调用 | 0 | 0 |

save-on的37次已有前缀恢复，首次计算待执行2–29位置，一次便返回新输出；全部正常decode的位置需求也能放进同一1024预算。剩余1次为首次无已保存前缀的冷恢复，3274位置、4计算调用，和off相同。41个就绪计算调用均满足 `q_min + 当次全部pending-one peers ≤ 1024`；off的152调用也满足，因此不给整数份额约束另跑GPU或扫阈值。

74个无目标计算调用对应37个commit/load-dispatch调用（快照PREEMPTED）及37个后续WAITING_FOR_REMOTE_KVS调用。**不是74次浪费或可免费删除的停顿。** 加载依赖和完成通知仍需兑现，peer同时在正常输出。另有37个首次执行调用的begin快照仍是WAITING_FOR_REMOTE_KVS，但原生随后消费已完成通知并执行；不能从computed位置或最终actual_scheduled反推它在begin已ready。脚本将这些转换调用仅归类为已观察的实际需求，不输入前态计划器，不把加载位置说成重算或免费计算。

纯模型的117个合法resident输入全部匹配实际分配。调用数约束的新增动作仍为零。这个事实只说明当前已测状态下没有少一次目标计算的份额机会；batch形状可能影响一次调用耗时，不能声称墙钟余量为零，也不能把跨臂peer输出总数不同当成损害，完整请求结果以原服务表为准。

合流接口保持简单：上层提供既定target、实际释放/持有KV及**已消费的原生ready状态**；加载未ready则等待原依赖，ready后按历史保留＋可行peer余额分配，只有真实新输出才解除。当前begin快照缺 `finished_recving_kv_req_ids/failed_recving_kv_req_ids`，因此未承诺能在线精确定位通知到执行的所有延迟。下一最小工作由共享资源模型处理**load完成通知→可执行→新输出**这一段；先补同一状态机中的ready交接，目标/victim及份额规则不变，不重做候选搜索。

最新 Verdict：`MEASUREMENT_ONLY`；证据是两格真实native诊断的全量38目标分析＋117合法resident调用的CPU分配核对。已测边界为单模型、同长强制输出、6656GPU块/16GiB host、两阶段保存；未知EOS和开放到达适用性仍引用前节自然EOS无动作结果，不改阈值制造问题。最强基线已吸收staged save-on；没有新GPU方法收益或全局Oracle。失败类别为当前份额规则在最新已测就绪状态无增量动作；若预先确定的新负载/预算下ready工作与peer预算冲突、或已开始恢复又被挤掉，才重新检验执行规则。

**直接回答：当前先保留KV、再共享计算即可；在新的保存强基线中，37/38恢复已加载到只剩一次小尾部计算，继续独立调份额没有已证实的调用收益。执行层应与资源方合流到原生ready和实际新输出的交接，而不是把旧自然恢复hook包装成新方法。**


## 统一指令后的职责与实质接续：原生 ready 后没有额外漏调度

本方继续承担恢复执行层，非主研究负责人；不接管资源方的target/victim与资源演进，不修改主会话的实验清单。主会话已完成保存能力对照并接受save-on强基线；资源方已完成容量过滤六格、独立状态模型与既定cohort3迁移准备；本方不重复这些实验或主表。全局cooldown20/0是主会话选定的下一组，原absence30/0与本方旧重算六格均不启动。

本轮唯一假说：**保存加载已经完成并报告后，是否因ready到实际执行的交接而漏过后续调度机会？** 这是前一轮计算份额没有增量动作之后的定向问题，不重新选择启动阈值。

本轮实际读取同一diag-on的全部43个真实load dispatch及completed-job通知，按job ID、请求身份、engine-call区间和新输出返回对齐。结果 [native_ready_handoff/analysis.json](native_ready_handoff/analysis.json)，可复用脚本 [probe_native_ready_handoff.py](../../../experiments/admission_capacity/probe_native_ready_handoff.py)。相对前一节38个主动commit范围，本次还覆盖另外6个原生自行恢复的load，不把同一raw增加为新重复。

- 43/43 load均在通知所在engine-call之后的**紧接调度调用**执行；漏过的调度机会为0。
- 43/43均在这次计算调用返回首个新输出；剩余工作2–29位置，中位数11。其中37个有主动恢复保护、6个没有主动保护。
- 43个执行调用的begin快照都仍是WAITING_FOR_REMOTE_KVS。此标签说明尚未消费promotion，而不能单独说明物理传输仍未完成。
- 通知观察入口到下一schedule开始为0.799–2.108ms，中位数1.398ms。观察入口在原connector处理之前，且包含详细host快照和其他CPU处理，不能称纯等待、可回收余量或生产开销。dispatch到通知36.229–39.547ms同样包括查询/调用边界，不是纯H2D时间。

本地封存原生源已与本组运行时来源合同核对；没有下载或重建后端。源码载体是 [native_source.json](../20260914_load_ready_contract_r01/native_source.json) 和 [native_offload_source.json](../20260914_kv_roundtrip_feasibility_r01/native_offload_source.json)，本组约定见 [runtime_source_hashes.json](../20260915_repeated_kv_service_r01/execution_weste_26862/readback/pkg/runtime_source_hashes.json)。按其中源码字符串解码后的行号：

| 执行环节 | 源码位置 | 本方依赖的语义 |
|---|---|---|
| load完成上报 | offloading/worker.py:328–363 | 查询transfer完成，返回finished_recving与completed job；不是提前修改本call调度 |
| worker通知返回 | kv_connector_model_runner_mixin.py:95–109 | 执行上下文退出时取完成通知 |
| scheduler消费 | scheduler.py:1888、2622–2647 | update_from_output先更新connector，再加入finished_recving集合，供后续schedule使用 |
| 原生promotion | scheduler.py:2586–2601 | WAIT_REMOTE先检查完成集合，再处理有效前缀、回到PREEMPTED/WAITING |
| 实际分配/计算 | scheduler.py:935–959、1011–1031 | promotion后仍须实际KV分配成功，才能登记scheduled tokens |

取消请求的迟到通知只触发释放，再次抢占不能沿用旧ready状态。通用scheduler虽有invalid-block修复路径，本版OffloadingConnector worker对transfer失败直接assert；本次全部成功加载不能外推成已验证故障恢复。没有增加故障测试或新适配器。

**改变的决策：停止本批ready交接机制投入。** 本轮曾写未接入的ready适配器草稿，发现全部43次均已在首个合法后续调度兑现后即删除；不再为它增加CPU测试、模型层或GPU包。当前缺乏证据要求修改ready交接或份额；不把相同状态重放得更细当成方法。

结论严格限于本固定负载/资源/同步后端的43次成功加载：排除的是“通知已到后额外漏调度”的解释，不是证明物理传输不可优化、所有调度等待已解决、墙钟无余量或研究问题死亡。自然EOS无反复恢复的既有范围继续作为无动作边界，不调低阈值制造问题。

主评价、基线和下一执行均合流主会话。下一动作是复用其cooldown20/0原始结果，按已选target观察更早启动是否带来新增恢复中断/短服务及peer代价；若未出现执行缺口，不重新启动本方份额/保护搜索。该组的GPU启动与主分析只归统一清单的原负责人。

复算（仓库根目录，输出路径须不存在）：

```sh
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/probe_native_ready_handoff.py \
  --cell refine-logs/expert_saturation/outputs/admission_capacity/20260915_repeated_kv_service_r01/execution_weste_26862/readback/results/diag-on \
  --output /tmp/NEW-native-ready-handoff.json
```

本轮Verdict：`MEASUREMENT_ONLY / 本批ready交接扩展暂停`。证据是既有真实native请求/加载事件的定向分析，非新GPU或另一轮重复；最强基线为共同staged save-on。未测故障恢复、异步引擎、质量、自由EOS和新启动策略收益，无全局Oracle。重新投入条件是在预先指定的新运行域或共同候选实际轨迹中发现通知已到而漏调度、恢复计算被打断或预算无法共同容纳的新证据。**当前执行层不缺新的ready规则；应让共同启动时机实验决定是否会产生值得修正的新执行状态。**


## 最新合流：更频繁切换是否产生计算份额争用

继承执行层职责，本轮唯一问题是**主方取消全局cooldown后，给定其已经选定并提交的恢复对象，恢复与正常decode是否开始争用计算预算？** 主方按统一合同完成 `20260915_saved_recovery_start_r01` 的1诊断＋4主性能格，共160请求全部完成并释放资源。本方只复用其正式回读，不启动、归档、下载或重算完整服务主表。固定6656 GPU块、16GiB host KV、1024计算预算，保存开启，目标/victim/absence30/residency30及首输出解除保护均不变；唯一上层变化为全局cooldown20→0。

结果：[eager_execution_boundary/analysis.json](eager_execution_boundary/analysis.json)。原探针只扩展cell标签和主表引用路径，分配模型与判定逻辑未改；实际新诊断执行通过原有身份、输出时刻、预算和前态分配检查，没有增加测试层。

| 给定主动目标的执行测量 | current旧诊断（引用） | eager新诊断 |
|---|---:|---:|
| commit / 首个新输出兑现 | 38 / 38 | 84 / 84 |
| 加载后仅1次计算的恢复 | 37 | 83 |
| 这些恢复首次计算的待执行位置 | 2–29 | 2–28 |
| 冷恢复的计算调用 | 1次恢复、4调用 | 1次恢复、4调用 |
| 保护内有目标计算的调用 | 41 | 87 |
| 最低目标份额＋全部resident peer需求超过1024 | 0 | 0 |
| 因KV保留延后的peer调用 | 0 | 0 |
| 合法resident模型调用 / 分配完全相同 | 3 / 3 | 3 / 3 |
| preserve_calls改变分配 | 0 | 0 |

旧列直接引用本报告staged_boundary原结果，不当作新组的额外性能重复。新83次加载恢复均可在一次调用放入全部目标剩余工作和正常decode；冷恢复仍为3274位置、4次计算，只有其中后3个resident前态输入纯模型。加载转换调用仅报告实际需求，不能倒推为调用入口已知的ready状态或在线预测。

新组另有166次保护内不执行目标计算的调用：83次PREEMPTED/加载派发及83次WAITING_FOR_REMOTE_KVS。其间peer正常执行；这些调用不能整体计成可删除的浪费。保护区间累计6771个peer新输出只用于确认peer在推进，区间覆盖随轮转数变化，不能与旧组累计数相减宣称服务增益。该探针不重复原方的短服务、再次抢占和完整请求损益分析。

主服务结论仅引用原方 [execution_weste_26862/analysis.json](../20260915_saved_recovery_start_r01/execution_weste_26862/analysis.json)：两对eager相对current最长engine-return gap为−9.7975%/−10.4133%，平均完成为+2.0417%/+1.4104%，吞吐为−0.5021%/+0.1244%。固定原主目标，报告降低停顿所付完成代价，不改选吞吐胜负、不设事后SLO、不宣称显著。引用版本SHA在本分析的full_service_reference中，未从诊断计时替换主表。

**改变的决策：本域的计算份额扩展停止，不再以更密集轮转为由增加规则或重跑旧包。** 新的实际动作频率已明显增加，但首输出前没有出现最低目标份额与正常decode不能共存的状态；因此“取消cooldown会暴露需新份额规则修正的争用”这一解释在本诊断中不成立。它不证明传输无成本、任意混合batch墙钟相同或首次输出后的服务稳定性已解决。后者属于共同主方已有的完整服务与资源演进问题。

本轮Verdict为 `MEASUREMENT_ONLY / 本域份额机制暂停`，证据是新真实native诊断的全部主动commit，非替代策略GPU结果；最强基线为staged most_output＋native重复KV保存。无全局Oracle/净墙钟上界，未新增自然EOS或其他预算域实验。原自然EOS无反复恢复边界保持，不降低阈值制造动作。唯一接续是主会话用该组完整请求损益决定启动节奏的适用范围；本方仅在其预先定义的新运行域实际出现计算预算冲突或首输出前再中断时重开执行修正，不独立搜索候选。

复算（仓库根目录，输出路径须不存在）：

```sh
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/probe_staged_execution_boundary.py \
  --service-bundle refine-logs/expert_saturation/outputs/admission_capacity/20260915_saved_recovery_start_r01 \
  --labels diagnostic-eager \
  --full-service-reference refine-logs/expert_saturation/outputs/admission_capacity/20260915_saved_recovery_start_r01/execution_weste_26862/analysis.json \
  --output /tmp/NEW-eager-execution-boundary.json
```


## 自然native-full：暂缓抢占能否等来资源释放？

按新的长期专项职责指令恢复研究，不再把空的next_group视为CPU研究禁令。本轮HEAD起点abf1d17c（独立工作区干净）；主工作树仍de64dae5且有并行修改，本方只同步自己的文件。主方已完成自然selected资格及native-full一格，本方复用其canonical原件与主分析，不重新统计保存/加载、生命周期或主性能。

当前正确强基线是 `20260915_natural_native_full_gate_r01` 的most＋native完整增量保存，实际4096 GPU块/16GiB host、1024计算预算、64异构完整文章、0.2s到达、EOS允许。原方已给出 `NATIVE_FULL_INCREMENTAL_SCOPE_QUALIFIED`；不再拿旧selected下1–2输出两例直接当作full下事件。其主分析已有36恢复、零0/1–2输出再抢占；本轮没有降低短段阈值，直接取**全部16个恢复后再抢占段**，按实际动作为9个上层forced（排除）及7个native（全部分析）。事件集合是事后定位，不能冒充未经选择的新负载或在线target选择器。

唯一假说：给定已恢复target，延后部分peer的增块是否能保持目标输出，并等来一个请求完成？替代解释是只推迟容量耗尽，把停顿从target转给peer。最小区分工作为当前状态容量界＋独立CPU状态分支；不读取未来EOS或实际完成次序。

### 可复用模型与动作边界

沿用 [recovery_execution_share.py](../../../experiments/admission_capacity/recovery_execution_share.py) 的Request/State，新增 `decode_release_envelope` 与 `allocate_completion_bridge`。仅含resident pending-one decode、无共享/null块，现有持块在完成前不返还，无新准入、外部释放或抢占。七个前态均验证 `sum(resident持块)+free=4096`，全部prompt＋声明cap不超过model length。

令C为已计算位置，A为持块数，U为声明上限尚余输出，b为块大小，F为当前实际free。U是合法最坏终止界，**不是实际剩余长度预测**。若模型长度可能先终止，caller须提供更早的有效上限。

```text
d_i = max(0, ceil((C_i + U_i)/b) - A_i)
若 min_i d_i > F：
    无提前EOS时，任何计算顺序都不能让一个请求率先完成；
    最多还能产生 sum_i(b*A_i - C_i) + b*F 个输出机会，随后容量耗尽。
```

证明：首个完成之前追加块x_i非负且Σx_i≤F，完成i须x_i≥d_i；将 `C_i+s_i≤b(A_i+x_i)` 求和即得机会界。它计入未写位置，不能把每请求余量都简化成不足一块。若允许无损回收未写整块、EOS提前结束或其它释放，该条件可能失效；因此它只否定**永久延后peer即可保证等到释放**，不否定暂时服务转移有价值，更不是延迟或全局Oracle。

若存在容量可行的完成者，组件选择声明剩余调用数最少的可行resident（同数按所需块、原顺序）。对target和该peer，只保留它们到该peer声明上限期间尚缺的块；正常peer按原序使用余额，每调用仍服务target。实际观察到任何完成后必须停止该义务，等待原生输出/保存/flush退休并读取真实free再重新决策。CPU分支在完成输出边界停止，**没有把该请求持块提前加回free**。这只是一条给定target的条件执行规则，没有改变victim或恢复对象选择。

### 新状态结果

[完整结果与逐请求机会](native_full_release_boundary/analysis.json)来自 [probe_decode_release_boundary.py](../../../experiments/admission_capacity/probe_decode_release_boundary.py)。所有输入SHA和排除原因已记录。两分支从同一前态各自推进C/A/输出计数，不沿用原策略后续轨迹；假定无提前EOS，既有到达请求仍保留为等待而不新准入。

| native再抢占调用 | target尾号 | free / 任一请求到cap最少新增块 | 无首次完成时的输出机会总界 |
|---|---|---|---:|
| 685 | 5122 | 2 / 25 | 243 |
| 800 | 5122 | 1 / 19 | 224 |
| 863 | 0406 | 0 / 19 | 196 |
| 1013 | 0287 | 1 / 15 | 234 |
| 1021 | 0748 | 1 / 16 | 190 |
| 1231 | 0748 | 0 / 8 | 182 |
| 1377 | 2680 | 3 / 1 | 存在首次完成路径，不适用该耗尽界 |

7/7目标的下一decode当前可资助，0/7整个resident集下一decode可共同资助；计算预算均充足。前六类的简单持续历史保护只将目标缺块推迟3–16次调用，仍无完成者，故不为它们加固定长窗口。这里7个前态相关，不能当作7次独立实验或估计总体发生率。

1377的可行分支无需按身份写规则：自动选择peer0433，其声明cap尚余16，target2680尚余109。两者各需额外1块，共保留2块，剩余1块给原序其它peer。比较如下：

| 同前态分支 | 到停止时target输出机会 | 停止边界 | 代价 |
|---|---:|---|---|
| 原native已观察调用 | 当步0，实际再抢占 | 后续原轨迹由主方分析 | 当步其余请求真实执行；不与CPU调用数换算性能 |
| 延长已有history escrow的CPU简单对照 | 6 | 第7次调用target缺块；无人完成 | 6次调用共112输出机会；这是延长规则，不能冒称原most完整策略 |
| 条件完成者保留CPU分支 | 16 | 第16次调用peer0433到声明cap | 16次共200输出机会；5103/0464/0406各16调用无输出，所有24请求明细保留 |

两个分支停止时刻不同，112→200不是吞吐或同窗口净收益。后者到边界free仍为0，0433持有134块仅是待检查的释放对象，未计为可用资源；host保存任务可能推迟回收。自然EOS终态直接引用原方本格58个length/6个stop；本探针7个事件的目标最终全部到cap。旧selected诊断的6个stop均早于首轮转是旧组边界，本轮不将其当成新的独立复现，仍未验证恢复途中EOS。模型允许EOS更早解除义务，但不能拿该允许性当作实测泛化。

### 改变的决策与唯一接续

旧份额无冲突结论继续保留；新域问题明确为**增长块分配与完成释放能否连接起来**。停止“任何短恢复都加固定保护窗口”的推理：六个前态已经给出有限机会耗尽反例；只保留一个条件组件，用可见上限判断有没有持续服务至完成的可行路径。容量界适用于一类状态，1377只是由该规则找到的正例，不宣称是最优动作或完整服务改善。

本轮交付为STRUCTURAL_FROM_NATIVE_STATES／CPU组件，GPU替代轨迹、传输退休、质量和完整效率均未测。6项测试通过：4项旧行为＋512个微型状态的所有合法顺序与界对照＋一个不能借未来释放的分配反例；没有新增审计层。主评价仍为相同停顿要求下的完整服务效率，当前不另设SLO。

下一最小工作：在同一native-full调度入口验证这个条件分配能否真正跨到完成及物理回收，先以1377同前态检查调用合法性和原生store/flush依赖；其后才提出完整episode对照。给资源模型方复用本容量条件与分配函数，不接管其target/victim搜索。主方新接受的selected/native_full轻量四格由其既有prepare_start_contrast代理唯一执行；本方不接管该组、不改变包或分配规则，先复用它回答强基线完整效率。未准备多个冻结包，未争用GPU。

复算（在含canonical原件和本方脚本的共享仓库根目录，输出须不存在）：

```sh
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/probe_decode_release_boundary.py \
  --repo . \
  --bundle refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01 \
  --output /tmp/NEW-native-full-release-boundary.json
```

## 补记：完成后可分配，与可安全覆盖分开

本轮继续C执行层职责。起点为独立分支c8eaef91；共享de64dae5及其他会话修改保留。已读统一指令、CURRENT、共享台账和原方主分析，容量过滤六格及保存/生命周期统计直接复用。唯一假说是上一节的条件分配能通过原生调度走到完成，但资源重用可能附带保存依赖。证据上限为原生方法CPU执行及既有基线的一个完成边界，不是新GPU策略结果。

**纠正上一节的退休解释。** 本版OffloadingScheduler.request_finished正常返回False，不要求等保存完成才归还GPU块。完成输出处理调用Scheduler._free_request/_free_blocks；下一调度若重用未完成store的source块，build_connector_meta登记jobs_to_flush，worker.handle_preemptions提交deferred store后wait。store完成通过completed_jobs，不通过finished_sending。因此上一节“等待原生输出/保存/flush退休并读取free”改为：**观察完成，原生输出处理后读取实际free并重规划；执行覆盖前保留原生flush屏障。** 不在计划到cap时提前释放，也不等待一个不会到达的finished_sending。

最小组件的算术分配不变，只修正caller合同；没有新controller、target/victim搜索或时间预测器。源码依据为相同哈希的Scheduler（_free_request:2207、_free_request_blocks:2248）及封存offloading源码（request_finished:1268、build_connector_meta:1122、worker.handle_preemptions:281）。以下CPU代码执行这些原始方法；allocator、输出、hash/store生成、DMA是显式替身。

### 新执行证据

[verify_completion_bridge_native.py](../../../experiments/admission_capacity/verify_completion_bridge_native.py)从原1377前态独立演进，不读取之后的原始调度轨迹生成动作。新输出是CPU占位返回，没有真实EOS采样；上层动作和新准入在条件切片内暂停，target身份是事后诊断输入，不是在线触发资格。

| 边界 | 无pending store替身 | 有pending store替身 |
|---|---:|---:|
| 与纯模型分配及free逐调用一致 | 16/16 | 16/16 |
| target新输出机会 | 16 | 16 |
| 第16次完成后归还池 | 134块 | 134块 |
| 下一次resident-only原生调度 | 23请求；分配22块 | 23请求；分配22块 |
| 原生重用屏障 | 无待flush任务 | submit_store → wait → 才能覆盖 |

[共享入口CPU结果](native_full_release_boundary/native_method_shared.json)的pending job900001是注入的依赖条件，不是替代GPU轨迹实际产生的保存。池0→134、后续剩112；worker的submit/wait执行到替身，GPU写入未执行。原先3个peer连续16调用无输出的代价不消失；全部请求机会沿用[原分析](native_full_release_boundary/analysis.json)，不把16调用或200机会换成毫秒、吞吐或净收益。

复算入口已实际跑通。首次独立工作区结果[native_method_fixture.json](native_full_release_boundary/native_method_fixture.json)保留，对应源码在本地提交f71343e3；共享复算首次遇到未合流protect_native_recovery参数，尚未执行分支即报TypeError。修正为只复用夹具状态构造、跳过随即卸载且从未使用的adapter安装，不改共享rotation代码。修正后两条件的全部逐调用结果与首次逐字段相同，补记两段复算命令均通过；这次兼容修正不是新增科学样本。

另做一个真实边界定位，[observed_completion_release.json](native_full_release_boundary/observed_completion_release.json)：**原基线**0433在step1392到1024声明上限，完成时间23.303997s；当步schedule后持134块，池204，下一schedule入口请求移除、池338，增加134。这支持正常完成后的池归还，不能证明替代分支时刻、DMA已结束或flush开销为零。本次是length终止，恢复途中EOS仍未验证。该单点没有重算原方生命周期或主表。

### 强基线与投入决定

原执行方已完成selected/full/full/selected轻量四格、256请求及唯一归档，本方未启动、下载或复算主表。其[主分析](../20260915_natural_save_scope_timing_r01/execution_weste_26862/analysis.json)（读取版本SHA d3cd388268aea345a4caf3768d982d6ed554e679e5792d2b729ab21cf7d92990）报告full相对selected：最大gap +13.19%/+63.96%，输出率+0.90%/−3.52%，均完成−1.92%/+1.49%，输出量−0.82%/−1.60%。不能以完整保存范围资格将full称为服务最强；当前主要停顿指标两配对均由selected较好，输出量不同亦不构成等工作量加速。本方只引用原分析，主裁决由原方整合。

**改变的决策：** completion-bridge具备条件资源路径，取消“须等store结束才可重规划”的额外等待；保留覆盖屏障，不绕过搬运依赖。6个无完成路径的前态仍不加固定长窗口。这不支持立即投入完整GPU对照：在线触发、上层动作衔接及同停顿要求下的完整代价仍缺证据，局部通过不升格成贡献。

唯一接续：把“分配池就绪/覆盖须等待”作为接口交给资源模型方，在其给定target且存在cap可资助完成者的状态上接入默认关闭的条件动作；commit重检和victim选择由原方负责。当前不封GPU包、不争抢后续实验；只有共同基线上出现有意义的动作差异，再由唯一执行方做真实资格。无全局Oracle或净墙钟上界；本轮排除的是本后端正常完成须等store才释放池的解释，未排除同步代价。

**给定恢复对象，本模型已能区分：何时只是转移有限decode机会，何时可以有代价地走到真实分配池释放。它尚不能判断这条路径是否改善完整服务。**

复算CPU原生方法（共享仓库根目录；输出须不存在；临时Scheduler来自封存源码）：

```sh
python3 -B - <<'PY'
import json, runpy, sys, tempfile
from pathlib import Path
base = Path('refine-logs/expert_saturation')
script = base/'experiments/admission_capacity/verify_completion_bridge_native.py'
carrier = base/'outputs/admission_capacity/20260914_load_ready_contract_r01/native_source.json'
with tempfile.TemporaryDirectory() as tmp:
    source = Path(tmp)/'scheduler.py'
    source.write_text(json.loads(carrier.read_text())['v1/core/sched/scheduler.py'])
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), '--repo', '.', '--scheduler-source', str(source),
                '--output', '/tmp/NEW-completion-bridge-native.json']
    runpy.run_path(str(script), run_name='__main__')
PY
```

真实池边界复算（raw只读；与observed_completion_release.json数值对应）：

```sh
python3 -B - <<'PY'
import json
from pathlib import Path
p = Path('refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/execution_weste_26862/readback/results/diagnostic-native-full/raw.json')
r = json.loads(p.read_text())
rid = 'measured/memory-train-article-0000433-9af76161'
q = next(q for q in r['requests'] if q['internal_request_id'] == rid)
c, = [c for c in r['engine_steps'] if c['returned_s'] == q['completion_s']]
k = c['scheduler_step_start']; assert c['scheduler_step_end'] == k + 1
frames = {v['attempted_step']: v for v in r['memory_trace']}
a, n = frames[k]['after'], frames[k+1]['before']
assert set(a['running_ids']) - set(n['running_ids']) == {rid}
held = sum(a['requests'][rid]['block_counts'])
assert n['pool']['free_blocks'] - a['pool']['free_blocks'] == held == 134
print(k, q['completion_s'], q['stop_reason'], a['pool']['free_blocks'], n['pool']['free_blocks'], held)
PY
```


## 补记：保住目标到完成的peer代价能否靠份额消除？

本轮起点fa863e4e，继续C执行层职责，原始数据只读。本轮核对原方已完成F四格主报告及下一G组建议：selected/current是自然域停顿优先参照，full仍为必要系统对照；下一G拟含native-full无额外rotation及selected current/eager。没有本方GPU执行身份，不改其包或主目标。唯一新问题是**在已有正例里，peer整段无输出是当前贪心分配造成的，还是资源条件本身要求付出的代价？** 不重复首输出/保存/生命周期分析。

先核对A的实际接口：swap_envelope只返回共同增长余量，choose_commit_action只在取消、直接恢复、原swap中选择；两成功分支的保护仍在首个新输出解除。C可以接受A已经选定的target，但必须等到其resident/pending-one；completion-bridge会增加跨首输出的target/finisher义务及peer hold，**并非兼容接口的小修正**。因此撤回上一节直接接入默认关闭组件的下一步，先判断这个新义务是否值得付费。未改A源码或另建controller。

### 条件成本模型

[recovery_execution_share.py](../../../experiments/admission_capacity/recovery_execution_share.py)新增decode_service_envelope(state,H,required_ids)。使用同一Request/State，没有新资源/策略模型。H是给定的成功调用窗口；required集合中的请求每调用各获得一个输出机会。所有请求必须resident pending-one、声明cap至少还能运行H次，计算预算足容当前cohort；不允许提前EOS、新准入、抢占、共享引用、现有持块提前归还。声明cap在第H次完成后可以归还，但不能拿它资助这H次调度。违反这些条件时拒绝套用本界。

令s_i=bA_i−C_i是当前持块剩余可写位置，R是required集合走过H次尚需新增的块，F为当前free，L=F−R。若L<0，义务本身不可资助。对于其余peer，在给k个新增块时最多得到q_i(k)=min(H,s_i+bk)个输出机会。每块边际收益递减；把全体边际收益从大到小取最多L项，得到这个限制下可达到的总输出机会最大值。B至少为cohort大小，所以该数量可按每请求至多一步/调用合法安排；它不是墙钟上界。

另令Z为s_i=0的普通peer数量。每个这样的peer想获得哪怕一个输出，也必须独占一块新KV，且在第H次结束前不能回收重用。因此：

```text
整个H调用窗口零输出的peer数量 >= max(0, Z − L)
总未服务机会 >= H × resident_count − 最大输出机会
```

这里的未服务机会以“当前集合每调用都推进一次”为计数标尺，该标尺可能不满足KV约束，**不是原native/most基线**。此界不能换算成延迟、效用或净服务损失；改变required服务频率、允许释放/提前EOS或其它动作会改变可行集。

### 本次计算改变的判断

[peer_service_bound.json](native_full_release_boundary/peer_service_bound.json)由[probe_bridge_service_bound.py](../../../experiments/admission_capacity/probe_bridge_service_bound.py)复用已保留的唯一cap完成分支及同一前态。目标/事件没有新增搜索，没有新实验样本。

| 同一1377前态、H=16 | 精确条件结果 |
|---|---:|
| 原free | 3块 |
| target2680与finisher0433所需新增 | 2块 |
| 其余peer可用 | 1块 |
| 当前块已无可写位置的其余peer | 4个 |
| 任意该类分配都无法避免的整段零输出peer | 至少3个 |
| 最多总输出机会 / 原分配实际达到 | 200 / 200 |
| 相对24×16计数标尺的未服务机会 | 至少184 |

原分配让5057得到最后一块，5103/0464/0406各16调用无输出。换顺序可以改变承担者，但在保留相同两个逐调用输出义务和相同KV资源时，不能让这3个整段无服务名额消失；这个上界已被现有简单分配达到，不支持继续搜索另一套固定计算份额来增加本窗口服务量。其它有块内余量的peer或能调整输出间隔，但全组至少3个零输出窗口不变，不据此宣称全局最大ITL不可改善。

8项CPU测试通过，其中新界与1,280个三请求微型资源状态的所有末态输出向量独立枚举一致；提前cap与计算争用情形拒绝套用。测试验证这个受限容量公式，不作为额外实验重复或性能优势。源码/输入hash、所有24个请求输出机会保留；没有真实替代GPU轨迹、EOS分支、未来释放、秒级Oracle或服务质量证据。

**投入决定：暂停把当前completion-bridge接入A或完整runtime。** 它并非实现错误：既有正例在受限空间中已经达到数量上界，但用目标连续服务换来了无法靠份额消除的peer停顿。资源释放路径成立不足以支持扩大该义务；保留执行模型及成本界供A评价动作，不将它包装成方法增益，也不判死所有恢复执行策略。

唯一下一实验是主方已在准备的自然启动节奏G组，本方不另起一组。若其诊断出现已恢复目标的计算/增长冲突，再用实际前态判断是否存在更低peer代价的执行条件；若无冲突，则保留无动作边界，不降低阈值加压。窗口与peer代价必须随目标同时改变，不能把本界中的200机会当全局最优，也不能仅因新请求ID就重启这条机制。

复算（共享仓库根目录，输出须不存在）：

```sh
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/probe_bridge_service_bound.py \
  --repo . --output /tmp/NEW-bridge-service-bound.json
python3 -B -m unittest discover \
  -s refine-logs/expert_saturation/experiments/admission_capacity \
  -p test_recovery_execution_share.py
```


## 自然selected/current：冷恢复的整数调用边界首次产生实际份额差异

本轮起点09d2ff9f。G自然节奏七格已接受、由原prepare_start_contrast独占执行/主分析，尚未交付本方canonical结果；本方不轮询其进程或改包。先检验现有探针能否直接迁移到已完成D同域current，新增问题仅为首输出前计算份额，不重做A的首输出后增长或B的生命周期/成本计数。这里复用D原数据，不增加独立样本。

旧[probe_staged_execution_boundary.py](../../../experiments/admission_capacity/probe_staged_execution_boundary.py)直接通过D的全部25个主动commit，结果保留在[current_diagnosis.json](natural_cold_share/current_diagnosis.json)。保护内pending-one peer假定在本次真实成立：成功commit拒绝mixed/skipped，保护时hold不放非pure decode，waiting钩子不准入其它请求；不能因为是开放到达就杜撰新prefill争用。原探针对target_terminal及无后续schedule的结束边界仍不适用，未来出现此类事件须按实际output/finished终止，不能跨episode寻找下一target_new_output。本次25项均有真实新输出解除，不涉及该未支持分支。

| 首输出前执行量 | D自然current |
|---|---:|
| 主动commit / 首输出兑现 | 25 / 25 |
| 仅1次计算的恢复 | 21 |
| 目标计算调用 | 37 |
| 纯resident前态模型调用 / 分配与实测一致 | 12 / 12 |
| preserve_calls改变分配 | 2 |
| 无目标计算调用 | 42（21 PREEMPTED＋21 WAIT_REMOTE） |

21个单调用恢复待执行2–20位置；另4个冷恢复为3030/3044/3078/3094位置、各4计算调用。只有target5122的前两次在resident途中跨整数预算边界。它们不是两条独立请求，也不是旧固定域83/84一次计算结论的反例；新运行域确实有不同的未保存恢复工作量。这里不新增完整保存的比较结论，最新F的selected停顿优先参照与full必要系统对照均保留。

### 最小模型与真实前态分支

给定resident恢复剩余R、pending-one peer数量n、预算B，目标在compute-only最少K=ceil(R/B)次调用内完成，留给peer的总预算最多KB−R。若保持peer每调用一次，至少要延后：

```text
L = max(0, K*n − (K*B − R))
```

这是成功计算调用内的工作预算下界；假定peer在窗口内仍存活，不计未来EOS/释放，不预测时间。KV还须逐调用验证。既有preserve_calls按R−(K−1)B保住本次最低目标份额，其余预算给原序可行peer；没有新规则、新阈值、target或victim选择。

[probe_cold_recovery_share.py](../../../experiments/admission_capacity/probe_cold_recovery_share.py)从两个实测resident前态分叉执行完整native schedule方法和既有hold钩子。各分支独立推进C/A/输出；原策略前两次分配与实际raw逐字段相同，候选不读取原后续状态。allocator与返回token为替身，固定这2次调用内不引入新准入、上层swap、EOS或store/DMA演进；未做GPU替代轨迹。

| 同前态、同2调用窗口 | step637 | step730 |
|---|---:|---:|
| 目标剩余 / peer数 | 2035 / 29 | 2048 / 28 |
| ready_first目标两次分配 | 995＋995 | 996＋996 |
| preserve_calls目标两次分配 | 1011＋1024 | 1024＋1024 |
| 原策略末态剩余恢复位置 | 45 | 56 |
| 候选首新输出 | 第2次调用 | 第2次调用 |
| 必需延后peer输出下界 / 候选达到 | 45 / 45 | 56 / 56 |
| 原策略/候选总新输出机会 | 58 / 14 | 56 / 1 |
| 两分支各自总scheduled位置 | 2048 / 2048 | 2048 / 2048 |

[完整CPU分支及全部请求损益](natural_cold_share/native_branches.json)保留。原观察首输出分别在639/732，即前态起第3次；候选只执行到第2次输出边界，没有借用之后的资源释放或原轨迹。候选末态pending=1是新token尚未写KV的正常下一decode，不是尚欠1位置的旧恢复。首组16peer两次都未输出、13peer只输出一次；次组28peer两次都未输出。阶段总新输出数量下降，内部计算进展增加，不能把‘少一次到首输出调用’称作总服务加速。

### 改变的决策与边界

不能再把‘计算份额在所有已测域均无动作’作为解释。**自然selected的特定冷恢复债务接近预算整数边界时，已有最简单规则确有可执行差异；但兑现首输出必须付出peer服务转移。** 当前证据只恢复这条首输出前规则的候选资格，不支持直接接入、GPU性能预告或新颖性声明。首输出后的completion-bridge暂停结论保持，不能把两种不同义务混为一条持续保护机制。

唯一下一步仍复用G既定诊断与完整服务结果，核对取消cooldown后冷恢复与这种整数冲突是否仍出现。若只剩保存后2–20位置的恢复，继续暂停此份额机制；若同类冲突仍存在，再决定是否值得在同selected/current或eager强底座做一个默认关闭的份额对照。当前不改变G、另开GPU组或用D的0.530/0.105s commit-to-output跨度预告可省时间；该跨度包含前置调用和诊断开销，不是候选saving。

复算诊断（输出须不存在；本函数增加的L字段是解释性预算下界，不增加性能样本）：

```sh
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/probe_staged_execution_boundary.py \
  --service-bundle refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_saved_recovery_gate_r01 \
  --labels diagnostic-current \
  --full-service-reference refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_save_scope_timing_r01/execution_weste_26862/analysis.json \
  --output /tmp/NEW-natural-current-execution.json
```

原生CPU分支复算（从封存源导出临时Scheduler；使用上述新诊断）：

```sh
python3 -B - <<'PY'
import json, runpy, sys, tempfile
from pathlib import Path
base = Path('refine-logs/expert_saturation')
script = base/'experiments/admission_capacity/probe_cold_recovery_share.py'
carrier = base/'outputs/admission_capacity/20260914_load_ready_contract_r01/native_source.json'
with tempfile.TemporaryDirectory() as tmp:
    source = Path(tmp)/'scheduler.py'
    source.write_text(json.loads(carrier.read_text())['v1/core/sched/scheduler.py'])
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), '--repo', '.', '--scheduler-source', str(source),
                '--diagnosis', '/tmp/NEW-natural-current-execution.json',
                '--output', '/tmp/NEW-cold-recovery-share.json']
    runpy.run_path(str(script), run_name='__main__')
PY
```


## G强基线后的冷恢复边界与最小计算钩子

本轮起点66dc83a7，职责仍为C执行层；共享HEAD de64dae5、多人dirty工作区保持，修改在独立worktree。已读主方CURRENT_EXPERIMENT、台账及G的[唯一主报告](../20260915_natural_recovery_cadence_r01/RESULTS.md)，直接继承完整服务结论：同selected保存下，eager/current最大gap下降63.43%/53.12%，吞吐+1.066%/−0.706%，平均完成−1.473%/+1.443%；不是全分布支配。selected/eager成为本域停顿优先简单基线，current和native_full_native其它目标优势保留。C没有重算主表、回读远端、增加实验样本或改接受包。

本轮唯一假说是：**更早启动已解决停顿主因后，冷恢复中的整数预算冲突是否也消失？** 反解释是D的两例只是较慢启动积累债务造成的特例。复用G全部主动commit的已有诊断即可区分，不需新GPU、调整压力或选择另一victim。允许的主张上限为已发生状态中的份额差异及CPU可执行性。

### 新诊断及同前态分支

[natural_eager_share/analysis.json](natural_eager_share/analysis.json)由原探针直接生成：54/54主动恢复返回首新输出，164个保护调用中60个实际计算、104个尚未计算（52 PREEMPTED＋52 WAIT_REMOTE）；52个恢复只计算一次，待执行2–34位置。另两个冷恢复的首次剩余为3003/3026位置，各用4个计算调用。6个纯resident前态的模型map全部吻合，其中2个preserve_calls改变动作；原策略没有hold peer。这里只研究主动保护范围，不将其54个目标扩写为全部自然恢复。

两处都来自文章5161，与D的5122不同，但仍为同一文章的两次恢复；D/G复用同一批文章且策略影响轨迹，**不是独立输入验证**。不因多一篇ID就宣称迁移成立。

| 实际resident前态 | G step740 | G step824 |
|---|---:|---:|
| 目标剩余R / peer数n / B | 1011 / 28 / 1024 | 2029 / 27 / 1024 |
| compute-only最少K次 | 1 | 2 |
| ready_first目标分配 | 996 | 997＋997 |
| preserve_calls目标分配 | 1011 | 1005＋1024 |
| 原策略窗口末尚欠恢复位置 | 15 | 35 |
| 候选首输出机会 | 第1次 | 第2次 |
| 延后peer机会下界 / 实际达到 | 15 / 15 | 35 / 35 |
| 原策略→候选总新输出机会 | 28→14 | 54→20 |
| 两分支各自scheduled位置总量 | 1024 | 2048 |

原实际首输出为741/826，即各前态起第2/3次。候选独立推进到第1/2次输出边界即停，不借用原后续资源释放。预算下界仍为L=max(0,R+Kn−KB)，是给定成功计算窗口、peer未提前终止时的机会代价，不是时间损失或全局最优。step740中15个peer少一次；step824中8个peer少两次、19个少一次。全部请求损益见[native_branches.json](natural_eager_share/native_branches.json)。部分恢复工作与peer新输出共批，两个分支总位置数相同；少一次到目标首输出调用不能写成节省一整次计算或服务加速。

### 最小机制与已验证接入边界

新增[recovery_compute_share.py](../../../experiments/admission_capacity/recovery_compute_share.py)，约80行，复用同一Request/State/allocate，没有新状态预测器或selector。`install(scheduler, enabled=False, block_size=16, diagnostic=False)`默认不修改回调；开启后包装已有`_rotation_begin`和`_rotation_hold`。每调用先由原begin处理准备/提交和目标，再仅对RUNNING、队尾、剩余>1的目标计算份额；所有peer须为resident pending-one decode。等待加载、尚未进入running、队列位置不符合及已进入正常decode时均不启用，不移动请求来制造适用性。

份额器先检查当前完整历史KV可资助，再按既有整数规则延后必要peer。原`hold`始终先执行，不能覆盖KV拒绝。每次begin清空额外held；首输出/terminal的识别与解除仍完全由base负责，不能用computed增长代替新输出。当前支持同步、无APC、不共享FullAttention、16-token块、无long-prefill clamp；完整staged base另有更严格配置资格，不能绕过其安装检查。本模块没有新增host保存或加载行为。

以下是将来可用的安装顺序，**当前G runner未接入**；native_full_native没有rotation hooks，不能直接安装开启版本：

```python
rotation_data, remove_rotation = install_rotation(scheduler, ...)
share_data, remove_share = recovery_compute_share.install(
    scheduler, enabled=compute_share_enabled, diagnostic=diagnostic)
# 随后安装观测包装并执行；退出时先解除外层观测。
remove_share()
remove_rotation()
```

扩展同一[probe_cold_recovery_share.py](../../../experiments/admission_capacity/probe_cold_recovery_share.py)增加`--service-bundle`、`--label`和`--runtime-hook`，复用完整native schedule方法。启用时base仍给ready_first，额外钩子实际经过native hold位置执行。G两例的map、分配后空闲块、输出机会及15/35额外hold均与模型分支一致，见[hook_branches.json](natural_eager_share/hook_branches.json)；D原两例同一钩子也达到45/56，见[D兼容结果](natural_cold_share/hook_branches.json)。这是已有四个前态的代码验证，不是四次服务重复。

三个针对性回调检查通过：默认关闭保留原回调身份；base解除目标后清除额外hold；等待加载/非队尾不动作且原KV拒绝保留。恢复生命周期的完整识别没有在此替身里重做；原生allocator、返回token、base回调是限定替身，store/DMA、上层新决策、自由生成和EOS没有执行。静态核对G封存base的释放/安装顺序没有发现接口冲突，仍不等于完整native offload组合已获运行资格。新增状态读取、规划、回调与必要计数成本必须进入将来处理臂的真实墙钟，不能事后扣除。

### 研究决定与适用范围

新证据排除了“eager已让全部恢复成为一次计算，所以份额永远无动作”这个解释；它没有推翻eager作为当前强简单基线。冷恢复整数边界有一个无需未来信息的最小动作，但当前只是2/54恢复、一个目标，且确定会转移peer服务。**交付默认关闭的执行组件，暂不增加GPU份额组或声称有完整效率增量。** 先沿用主方已选的独立128篇、较长到达episode迁移验证，C仅在其提供的必要诊断上检验这一条件是否持续；不要求主方为本模块额外插桩或筛选有利文章。

G主报告已明确所有EOS/stop请求自身均未被成功抢占，诊断无target_terminal。因此自然EOS允许仍未覆盖恢复对象在保护中自然结束。此前0.5s到达、零主动动作范围继续作为无动作边界，不降阈值制造冲突。若新独立范围同样无冲突，继续搁置份额性能扩展；若冲突持续且对预先固定停顿目标有可见影响，再在同强底座提出唯一on/off资格，随后才考虑同停顿要求下的平均完成或goodput。少调用或一条目标改善本身不触发性能GO。

Verdict：`CPU_NATIVE_METHOD_BRANCHES / GPU_SHARE_UNRUN`。实测为G已发生执行状态，替代份额仅CPU；最强基线为G selected/eager，native_full_native系统参照保留。无秒级Oracle、完整服务headroom、恢复期EOS/语义质量或新颖性证明。当前限制属于**局部机会尚未证明有完整服务价值**，不是实现失败或问题级NO-GO；首输出后completion-bridge继续暂停。

直接答案：给定恢复目标，正常情况下继续采用现有可行decode优先、恢复用余额；当resident重算剩余接近B的整数边界时，可用`R−(ceil(R/B)−1)B`作为当次目标最低份额，明确承担必要peer机会转移。这个规则已能通过真实native调度方法表达，但是否值得用于服务，尚须由独立运行域和固定停顿目标下的完整成本决定。

复算（共享仓库根目录，输出路径须不存在；保留原结果）：

```sh
python3 -B -m unittest discover -s refine-logs/expert_saturation/experiments/admission_capacity -p test_recovery_compute_share.py
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/probe_staged_execution_boundary.py \
  --service-bundle refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_recovery_cadence_r01 \
  --labels diagnostic-eager \
  --full-service-reference refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_recovery_cadence_r01/execution_weste_26862/analysis.json \
  --output /tmp/NEW-natural-eager-execution.json
python3 -B - <<'PY'
import json, runpy, sys, tempfile
from pathlib import Path
base = Path('refine-logs/expert_saturation')
script = base/'experiments/admission_capacity/probe_cold_recovery_share.py'
carrier = base/'outputs/admission_capacity/20260914_load_ready_contract_r01/native_source.json'
with tempfile.TemporaryDirectory() as tmp:
    source = Path(tmp)/'scheduler.py'
    source.write_text(json.loads(carrier.read_text())['v1/core/sched/scheduler.py'])
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), '--repo', '.', '--scheduler-source', str(source),
        '--service-bundle', str(base/'outputs/admission_capacity/20260915_natural_recovery_cadence_r01'),
        '--label', 'diagnostic-eager', '--runtime-hook',
        '--diagnosis', '/tmp/NEW-natural-eager-execution.json',
        '--output', '/tmp/NEW-natural-eager-hook.json']
    runpy.run_path(str(script), run_name='__main__')
PY
```

## 补齐简单计算基线：比较必须从首个可接入状态开始

本轮起点85b6322e。已重读共同goal、PAPER_ARGUMENT、CURRENT和最新台账：G主结果完成，独立128篇迁移尚为提议、没有新raw；C未获GPU执行身份。上一轮是真实代码/证据进展。此轮唯一未决问题是**preserve_calls相对更简单的restore_first到底剩下什么**，而非扩大controller。restore_first早已在同一allocate中实现，末次余量正常服务peer；没有新算法、长时预测、参数或保存能力。

### 先排除一个不公平的比较起点

先将原来的四个map-change前态增加restore_first分支，结果在[D晚期窗口](natural_cold_share/simple_baseline_branches.json)与[G晚期窗口](natural_eager_share/simple_baseline_branches.json)。四例目标首输出调用、各请求累计输出、computed/history及持块计数都相同；两例仅把13/19个peer输出从末次移到首次，另两例map也相同。

但这**不能**支持完整resident阶段的两组件等价：G第一冷恢复最早可接入是739，740只是preserve_calls首次不同于原策略的位置。restore_first若从739启用，会在739就扣留peer；只从740比较会漏掉这部分代价。因此保留这四个限定窗口结果，同时把正式组件比较的选择规则改为：每个原主动保护episode的**首个已被模型核实、RUNNING且剩余重算>1的resident前态**，包括preserve_calls尚未动作或整段不动作的恢复。此前PREEMPTED/WAIT_REMOTE及首次waiting准入不在钩子作用域，各臂继承同一前态；不因反事实好坏筛选。

[probe_cold_recovery_share.py](../../../experiments/admission_capacity/probe_cold_recovery_share.py)增加`--include-restore-first --branch-start first_resident`，逐episode保留选中step或空列表。D全部25次保护中4个可用冷恢复、G全部54次中2个；其余已知单次加载计算范围没有可用的resident重算起点，不人为扩展它。三规则各自执行原生schedule/hold和独立KV计数、计算/输出演进，ready_first前缀继续与raw吻合；同窗口K=ceil(R/B)在运行前由当前状态确定，不读取候选未来EOS或完成来选时域。此次各窗口均未发生完成/新准入/额外抢占，候选在K次末有输出；ready_first可以在该窗口尚未输出，原后续轨迹仅作事实核对。

### 六个完整resident入口的结果

下表统计相同K次成功计算调用内的peer输出机会，不含目标自己的一个新输出。`PC`为preserve_calls，`RF`为restore_first。完整结果包含全部请求计数末态、逐调用分配、输出身份及原输入/源码hash：[D四例](natural_cold_share/resident_baseline_branches.json)、[G两例](natural_eager_share/resident_baseline_branches.json)。这是已有两个诊断的六次恢复，非六个独立运行；D同一5122反复恢复，G同一5161反复恢复，输入文档集相同。

| 首resident前态 | R / n / K | ready_first peer | PC peer | RF peer | PC/RF目标首输出调用 | PC/RF末态free块 |
|---|---|---:|---:|---:|---|---|
| D637 | 2035 / 29 / 2 | 58 | 13 | 13 | 2 / 2 | 28 / 28 |
| D730 | 2048 / 28 / 2 | 56 | 0 | 0 | 2 / 2 | 63 / 63 |
| D840 | 2080 / 26 / 3 | 78 | 78 | 26 | 3 / 3 | 26 / 28 |
| D970 | 2095 / 25 / 3 | 75 | 75 | 25 | 3 / 3 | 211 / 213 |
| G739 | 2007 / 28 / 2 | 56 | 41 | 28 | 2 / 2 | 37 / 37 |
| G824 | 2029 / 27 / 2 | 54 | 19 | 19 | 2 / 2 | 102 / 102 |

D637与G824两者末态逻辑计数相同，但PC将13/19个peer从RF第二次提前到第一次；D730分配完全相同。D840/970中PC与现有ready_first整段相同，各26/25个peer每调用都输出，而RF只在第三调用给一次；PC多52/50输出机会同时各多占2块KV。G739中PC为目标996＋1011、peer28＋13；RF为目标1024＋983、peer0＋28。PC使全部28peer先输出一次，其中13peer再输出一次，比RF多13机会，但相对ready_first同期56仍须延后15。两臂free同为37不代表各请求进度或数值KV相同。

这区分了三个事实：目标最少计算调用数不需要整数规则才能达到；“保留KV不必独占计算”的优势有一部分原most余额规则已经提供；余量跨调用分配在G739这个实际resident入口确有额外peer机会，但仍没有完整服务净收益证据。不能把D840/970的52/50写成新钩子收益，也不能把G739额外13写成相对most增加13——它只相对RF增加。

### 简单条件模型与投入决定

令S=KB−R。若全部n个pending-one peer在K调用内持续存活、计算与KV均允许，RF仅在末次给peer最多min(n,S)个输出机会；PC可提前使用累计S余量，最多min(Kn,S)。因此S≤n时两者累计量可能相同，仅先后不同；K>1且S>n时RF把太多工作集中到目标，末次又受每peer一位置限制，不能自动用完可用预算。实际KV边界可能降低这个数量；本次六分支逐调用验证，公式不替代allocator或实际返回。

同K和同目标工作量不是同成本。PC在三例执行了更多peer位置、两例多占KV，混合batch的上下文、形状、计算/调度及未来增长代价随策略变化。CPU输出为占位符，未执行GPU forward、原生offload组合、自由生成、实际EOS或完整请求尾部。块计数相同也不证明物理块身份、数值内容、最近输出时间或后端状态相同；不能对这六分支作毫秒排序或未来轨迹合并。

**改变的决定：将restore_first保留为必要计算组件对照，但不以它替换已有most余额基线，也不把整数规则的最少调用保证包装成独立贡献。** 当前新模块只在真实预算边界提供额外约束，尚未证明同停顿要求下的平均完成/goodput价值；不增加第三策略GPU矩阵。主方仍先完成独立输入/较长到达迁移。只有新范围留下与主目标相关的执行缺口，才比较同入口RF、原余额与最小份额的完整成本；无动作或恢复期EOS仍缺失就如实保留，不调阈值制造事件。

Verdict：`CPU_COMPONENT_COMPARISON_ONLY`，不是方法GO或问题NO-GO。最强服务基线仍为G selected/eager，RF仅为此执行层的必要简单组件。无完整服务Oracle；当前限制是机会与成本尚未兑现为完整服务证据。直接回答本轮问题：**恢复优先已能达到相同首输出调用数，但不能普遍替代跨调用的余额分配；后者的剩余价值是peer在恢复期间得到多少输出、何时得到以及为此增加多少KV/计算成本。**

复算沿用上一节便携命令，在`sys.argv`中加入`'--include-restore-first', '--branch-start', 'first_resident'`，输出改为新的`/tmp/NEW-natural-eager-resident.json`。D使用`20260915_natural_saved_recovery_gate_r01`、`diagnostic-current`和保留的`natural_cold_share/current_diagnosis.json`，输出另取新路径。保留默认`changed_call`可复算四个晚期窗口；不得用它们替代六个首resident比较。
