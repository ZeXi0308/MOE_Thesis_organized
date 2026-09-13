# 请求进度与 KV 预算：实验记录

**当前执行状态：** 目标恢复后连续三个回合实时确认同一Qwen3 PID13275占用GPU，再次标记BLOCKED_RESOURCE；最新已处理4/16分片，第5片3.10GB。新32文档四臂八项保持STAGED、GPU 0/8，本地及远端无已启动结果；准备与检查已完成，无后台等待或执行器。主问题OPEN，资源释放后接续同一包。详见[状态与接续入口](../../outputs/admission_capacity/20260913_rotation_strong_baseline_r01/STATUS.json)。

新一轮先查 [共享结论台账与四臂对照修正](RESULT_LEDGER.md)（2026-09-13）：复用已算结论，比较最新 headroom 基线，初始化前检查 GPU 并留痕。

主文档：[RESEARCH_NOTES.md](RESEARCH_NOTES.md)。原始运行保留；本文件维护同一问题的命令与接续点。

当前接续：首次交换不足以保留持续排序的吞吐结果；主问题OPEN。新文档cohort上native/native A/A/headroom/持续most_output四臂×两block已STAGED，空卡后执行既有RESUME_COMMAND.sh。历史条目和各轮审计状态分别保留。

2026-09-13定向动作查新已核对FastServe正式版/代码、FastSwitch全文、VTC正文/抢占附录及Andes v2/作者代码：token抢占、aging、KV预留和重算成本计费均非本轮独有。Andes公开实现默认RECOMPUTE，但固定时延/slack限抢占代码不能等同论文refiner；server max-ITL未升级为消费QoE。详细矩阵与迁移边界写入[主文档](RESEARCH_NOTES.md#当前动作与已有工作的重合2026-09-13定向核查)。20项仍是输入迁移测量，不代替最近邻策略基线；本轮未修改冻结包或新增Controller。有界查新至此结束；新文本迁移已实跑，下一信息缺口是具体动作贡献的分离。

负控统计解释的定向复核已完成，见[独立审阅及配对集合澄清](../../outputs/admission_capacity/20260913_runtime_drift_review_r01/EXPERIMENT_AUDIT.md)：总体WARN，真实测量保留；有限相关差值不能当跨负载噪声界。本问题不再追加同批raw的完整性审计。

此前授权的四项恢复准入对照已全部完成、回传，128/128 请求。
同池 chunk 臂两次均从 2 次抢占增至 70 次，吞吐下降 6.89%/4.74%；见[结果与源码定位](../../outputs/admission_capacity/20260912_recovery_admission_r01/REPORT.md)。下文授权阻塞条目保留为历史经过。

## 2026-09-13：新文档强基线八项准备并上传（STAGED / GPU 0）

问题：持续most_output在新32文档上，是否仍改善相对native与fast headroom的最长停顿—完整吞吐权衡。固定同池/长度/到达，block0 native/headroom/most/nativeAA，block1反序；主配对6、nativeAA2、同角色重复4。全部八项合格前不输出数值配对。

从固定shard源行12128–17106选择下一32篇足长文章，执行3072-token前缀；此前96篇逐文本/token重现，新旧128篇文档及输入hash/源区间均互斥。输入脚本prepare_rotation_fresh_inputs.py，冻结脚本prepare_rotation_strong_baseline.py，分析入口analyze_rotation_strong_baseline.py。12组CPU检查和实际UNRUN状态检查通过；两处准备期兼容/序列化问题已在封包前修复，未生成测量raw。

N=outputs/admission_capacity/20260913_rotation_strong_baseline_r01。包SHA `7cbd2f0e595327ddcba427a3d3599f46594e3c67ca96b45b4ec6f1e1a591d9fe`；N/STAGE_COMMAND.sh已在现有weste:23478上传并校验，运行环境/模型hash/目标GPU身份符合既有配置。GPU当前由另一Qwen3任务PID13275持有，因此状态STAGED、cells空；没有下载模型或初始化GPU。唯一下一步空卡后执行N/RESUME_COMMAND.sh；先复核暂存hash/无已有cell/空GPU，再串行八项并逐格回传。

当前新证据仅离线输入与执行准备，不升级为性能、质量或方法结论；没有新参数搜索或其它controller。详见[报告](../../outputs/admission_capacity/20260913_rotation_strong_baseline_r01/REPORT.md)。

## 2026-09-13：首次交换六项实测完成（weste:23478）

用户改回weste:23478后，使用原冻结包，新增execution02_weste_23478并保留旧westc STAGED目录。六格A/C/B、B/C/A全部returncode0、COMPLETE/READ_BACK；192/192请求、196608输出，实际KV16,089,350,144字节、7671块，六项资格通过。

结果：C/A吞吐+0.166%/−0.425%、平均完成−0.032%/+0.644%；C/B吞吐−1.332%/−1.513%、平均完成−1.065%/−0.850%。C只把双请求尾段87→58调用，却把最后一请求独跑12→45，局部时间减少被抵消约九成，完整收益未一致。B/A仍为吞吐+1.518%/+1.105%、平均完成+1.044%/+1.507%。一份复用cohort、两个顺序block；同角色漂移不是噪声界。fresh GPT-5.6-Sol ultra独立重算六raw指标零差异，完整性PASS、P0/P1=0，same-family/provisional。

命令：F/RUN_WESTE_COMMAND.sh；主分析analyze_rotation_first_swap.py --run-dir F/execution02_weste_23478 --output-dir F/analysis02_weste_23478；路径diagnostics/diagnose_paths.py另加--run-dir同目录，输出diagnostics/measured02_weste_23478。F=outputs/admission_capacity/20260913_rotation_first_swap_r01。完整结果与条件尾段模型见[RESULTS_WESTE_ADDENDUM](../../outputs/admission_capacity/20260913_rotation_first_swap_r01/RESULTS_WESTE_ADDENDUM.md)。

唯一下一项：新文档cohort的native/native A/A/completion_headroom/持续most_output八项对照，先验固定同预算/长度/到达；尚未准备或运行。首次交换组合目标未获支持，不扩写为问题族NO-GO。

## 2026-09-13：首次交换与持续改序（六项准备完成，BLOCKED_APPROVAL）

假说：一次不同的实际驱逐是否足以压缩原末两请求收尾，同时不必持续延后前面的请求。C按此前成功forced交换数决定首次most_output、之后least_progress；noop/资金拒绝/自然抢占不消耗首次机会。计数在真实schedule和资金/保护断言成功后更新，各臂独立演进。

复用cohort0，block0 A,C,B、block1 B,C,A，六格192次请求；A/B也在同底座重跑，旧八项不替代配对。25项选择器、4项原生方法/分配夹具、12项分析器检查通过；分析器仅额外允许rotation_victim_order配置差，校验全部step的计数与模式；无新GPU性能。

包SHA256 `fd1342a6aea8f2cc886a0a6957d4059dcc2d3ffad1797b0428fb23d8be162b4c`；[具体报告与拒绝记录](../../outputs/admission_capacity/20260913_rotation_first_swap_r01/REPORT.md)。现有GPU观察到PID68167占用后退出，最后一次检查为空闲；自动审批仍拒绝新六项包数据上传执行，execution目录未创建，uploaded=false、GPU=0。不改路径/拆命令绕过，保留一项审批阻塞。

准备命令：`prepare_rotation_first_swap.py --victim-run <V/execution> --output-dir <F/preparation>`；新分析入口`analyze_rotation_first_swap.py --run-dir <F/execution> --output-dir <新目录>`。唯一下一步是获准后完成本冻结六项，不追加阈值或次数搜索。

## 2026-09-13：驱逐服务量排序消融（8/8 COMPLETE）

唯一假说：旧轮转的暂停—服务量权衡部分依赖“驱逐最少进度者”的选择，而不是只有最长缺席恢复。新开关`most_output`在相同合格集合内选择当前已生成输出最多者；不计重算，保留其它触发/保护/资金规则。原默认仍为least_progress。

选择器24项CPU检查与三个原生方法/分配夹具通过；后者覆盖两种驱逐、资金不足拒绝、缓存消息及恢复保护，不含GPU worker证据。新分析器复用请求/成本/恢复账本，唯一额外允许差异为已验证的rotation_victim_order；旧H raw适配一致、错误配置/跨cohort/未完成配对拒绝。代码与原始本地结果见[准备报告](../../outputs/admission_capacity/20260913_rotation_victim_order_r01/REPORT.md)。

八项：cohort0/0 A,B；cohort1/0 B,A；cohort0/1 B,A；cohort1/1 A,B。复用两个已见文本cohort，故为探索性组件消融，非新holdout。冻结包SHA256 `0049d6e125dadb37655c3580339a667979f43da5290c0c0afa6edfed1df8ff00`；24个载荷文件中18个与旧已授权包逐字节相同，其余只改排序/参数传递/清单/协议。

用户具体授权后，包上传至moe-rotation-victim-order-20260913-r01，8格全部完成、回传核验、driver退出0；此前审批拒绝保留。分析器8格资格与四配对通过：2026-09-13驱逐服务量排序消融8/8完成、256/256次请求。most_output相对least_progress四配对吞吐+1.134%–+1.560%、最大ITL减少19.638–24.030ms，但平均完成+1.004%–+1.471%；额外4,910重算位置与减少104次纯decode调用同时出现，最后两请求width2收尾87→8次。见[本轮报告](../../outputs/admission_capacity/20260913_rotation_victim_order_r01/REPORT.md)。这是复用文本上的组件消融，非新holdout、显著性、全面改善或方法GO。 首次实际victim分离在step836；A/B均合法且各自演进。强制驱逐8→9、自然均2、held均0；前四完成名次1/2/3/4→23/9/25/13。路径及互斥host成本随报告保存。

复算：`analyze_rotation_victim_order.py --run-dir <execution> --output-dir <new-dir>`；原始文件保留。 用户授权清理5090后，四轮40份远端重复raw逐文件匹配本地原件和归档后删除，释放11.1515 GiB；本地科学证据与远端归档保留，[清理清单](../../outputs/admission_capacity/20260913_rotation_victim_order_r01/remote_cleanup/README.md)可追溯。唯一下一因果实验是仅首次合法交换使用B，随后回到A，检验一次选择是否已足以改变同两尾请求收尾，而无需持续延后早完成者。该新诊断尚未实现/运行，不能预报收益。

## 2026-09-13：固定参数独立文本与A/A（20/20 COMPLETE）

假说：同KV、同长度/到达与固定轮转参数，暂停—服务量权衡可迁移到两组新文档。两个cohort各两个随机顺序block，每block四策略加native A/A；主native标签预先固定。执行包SHA256 `0ead2b4c9cf935ddda491646139e73073164f6e0250384066226f8b2fb9718f7`，现有远端 `moe-rotation-holdout-20260913-r02`；全部归档回传核验，driver退出0。

新增信息：两个新文本cohort保持四次8强制+2自然抢占和相同计数路径；额外30,876重算位置与减少123次纯decode调用抵销，native吞吐差依旧变号。平均完成劣于native、两条新暂停约0.953–0.960s的代价保持；headroom对照方向保持。A/A全部输出相同仍出现吞吐漂移，不能以输出相同替代运行重复。

分析入口：`analyze_rotation_holdout.py --run-dir <execution03> --output-dir <new-dir>`及`analyze_rotation_recovery.py --run-dir <execution03> --analysis <analysis.json> --output <new-accounting.json>`；全部原值、请求变化和成本见[报告](../../outputs/admission_capacity/20260913_rotation_holdout_r01/REPORT.md)。本轮640次执行是64篇新文档，不是640个独立workload；没有新长度/到达/KV/模型域，也无方法GO。 独立GPT-5.6-Sol ultra完整性复核PASS（same-family/provisional，P0/P1=0），审阅原文及指纹见同bundle的EXPERIMENT_AUDIT.md/.json；未升级科学裁决。

下一动作：仅改变同合格集合内驱逐排序，最少进度与已交付输出最多者对照；共同最长缺席恢复、30/20/30、近完成保护与资源/恢复语义不变。此为排序消融，尚未准备/上传/执行。若实际选择没有分离，记无动作差异，不判机制失败。

只读动作空间检查：在新cohort0/block0轮转首次动作前step836，31个running均未曾抢占且进度低于0.90；当前驱逐文章7877（742已输出token、239块），按已交付输出最多排序会选择文章3733（834token、245块）。空闲179块、恢复需237块，两种选择都能资助完整历史。这只证明该记录前态存在不同合法选择，不是另一策略已执行、性能估算或完整反事实；随后状态必须另行GPU推进。

## 2026-09-13：四臂共享对照（8/8 COMPLETE）

协议与结论登记于 [RESULT_LEDGER.md](RESULT_LEDGER.md)。`native32 / safe29 / headroom32-fast / rotate32-c20`按正反序执行，各格独立引擎、固定实际池。全部已上传/执行/回传哈希验证，driver退出0；见[execution.json](../../outputs/admission_capacity/20260913_rotation_fourarm_r01/execution/execution.json)。旧headroom数据不代替本轮基线。

新增信息：交换间隔20不构成每请求停顿上界，2名缺席者排队加30步门槛导致最长45步等待，再经4调用恢复，实测约1.006s。多重算30,876位置，纯decode调用少123、总调用少91，抵销大部分开销。原生吞吐差变号，未证明非劣；本轮无held，保护本身贡献未分离。

执行命令与三项重算入口见[REPORT](../../outputs/admission_capacity/20260913_rotation_fourarm_r01/REPORT.md#重跑与原件)。恢复链可复算：`python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_rotation_recovery.py --run-dir <execution> --analysis <analysis.json> --output <new-recovery.json>`。准备状态文件保留为运行前快照，当前状态以execution为准。

GPU预检查及每cell加载前均执行，发现占用/查询失败即ABORT，不自动重试。当前八项已终止完成，没有本轮后台待跑任务。一次fresh完整性核对另见该目录EXPERIMENT_AUDIT；没有方法GO或问题族判死。

## 2026-09-13：同动作 fast 保 KV（4/4 COMPLETE）

假说：减少完整块 ID 列表提取能降低观察成本，是否足以保住吞吐？固定动作的 CPU adapter 回放与实测历史决策均一致，优化后决策子区间 0.221/0.219s，但完整吞吐仍 −3.25%/−2.86%，平均完成 +7.61%/+7.23%。详见[报告](../../outputs/admission_capacity/20260913_headroom_fast_r01/REPORT.md)与[执行记录](../../outputs/admission_capacity/20260913_headroom_fast_r01/execution/execution.json)。原始记录没有覆盖。

成本模型新信息：纯新 decode +100 次、含重算调用 −8 次；width=1 调用 129→290。额外非调度引擎区间 0.497/0.438s，已不能全部归因于块表观察。下一动作是上述共享四臂，不是继续重复同一优化对照。

以下为已执行命令记录；复算另选空目录，勿向完成目录重跑：

```bash
python3 -u refine-logs/expert_saturation/experiments/admission_capacity/run_frozen_kv_remote.py \
  --source refine-logs/expert_saturation/outputs/admission_capacity/20260913_headroom_fast_r01/preparation \
  --output refine-logs/expert_saturation/outputs/admission_capacity/20260913_headroom_fast_r01/execution \
  --host root@connect.weste.seetacloud.com --port 23478 \
  --control-path /private/tmp/moe-paging-continuation-23478.sock \
  --gpu-uuid GPU-389be666-aeaa-c602-1504-85bf1dd3ac9f \
  --remote-dir /root/autodl-tmp/moe-headroom-fast-20260913-r01 \
  --labels repeat0-native repeat0-headroom repeat1-headroom repeat1-native
```

复算工具为 `analyze_completion_headroom.py --run-dir <execution> --output-dir <new-dir>`、`analyze_headroom_cost.py --run-dir <execution> --output <new.json>`、`analyze_headroom_work_cost.py --run-dir <execution> --output <new.json>`；均位于本目录。执行包 SHA256 `6ed591759959f8e354c1dae73ceab959f9b1ae45c54b62ae6d4d0b92e3e0b5c6`。

## 2026-09-12：实际 KV 池与长暂停

- 假说：秒级暂停与实际 KV 容量约束有关，不能全部计为重算计算。
- 运行：[20260912_kv_budget_r01](../../outputs/admission_capacity/20260912_kv_budget_r01/RESEARCH_REPORT.md)，具体命令见该目录 `COMMANDS.md`，四项均完成，128/128 请求。
- 结果：低预算每次 2 次抢占/7,685 重算位置，最长 ITL 4.591/4.560 秒；较大池无抢占，吞吐增加 3.99%/4.70%，但平均完成时间增加 1.02%/0.15%。
- 解释：增加资源确实改变请求结果；尚无同预算机制收益。最长暂停约 97% 在首次重算调用之前。
- 下一动作：同实际池检验原生恢复准入检查。

## 2026-09-12：完整历史准入 vs 首 chunk 准入（4/4 COMPLETE）

- 假说：解除完整历史检查可能缩短恢复等待，也可能造成恢复期间反复抢占。
- 执行前定位：[SOURCE_LOCALIZATION.md](../../outputs/admission_capacity/20260912_recovery_admission_r01/preparation/SOURCE_LOCALIZATION.md)。旧两次运行均有 187 次首 chunk 可容纳、完整历史不可容纳的拒绝，本轮对此执行真实配置干预。
- 固定：OLMoE BF16/vLLM0.26.0、cap32、3072/1024 token、50ms 到达、budget1024、实际 KV 16,089,350,144 bytes/7,671 可用块。
- 动作：只改 `scheduler_reserve_full_isl`；顺序 `full, chunk, chunk, full`，各策略独立完整执行。
- 包：[preparation/execution.tar.gz](../../outputs/admission_capacity/20260912_recovery_admission_r01/preparation/execution.tar.gz)，SHA256 `e2886420916c96f0793fbea52f22142b70e486460d447f3f69afd8863d9f08a0`。
- 源码检查：输入与原始冻结包一致；capture、memory telemetry、metrics、safe qualification 与已执行 KV 包逐字节一致；核验实际 scheduler flag 与实际块数。
- 本地验证：`python3 -m unittest test_pause_ledger test_paging_cost_model -v` 通过 11 项；新分析器对旧 `repeat0-budget90` raw 的兼容性 fixture 得到 2 次抢占、7,685 重算位置、4.590642666 秒最大 ITL。fixture 只适配新增配置元数据，不修改 raw，也不算新实验。
- 早期尝试：上传/远端执行曾被自动审批拒绝，GPU 0 次；用户随后明确授权本包上传与四项执行。
- 实测：`full → chunk → chunk → full` 全部完成，实际池均 7,671 可用块。两次 chunk 的最大 ITL 从 4.589/4.572 秒增至 5.973/5.866 秒，吞吐下降 6.89%/4.74%，平均完成延迟增加 7.83%/5.68%。
- 原因：两轮各 68 次没有新输出就再次抢占；第一个失败恢复末 chunk 需 50 块而仅剩 41 块。额外重算 116,242 位置与反复丢弃的恢复工作闭合。完整输出每对 31/32 相同，未做质量验证。

仓库根目录执行，使用现有已认证 SSH control socket；凭据不写入文件：

```bash
python3 -u refine-logs/expert_saturation/experiments/admission_capacity/run_frozen_kv_remote.py \
  --source refine-logs/expert_saturation/outputs/admission_capacity/20260912_recovery_admission_r01/preparation \
  --output refine-logs/expert_saturation/outputs/admission_capacity/20260912_recovery_admission_r01/execution \
  --host root@connect.weste.seetacloud.com --port 23478 \
  --control-path /tmp/moe-research-20260912-ssh \
  --gpu-uuid GPU-389be666-aeaa-c602-1504-85bf1dd3ac9f \
  --remote-dir /root/autodl-tmp/moe-recovery-admission-20260912-r01 \
  --labels repeat0-full repeat0-chunk repeat1-chunk repeat1-full

python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_recovery_admission.py \
  --run-dir refine-logs/expert_saturation/outputs/admission_capacity/20260912_recovery_admission_r01/execution \
  --output-dir refine-logs/expert_saturation/outputs/admission_capacity/20260912_recovery_admission_r01/analysis
```

请求级吞吐/暂停与全部延迟共同解释；原生 full 为本轮两个配置中更强的基线。
下一项是针对后续 KV 增长保障的最小保留/暂缓动作，而不是继续放松检查。尚无该新动作实现/实测、跨策略 Oracle 上界或同预算方法收益。

## 2026-09-12：修正预测输入与跨策略下界（CPU 完成）

- 假说：原耗尽趋势可能仍可从早期状态复现，但受害者数估计依赖原调度，不足以排除其他动作。
- 改动：同 cell 初始化 block size、仅窗口内宽度、窗口结束后发布预测；未来完成时间只用于等待重建；终态容量与固定路径受害者估计改为准确字段。
- 结果：[ADDENDUM.md](../../outputs/admission_capacity/20260912_kv_deficit_law_r01/ADDENDUM.md)。两个精确资源状态反例推翻“终态不足必然抢占”和“固定增长路径的 victim 数是跨策略下界”；40 项测试通过。
- 重算：复用上次 `analyze_kv_deficit.py` 四项 `--cell/--pause-ledger` 命令，仅将 `--output` 设为 `20260912_kv_deficit_law_r01/analysis/causal_cutoff_addendum.json`。两次低池在第 519 步后仍预测 807，实测 806；原始结果不覆盖。
- 解释：仅对原路径得到条件趋势预测；尚未验证动作排序或在线策略收益。下一项仍是固定实际池 full/chunk 原生对照，上传执行授权待回复。

## 2026-09-12：读取新增 pager 接入结果

- 新回传证据：[attempt02](../../outputs/admission_capacity/20260912_native_pager_r01/attempt02/readback/results/execution.json) 中 cap64/cap24 各 4/4 请求完成；[attempt03](../../outputs/admission_capacity/20260912_native_pager_r01/attempt03/readback/results/cap24_validation/pager_summary.json) 完成 16 层首个测量调用、每层至多 16 行的同输入 kernel 校验。
- 能力更新：真实 OLMoE BF16 / vLLM0.26 eager Triton / WiSP paging 及请求加载账本已跑通，不再只有人工专家 smoke。
- 范围：cap64/cap24 的专家 scratch 为 12/4.5 GiB、实际 KV 均 1 GiB；不是同显存预算策略比较。局部 kernel 校验也不代表完整质量验证，其校验运行不纳入性能结果。
- 成本线索：cap24 出现混合 prefill/decode 时约 1.33 秒 incumbent ITL，无 KV 抢占；分组导致的重复权重加载需计入成本。这是单 arm 观察，不能替代原长上下文恢复实验或证明动作收益。
- [attempt04](../../outputs/admission_capacity/20260912_native_pager_r01/attempt04/readback/results/execution.json) 的同 pager prefill 对照在 CUDA 初始化前因其他 GPU 进程退出，没有新测量或可等待的活跃执行句柄。
- 接续：恢复准入对照仍待明确上传/执行授权。授权阻塞已连续三轮出现；本地准备、模型修正与现有新数据核查已完成，后续需要新的 GPU 干预证据。研究结论仍 `OPEN`，执行目标记为等待授权的 `blocked`，不判死问题。

## 2026-09-12：保 KV 完成余量（PREPARED_UNRUN）

- 假说：耗尽前暂缓部分 decode 并保住一个请求完成所需 KV，可避免销毁/重算；完整请求收益未测。
- 代码：`completion_headroom.py`，198 行。`python3 -B -m unittest test_completion_headroom -v`，5/5 通过；CPU 状态不替代真实执行。
- 原始动作窗口：两轮 baseline step799，F16/H14/本步新增3；[source_probe.json](../../outputs/admission_capacity/20260912_completion_headroom_r01/source_probe.json)。
- 状态：自动审批拒绝上传/执行，GPU 0 次。现有 GPU 只读快照 UUID 匹配、2MiB、无 compute PID；执行前需重新检查。该限制是操作授权边界，不是科学 NO-GO。
- 冻结包与解释：[REPORT.md](../../outputs/admission_capacity/20260912_completion_headroom_r01/REPORT.md)。不覆盖旧四项结果。

仓库根目录执行；复用现有已认证 control socket，若失效先恢复同一连接：

```bash
python3 -u refine-logs/expert_saturation/experiments/admission_capacity/run_frozen_kv_remote.py \
  --source refine-logs/expert_saturation/outputs/admission_capacity/20260912_completion_headroom_r01/preparation \
  --output refine-logs/expert_saturation/outputs/admission_capacity/20260912_completion_headroom_r01/execution \
  --host root@connect.weste.seetacloud.com --port 23478 \
  --control-path /private/tmp/moe-paging-continuation-23478.sock \
  --gpu-uuid GPU-389be666-aeaa-c602-1504-85bf1dd3ac9f \
  --remote-dir /root/autodl-tmp/moe-completion-headroom-20260912-r01 \
  --labels repeat0-native repeat0-headroom repeat1-headroom repeat1-native
```

判定仍看同任务完整吞吐与每请求最大 ITL，同时保留 TTFT、平均完成、墙时和全部失败/暂停。只减少抢占不算成功。通过状态检查并获得实测后再决定机制继续、改动或停止；不预判整个研究问题。

### 用户授权后执行结果（4/4 COMPLETE）

用户明确同意本次上传执行。第一尝试 execution 在预检发现 PID30210 后退出，0次模型加载；进程与GPU上下文确认结束后，同包在新 execution02/remote r02 运行。四项均返回0、32/32完成，raw归档回传SHA匹配。

native/headroom/headroom/native 墙时分别22.913784/23.828437/23.803814/22.875082s；headroom两轮零抢占、零重算、maxITL1.381322/1.381332s，对比native4.468961/4.448064s。吞吐−3.84%/−3.90%、平均完成+8.28%/+8.46%，每轮29/32请求自身maxITL更大、31/32完成更晚。各对31/32完整输出相同，同策略重复32/32；不主张质量等价。

成本分解：调度区间多0.467/0.577s，引擎其余区间多0.449/0.343s；引擎调用1348→1440，真实重算7685→0。决策子区间已包含于调度，不能重复计费；其余引擎区间不是纯GPU时间。下一步只优化不改变held选择的监测/检查成本，未执行该优化，不能先扣除开销宣称收益。

```bash
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_completion_headroom.py \
  --run-dir refine-logs/expert_saturation/outputs/admission_capacity/20260912_completion_headroom_r01/execution02 \
  --output-dir /tmp/headroom-analysis-new
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_headroom_cost.py \
  --run-dir refine-logs/expert_saturation/outputs/admission_capacity/20260912_completion_headroom_r01/execution02 \
  --output /tmp/headroom-cost-new.json
```

两个分析输出均要求新路径；原analysis不可覆盖。执行命令使用前文同一参数，仅实际output目录为execution02、remote目录后缀为r02。结果是MEASUREMENT_ONLY；主问题仍OPEN，没有方法GO或问题NO-GO。
