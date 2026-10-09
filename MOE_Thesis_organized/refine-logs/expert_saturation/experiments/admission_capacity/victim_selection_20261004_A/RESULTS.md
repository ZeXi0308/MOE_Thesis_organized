# A victim selection — 2026-10-04

## 收束研究卡与贡献判断（2026-10-09，本地 CPU 复算）

**贡献说明。** 中心问题是：有限 KV 下，victim 释放的容量能维持多久、让哪些请求取得进展，以及这如何影响全部请求的等待。CacheOPT 已考虑完成前容量可达性、剩余输出及 KV 分桶，FastSwitch 已复用有效 host 片段；本线尚未证明“原生恢复重占”提供了它们及强简单规则之外的新决策原则。已有贡献资产是原生真实干预、完整服务取舍和有条件的容量解释；**当前没有获支持的新优化方法或独立论文中心主张**。本轮停止排序公式、host 代理和首次重入门槛评分的继续搜索，不把负结果自动包装成测量论文。

**服务与主目标。** 当前主要证据面向单卡文章续写批处理，用户等待完整结果。预先固定主指标为全部外部到达请求的平均 completion flow；TTFT、每请求最大生成间隔、输出吞吐、排空和个体损害作为必要副作用。没有可靠应用 SLO，未事后挑 goodput 阈值。允许取舍：max-release 的输出吞吐和 gap P95 改善有工程意义，但未改善本组声明的平均完成目标。输出数一致也不代表重算量或任务质量等价。

**部署与动作。** 三个主要完整规则在 OLMoE-1B-7B-0924 base BF16、vLLM 0.26.0、单 RTX PRO 6000 上比较；GPU KV 77,076,627,456B（36,752 可用16-token页）、host KV16GiB、context4096、maxseq384、batch1024、同步 FCFS。320 篇文章开环每0.01s到达，40个cap128、280个cap1024，允许 EOS；固定32/384/2预热并观察排空/600s截止。只改变 allocation failure 时原生未处理 suffix 内的 victim；恢复目标、触发、Q1、传输及准入不变。表中 prefix-ON 单次探针和较早 host-near 均单列；host-near 是此前 PRO 主机、GPU KV77,135,347,712B 的另一组运行域，不能跨行用绝对时延构造对照。

**三个假设分别判决。** H_problem：抢占确实带来长停顿和显著个体损害，合法动作能改变它们；但强简单预算之后仍有可稳定改善的总体损失，**尚未确定**，竞争解释是既有原则已覆盖当前可控空间。H_model：即时释放、重占及输出进展不同量，源码与轨迹支持这一点；但必要重入页数减可释放页数在强预算轨迹的三种建议中仅0–1页，尚不能预测有价值的窗口差异。H_method：新增 host/阶段/分桶或重入信号没有建立超过强简单方法的稳定主目标收益；**当前候选停止**，这不否定其他运行域的 victim 问题。

**本轮新增证据与资源。** 仅本地 CPU 复算下面8组已完成对照，没有启动、访问或轮询远端任务。清单中19个含 raw 的组均已有分析，未发现本地已完成但漏分析的组。自然摘要 session、原始结果包和分析均未在本地出现：只确认准备材料存在，远端终态保持未核实；文末旧排队状态是当时记录，不能当作现在仍在等待或已完成。历史6组启动前终止保留，不计作方法负结果。

### 精简证据表

A为该行基线，B为候选；每个数对按正序、反序给出 B−A，host-near 保留4个运行对，**不以请求或事件充当独立重复**。合法suffix列为每运行候选数的中位（多次运行不同则显示范围）及所有该臂观测的最小–最大值；包括原生合法的当前/partial/pending请求，旧 `qualified` 字段不作为合法性过滤。改选数是与成功原生 preempt/free 精确关联的非tail执行数；A→B不是两条分化轨迹间逐事件配对的改选数。

释放中位取所有实际动作；“B改选Δtail”比较该动作的真实free增量与同一前态原tail的引用计数推导可释放量，后者没有实际执行，不能当两种释放的因果差。抢占→下次输出以原生preempt返回至host收到后续token计时，包含排队、重算/加载和后续再次抢占，是重叠的观察停顿，**不能求和成独立恢复成本**。本表所有抢占均观测到后续输出。拒绝/超时没有独立计数字段，保留未知；各格全部到达请求均完成。完整每格吞吐、flow/TTFT均值及P95、最大gap、发送滞后、选择开销和load ACK数可由同一入口 `--details` 读取。

<!-- A_BOUNDARY_TABLE_BEGIN -->
| 完整策略对照 B / A | 合法suffix数；实际非tail改选 A→B | 容量（页） | 再次抢占与停顿 | 全请求 Δ秒：flow均值；TTFT P95；gap P95 | 输出率/排空与工作量 | 当前证据状态 |
|---|---|---|---|---|---|---|
| remaining-budget / tail [1](session-native-remaining-budget-westd53005-20261008-r02/) | A 139[17–241]；B 136[17–236]<br>改选 0→10（各对） | 释放中位 135→135（各对）<br>B改选Δtail中位 21/21 | 抢占 41→41（各对）<br>重复victim 1→4（各对）<br>抢占→下次输出P50 17.04→15.50s/17.08→15.54s | flow -0.364/+0.061<br>TTFT -0.500/-0.100<br>gap -1.671/-1.603 | token/s +0.209/-0.388%<br>排空 -0.206/+0.385s<br>Δtokens 0/0<br>序列/长度/终止差 19/0/0；19/0/0 | 主指标反序翻转；简单预算有个体收益，无稳定总体优势<br>各格 到达/完成/失败/未完 320/320/0/0 |
| max-release / remaining-budget [1](session-native-max-release-westd53005-20261008-r01/) | A 136[17–236]；B 134[8–234]<br>改选 10→24（各对） | 释放中位 135→214（各对）<br>B改选Δtail中位 30.5/30.5 | 抢占 41→27（各对）<br>重复victim 4→3（各对）<br>抢占→下次输出P50 15.46→15.65s/15.31→15.61s | flow +0.786/+0.577<br>TTFT +0.189/+0.232<br>gap -8.711/-8.512 | token/s +0.144/+0.588%<br>排空 -0.141/-0.572s<br>Δtokens 0/0<br>序列/长度/终止差 26/0/0；26/0/0 | 两对平均flow更差、gap P95更好；更少抢占不等于更快完成<br>各格 到达/完成/失败/未完 320/320/0/0 |
| cap-bucket / remaining-budget [1](session-native-cacheopt-cap-bucket-westd53005-20261009-r01/) | A 136[17–236]；B 111[16–238]<br>改选 10→36（各对） | 释放中位 135→109（各对）<br>B改选Δtail中位 -63/-63 | 抢占 41→43（各对）<br>重复victim 4→2（各对）<br>抢占→下次输出P50 15.29→18.41s/15.31→18.59s | flow +0.161/-0.043<br>TTFT -0.200/-0.373<br>gap +2.561/+2.791 | token/s -0.402/-0.110%<br>排空 +0.212/-0.074s<br>Δtokens -525/-525<br>序列/长度/终止差 25/2/1；25/2/1 | 主指标翻转、gap P95更差、工作量改变；仅CacheOPT式组件<br>各格 到达/完成/失败/未完 320/320/0/0 |
| 等释放 host once / tail（prefix ON） [1](session-native-equal-release-host-once-westd53005-20261008-r02/) | A 128[1–269]；B 121.5[11–269]<br>改选 0→1（各对） | 释放中位 138→139（各对）<br>B改选Δtail中位 0/0 | 抢占 27→26（各对）<br>重复victim 3→3（各对）<br>抢占→下次输出P50 8.15→9.43s/8.11→9.61s | flow -0.050/+0.205<br>TTFT -0.012/-0.089<br>gap -0.493/-0.403 | token/s +0.085/-0.222%<br>排空 -0.074/+0.195s<br>Δtokens 0/0<br>序列/长度/终止差 2/0/0；2/0/0 | 真实等释放干预；主指标翻转，替代victim受损<br>各格 到达/完成/失败/未完 320/320/0/0 |
| partial-restore once / tail（prefix ON） [1](session-native-partial-restore-once-westd53005-20261008-r01/) | A 128[1–269]；B 134.5[11–269]<br>改选 0→1（各对） | 释放中位 138→138.5（各对）<br>B改选Δtail中位 1/1 | 抢占 27→26（各对）<br>重复victim 3→3（各对）<br>抢占→下次输出P50 8.13→9.50s/8.30→9.50s | flow +0.411/+0.003<br>TTFT +0.251/+0.046<br>gap +0.020/-0.059 | token/s -0.573/-0.259%<br>排空 +0.503/+0.227s<br>Δtokens 0/0<br>序列/长度/终止差 2/0/0；2/0/0 | 两对平均flow无改善；替代victim再次抢占<br>各格 到达/完成/失败/未完 320/320/0/0 |
| 严格等释放 budget / tail [1](session-native-equal-release-budget-westd53005-20261008-r01/) | A 139[17–241]；B 139[17–241]<br>改选 0→0（各对） | 释放中位 135→135（各对）<br>B改选Δtail中位 不适用（零改选） | 抢占 41→41（各对）<br>重复victim 1→1（各对）<br>抢占→下次输出P50 17.19→17.15s/17.16→17.09s | flow +0.036/-0.196<br>TTFT +0.131/-0.299<br>gap -0.049/-0.057 | token/s -0.029/+0.112%<br>排空 +0.029/-0.111s<br>Δtokens 0/0<br>序列/长度/终止差 0/0/0；0/0/0 | 实际零改选；时延波动不是机制效果<br>各格 到达/完成/失败/未完 320/320/0/0 |
| host-near seeded / tail（4对） [1](session-native-host-near-seeded-20261008-r01/),[2](session-native-host-near-seeded-20261008-r02/) | A 101.5[1–252]；B 94–119[1–252]<br>改选 0→34/0→32/0→32/0→33 | 释放中位 135→137/135→146/135→146/135→144<br>B改选Δtail中位 未知/未知/未知/未知（未记引用计数） | 抢占 50→49/50→47/50→47/50→48<br>重复victim 2→2（各对）<br>抢占→下次输出P50 20.91→22.76s/20.94→23.18s/20.77→23.29s/20.82→23.25s | flow +0.318/-0.293/+0.177/+0.300<br>TTFT -0.037/-0.511/-0.060/+0.033<br>gap +0.066/-0.173/+0.081/+0.286 | token/s -0.298/+0.307/-0.212/-0.322%<br>排空 +0.324/-0.332/+0.231/+0.352s<br>Δtokens 0/0/0/0<br>序列/长度/终止差 24/0/0；29/0/0；29/0/0；26/0/0 | 3对平均flow变差、1对变好；非等释放、输出序列不同<br>各格 到达/完成/失败/未完 320/320/0/0 |
| 启动失败（历史保留） | 初始化前GPU忙 2 组；磁盘不足 4 组 | 未进入测量 | 无请求轨迹 | 不计入性能对照 | 不能记成请求失败或方法负结果 | receipt均ABORTED、cells为空 |
| 未测/未确认 | 自然摘要：本地未见原始结果；终态和服务效果未核实 | 新窗口评分器未执行 | 精确重算/复制成本未隔离 | 应用SLO goodput与独立确认未测 | 输出质量等价未验证 | 完整CacheOPT、第二模型泛化均未证明 |
<!-- A_BOUNDARY_TABLE_END -->

**稳定取舍和谁受损。** max-release 将抢占41→27、释放中位135→214页，却使平均flow增加0.786/0.577s；gap P95降低8.711/8.512s、输出率提高0.144%/0.588%，最大单请求gap反而增加0.489/0.758s。请求0042067的完成损失为21.773/21.652s。cap-bucket 实际非tail改选36次，但mean flow +0.161/−0.043s换号、gap P95增加2.561/2.791s，0066842稳定晚完成6.527/6.256s；每臂少525输出，不能宣称等工作量提速。剩余预算相对tail、等释放host的主指标均反序翻转；host-near的4对有3负1正。这些开发重复不提供显著性或独立负载确认。

**恢复边界。** max首格0064312在673释放203页，首次lookup及实际allocation external均0、无LOAD；675持48页是冷补算快照，679持192页且未新增输出又被抢，直到1126才输出下一token，完整生成间隔33.079s。它证明了容量快速重占与无新输出的重复抢占可以共存；不能把48页写成host复用或精确新增分配量。同事件普通remaining已建议另一完整decode候选，强预算参考的41次决策没有partial候选，故此坏例子尚不支持新保护器优于预算。

cap首格0021202在673/679动作前都host-ready49页、首次offer784 tokens，但前者实际请求784并有ACK，后者实际external0、无LOAD。Host保存量只有到真实allocation及ACK才证明兑现，且不等于服务节省；未兑现的具体原因仍未知。首662在三个已执行参考轨迹分别释放50/83/192页，首次allocation都约0.211s，保留suffix均36请求各新增3 token、无完成；这只是有限跨轨迹描述，不是同状态干预或一般窗口等价证明。继续解释这些事实可复用[容量窗口诊断](diagnose_capacity_window.py)，不再为它新增实验。

**开销与解释限制。** 主要三组每格选择器外层累计0.258–0.373s，含共同候选观测，已在端到端时延中；内部helper、观测与外层不能重复相加，也未测无观测版本的完整增量开销。部分旧组及cap候选仍出现JIT警告，具体记录保留在原分析，不能据小幅时钟差作机制归因。表中零改选组的时延差仅是运行波动；缺失字段（如旧host-near引用计数、精确重算量）不补成0。

### 可直接用于论文的边界结论

In the evaluated single-GPU article-continuation domain, victim selection substantially changed execution without establishing a new policy with repeatable improvement in mean completion time from external arrival. Remaining-budget, maximum-release, and cap-bucket rules replaced the native tail 10, 24, and 36 times per run in their respective comparisons. Maximum-release reduced preemptions from 41 to 27 and improved generation-gap P95 by 8.5–8.7 s, yet increased mean flow by 0.786 and 0.577 s, with individual completion losses approaching 22 s. A fixed CacheOPT-inspired cap-bucket component also changed execution, but its mean-flow effect reversed sign (+0.161/−0.043 s); it generated 525 fewer tokens and changed 25 request sequences. Equal-release host interventions likewise did not establish repeatable mean-flow gains. These observations weaken immediate release and host-ready capacity as sufficient proxies for victim value, rather than demonstrating scarce opportunities to act. Scope remains narrow: all 256 distinct legal candidates in the reference trajectory reached their declared output caps. Under private decode without prefix caching and native full-sequence reentry checks, required reentry capacity minus releasable capacity was zero or one page for the three recorded rule proposals at all 41 reference decisions. This is a necessary capacity relation, not a recovery-time prediction. We therefore stop developing this reentry-offset score in the tested domain. The evidence neither evaluates full CacheOPT nor establishes that victim selection lacks value generally; independent workloads and task-quality equivalence remain unverified.

### 最小复算与复用资产

在本目录执行，Python标准库即可，仅读已有 `plan.json`、`raw.json`、`selective-store.json`、`metrics.json`、`offload-events.json` 和失败receipt，不导入GPU运行时、不写结果、不联网：

```sh
python3 -B summarize_victim_boundary.py          # 从原始记录生成本表
python3 -B summarize_victim_boundary.py --check  # 与RESULTS表逐字比较
python3 -B summarize_victim_boundary.py --details # 每格绝对值及完整运行对差异
```

每行链接指向冻结组，组内plan/receipt及保留的候选包记录代码与输入版本；基线、配置和原始失败均未覆盖。等释放budget旧分析修正前后两版仍在，原始数据相同；本表直接读原始记录，避免沿用旧分析的prefix coordinator判定错误。复算脚本首次兼容旧offload schema时发现`allocated`字段缺失，现明确输出未知而非0；这是本地分析修复，不是GPU实验失败。

主论文可复用三类资产：原生合法suffix干预入口及冻结的强简单/近邻组件；覆盖外部等待、全部完成及输出差异的原始请求与事件数据；[必要重入模型](native_reentry_fit.py)、[容量窗口观察](diagnose_capacity_window.py)和[范围/事后终点诊断](shadow_remaining_scope.py)。它们支持相关系统的基线、取舍和边界论述，不支持“击败完整CacheOPT”或新选择器收益主张。

**收束决定。** 当前候选停止，不再用机制检查、近邻阈值或新增评分器延长本线。潜在重新投入条件是：另有真实服务证据表明强简单规则下仍存在重要可控损失，而且在合法选择时出现当前cap主导、完整私有decode集合没有的事件次序差异（例如自然完成或共享/部分驻留真正改变容量归还），能事前产生不同动作并预期区别于remaining-budget及最近方法。它与已失败候选的实质区别必须先明确；本轮只列条件，不启动该域实验，也不声称新机会已成立。

---
以下全部为历史记录。旧“当前”“下一步”、排队状态及GPU命令仅保留追溯，当前判决以上述收束卡为准；本轮没有访问远端，未声称历史任务已终止。

## 收束前研究卡（历史版本，以下计划不再自动执行）
**服务对象和目标。** 声明的部署假设是共享单卡上的有生成预算的文章续写作业，用户等待完整结果；现有文章回放不是生产流量或质量基准。主目标固定为全部外部到达请求的平均完成flow，研究短长作业竞争下的整体等待；TTFT、每请求最大生成间隔、吞吐、排空及个体损害是必要副作用，不要求所有指标同时改善。没有应用SLO，不用事后阈值宣称goodput成功。不得增加拒绝、截短、降精度或隐藏排队；失败和截止未完成必须保留，不能只比较完成者均值。输出差异允许诊断为浮点/调度差异，但未经任务质量评估不声称质量等价。

**部署域。** OLMoE-1B-7B-0924 base BF16、revision6d84c485…，vLLM0.26.0/torch2.11.0/Triton3.6，单PRO6000 97,887MiB、110GiB host cgroup；GPU KV77,076,627,456B（36,752可用16token页）、host KV16GiB，context4096、maxseq384、batch1024、同步FCFS。320请求开环i/100秒到达，40个声明cap128、280个cap1024，允许自然EOS；为既有合成长短预算压力诊断，不能外推真实对话。共享前缀版本另有64文章对，明确区分。固定32/384/2请求预热、私有编译种子、测量至排空/600s截止，源不等待引擎。正常容量来自本机profile，历史限容与旧GPU时间不作本机收益对照。

**损失、动作与最薄弱环节。** 普通预算保护可稳定让两个短请求提前约30s完成，但相对tail的全请求mean flow差−.364487/+.061444s仍不确定。最新完整强简单端点ABBA中，max-release将抢占41→27、受害请求37→24，gap P95改善8.5–8.7s，却使mean flow增加.785830/.576574s，并造成约22s的稳定个体完成损失。按主目标先保留预算原则；不把更少抢占当服务收益。尚未证明强近邻之后仍有重要可控剩余空间。合法动作仅是allocation failure时未处理suffix中的victim选择，原生free/preempt/waiting、目标、1秒触发、Q1、传输和准入固定。

**最小模型与边界。** 检查释放能否在原生恢复重新进入前支持保留集合的近期完成或跨过压力阶段，只比较现有三个规则提出的候选。新源码/轨迹诊断表明：当前prefix-OFF、完整私有decode、full-ISL-fit条件下，重新进入必须容纳已有序列，必要页数Q=ceil((prompt+output)/16)；动作前引用计数推导的可释放页J与Q只差0–1页。预算基线41个决策的三份建议均如此，所有合法候选中无部分恢复；因此这项门槛状态在强基线轨迹上尚无足够增量信息，不能据此开发窗口评分器。Q−J只是固定其他队列/预留条件的容量偏移，不预测首次重入时刻或服务窗口；实际轨迹还受保留集合增长、完成归还、其他恢复、host兑现和控制阶段影响。容量记账关系仍为初始free+即时释放+后续归还−running新增占用−准入/恢复新增占用。等释放pure decode在共同推进k token、无完成/准入/再次抢占下新增页需求最多差1页，不是调度步数保证。cap−output是输出上界，未知自然stop不能成为在线预计完成时间。本组prefix-OFF没有GPU本地前缀重叠；先前prefix-ON重叠现象不能沿用解释当前复用减少。当前host-ready仍会在等待中失效，只有实际allocation接受的external KV及ACK可证明兑现，且不等于服务节省。

**近邻与贡献边界。** [CacheOPT §3.3–3.4](https://arxiv.org/html/2503.13773v2#S3.SS3)已检查请求共同推进到预计完成前的容量可达性及及时释放供给，victim按SLO桶降序、预测剩余输出桶降序、同剩余桶内KV占用桶升序；不能声称它只看静态占用。预算排序只是简单组件适配，“终点有释放但途中不能超容量”也不是新原则。[FastSwitch §3.3](https://arxiv.org/html/2411.18424v1#S3.S3)已复用有效host片段，[TOPAS](https://arxiv.org/html/2608.25523v1#S4)联合前缀驻留、工作流进度、decode reservation与重建/移动成本；“成本感知”和“共享引用计数”均不是新原则。[UniBoost](https://arxiv.org/html/2606.18431v1#S3.S3)已有输出进度保护，当前stage与首输出规则106/106同选，撤销stage增量。尚待证的具体差别是：为每个victim考虑保留集合跨完成事件的容量轨迹和原生恢复重占，是否有已有排序之外的增量决策价值；近邻段落未显式展示此计算，不能仅凭未见写法宣称创新。目前仅有局部经验事实，无中心方法贡献、独立确认或可投稿结论。

| 假设及状态 | 当前支持证据 | 最强竞争解释 | 判别与削弱条件 |
|---|---|---|---|
| H_problem：强简单方案后仍有重要可控等待。**尚未成立** | 预算有个体价值；max少14次抢占仍使mean flow+.786/+.577s；cap分桶主效应换号且gap P95更差 | 已有简单原则已覆盖可控空间；当前合法候选全达cap，未覆盖自然完成异质性 | 当前预算轨迹无partial，扩全running/替换真实最终长度均41/41同选。下一步只在固定摘要服务域核实自然质量、正常容量压力及合法动作；无机会即收束，不恢复退休评分器 |
| H_model：可见状态能预测有意义的全局差异。**必要容量门槛获支持，服务预测未成立** | Q−J在预算41决策三规则建议中仅0/1；旧max唯一partial为11页，普通remaining已选另一完整候选 | 门槛变化只是已有排序的重述；实际窗口更多由队列/其他完成与恢复共同决定 | 不把静态抵消当相同实际时间；只有在强基线状态出现不同且可解释的事件次序预测，才值得新增模型 |
| H_method：新增方法超过强简单与近邻。**方法未获支持** | 两完整简单原则及固定分桶组件均真实执行；分桶36次实际改选，但mean flow+.161/−.043s | 静态排序没有稳定增量；输出减少、时钟漂移和损失转移可解释部分表象 | 新窗口原则必须先产生不同且有价值的动作，再开发online选择；固定组件结果不是完整CacheOPT复现/击败，也不是新贡献 |

**当前判决。** 普通预算在上一组mean-flow优于最大释放；固定CacheOPT式cap分桶主效应换号且gap P95更差，停止此代理的调参，不能据此否定完整CacheOPT。新增释放窗口诊断没有建立预算之外的动作价值：纯decode必要fit偏移退化到0–1页，唯一明显partial例外在较差max轨迹中且普通预算已避开。收束当前“纯decode＋首次重入门槛”评分方向，不为它增加GPU组、预测器或相邻阈值；这不是所有victim选择无价值的结论。未来继续投入须出现强简单方案未覆盖的具体损失和不同动作预测；先前已失败的完成等待机制也不改名复活。当前保留可复现原型、正负结果与有条件的容量解释，不包装成新方法论文。

**最新状态：** 上一cap-bucket r01四格均320/320完成、失败/未完成0；分桶少525输出，25序列/2长度/1终止变化，质量等价未验证；112原始文件和唯一分析完整保留，57669/SSH4613已终态。新CPU诊断确认256个合法suffix候选均最终到cap，扩候选范围或给定真实终点也不改变41次预算建议，故不继续该域的预测器/范围调参。现成Instruct的固定自然摘要输入与单组可行性原型已冻结，唯一74306/SSH12296于02:25:07 CST实查等待公共锁（73467持有、73874在前），尚无A CUDA/session或新GPU结果。源、原始、版本、命令与预算见文末；新版AGENTS已重读，目标ACTIVE，独立中心贡献和可投稿证据尚未成立。

以下为按发生顺序保留的历史假设、实施与结果；当前研究判断以上述贡献说明及文末最新证据为准。

## 本轮收束复算（2026-10-09，仅本地 CPU）
当前未知：已完成实验中，实际改选、容量差异和完整服务效果是否支持同一结论；尚未分析的本地结果是否改变判决。
主要竞争解释：没有合法机会／规则未执行，或真实改选只转移损失、改变工作量而没有稳定主目标收益。
最小行动：仅复算已有 raw/store/metrics，按运行对保留正反序、失败和未测项；预计本地 CPU 数分钟，GPU 0，不访问远端。
不同结果将如何改变决定：有稳定取舍则保留为边界证据，零干预不归因；新的机会只列重新投入条件。本轮停止当前候选，不启动新实验。




## 假设、修改与版本

当前 host KV 的有效连续前缀能否预测 victim 后续恢复损失？只替换 recovery funding victim；原规则为逆序第一个合法单请求 funder。`candidate/pkg/funding_victim.py` 的 `host_missing` 在原合法集合内最小化“完整物化页数−连续 ready host 前缀页数”，同分原尾序，未知状态回退。只读缓存字典；pending store 不算 ready，不调用原生 lookup/touch。目标、1秒触发、Q1、传输、commit、准入和分配失败 tail 全固定。

原生增量 store 的 cursor 单调前进，LRU 淘汰的早期洞不会由普通 native_full 准备步补齐；full-attention lookup 遇首个 MISS 停止。因此信号有代码依据，但不保证本域有异质性，也不是精确恢复耗时。

工作起点 `agent/publish-current-moe-code` / `5593b5ff0fb602f479b10f82ca682ba8f9adc86d`。原冻结包不修改；本轮代码完整哈希见 [plan.json](plan.json)、[manifest](candidate/manifest.json)（`928dc81c…54c4c5`）。已读 AGENTS、A_progress、近期共享台账、GPU 协调和 recovery_quantum 索引；旧大小排序、residence-density 和 h/d 失败不重跑。

## 固定实验口径

**已完成首组属于显式限容的受控机制实验，不是 PRO 6000 默认主实验配置。** OLMoE-1B-7B-0924 base，固定 revision `6d84c485…`，BF16；4096可用GPU KV页、host16GiB；128篇既有文章、外部每0.2秒到达、自然EOS允许/输出上限1024。相同长短预热，每臂独立初始空编译缓存。新授权 RTX PRO 6000、120GiB cgroup；不混合旧5090时间。

顺序 tail / host_missing / host_missing / tail。主观察量为全部请求从外部到达到完成的平均 flow；所有 TTFT、完成时间、每请求最大生成间隔及输出差异完整保留。无应用SLO；原20点 goodput 网格仅开发诊断。统计单位是运行，不是200个决策事件；非同状态重放或等工作量加速。

## 首组真实结果

四格均128/128完成，失败0、未完成0，整组954.55秒。原件 [session-r01](session-r01/)，唯一完整分析 [analysis-host-r01.json](analysis-host-r01.json)，108个输出文件哈希全部核对。归档 SHA256 `35990f50e69dd82a09a24980a0ee5931721f76f4fa1ff1be13de0e449e834387`。

| 次序/策略 | 输出总数 | token/s | 完成req/s | 平均flow秒 | TTFT P95秒 | flow P95秒 | 请求max-gap P95秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 tail | 127373 | 1493.41 | 1.501 | 38.316 | 47.759 | 61.317 | 1.764 |
| 1 host_missing | 127289 | 1473.98 | 1.482 | 39.040 | 48.087 | 61.767 | 2.002 |
| 2 host_missing | 128290 | 1479.35 | 1.476 | 39.628 | 48.772 | 62.651 | 2.143 |
| 3 tail | 128290 | 1495.83 | 1.492 | 38.704 | 47.924 | 61.466 | 1.601 |

host_missing 对各自 tail：输出率−1.30%/−1.10%，平均flow+1.89%/+2.39%，gap P95+13.47%/+33.86%。86/128、101/128请求完成更晚；最大flow增加28.780/21.066秒，均为非funding目标/非victim的0052593，TTFT分别增加29.829/21.509秒。这些是独立轨迹差异，不能归因给未发生的 victim 替换。序列差异63/58，stop差异2/0；开发goodput网格分别5胜15负、1胜19负，不选择有利阈值。

四格 funding 决策53/47/47/53，**全部合法候选的missing suffix均为0，unknown0，实际选择改变0/200**。原生实际抢占126/109/112/117；存在重复抢占的请求27/27/19/27。200个commit关联raw无错误；199次强制抢占的实际释放均等于victim当时持页，另1例direct-ready无抢占、释放0。目标anchor到下一输出中位约69ms；实际victim输出gap均值0.996/1.154/1.195/1.033秒，均为绝对停顿而非增量因果损失。所有目标和victim最终完成。

成功选择边界的观察+选择墙时总计55.49/46.84/48.74/54.22ms，约捕获时长0.054%–0.065%，两臂都有同一观测。它不是完整controller成本，也不足以解释全部运行差异。时间触发、批处理和输出轨迹可分化；当前证据不区分具体波动来源。准确失败类型是**动作未改变/信号退化**，不是“host成本预测已被有动作实验推翻”。Oracle、同状态干预、自然任务质量、第二模型均未测。

## 已准备、暂缓：小容量原始 arrival 简单基线

两次真实tail运行各53次动作，按原始到达时间选最晚请求均会改变19次：正序16次少释放/1次同容量/2次更多，反序15/1/3；改变时持页差中位−80/−32。复现 `python3 -B shadow_existing_arrival.py --session session-r01`，按runner原式epoch_origin+external_arrival重建原始到达时间，只向stdout输出，不新增一套结果汇总。这只是一条轨迹上的影子机会，不是执行结果。step1028原tail0042067恢复后仅再输出1便被选，持205页；arrival候选0042542持85页。step2781两候选同为65页。保留容量差异，不预称同容量损失降低。

新理由是恢复后append到running尾部会改变victim次序；原始arrival保持FCFS年龄，不使用沉没恢复成本。此规则是已有强简单基线，不是新颖方法。只改这一信号，其他配置及主指标原样；tail/arrival/arrival/tail，各128。版本 [candidate_arrival_r01](candidate_arrival_r01/) / manifest `5bd535e9…bd2396`，计划 [plan-arrival-r01.json](plan-arrival-r01.json)。选择及未知状态回退CPU检查通过。实际状态 **UNRUN_LOCK_BUSY**：另一B组controller8302/driver8311已持共同锁，A提交在初始化前退出，未创建session-arrival-r01；[原始失败日志](lock-busy-arrival-launch.log)。不等待其他研究交付，不干扰其进程，也不创建候卡任务。

## 复现与失败保留

远端根 `/root/autodl-tmp/moe-a-victim-20261004`，共同锁 `/root/autodl-tmp/moe-research-gpu.lock`，整组非阻塞flock，逐格查GPU占用。首组已释放锁。不要重跑已存在session路径；新重复复制计划并换全新session_dir。下列是已准备的小容量复现命令；当前优先执行后文正常容量表征，勿将小容量配置当作主实验：

```sh
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
/root/miniconda3/bin/python run_arrival_group.py --plan plan-arrival-r01.json
# 本地重新分析必须使用新的输出文件名
python3 analyze.py --session session-r01 --output NEW-analysis.json
```

远端torch目录混有75个未列入RECORD的旧.py，首次import报duplicate template。本任务 [prepare_runtime_overlay.py](prepare_runtime_overlay.py) 建立11678个白名单文件软链接，未改全局；[runtime_repair.json](runtime_repair.json)保存失败栈与CPU导入成功证据。runner为所有臂显式设置同一overlay。新主机复现须先用该脚本的`--output`创建全新overlay，并按真实设备填写计划。

最初27项CPU检查有6项因复制后的测试相对路径失败，修正本线路径后27/27通过。首上传tar含macOS生成的`._*`，exact-file-set拒绝；只删除本线未启动包内元数据，原tar保留。首组首次提交也遇共同锁忙，保留 [lock-busy-launch.log](lock-busy-launch.log)。上述均非GPU性能结果；没有覆盖历史失败或推送仓库。

最近邻：[CacheOPT §3.4–3.5](https://arxiv.org/html/2503.13773v2#S3.S4) 的SLO/剩余量/KV分桶与成本目标相关；[UniBoost Algorithm1](https://arxiv.org/pdf/2606.18431#page=13)显式传入swap-cost而未展开selector，不能声称其忽略迁移成本。下一组若有一致净收益，先独立到达序列和去信号对照，再讨论局部干预、最近邻方法、自然EOS质量与第二模型；当前不包装独立贡献。

## PRO 6000 正常容量准备与旧入口阻塞（2026-10-04历史）

保持既有 OLMoE BF16/原生上下文4096、host KV16GiB、batch token1024、原生准入规则、Q1及1秒触发；并发上限统一提高到384，作为新运行域的固定配置。先以 gpu_memory_utilization=.90 实测可用KV池，随后同组各测量格显式固定该实际字节预算。预计容量不作为实测数据。128个请求即使全部到达4096token也只需64GiB，因此旧128/低并发输入不能代表本卡容量压力。

从三个既有互不重叠的文章池按输入长度分层交错，构造嵌套128/256/320请求，统一外部0.01秒到达、允许自然EOS/上限1024；只按输入长度构造，不看未来完成/输出。同一宽批预热并清空connector缓存。初次仅tail基线做三个点，记录实际峰值占用、同时在途请求、真实抢占/恢复与合法候选；“低/拐点/高”是计划标签，是否达到压力以实际数据判定。不存在动作也保留为有效边界，不缩KV制造结果。

实现位置 [candidate_pro6000_r01](candidate_pro6000_r01/)、控制器 [run_pro_group.py](run_pro_group.py)（复用既有整组共同锁），不改历史包。新增只读O(1)调度边界资源极值，保留真实候选与host/arrival影子机会。 `analyze_pro.py`现额外比较host选择与最少computed整页/最少持页排序，记录零或一致host前缀，以及相同持页/相同computed层内是否仍有host差异；UNKNOWN单列。若信号仅退化为已失败的大小排序，不把它改名当新候选。该补充只改分析器，GPU冻结包未变。正常容量表征不做不同工作量间的策略收益宣称；根据实测动作和信号区分度，才选一个有信息的压力点做两策略反序对照。

输入构建器 [build_pro6000_inputs.py](build_pro6000_inputs.py)：三个既有探索文章池共384条，原ID/doc/content/token hash均唯一，未重分词或截断，不是新holdout。64个输入长度层交错，128/256/320每层分别取2/4/5条；prompt-only页需求30.027/60.057/75.076GiB，满1024输出潜在46.027/92.057/115.076GiB，均非实测占用。32/384宽预热输出16/93，记录实测纯decode宽度，不预称已覆盖384。

版本 [plan-pro-capacity-r01.json](plan-pro-capacity-r01.json)，38文件manifest `ffdbab66f6507936bceec5dbd5d16854adb0282918c0248cd5d9e71ec237fa64`；部署包SHA256 `2455b3b6eec8f8655cb025c173bafefd54ea875ee2cb1001ecc885769bf646ed`。27项既有CPU检查通过；新增资源观察最初遇mock缺num_gpu_blocks，已仅在本线测试fixture补真实pool属性后整套通过，非GPU错误。输入hash/上下文、profile/pin路径检查通过；宽预热与正常KV分配仍须实际GPU验证。

首次非阻塞提交在共同 `flock(LOCK_EX|LOCK_NB)` 返回EAGAIN，保留[原始失败日志](lock-busy-pro-capacity-launch.log)。随后唯一一次有限候锁由 `run_pro_group_wait.py` / PID15254 / SSH exec40366执行，900秒到期在flock处退出（exit1），保留[原始超时日志](wait-timeout-pro-capacity-launch.log)。日志中的通用“Group budget”来自共用SIGALRM处理器，实际触发的是获取锁前900秒等待预算，GPU整组6000秒预算尚未开始。

2026-10-04 09:02:28UTC实查：PID15254已消失，`session-pro-capacity-r01` **未创建**；B controller13233仍存活，实际GPU worker18195使用88092MiB。A无GPU子进程、候卡者或持有锁，不自动重试。 09:07:37UTC最新实查：B已结束，C PID15300实际使用80186MiB；A session仍未创建。恢复后连续三轮共享GPU不可用，目标再次BLOCKED，需真实资源释放后继续。这是资源占用阻塞，正常容量profile/宽预热/三档请求实验全为UNRUN，不能据CPU检查或候锁过程宣称性能结果。

GPU释放后，在已部署远端本线根目录运行以下整组命令（profile→low-tail→knee-tail→high-tail，单格上限1200秒、整组6000秒；忙则退出，无自动重试）：

```sh
cd /root/autodl-tmp/moe-a-victim-20261004
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python run_pro_group.py --plan plan-pro-capacity-r01.json
# 取回完整session后，在本地本线目录生成唯一新分析；不得覆盖旧输出
python3 -B analyze_pro.py --session session-pro-capacity-r01 --output analysis-pro-capacity-r01.json
```

不增加请求上限之外的新控制器，不更改host预算/触发/量子/传输；同组所有测量格以首次正常profile的实际KV字节值固定。下一步只运行这三个有判别力的点，不扫描全参数空间。

唯一有限等待已结束，不重复排队。等待入口只改变锁获取方式，冻结包、输入、profile/pin和组内配置均未改；后续需实际GPU整组释放后，使用上面的既有非阻塞命令执行。

## 2026-10-07 新入口恢复

westb:25495，实际PRO6000 UUID `GPU-94203fc3-1021-3a9c-a367-cff792479616`，97887MiB/120GiB cgroup。原离线模型、runtime版本及38文件冻结包校验通过；历史结果不作新卡基线。新计划[plan-pro-capacity-20261007-r01.json](plan-pro-capacity-20261007-r01.json)，只更新授权物理UUID、主机说明和独立session目录，科学配置/代码/输入不变。唯一候锁PID4153等待813.667秒后取得共同锁，整组已完成、SSH exec76258退出0并释放锁。日志group-pro-capacity-20261007-r01.log；本轮实际结果见下。

最近邻补充核对（2026-10-07）：[CacheOPT v2 §3.4–3.5](https://arxiv.org/html/2503.13773v2#S3.S4)在同质SLO下仍按预测剩余输出量分桶，再按KV占用排序；自然EOS下不能把1024−produced称其预测器。统一硬上限时该差值排序仅等价于反向排序已生成量。[UniBoost v1正文与算法](https://arxiv.org/html/2606.18431v1#S3.S2)采用已获服务/到达优先级及服务保护，SelectVictim虽接收swap-cost但未展开完整函数，且正文与附录服务信号表达存在差别；不能把单一已生成量规则称其完整复现。冻结组不变；后续若用此简单规则，按“最少已生成输出/剩余硬额度”命名并报告局限。

## 2026-10-07 正常容量实测：有抢占，funding入口没有执行

状态 **CHARACTERIZATION_COMPLETE**。profile+三档tail全部exit0，704/704请求完成、0失败/未完成。实际物理KV77135347712bytes（71.8379GiB，含1个null页），36780可用页；各测量格PIN_VERIFIED，同模型/BF16、host16GiB、maxseq384、batch1024、Q1/触发/准入/传输不变。三格相同宽预热均实际达到384 running及384 pure-decode。所有返回chunk均1token，无ITL插值。

| 请求数 | 输出token/s | req/s | TTFT P50/P95(s) | 完成flow均值/P95(s) | 请求最大gap P95/全局max(s) | 原生抢占 | 最大running/已用页 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2722.39 | 2.773 | 2.381/5.985 | 41.057/44.851 | .086/.086 | 0 | 123/21521 |
| 256 | 2890.00 | 2.912 | 6.563/20.058 | 70.391/83.698 | 16.392/27.860 | 31 | 247/36780 |
| 320 | 2917.29 | 2.937 | 9.331/70.101 | 79.254/105.512 | 27.216/39.832 | 50 | 265/36780 |

完成时间从外部到达计。输出总量125671/254106/317800，EOS数6/9/11，其余到1024上限；三种不同负载不作等工作量或策略加速比较。各1次运行，无运行级方差估计；无应用SLO，不从20点诊断goodput网格挑有利阈值。占用是调度边界极值，非连续峰值。

funding选择/anchor均0。256/320格达到1秒年龄检查后的366/531次尝试均在全局`KEEP_PENDING_TRANSFER_JOBS`被拒（`staged_store_rotation.py:1048`），没有进入容量判断和victim选择；不是“host信号没区分”或“Q1失败”。gate计数只记首拒绝原因，不是阻塞秒数。低压确实无抢占；高两档确实有长停顿，不能将零funding归为没压力。

原生victim合法suffix最多219/252候选，但original-arrival影子与tail为0/31、0/50差异（high1次未知沿原fallback），不跑这个同动作对照。外部到达2.55/3.19秒即结束，首次抢占37.169/27.664秒。存在与tail同持页替代的决策10/31、19/50，说明仍有选择空间。high的0025181仅输出8token后被抢占，最大gap39.832秒；0026292零输出被抢占，40.474秒后首次输出，不合funding目标资格；0070086/0030422各被抢占两次。均为本轨迹绝对损失，尚无反事实增量伤害。

原始[session-pro-capacity-20261007-r01](session-pro-capacity-20261007-r01/)，唯一分析[analysis-pro-capacity-20261007-r01.json](analysis-pro-capacity-20261007-r01.json)；95输出文件本地逐一SHA256校验通过，原始压缩包SHA256 `0d8ce3f64cc68010d5220e3e04c1942ed65c4e89e60399d35bc8b0d28d250ca7`。首次本地解压因系统Python不支持`filter=`在写文件前失败，改用逐成员安全检查后成功；不是GPU实验失败。冻结包manifest仍ffdbab66…7fa64；计划4c3ca230…9743。复现使用新session路径，不覆盖已有结果：

```sh
# 远端：复制plan-pro-capacity-20261007-r01.json并仅换全新session_dir
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python run_pro_group.py --plan NEW-plan.json
# 本地唯一分析的原始命令；重分析须换输出名
python3 -B analyze_pro.py --session session-pro-capacity-20261007-r01 --output analysis-pro-capacity-20261007-r01.json
```

下一步：固定本次high负载/正常预算，仅补原生allocation-failure候选的host ready-prefix和准确释放页数观测（现有funding观察未覆盖该入口），做单格tail探针。先判断信号是否有区分、是否只是大小排序，再决定最小策略。保持全局pending条件和恢复目标/Q1不变，不借改变触发来制造动作。

### 原生候选观察探针（GPU UNRUN：取得锁后入场占用检查退出）

[candidate_native_host_probe_r01](candidate_native_host_probe_r01/)只改两个源码：普通suffix候选增加现成host_prefix、computed/output/max_tokens、`pending_native_store_dependencies`及合计观察/选择耗时；现有稀疏抢占wrapper前后各读一次free pool，记录`actual_released_blocks`。不修改资格、排序或生命周期。依赖计数只指未收到全worker确认的本请求STORE jobs；原生flush发生在后续worker.wait，既非该计数也非_preempt_request墙时可充当暴露等待。

13项既有host/native/lease CPU检查通过；5项请求测量检查复用原fixture补只读pool后通过，成功/失败事件都验证7→10释放3页和hook恢复。实际adapter fixture验证host[6,1]、依赖[1,0]或UNKNOWN都仍tail。仅CPU正确性证据。manifest `8568fa3ec96b8597b40caed44cd9db2d57c101ac047ddcea777671eab5b23103`；[r01 plan](plan-native-host-probe-20261007-r01.json) SHA `1db2e71b9547085eb2c2798347026a41cdb275cba35d984b9424ce2b06b01459`。同320 high、同实际77135347712bytes、同384宽预热，单格tail，无策略收益对照。

唯一controller13301等待135.073秒取得共同锁；2026-10-07 08:39:54UTC入场仍发现前一作业PID11853占14464MiB，按原安全检查在CUDA初始化前exit1并释放锁。保留[r01失败log](group-native-host-probe-20261007-r01.log)及[原始入场记录](session-native-host-probe-20261007-r01/gpu-start-occupancy.json)。receipt的RUNNING是异常发生在runner初始化try之外遗留字段，真实状态是**ABORTED_BEFORE_CUDA/零测量请求**，不能当运行中；没有cell目录或A GPU子进程。后查同一inode2304:15049831297由PID13431持有，GPU80052MiB；不干扰，不重复排候卡。r02仅换新session目录，科学包/配置不变，已准备[计划](plan-native-host-probe-20261007-r02.json)（SHA b67882aa…01262），未提交。

```sh
cd candidate_native_host_probe_r01
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_funding_victim_cpu test_bidkv_full_running_cpu test_native_lease_cpu
# 远端本线根目录，整卡真正空闲后执行，LD_LIBRARY_PATH与前述相同；忙则退出
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -u run_pro_group.py --plan plan-native-host-probe-20261007-r02.json
# 完整结果取回后，唯一新增分析（当前不能运行，因为没有测量数据）
python3 -B analyze_native_probe.py --session session-native-host-probe-20261007-r02 --output analysis-native-host-probe-20261007-r02.json
```

为保留磁盘余量，只清理已完成A正常容量组302673137bytes私有编译缓存；95个output与archive经双侧hash相同后以硬链接去重618711258bytes，所有原路径、原始数据及本地压缩包保留，未操作其他会话文件。

续轮CPU分析：同持页替代中，进一步要求与tail的`floor(computed_tokens/16)`也相同，knee仍有10/31次（15对），high有18/50次（27对）；例如high step600有两候选都持184页、已计算183整页。`analyze_native_probe.py`补此严格层内比较，未知computed保持UNKNOWN；避免把同持页下多算一整页造成的missing-suffix差异误归于host前缀。这是已有轨迹的选择空间，非已执行替换或收益。2026-10-07 08:47UTC实查13431仍持同一共同锁并用80052MiB，A r02 session不存在、无A作业或候卡者。

第三项紧邻核对：[FastSwitch v1 §3.3](https://arxiv.org/html/2411.18424v1#S3.S3)已经跟踪CPU副本有效块组并避免不必要换出；[§3.2 Algorithm1](https://arxiv.org/html/2411.18424v1#S3.S2)按完成事件/块组冲突同步，不能把CPU复用或依赖同步本身称本线创新。其[§4](https://arxiv.org/html/2411.18424v1#S4)请求优先级来自预生成轨迹，所读算法未把连续ready前缀或pending STORE计数作为victim排序键；这不证明无人研究同一规则，也不能把min-host-missing命名为FastSwitch复现。另限读[FastServe §4.2](https://www.usenix.org/system/files/nsdi26-wu-bingyang.pdf#page=8)的预计再次调度时间ENST交换顺序、[InferCept §4.3](https://arxiv.org/html/2402.01869v2#S4.SS3)的API暂停请求内存浪费预算，候选/决策变量均不同。此后不扩大首次实验的文献前置条件。

2026-10-07 08:52:47UTC再次验证13431进程存活、实际持共同锁/GPU80052MiB；r02未创建。当前只有GPU操作受阻，补丁、严格配对分析及可运行命令已就绪；未降低研究目标、未标完成、未再次排候卡。

2026-10-07 08:55:16UTC第三个连续goal轮次验证共享GPU阻塞：13431已终止，存活进程13862已接力持同一inode2304:15049831297，GPU worker14278/13928MiB；A r02不存在，13301已消失，无A作业/候卡。目标现为 **BLOCKED**，不是研究完成；仅待真实资源释放后执行已准备r02。正常容量704请求结果保持有效，探针仍零GPU测量，不能宣称host或保存依赖策略收益。

### 恢复与磁盘阻塞处理（2026-10-07）

11:54:06UTC确认旧GPU进程退出、共同锁无人持有、0MiB/0%，直接提交r02；在CUDA初始化前因共享数据盘仅317MiB而退出，最终receipt **ABORTED**、cells为空。[失败日志](group-native-host-probe-20261007-r02.log)与[原始r02目录](session-native-host-probe-20261007-r02/)保留。重取r01最终receipt也已为ABORTED；此前对RUNNING的描述是早期观察，不能据此声称runner没记录异常。

仅在已完成A目录内回收：2126842861bytes私有编译缓存；797对output/archive双侧hash验证后，702个重复文件硬链接去重860766248bytes；74个大raw/warmup JSON无损gzip（1371688077→83023453bytes），解压后逐一匹配原`output_sha256.json`。两个别名均保留`.json.gz`，原始hash未改，本地原始JSON/压缩归档仍在；读取旧远端plain路径前用gzip解压即可。模型、冻结源码及其他会话数据未改；base与Instruct三对权重SHA全不同，合理占26GB，未删除。磁盘余量恢复4627476480bytes，高于原4GiB门槛。

科学配置未改，新[r03计划](plan-native-host-probe-20261007-r03.json) SHA `9d8c34884b516dace9155d576ec01dfed3fc51d7f3929bbceb346aa8356fb654`。唯一controller43175/SSH exec75811于同一共同锁限时等待1800秒，12:44UTC核实活跃；前方42868持锁、43152等待。无第二A等待者；不是性能结果，也未重跑已完成容量组。准确执行命令：

```sh
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -u run_pro_group_wait_20261007.py --plan plan-native-host-probe-20261007-r03.json
# 测量完成并取回后：
python3 -B analyze_native_probe.py --session session-native-host-probe-20261007-r03 --output analysis-native-host-probe-20261007-r03.json
```


### 原生探针完成与下一步依据（2026-10-07 13:10UTC）

r03 **CHARACTERIZATION_COMPLETE**，controller43175/SSH75811退出0，共同锁已释放、末边界GPU为空。候锁1324.373秒不计GPU组；组303.854秒，单格290.820秒，测量窗口108.847413秒。[原始session](session-native-host-probe-20261007-r03/)、唯一[分析](analysis-native-host-probe-20261007-r03.json)；28输出SHA逐一核对，完整tar SHA256 `d61abc8c6869da84b9c82167eaf959ee68df8fff979bcabd89ccb435d18c9266`。前述r01/r02均是入场前ABORTED，r03是真实GPU完成；不混合为重复。

320/320完成、失败/未完成0，输出317800（309 length/11 stop）。吞吐2919.684 token/s、2.940 req/s；外部到达计TTFT P50/P95 9.346/70.164秒，完成flow均值/P95 79.280/105.427秒；逐请求最大gap P95/全局max 27.341/39.883秒。50次抢占、50个决策全部唯一关联，实测释放均等于当时held，释放页均值132.88、最大203。所有原始请求数据保留。这是一个tail观察运行，不是策略比较；与前一high吞吐相近不能据此估计完整观察开销。正常71.8379GiB/host16GiB及全部控制机制未变，funding仍0。

5807候选观察中host ready-prefix为零3939、非零1868；host与pending均无UNKNOWN。41/50次在某个“同held且同computed整页”层内有host差异，因而host并非单纯大小别名。但tail自身同held层有更低missing替代的次数为0；同held且同computed整页也为0。严格同容量规则不会改变本轨迹，不运行无动作臂。不限容量的min-missing影子改变22/50，其中15次多释放、7次少释放，held静态差额合计+448页；仅要求held≥tail则改变20次、差额+937页。以上是动作前影子持页数，不能写成已执行释放或同状态收益。

真实例子：step600，tail0054011与0068294同持184页、同183整计算页，ready前缀183/0、missing0/183，tail已选较易恢复者。step956，tail0071017持203页、missing202；候选0068250持210页、missing32。后者以多7页换取少170页missing，支持继续检验host信号，但保留容量混淆。恢复与完成损失尚未做反事实验证。

pending为0/1的观察5380/427，**5807/5807均等于int(computed_tokens % 16 == 0)**；tail47次为0、3次为1，同held且同computed整页从无pending差异。其元数据含义不是暴露DMA等待，不跑独立pending臂。原arrival影子仍0/50差异，与tail合并。

选择函数累计0.209095秒（测量窗口0.1921%），候选新增观察0.074093秒（0.0681%，包含在前者内，不能相加）；单次最大8.603/3.378毫秒。未覆盖完整序列化、free-pool观察或全部controller开销，不据此声称端到端零开销。

下一步只改变普通allocation-failure suffix的victim选择，继续固定high输入/正常预算/恢复目标/1秒触发/Q1/传输/准入。避免无约束min-missing把选择差异混同大幅容量差：先检查无阈值的最小额外释放形式（host missing严格改善、held至少tail，在其中选最接近tail容量者）。该形式是否值得执行由此探索数据决定；不是确认集预注册或已证明收益。若执行，将保存实际释放、全部请求损失和反序运行，新增信号不进入recovery prepare/protection。


### 最小额外容量 host 规则：已冻结并唯一候锁，尚未GPU测量

[candidate_native_host_near_r01](candidate_native_host_near_r01/)从探针复制，仅两处运行源码改变：普通suffix selector加入`host_near`，CLI/安装入口窄范围允许该规则。规则先要求missing严格低于tail且held≥tail，再按额外held最少、missing最少、原尾序排序。没有容差参数，不用历史成本/恢复次数，不改native释放/flush/再次allocate流程。既有qualified未知、任意host未知、或prepare/protection激活时回tail；funding victim始终tail。

在冻结前探索轨迹上重算20/50改变；额外held中位4.5页、范围1–17、总142页（相对tail中位+4.60%、最大+18.42%），missing减少中位41页、总997。20个选择均非全候选纯min/max-held，且已输出更多，可能造成完成损失转移。最少已生成输出规则与tail50/50重合，原arrival也50/50重合，所以只保留tail一个简单对照。不能将新规则称为CacheOPT/UniBoost复现，也不能把此静态差值当实测额外释放或收益。

15项相关CPU检查通过（既有host/native/lease测试中增加同容量优先、较小候选排除、未知/phase fallback、实际AST suffix弹出不撤销已排定prefix）；probe50事件用实际新函数重算为20动作。CPU检查不替代GPU实验。manifest `5226b31a2ed1faf2ff437272a1d58ce5d523c234516b47f3916842109440d2a2`，仍38文件；部署tar SHA `1dc7a781270f6e7478f2db3496d7fd6b424786c9cb9aad500092e27daeb6074e`。新[计划](plan-native-host-near-20261007-r01.json) SHA `479f159d9abf924b8876b3243643ce24463109b3814bf82a4823f9ef13d65da8`记录三控制器文件hash及原输入hash。旧包/结果不改。

顺序tail/host_near/host_near/tail，每格相同320 high、外部0.01秒、自然EOS/1024上限、同384宽预热和空私有编译缓存；正常GPU77135347712bytes/host16GiB、maxseq384/batch1024/4096context、原目标/1秒触发/Q1/准入/传输均固定。主要观察为全部请求外部到达计平均完成flow，两块1vs0与2vs3；保留完整吞吐/TTFT/flow/maxgap/失败未完成/输出/EOS差异。无应用SLO、不新挑goodput阈值；本组仍探索，不作运行级显著性推断。

数据盘余量低于4GiB，新session直接写已有系统盘/root/moe-a-victim-20261007（部署时15.05GB空闲），不改模型/源码/overlay/共同锁、不降低余量检查。新控制器只复用原控制器的env与receipt字段：清继承A_*并显式设定，receipt记录真实native rule；实验包四格相同。单格1200秒、整组4800秒，候锁3600秒单独计。

唯一controller53702/SSH exec29836已提交并实查为共同flock等待者；session尚未创建，无A CUDA初始化。前方49617持锁、51255等待，不重复排队。当前 **WAITING_SHARED_LOCK**，不是GPU结果。命令：

```sh
# 本地CPU检查
cd candidate_native_host_near_r01
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_funding_victim_cpu test_bidkv_full_running_cpu test_native_lease_cpu
# 远端本线源码根；此计划已提交，不得再执行同一session
cd /root/autodl-tmp/moe-a-victim-20261004
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -u run_native_host_group_wait.py --plan plan-native-host-near-20261007-r01.json
# 完成取回后，仅生成一次新分析
python3 -B analyze_native_group.py --session session-native-host-near-20261007-r01 --output analysis-native-host-near-20261007-r01.json
```


候锁期间已有数据解释（未改冻结组）：probe的raw只记录稀疏抢占，recomputed_tokens/recovery_count均null，residency_admissions没有实际命中/加载起点，故恢复时真实复用与重算量UNKNOWN。0030422在step661抢占前ready170/missing7，0.715秒后准入held178、再0.071秒新输出；同请求step680抢占前ready177/missing1，31.849秒后准入仅held6、再0.353秒新输出。0071017 step956 ready0/missing202，5.956秒后准入held9、再0.368秒新输出。单组full-attention owned计数排除了第二例已完整恢复177页前缀，但不能据此算LRU淘汰量、实际命中或重算页。准入到输出也含正常调度，不是隔离恢复时间。host信号在长排队中可能失效，是待验证代价线索，不是已知成本；无需因此修改正在候锁的探索组。

新分析入口analyze_native_group.py已实现并用既有probe验证接口：保留全请求指标、native决策→下一host输出绝对wait、受益候选/实际victim/其他角色的跨轨迹损失、实际free-pool释放、重复抢占、原始guard下新规则重算、四格配置与代码一致性。该CPU验证不是新GPU对照；正式只输出analysis-native-host-near-20261007-r01.json这一份分析。

14:04:49UTC实查controller53702仍存活，wchan=locks_lock_inode_wait，同一共同锁由49617持有、51255在前；A session尚未创建，无CUDA初始化、无第二A等待者。唯一SSH29836仍待原进程结束，不重启。上一goal轮有实质进展（探针证据、冻结补丁/部署、实际提交），当前阶段为已验证等待，目标ACTIVE。


2026-10-07续轮已有probe全量准入分解：50个victim均按内部ID与动作后第一个residency_admissions精确关联，准入output_count均与动作时相同。43个动作前ready-prefix>0，其中38个下次准入held小于该ready-prefix；只能排除完整装回当时前缀，不能反推出精确cache淘汰/hit/recompute量。decision→准入均值/中位20.278/20.600秒，准入→下一host输出均值/中位0.252/0.225秒；前者包含排队和原生load等待，后者也包含正常调度，不是隔离恢复耗时或copy瓶颈测量。此解释加入同一新组分析入口，冻结GPU包不变。


### 当前最薄弱环节与真实断点（2026-10-07追加准则后）

本轮仍不知道：真实改变普通victim后，较少host missing能否影响全部请求服务结果，还是被长等待期间的host状态变化吸收、并把损失转给更接近完成的请求。最小判别实验保持已冻结tail/host_near/host_near/tail，正常容量同320输入；不依据新观察改规则/阈值。若实际0动作，先核对guard/入口与当时合法机会，不铺重复矩阵；若改变且两次均有完整净收益，下一步做结构匹配容量消融并独立到达序列确认，排除“只是多释放容量”；若只改善局部指标或转移完成损失，停止本host规则近邻调参，按真实受损与等待分解决定同问题内下一机制；符号翻转先按运行级波动设有限复测，不能无限测到有利。

此次新规则仍 **GPU_UNRUN**。r01 controller53702等待1083.987秒后在14:08:00UTC取得共同锁，GPU边界记录原PID51255仍14464MiB；0.121秒内ABORTED、cells=[]，无A CUDA，锁已释放、SSH29836退出1。保留[完整r01失败目录](session-native-host-near-20261007-r01/)与[group log](group-native-host-near-20261007-r01.log)。14:11UTC随后实查GPU/锁为空，按新独立r02目录直接非阻塞提交；56960已接力持锁，因此flock返回EAGAIN，session未创建，保留[r02失败log](group-native-host-near-20261007-r02.log)。未重跑任何完成实验，也没有候卡或运行中的A任务。

r02仅变session路径与失败说明，科学配置、源码、顺序、输入和资源预算原样，计划SHA256 `37f130770976e89b46f49a54c1d334ef0f6ebe11a18f4186f5bfb5254d381141`。已部署，可在真实空卡时直接运行下列既有非阻塞入口；忙则退出，不再自动排队。r02尚未创建，仍可作为下一次真正启动的结果目录：

```sh
cd /root/autodl-tmp/moe-a-victim-20261004
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -u run_native_host_group.py --plan plan-native-host-near-20261007-r02.json
# 完成并取回本地之后；唯一分析输出，不覆盖r01失败
python3 -B analyze_native_group.py --session session-native-host-near-20261007-r02 --output analysis-native-host-near-20261007-r02.json
```

追加口径沿用：无可靠应用SLO，当前探索报告原始分布与完整损失；正式确认前冻结联合SLO阈值、分母及截止规则。既有capturer记录外部arrival、admission与engine_add_return，所有due请求进入native队列、不暂停源或设客户端inflight cap；分析补发送滞后及排空口径，不改变冻结测量包。没有消费遥测，共享总额度余额未知；计划4800秒是本组进程终止上限，不是费用或可用总额度预测。最终英文论文/独立确认/质量/关键消融等尚未完成，不标可投稿；方法失败不会自动变成测量论文。


分析器发送/排空补充已用原probe验证：320计划请求全部到达、提交、完成；提交滞后均值11.885ms、P95 23.927ms，engine.add_request均值0.389ms；外部最后到达后105.657281秒全部完成。拒绝与逐请求超时未被raw独立分类，保持UNKNOWN，不能从failed计数反推；capture runtime_limit单列。这是既有GPU数据新增解释，未生成第二份结果或改变GPU包。分析器现有唯一入口支持完整请求口径，下一真实组完成后再一次性写入其唯一分析。

14:22:37UTC资源实查：56960已退出，57657已接力持同一共同锁2304:15049831297、GPU88188MiB/90%，进程确认存活；A53702已消失、无A进程/候卡，r02结果目录仍未创建。原任务已终止，不重新启动它；当前只暂停GPU操作，没有新的策略执行或科学结果。相同资源阻塞延续，现成代码/命令及分析准备已完成。

14:24:25UTC第三个连续goal轮次核实同一共享GPU阻塞：57657处于running并持共同锁2304:15049831297，GPU88188MiB/93%；A53702消失、r02 session不存在，无A任务/候卡。实现、输入、分析及命令已完成准备，下一有信息的步骤需要真实GPU资源释放；目标已记BLOCKED，不是研究完成或机制负结果。

2026-10-07 14:36:19UTC恢复后第三个连续goal轮次实查：58138与GPU worker58293均存活，共同锁2304:15049831297由58138持有，GPU88230MiB/89%；r02未创建，A53702消失、无A进程/候卡。没有新科学数据或策略执行。代码/输入/分析与非阻塞命令均已准备，无需继续离线准备；下一有信息动作依赖整卡释放，目标再次BLOCKED，不是研究完成。

### r03：唯一接力提交，科学组仍冻结（2026-10-07 15:32UTC）

旧 r01 的真实失败是前一 CUDA 上下文晚于 flock 释放；r02 非阻塞入口又失去接力机会。此次只增加 [run_native_host_group_handoff.py](run_native_host_group_handoff.py)：一次共同锁候等上限3600秒，取得原锁后只在首次边界每5秒读GPU、最多等300秒，仍忙/未知则原样失败；不发信号、不初始化CUDA、不自动重试，300秒计入原4800秒组上限，后续边界不变。4条CPU模拟路径通过，不是GPU性能证据。新 [r03计划](plan-native-host-near-20261007-r03.json) SHA `7ddb442b839dcc07608a130e0c8aa8f43490a38e8cb9db73ba0fc48171727653`；与r02的科学配置、四格顺序、源码包、输入、KV和运行时预算逐项相同，只换独立session并记录交接入口hash。

15:32:22UTC实查唯一controller66669/SSH exec95726存活，共同锁2304:15049831297由66402持有，66669为flock WRITE*等待者，r03 session尚未创建、无A CUDA。提交前无其他A控制器，系统盘余量10089275392bytes；未覆盖r01/r02。当前是已实际提交的有限等待，不是GPU运行/策略负结果。以下命令**已经提交，不得重复启动**：

```sh
cd /root/autodl-tmp/moe-a-victim-20261004
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -u run_native_host_group_handoff.py --plan plan-native-host-near-20261007-r03.json
# 完成、释放锁并取回之后才生成唯一分析：
python3 -B analyze_native_group.py --session session-native-host-near-20261007-r03 --output analysis-native-host-near-20261007-r03.json
```

本轮问题与预定下一步沿用上节：检验真实host驱动动作的全部请求净收益；正信号先做容量结构匹配消融和独立到达，零动作先诊断门控，损失转移则停止本规则近邻搜索。候锁/交接处理不构成研究贡献。新目标文件已读取，与本线范围一致。

### r03真实结果：有执行、无净收益；发现测量期JIT（2026-10-08 00:05 CST）

[完整原始组](session-native-host-near-20261007-r03/)已完成、四格exit0/各末GPU EMPTY，controller66669消失、SSH95726退出0，共同锁在15:54:56UTC已交68589。组1179.3945秒（候锁189.909秒另计），首次上下文交接5.290秒。112个输出在远端和本地逐一SHA通过；原始归档99895290bytes，SHA `31eb12537e0fb783e5b941a4249ef7a64d7904aac6646e8ddae9d224d1dffd98`。唯一[分析JSON](analysis-native-host-near-20261007-r03.json)为FOUR_CELLS_COMPLETE_COMPARABLE，四格配置/源码/输入/预热/物理KV一致，197次决策全部与执行函数匹配。没有新GPU作业/候卡。

| arm | token/s | 外部到达计mean flow (s) | TTFT P50/P95 (s) | flow P95 (s) | 每请求maxgap P50/P95 (s) | 实际抢占/改选 |
|---|---:|---:|---:|---:|---:|---:|
| tail-first |2920.898|79.217|9.311/70.098|105.381|0.160/27.377|50/0|
| host-near-first |2904.939|79.847|9.313/70.416|105.979|0.702/28.069|49/34|
| host-near-reverse |2911.480|79.607|9.269/70.287|105.728|0.706/27.944|48/33|
| tail-reverse |2930.248|78.895|9.280/69.782|105.032|0.152/27.105|50/0|

每格320/320完成、failed/unfinished=0，全部外部到达与提交均320，输出均317800、309 length/11 stop；两对各26个输出序列变化，因此不宣称质量等价。最后到达后排空分别105.612/106.210/105.964/105.265秒，客户端提交滞后均值11.652/12.421/12.374/11.892ms，非引擎外隐藏排队。无可靠应用SLO，未新选goodput阈值；拒绝/逐请求超时未单独分类仍UNKNOWN。请求级原始分布和个体结果完整保留。

两次run-level对比mean flow **+0.630s(+0.796%)/+0.712s(+0.903%)**，实际吞吐 **−0.546%/−0.641%**，maxgap P95 **+0.692/+0.839s**。完成更早26/32请求、更晚294/288请求；maxgap更差245/287请求。只有两个探索配对，不作显著性声明。两个tail mean flow差0.322s、两个host差0.240s，符号未翻转但并非独立确认集。

真实动作34/49、33/48全部唯一接到原生preempt/free-pool增量，selected held等于实际释放；不存在覆盖、取消或phase保护阻断。每个host运行1次unknown回tail、14次无合格改善回tail。least-generated仍与tail在197/197当前状态建议相同，无需另跑同动作臂。funding仍0，目标/触发/Q1/准入/传输均未变。两种策略每次仍有0070086、0030422各两次抢占，再次抢占未改善。

**容量接近假设没有沿真实轨迹保持。** 改选时selected−tail held中位52.5/59页、最大97/99，合计1779/1810页；此前单tail影子的中位4.5页不代表闭环行为。实测全运行总释放为6644/6648/6584/6644页；逐动作差额不是运行净增释放，更不是测得的另一动作反事实。34/33次改选中33/32次绕开同一个tail0058251，selected的输出进度中位多69/66tokens，missing中位少55.5/47页。

具体损失转移：0058251由抢占1次变0次，flow早11.683/11.550s；0037640由0次变1次，flow晚13.787/13.842s，maxgap约0.15→29.6s。后者在两host轨迹step712选择时与tail同持80页、ready59/missing20，实际释放80页。这些请求均1024输出；跨轨迹差异不是严格同状态因果效应。host臂42/49、44/48个正ready victim下次准入held少于之前ready；改选子集30/34、31/33，实际hit/recompute仍UNKNOWN，不能把当前ready视为持久可复用保证。

selector累计0.197/0.195/0.171/0.186秒（窗口0.157–0.181%），其中候选观察0.063/0.063/0.054/0.060秒，包含关系不相加；这部分未显示新增策略CPU成本足以解释共同停顿，也不是完整开销计量。

**新发现改变下一步：** 两host的launch.log均在PHASE MEASUREMENT_BEGIN后出现一次fused_moe_kernel JIT，两个tail均无。两host最长相邻输出返回间隔0.701780/0.706410秒，分别落于15:43:24.951–25.652UTC、15:48:14.807–15.513UTC，与JIT警告同一秒；末端有217/216个请求输出，起点308/307请求尚未完成。tail最长仅0.159537/0.151541秒。未记录编译精确起止，不能直接把0.70秒当编译时长或从flow扣除。当前结果是统一冷私有编译缓存与既有384宽预热下的真实总损失，保留不修饰；但尚不能全归因于host代价判断。

下一最小诊断只改变**所有臂相同的Triton预编译缓存初始状态**，源码包、victim规则、到达、预热请求和其余控制器仍冻结；先核实可安全使用已完成探索组的公共kernel集合，每格私有相同副本，绝不只给candidate暖缓存。若测量JIT消失且仍无净收益，停止host-near近邻调参，根据稳定损失转移研究同问题内下一机制；若收益仅来自消除JIT，不归入victim创新；只有剩余净收益才推进结构匹配消融/独立负载。若JIT未消除，本次只能判缓存诊断未成功，不添加近邻阈值。当前此诊断尚未实现/提交，A无新候卡。

[图PDF](figure-native-host-near-20261007-r03.pdf)由唯一分析自动生成，含四运行ECDF、完整分母和两对逐请求差异；不汇集请求置信区间。复现分析/图：
```sh
python3 -B analyze_native_group.py --session session-native-host-near-20261007-r03 --output analysis-native-host-near-20261007-r03.json
python3 -B plot_native_group.py --analysis analysis-native-host-near-20261007-r03.json --output figure-native-host-near-20261007-r03
```
输出已存在，入口拒绝覆盖；新环境使用归档和同名未存在输出。首次本地解包命令因Python3.9缺hashlib.file_digest失败于解包前，改用分块SHA后成功，未改任何实验文件；不是GPU失败。英文投稿证据尚不完整，未标可投稿或目标完成。

当前未知：host-near负结果中，测量期JIT与持续victim损失各承担多少；现有记录不能给出编译精确时间。
主要竞争解释：缺少两个fused_moe编译key造成共同停顿；或绕开同一tail导致的损失转移在热缓存下仍无净收益。
最小实验：一次相同tail/host_near/host_near/tail，每臂320；四臂均由同一冻结Triton并集种子复制到独立私有缓存，其余源码/到达/预算/预热原样。
不同结果的下一步：JIT消失仍无净收益则停止本host-near规则的近邻修正；仅启动差异消失不作victim创新；剩余稳定净收益才做容量匹配消融与独立确认；JIT仍出现则该缓存诊断未解决问题，不继续盲目扩展。

统一缓存诊断已执行：计划 [plan-native-host-near-seeded-20261008-r01.json](plan-native-host-near-seeded-20261008-r01.json) SHA `21ccc410bb3523b2f2fe9d191007f14b2b535ac87fbf3cb14bd548607e7cd051`。仅缓存初始状态改变；原运行时manifest5226b31a…40d2a2、同ABBA每格320与全部机制不变。新入口[run_native_host_group_seeded.py](run_native_host_group_seeded.py)复用原共享锁/交接/预算；base仅把固定EMPTY_PRIVATE收据改为准确的TRITON_SEEDED_PRIVATE_OTHER_CACHES_EMPTY。种子279文件/41key/13937676bytes，规范摘要a48b8786…093df，39完整key来自已完成tail-first，2个fused_moe key来自host-near-first；238个group child路径按私有目录重定位，二进制/extern_libs原样。源码与seed摘要、路径重定位CPU夹具通过，无新manifest体系。

16:37:18UTC实查唯一controller73861/SSH95639仍存活并持同一共同锁2304:15049831297；候锁101.649秒、首次空卡确认0.144秒，前三格exit0，最后tail运行中，各收据缓存状态正确。以下命令已提交，不得重复启动；整组结束后释放才取回。正式分析将报告测量阶段JIT警告，原cold-r03分析源码已按其原SHA保存在原session，不覆盖原结果。
```sh
cd /root/autodl-tmp/moe-a-victim-20261004
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -u run_native_host_group_seeded.py --plan plan-native-host-near-seeded-20261008-r01.json
# 完成取回后：
python3 -B analyze_native_group.py --session session-native-host-near-seeded-20261008-r01 --output analysis-native-host-near-seeded-20261008-r01.json
```

### 统一Triton种子 r01：共同长停顿消失，服务效果翻转（2026-10-08）

[seeded-r01原始](session-native-host-near-seeded-20261008-r01/)四格全部完成/exit0，controller73861与SSH95639已终止，00:41CST释放原共同锁；组1118.1104秒。112原始输出双侧SHA通过，归档102646975bytes/SHA `4f276d2aad4298409e929c2d29f93c5e795503f9c3467ebea563e1c541406b36`；归档含同一279文件种子，规范摘要a48b8786…093df，本地已验证。唯一[分析](analysis-native-host-near-seeded-20261008-r01.json)与[图PDF](figure-native-host-near-seeded-20261008-r01.pdf)保留，分析源码按执行版保存在session中；配置、原运行时源码、种子和实际选择均匹配。

| arm | token/s | mean flow (s) | TTFT P50/P95 (s) | flow P95 (s) | maxgap P50/P95 (s) | 抢占/改选 |
|---|---:|---:|---:|---:|---:|---:|
| tail-first |2929.952|78.967|9.265/69.887|105.046|.158/27.330|50/0|
| host-near-first |2921.213|79.285|9.330/69.851|105.373|.159/27.396|49/34|
| host-near-reverse |2930.239|78.901|9.258/69.555|105.035|.156/27.208|47/32|
| tail-reverse |2921.284|79.194|9.266/70.066|105.367|.152/27.381|50/0|

每格仍320/320到达/提交/完成、failed/unfinished0、317800tokens、309 length/11stop；序列变化24/29请求，不称质量等价。排空105.276/105.600/105.265/105.598秒；发送滞后均值12.198/11.778/12.011/11.254ms。正序host mean flow **+0.317870s(+0.403%)**、吞吐−0.298%；反序 **−0.292804s(−0.370%)**、吞吐+0.307%。完成更晚296/28、更早24/292；maxgap更差270/183、更好50/137。两对mean flow差平均+0.012533s，范围−0.292804到+0.317870，不能称稳定净收益或无效证明，事件/请求不是重复。

所有196个决策与真实preempt唯一关联、实测释放等于selected held；funding0，least-generated与tail仍196/196同建议；两个重复抢占请求仍相同。候选34/32次改选分别33/31次绕开0058251，额外held中位52.5/60页、最大96/100，容量接近的离线预想仍未保持。0058251 flow早11.937/12.679秒；0037640晚13.448/12.769秒，损失转移在两块保留。

两host首次改选均step692：0060608 held164/computed2616/output136/ready154/missing9替代tail0054922 held148/computed2368/output131/ready138/missing10，实测free0→164；下次准入held仅2/44页，等待30.436/30.380秒，再次输出距动作30.798/30.663秒。动作前14次抢占step+identity与各自tail相同，step<692的108198条输出身份/ordinal/token一致，但截至首动作1493条候选有24条host状态不同；不是严格同状态实验，也不以未来准入状态定义在线规则。

原先将“没有JIT警告”作为缓存诊断成功条件过严：安装版本vLLM `utils/jit_monitor.py` 的warning_once挂接Triton `jit_post_compile_hook`；Triton3.6 `runtime/jit.py:849–852` 在compile()返回后无条件调用hook，而 `compiler/compiler.py:265–276` 在磁盘命中时同样返回CompiledKernel。故磁盘命中首次载入仍可有相同警告。现场检查四格均41key、无新增key，两个额外fused_moe key的文件均无安装记录之后的重写；本次每host仍1警告，但原共同0.70s停顿已消失，maxgap中位回到约.16s。不能据警告断言重新编译，也不把观察到的跨组改善当精确编译时长因果扣除。原plan警告条件及原结果保留，此处解释测量限制，不再改seed追加“直到零警告”的实验。

当前未知：相同种子下约±0.30秒的mean flow翻转是运行波动，还是足以重复的小收益/损失。
主要竞争解释：没有稳健净收益、只有个体损失转移；或真实小效应被运行级变化遮蔽。
最小实验：最多追加一组原样seeded ABBA（新增2配对，累计4；每格320），源码/种子/规则/到达/预热/计时全冻结，不增加阈值。
不同结果的下一步：若追加仍翻转或无一致服务优势，收束本host-near＋本运行域的尝试，不再重复到显著；若两新配对同向净改善且全4配对支持继续投入，再用有限局部干预区分首次选择与随后持续保留的作用，仍不称独立确认/投稿主结果。

限定重复 r02 已单次提交：plan SHA `e67d9f80c4969a3c69f1d420906d127a331ea5b42915cbf39294da9c1c236963`，仅换独立session与有限重复说明，源码/种子/ABBA/输入/预算均未改。第一次启动前检查因3153420288bytes余量不足退出，runner未exec、未创建session/log；[失败原文](group-native-host-near-seeded-20261008-r02-prelaunch-reserve-failure.log)保留。只将A两个已完成组224对output/archive在双侧SHA匹配后硬链接，路径/原始内容全保留；删除536259996bytes可再生私有非Triton缓存，所有Triton/source/model/其他会话保留，余量恢复5608009728bytes，未降低4GiB门槛。

17:16:05UTC核实唯一controller82522/SSH64331存活，为原共同锁2304:15049831297的WRITE*等待者，80836持锁，r02尚未创建、无A CUDA/第二任务。此为有上限的最后一组原样重复，最多新增2配对，禁止再排同类组测到显著。命令已提交：
```sh
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -u run_native_host_group_seeded.py --plan plan-native-host-near-seeded-20261008-r02.json
# 整组结束释放并取回后：
python3 -B analyze_native_group.py --session session-native-host-near-seeded-20261008-r02 --output analysis-native-host-near-seeded-20261008-r02.json
```

17:25:59UTC实查r02唯一82522已RUNNING首tail；17:24:14.736UTC取得原锁，候锁522.927s/交接5.292s，空卡边界True。此前候锁描述是历史，无第二A作业，仍严格最多这一个追加ABBA。


## 2026-10-08 限定 seeded-r02：完成并收束 host-near

**新增证据。** [原始 session](session-native-host-near-seeded-20261008-r02/) 四格退出0，整组1126.869秒；112个原始输出在两侧SHA一致。唯一[分析](analysis-native-host-near-seeded-20261008-r02.json)与[分布图](figure-native-host-near-seeded-20261008-r02.pdf)由既有脚本生成，源副本保存在session。tar SHA `b81f43fa05b2cd775a681dedd5c9687e59bb765170fba4ddc632dc7f804b7096`。代码manifest `5226b31a…40d2a2`、plan `e67d9f80…236963`、种子 `a48b8786…093df`、正常GPU KV71.8379GiB/host16GiB和原输入/预热全冻结且组内一致。

| 顺序 | token/s | mean flow s | TTFT P50/P95 s | flow P95 s | 每请求max-gap P50/P95 s | 最后到达后排空 s |
|---|---:|---:|---:|---:|---:|---:|
| tail first | 2933.052 | 78.861 | 9.258 / 69.757 | 104.931 | .159 / 27.185 | 105.161 |
| host first | 2926.821 | 79.039 | 9.293 / 69.696 | 105.161 | .158 / 27.267 | 105.392 |
| host reverse | 2913.378 | 79.422 | 9.331 / 70.043 | 105.657 | .155 / 27.498 | 105.893 |
| tail reverse | 2922.804 | 79.122 | 9.266 / 70.010 | 105.310 | .159 / 27.212 | 105.541 |

全部4×320到达/完成，失败、拒绝、超时、未完成均0；每格317800输出，309 length/11 stop，逐请求完整分布留于分析。两对输出序列差29/26，不能宣称质量等价；没有应用SLO，不挑goodput阈值。发送滞后均值11.616/11.596/11.952/11.989ms已计入外部到达计时。

**执行变化。** native抢占50/47/48/50，host实际改选32/33；195/195唯一匹配真实preempt，实测释放均等于held，释放总页6644/6540/6584/6644。funding动作0、least-generated影子与tail全部重合。host分别31/32次绕开同一个0058251，额外held中位60/59、最大110/99页。首次仍step692、164页替代148页，后续准入仅3/17页且约30.9秒后才有输出，实际host命中UNKNOWN。再次抢占仍0070086/0030422各2次，未改善。选择器总时196.389/183.100/184.956/222.124ms（包含候选观察，不相加），约测量时长0.169%–0.204%，非完整额外成本。JIT警告0/0/1/0；它不是实际重新编译次数，先前约.70秒共同停顿未复现。

**完整收益与代价。** 新两对mean flow +.177283/+.300305秒（+.225%/+.380%），rate−.212%/−.322%；33/23请求更早、287/297更晚，总flow +56.731/+96.098 request-seconds。0058251提前12.107/12.007秒，新增victim0037640延后13.297/13.464秒、max-gap约29秒；均1024输出。减少2–3次抢占没有转成服务净收益。

统一缓存四配对的flow差为 +.317870、−.292804、+.177283、+.300305秒，均值+.125663、样本SD.285904、范围[−.292804,+.317870]；吞吐差−.298%、+.307%、−.212%、−.322%。统计单位为运行配对，只有两组ABBA，未给事件级显著性；冷缓存组单独保留。按预定上限，**封存host_near＋当前正常容量burst域的收益主张**，不再原样重复或做host阈值/连续保护近邻调参。不能据此否定整个victim选择问题，也不能把小幅平均损失断言为普遍规律。

复现原组命令沿用上一节（已有session禁止覆盖）；本地重分析须指定新的输出名：
```sh
python3 -B analyze_native_group.py --session session-native-host-near-seeded-20261008-r02 --output NEW-analysis.json
python3 -B plot_native_group.py --analysis NEW-analysis.json --output NEW-figure
```

当前未知：固定实际释放容量时，改变一次合法victim身份是否具有完整服务动作价值；host-near的容量变化与持续反馈尚未回答它。
主要竞争解释：选择身份有可利用影响，或本burst域的tail/最少已输出规则已足够；持续host保护的负结果不能区分二者。
最小实验：同输入、同种子、同原生路径，在首次同held且同computed整页数的合法机会，仅把tail换成同层中index最近的候选一次，其余全tail；tail/probe/probe/tail，最多这一组，真实执行与释放容量逐项确认。
不同结果将改变：若净改善且动作真实，才投入独立前缀/负载解释；若无改善/仅转移损失/接近噪声，则结束这一单burst动作探针，不增加预测器或原样重复，下一投入转向有新机会证据的运行域。

现有r02两tail各19次同held机会，第一次step600有4个替代：tail0054011 held184/computed2932/output13，最近候选0035449 held184/computed2931/output73。规则不使用host或未来EOS、不按结果选请求；这只是动作前合法机会，不是干预结果。独立包`candidate_native_equal_held_once_r01`已实现/冻结/部署，尚未GPU运行。


容量中性探针版本：manifest `33d0b83c567a65f5274bd74d980f0a35be1b22516961f48fadd91780842b1d17`；staged `655beaa7…8ce59a`；plan `c5249676643d3711d9ce0318e03eb3a55ec4150c8d98a4734432459ca474ced9`。仅新包picker及runner允许列表改变；4个针对性CPU helper检查通过、原tail两轨迹各只产生一次影子proposal。策略安装在预热/drain/reset之后，故预热不消耗唯一动作；baseline与candidate均记录proposal/requested/consumed，真实执行仍由原preempt/free确认。新的薄入口复用原shared-lock/种子/交接，四臂环境固定检查通过，未改KV/生命周期。当前为诊断探针，不是有收益的候选论文机制。

18:07:26UTC已单次提交并实查唯一controller90592/SSH62598为共同锁2304:15049831297的等待者，90323持锁，session未创建；A无CUDA/第二任务。最多等3600秒、整组原4800秒（含首次交接）上限，仅此ABBA，不重复候卡或自动重启。启动命令已在执行，勿再提交：
```sh
cd /root/autodl-tmp/moe-a-victim-20261004
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -u run_native_equal_held_group_seeded.py --plan plan-native-equal-held-once-20261008-r01.json
# 终态、释放和取回后：
python3 -B analyze_native_equal_held_group.py --session session-native-equal-held-once-20261008-r01 --output analysis-native-equal-held-once-20261008-r01.json
python3 -B plot_native_equal_held_group.py --analysis analysis-native-equal-held-once-20261008-r01.json --output figure-native-equal-held-once-20261008-r01
```

磁盘断点已实际处理：共享盘一度1.77GB，未降低4GiB入口。低优先级校验后，仅回收三个A已完成组cold-r03/seeded-r01/r02的48个raw.json/warmup-1.json远端展开副本；每组完整远端tar及本地raw+tar都保留，原始证据未丢弃。r02其余112对output/archive硬链接、268047758B私有非Triton缓存回收；所有Triton种子/source/model及他线不动，部署后余量5460549632B。远端旧展开目录中两类大raw现在需从对应原tar解出；本地分析路径完整。无消费遥测，未知共享余额，4800秒是进程上限而非成本估计。

18:10:01UTC再实查90592存活、wchan=locks_lock_inode_wait，90323持同一锁；新session仍不存在，系统盘5396856832B，没有新GPU数据或第二A任务。薄分析`analyze_native_equal_held_group.py`（SHA `fac8716e…10f9464`）已完成，取真实profile页大小、按SHA匹配helper逐条重算状态；实际执行另用preempt/free join。CPU夹具接受一次提案，拒绝重复动作/错误提案/缺页大小/错源码/shadow执行。原base只增加kind参数和页大小传参，旧三组分析脚本副本及原始JSON未重写。下一步接续唯一SSH62598，不重新提交。

当前单次干预的具体可信性风险：相同输入/规则前缀仍可能因异步host状态或调度而分化。已只在本地薄分析入口补pre_intervention_prefix，比较首次proposal前的输出token/调用/真实victim序列，并逐候选报告包含动作位置的状态差异与测量相对时间；不改GPU包、不另跑检查组、不声称隐藏状态相同。CPU夹具能区分host变化、仅时钟偏移、输出token改变、缺失映射和零proposal；它们不是新GPU结果。

18:17:46UTC：唯一90592已持原共同锁，receipt RUNNING首tail；取得锁时间1791397019.184（候锁587.069s），首次上下文交接5.246s后确认空卡才初始化。下方/此前等待状态为历史；不启动第二任务，组内源码/配置不改。


## 2026-10-08 同容量单次替换：局部延缓被吸收，损失仍保留

**新增证据。** [原始session](session-native-equal-held-once-20261008-r01/) 全部exit0/VERIFIED；整组1136.190秒，112原始输出两侧SHA通过。唯一[完整分析](analysis-native-equal-held-once-20261008-r01.json)、[分布图](figure-native-equal-held-once-20261008-r01.pdf)已生成，分析/绘图源码保存在session。tar SHA `8c2d878acb0e9f5dda64c6f02d278e3700a3a85787d96a2bc4cf2c3e8cbd26b5`；manifest/plan/种子与上述冻结版本一致，正常GPU KV71.8379GiB、host16GiB、320相同外部请求及预热全部可比。

| 顺序 | token/s | mean flow s | TTFT P50/P95 s | flow P95 s | 每请求max-gap P95 s | 最后到达后排空 s |
|---|---:|---:|---:|---:|---:|---:|
| tail first | 2907.947 | 79.620 | 9.562 / 70.321 | 105.866 | 27.392 | 106.097 |
| once first | 2894.549 | 80.081 | 9.721 / 71.011 | 106.366 | 26.961 | 106.602 |
| once reverse | 2922.428 | 79.143 | 9.341 / 70.021 | 105.319 | 26.919 | 105.555 |
| tail reverse | 2916.674 | 79.250 | 9.282 / 70.131 | 105.537 | 27.178 | 105.770 |

全部4×320到达/完成，失败/拒绝/超时/未完成均0；每格317800输出、309 length/11 stop，输出序列各33请求不同，质量等价未验证。没有应用SLO，不从诊断frontier选有利goodput阈值。发送滞后均值12.184/13.589/11.767/12.236ms，外部等待完整计入。JIT警告0/1/1/0，不能据此当实际重新编译次数。选择器总时208.854/226.139/206.670/205.112ms（含候选观察），不等于完整额外成本。

**实际动作与前缀。** 两候选各唯一请求并执行一次替换：step600，0035449取代0054011；原tail与替代均held184、computed整页183，基线与候选该次实测均free0→184。198/198全组抢占唯一对应选择且释放等于held，funding0。动作前两次抢占和84564条输出的call/request/token序列逐条一致；226候选状态中各2条host状态不同（0043610、0058251），动作时间候选已比配对tail晚.265/.100秒。首动作的原tail和所选victim上述状态一致，但这仍非严格同状态因果实验。

**谁获益/受损及完整代价。** 原tail在下一抢占step611就被选，输出13→24，仅延缓.793/.788秒，最终flow差+.139/−.568秒。0035449原本step811、284输出才被抢占，现在step600、73输出即暂停；约39.846/39.531秒后才有新输出，下一准入held26低于动作前ready177，实际host命中仍UNKNOWN。它晚完成8.012/7.442秒，max-gap增加20.121/19.745秒，均仍1024输出；这不是新增victim身份，而是把其原有抢占提前。最大赢家0042818早完成.456/1.165秒；分配失败请求0029338反而晚完成.528/.155秒。唯一victim集合仍相同48个，次数50→49；重复者由0070086/0030422变为0028275。从step824起最后21个victim的step、身份、held/computed/output恢复一致，但未证明完整缓存状态合流。

整体mean flow差+.461564/−.106976秒；9/265请求更早、311/55更晚，总flow +147.700/−34.232 request-seconds；rate−.461%/+.197%。max-gap P95虽下降.431/.259秒，整体完成收益仍翻转且有明确个体损失，不能事后切换主指标宣布成功。按预定最多这一ABBA收束，不调单次动作位置或追加同规则重复。结论是此动作的保护仅短暂、抢占顺序的损失能持续；既非动作零执行，也非“完全没有后续影响”，更不否定所有victim选择。

新的机会依据：四格已知tail剩余输出预算均在475–1016，动作处还余1011，远多于本次取得的11个输出步；均匀1024上限下，tail也一直等同最少已输出/最多剩余预算。这里缺少“短暂避开抢占即可完成并释放KV”的机会，不能靠继续调当前host信号补出它。

当前未知：引入预先指定的长短生成预算后，正常容量下是否出现临近预算结束的合法tail，普通最大剩余预算规则是否仍与tail同动作。
主要竞争解释：当前机会缺失源于统一长输出预算，或混合预算下短请求已在容量压力前完成、同样没有可利用机会。
最小实验：只跑一格tail，保持原320 prompts/到达/.01s/正常GPU与host预算/预热；按固定request-ID哈希排序选40个上限128，其余280仍1024，naturalEOS允许，不参考任何已有请求结果选这40个。
不同结果将改变：有不同合法动作且临近完成的机会，才做新的有界真实动作探针并面对最大剩余预算强简单规则；无压力/无机会则保留该正常容量负结果，停止此混合点，不扫描短预算或重启host-near。

这是一组**合成生成预算混合的开发性机会诊断**，不是应用质量/独立确认或相对旧工作量的性能收益。新包candidate_native_mixed_budget_probe_r01冻结；除保留并验证已有per-request budget输入外不改变运行时机制。40个短预算按SHA256("A-budget-mixture-v1:"+request_id)排序取前40，未参考输出/抢占结果；prompt444–3066、中位1855.5，五个64请求到达阶段分布8/7/10/8/7。两项针对性CPU检查及38项payload校验通过，不是GPU结果。

冻结计划 [plan-native-mixed-budget-probe-20261008-r01.json](plan-native-mixed-budget-probe-20261008-r01.json) SHA `e36389ad…b9f6492`，包manifest `8e6d3dcf…b8da94`、runner `c599834a…5d34e740`，config `5adcad20…808edb5`；原prompt/到达workload SHA `f5ce97da…7ddcb9`不变。沿用原控制器/共同锁/种子，不新建基础设施；kind沿用入口标签，实际experiment_role为MIXED_BUDGET_NATIVE_TAIL_CHARACTERIZATION，唯一cell为tail。

```sh
cd /root/autodl-tmp/moe-a-victim-20261004
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
/root/miniconda3/bin/python -u -B run_native_equal_held_group_seeded.py --plan plan-native-mixed-budget-probe-20261008-r01.json
```

2026-10-07 19:09:23UTC实查唯一controller99819/SSH79451存活，为共同flock2304:15049831297的WRITE*等待者；C98403持有，无A CUDA/session、无第二候卡。最多候锁3600s，取得锁后整组1800s/单格1200s，初始上下文交接最多300s计入组上限；无自动重试。没有消费遥测，过程上限不等于共享费用余额。

磁盘处理：仅清理已终态equal-held-once-r01的16个远端raw/warmup-1重复展开副本（1,865,809,666B）和268,048,521B可再生非Triton缓存；本地原始/本地tar/远端tar逐一核对后才删除，完整tar SHA `8c2d878a…cbd26b5`与全部Triton/模型/他线数据保留。根盘恢复4,440,940,544B，未降低4GiB启动余量。旧远端raw展开路径可由tar恢复，本地原始仍在。

## 混合预算单格结果：有普通排序差异，无既定的临近完成tail机会

[原始数据](session-native-mixed-budget-probe-20261008-r01/) / [唯一分析](analysis-native-mixed-budget-probe-20261008-r01.json) / [分析入口](analyze_native_mixed_budget_probe.py)。本地28个archive文件与远端output/archive全部SHA相符；tar23,902,673B、SHA `69c4e12ced0367a5de33e1a0eef16a2f4ef63fb6ab89b93c9a7c20d9502048d3`。分析器 `8e069e5c…e639619`与依赖源码已存session。预算逐ID核验VERIFIED，40×128/280×1024进入真实性能请求，原三段预热32×16/384×93/2×16不变。实验控制器终态CELLS_COMPLETE，单格267.112s/整组285.130s；19:18:20UTC空卡后释放锁，下载分析均在锁外。

```sh
python3 -B analyze_native_mixed_budget_probe.py --session session-native-mixed-budget-probe-20261008-r01 --output NEW-analysis.json
```

**完整服务结果（仅描述此工作量，无策略对比）：** 320到达/提交/完成，失败与未完成0，无显式拒绝/超时记录；全部请求排空，未触发600s观察截止。282868输出、309 length/11 stop；cap128生成不是应用质量任务。测量98.138s，2882.358token/s、3.261req/s；mean flow68.130s，TTFT P50/P95=9.032/62.729s，flow P50/P95=74.502/93.594s，每请求max-gap P50/P95/max=.158/20.217/32.359s；最后外部到达后排空94.948s，发送滞后均值12.214ms/P9524.231ms。不同于旧统一预算的输出工作量，不计算相对其“加速”或goodput收益；没有应用SLO。

**真实动作与合法空间：** 40次原生tail实际执行全部MATCH、实际free增量等于held，39个不同victim，0065237抢占两次；funding仍0。40次均可解析，39次不止一个合法suffix候选，同held+computed整页候选机会21次。实际策略改选0是该tail表征的设计，不能归因任何时延变化给影子规则。在线选择/候选记录总时191.402ms（含观察，约测量.195%，非独立新增策略开销）；测量期JIT警告1，不能当真实重编译次数。

**新证据及受损请求：** 第一次实际抢占step664、31.723s前，40短请求已完成32（31 length/1 stop）。当时余2/11token的0024322、0036565未被原tail选中，随后step665/674、约.145/.798s完成至128上限，始终未抢占；这些机会已由简单tail自然保留。实际短tail0049936在step706余66token、释放100页，0052099在step712余58token、释放136页，各抢占一次；恢复输出step1118/1116，max-gap30.003/29.420s，完成flow66.601/65.935s。它们确实停顿很久，但不能把全部停顿计成一次改选可消除的代价。

最大剩余**已声明预算**影子不同3次：step681为重准入的0065237(held84、remaining996)改选0053843(held150、remaining1011)，多释放候选66页；step706/712短tail对应候选分别少33/69页。同held+computed整页版本在后两处也可改选，remaining增加424/837，但没有tail remaining<=16机会（全40次范围58–1009）。suffix任意候选<=16仅第一次的两例，原tail已保留。单页仅描述机会，不是SLO或事后调出的选择阈值。

原始到达排序与least-produced影子均只在step681不同；恢复后回running尾部确实出现新的排序偏离，但它同时涉及多66页释放。这个事实可提出后续问题，不能当作恢复感知策略的收益。影子都不是另一策略的轨迹。

**决定：** 按四行预声明收束这个混合突发点，不把16改成64来宣布机会、不调短预算/比例、不复活host-near或追加本点性能矩阵。当前结果支持“多数短请求在容量压力前已完成，真正临近结束者已被tail保留”；不否定分散到达/不同执行阶段中的victim选择。下一项有判别力的工作只改变到达时间分布、沿用既有0.2s节拍做一格正常容量机会诊断，检验持续进入的长短请求是否打破这一时序；保持这40/280预算分配和全部运行时规则。若压力或新机会仍缺失，保留该运行域负结果，不扫描到达率；若出现合法临近完成tail，再做有界真实干预。此后续尚未构建/提交，当前无A GPU进程。没有可信主结果，因此尚未进入独立确认或论文可投稿阶段。

当前未知：把原0.01s突发改成既有0.2s外部节拍，短预算请求是否仍先于容量压力完成，tail是否仍保留全部临近完成机会。
主要竞争解释：缺机会由突发内的年龄排序和短请求提前结束造成；或在持续到达下仍由正常容量/原tail吸收，无需新增选择器。
最小实验：只一格原生tail，320相同IDs/prompts、固定40×128+280×1024、相同正常KV/host/预热/seed；唯一主要因素为外部arrival=i×0.2s（总63.8s），不改准入/恢复/量子。
不同结果将改变：有压力且出现临近完成tail的合法替代，才设计有界实际干预；无压力/无机会则停止这个时间分布分支，不扫描相邻到达率、不继续调预算或信号。

### 分散到达诊断已冻结并单次提交

新包 [candidate_native_mixed_budget_spread_r01](candidate_native_mixed_budget_spread_r01/) 只改pro_high的3个JSON：workload实际steady到达数组/说明，config gap/span/内容hash，stats span/hash；全部运行时代码、IDs/prompt、预算分配及预热逐字节不变。`build_spread_arrivals.py --check`可重建；原measure_episode的CPU假引擎检查确认320个arrival_s及engine.add_request绝对到达时间均使用该数组，不是GPU证据。

manifest `39bf38a2…715fa0`，计划 [plan-native-mixed-budget-spread-20261008-r01.json](plan-native-mixed-budget-spread-20261008-r01.json) SHA `e5b66fa7b46f2f467ff07855353f3c6128f0206978c6b1ae6609d77e38deeb66`；workload文件 `7d40e7ce…b9f3b78`，config `19275e83…51f1919`，tar `3c8a0e81…d3ce43`。沿用原控制器/正常容量/种子/整组1800s与单格1200s；kind和label为复用入口，arrival_diagnostic显式标明SAME_MIXED_BUDGET_SPREAD_0_2_SECONDS。薄分析仅新增逐ID真实外部arrival核对：旧.01s输入会被拒绝；不重复验证未变化的运行时。

```sh
cd /root/autodl-tmp/moe-a-victim-20261004
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
/root/miniconda3/bin/python -u -B run_native_equal_held_group_seeded.py --plan plan-native-mixed-budget-spread-20261008-r01.json
# 下载完成后，在本线目录分析；输出必须是新路径
python3 -B analyze_native_mixed_budget_spread.py --session session-native-mixed-budget-spread-20261008-r01 --output NEW-analysis.json
```

2026-10-07 19:40:20UTC实查唯一controller102853/SSH69228存活、共同flock2304:15049831297 WRITE*等待，102264持有；session未创建，无A CUDA/第二候卡。候锁最多3600s，取锁后仍重新检查空卡和4GiB可用盘，不降阈值、不自动重试。尚无本轮科学结果。

磁盘断点处理只涉及本线已终态副本：上轮mixed-probe的28个output/archive对经SHA匹配后硬链接，回收67,012,473B非Triton缓存；之后raw/warmup-1共4个展开路径移除，唯一inode218,194,460B，本地完整raw+tar和远端tar仍在。四个已完成ABBA组cold-r03/seeded-r01/r02/equal-held-once各4份执行包逐文件与各自完整远端tar比对后移除，共492,210,476B；本地展开执行包、远端完整tar和独立冻结源全部保留，所有Triton原件不动。旧远端展开package路径需从tar恢复，不改变原始结果。入场提交时根盘4,352,368,640B；共享他线后续写入仍可能阻塞入场。无消费遥测，不声称共享余额。

r01真实终态：102853候锁29.534s后取得锁，5.296s交接确认空卡，但19:40:43UTC根盘低于原4GiB要求，`ABORTED / ValueError: Insufficient new-host disk space / cells=[]`；无CUDA、无科学测量，SSH69228 exit1、PID消失。原始[失败目录](session-native-mixed-budget-spread-20261008-r01/)连同controller.log已下载，旧运行目录不复用。

资源已具体改变：将四个已完成ABBA的完整压缩tar（共402,247,074B）SHA核对后移到同机A数据盘`/root/autodl-tmp/moe-a-victim-20261004/completed-archives/`，原root路径以原子替换的符号链接保留，全部内容与本地副本不变。根盘恢复4,750,159,872B、数据盘1,362,993,152B，不删除原始证据或调整实验容量。基于该资源恢复才提交独立r02；[r02计划](plan-native-mixed-budget-spread-20261008-r02.json) SHA `b08cc1ccd67fd884d81349fcbc9d5ba89b40f5bf07bb4a01cd86866026731291`，仅session_dir和资源重入说明变化，其余冻结代码/输入/控制不变。复现命令中的plan换为r02，分析session也换为r02。

19:47:26UTC实查唯一103848/SSH61064存活，原共同锁2304:15049831297 WRITE*等待，103008持有；新session未创建，无A CUDA/第二等待者。后续接续这个进程，不再启动r01或重复r02。此轮至此仅实现/资源恢复及一次0格失败，没有新策略数据。

资源接续（同一r02，无新提交）：四个已完成ABBA的剩余展开目录以cp -a迁到A数据盘`completed-expanded/`，逐文件bytes/SHA/mtime核对，旧路径保留符号链接，Triton原件/历史元数据均仍可访问；本地完整原始不变。迁后根盘4,535,975,936B、数据盘824,336,384B。现场1791403354.999再次确认103848存活、locks_lock_inode_wait（15:33），session仍不存在，根盘随共享写入降至4,392,992,768B；这仍是等待/资源记录，非GPU负结果，不重启任务。

## 分散到达r02完成：正常容量吸收负载，停止到达率分支

[原始结果](session-native-mixed-budget-spread-20261008-r02/) / [唯一分析](analysis-native-mixed-budget-spread-20261008-r02.json)。28个输出文件远端output/archive及本地SHA一致；tar23,380,371B、SHA `9cf63af8b3e2ca6586a0ffe9cd36dd4fd467a5c772222583bb4887fa10b91ec0`。分析入口SHA `8f869be1…e2dee98`及两项依赖源码已随session保存。r01资源失败原样保留；r02仅因资源确已恢复而使用独立目录，未改变实验配置。

r02候锁937.579s，20:02:39UTC取得共同锁2304:15049831297，空卡检查.164s；唯一格275.088s、整组288.205s。20:07:27UTC `CELLS_COMPLETE / exit0 / VERIFIED / GPU after EMPTY`，103848/SSH61064退出，A无在途/候卡。解包和分析在释放GPU后执行。

**完整请求结果：** 320到达/引擎提交/完成，失败/未完成0，无显式拒绝/超时记录、未触发600s截止。全部预算、输入SHA和实际外部arrival逐ID确认，最后到达63.8s。输出282775，309 length/11 stop；不同到达导致实际输出量也与突发不同（原282868），不声称等工作量提速或质量等价。无应用SLO，不从frontier挑阈值。

| 描述量 | 结果 |
|---|---:|
| 测量持续 / 最后外部到达后排空 | 106.092 / 42.292 s |
| 输出吞吐 / 完成吞吐 | 2665.380 token/s / 3.016 req/s |
| 全请求平均flow / P50 / P95 | 40.638 / 48.613 / 55.498 s |
| TTFT P50 / P95 / max | .122 / .258 / .311 s |
| 每请求max-gap P50 / P95 / max | .132 / .141 / .141 s |
| 发送滞后 mean / P95 / max | .019 / .046 / .069 s |

**实际动作：** 原生抢占0，funding0，再次抢占0，victim选择/替换0。性能模式未采集逐步KV峰值，因此不虚构利用率；实际KV容量、模型、host和控制参数核对与计划一致。selector没有调用观测，不能把未记录计时写成整个runtime零开销。测量期JIT警告1，仍非实际编译次数。

**决定：** 这是正常容量、分散外部到达下不需要抢占的有效边界。按预声明停止本时间分布分支，不继续0.1/0.15等近邻到达率扫描，也不在此零机会负载上跑候选矩阵。两种到达的描述性时延差不构成victim机制收益；该结果不否定突发/其他状态下的问题。

### 下一项具体代码依据：等待集合随victim变化，实际暴露成本尚不知

已安装vllm与固定源SHA核对一致。`offloading_scheduler.py:1133–1152`把被抢占请求的transfer_jobs并入jobs_to_flush，另加block-reuse依赖；`offloading_worker.py:295–303`仍提交全部deferred STORE，但只等该集合。`v1/kv_offload/cpu/gpu_worker.py:379–383,443–447,546–548`让STORE串接前一STORE，再对指定job做event.synchronize。因此换victim可能改变必须等待的STORE前缀，不会自动减少总保存量；其他更晚依赖可覆盖这种差异。当前GPUworker源SHA `a3957e8f…3a3ec98`。

现有pending计数是尚未收齐worker确认的任务数（scheduler1225–1249），可已物理完成；preempt方法耗时也不包含后续worker wait。不能将pending=1当作实际暴露停顿或新颖性。旧mixed burst的同held异质性方向已核查：step681原tail pending0、alternative1，不是可降低依赖的机会；step847原tail0069357 pending1、held149/computed2384/output245，0050606 pending0、held149/computed2383/output563，才有方向正确的合法机会。两个候选的已输出进度差很大，不能忽略改抢旧请求的完成损失。

当前未知：真实victim相关flush等待是否暴露在主执行路径上，以及现有pending计数对应的是未完成传输还是仅未回ACK。
主要竞争解释：选择确能改变有分量的必要等待；或任务早已完成/被其他flush前缀覆盖，计数差异没有可消除的暴露成本。
最小实验：在原混合突发320请求、固定tail和全部预算/输入/恢复设置下，仅加测量期真实worker.wait只读计时与jobIDs/现成engine_call_index；一格，原方法原参数调用一次，不新增CUDA query/sync，不改传输，返回前卸载。
不同结果将改变：若没有实质暴露等待，收束pending依赖这条即时成本解释，不开发selector；若确有等待且合法victim可改变依赖，才做同释放容量的有界真实替换，并完整计入受害请求损失。计时不是收益，整段集合等待不分摊为各job独立成本。

新观察包candidate_native_flush_wait_probe_r01已冻结，仅2个Python payload改变；38文件核对通过。manifest `0cd00cd4b3e68493fd589edb9bca623dcec35f336e1c9215cd569bbf341623f0`，runner `de78cb387ad6606da0ddfb34824c299484ce4ace769957c10b4325c27a7f58aa`。两个必要CPU生命周期夹具通过，GPU尚待执行。无新增CUDA query/sync，UNKNOWN显式保留，观察开销未隔离，不作与旧无observer运行的性能收益对比。

冻结计划 [plan-native-flush-wait-probe-20261008-r01.json](plan-native-flush-wait-probe-20261008-r01.json)，SHA `e5deb532689cb1e3cd0904798226af354204a31b7dab76d73f216e4ac0e45081`；部署tar SHA `980578fd59b61196f5fa9198b1cc42cea2ae218521d55ffef848bb052bd6b8d5`。本次复用既有单格runner、.01s原突发及40/280预算；GPU71.8379GiB/host16GiB/全部机制不变。近期单格产物约0.6GiB，本次新计划磁盘入口2GiB（保留约3倍空间），不修改旧4GiB冻结计划；其他会话磁盘写入仍可能触发真实资源失败。无费用遥测。

```sh
# 远端source root；只运行到全新session路径
LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib /root/miniconda3/bin/python -u run_native_equal_held_group_seeded.py --plan plan-native-flush-wait-probe-20261008-r01.json
```

## flush等待诊断结果：pending非零不等于暴露保存瓶颈

[原始session](session-native-flush-wait-probe-20261008-r01/) / [唯一分析](analysis-native-flush-wait-probe-20261008-r01.json) / [分析器](analyze_native_flush_wait_probe.py)。整组278.819s、单格265.951s，20:43:43UTC终态CELLS_COMPLETE，GPU after EMPTY、109113/SSH74245退出0后释放；无A任务/候卡。28输出远端output/archive与本地SHA一致；原始tar23,913,309B，SHA `050cfe3409be55240104ac535ee4cc294fa36e5c5732c99549a3c36ba8936ec7`。分析入口SHA `fe6770111573b741d3098b22650fe8101c93b80f6ef7132e468e02f6ee3259fe`与两依赖源码随session保存。冻结输入、40/280实际预算和32/384/2固定预热VERIFIED，正常GPU71.8379GiB/host16GiB不变。

**新增证据。** observer为INSTALLED_AND_RESTORED，无安装/关联/恢复错误；35个非空flush均调用原wait恰一次且正常返回，40个实际victim的STORE依赖全部KNOWN，非空依赖均找到同step等待。35段原wait合计6.608ms（捕获98.104s的约0.0067%），中位28.8μs、P95 1.815ms、最大2.228ms。handle整体合计约29.053ms，包含原有deferred STORE提交及观察工作，不能当作额外可移除成本。计时区间非重叠，但总和也不是反事实加速上界。

两次直接包含victim job的等待：step838/0053243/job10248为28.8μs；step847/0069357/job10371为35.1μs，均是单job flush集合。其余33次无当步victim job直接重叠，仍可能有STORE前驱间接依赖。step847已知合法同held149页替代0050606，pending0对tail1，其他状态为computed2383对2384、output563对245；该方向正确的计数差并未对应实质暴露等待。不能从计时确定DMA恰何时完成，但数据不支持为本域的pending计数差开发即时等待成本选择器。

**实际执行变化。** 固定tail，40/40选择与原规则一致、0替换，funding0；40次实际释放全部等于held，总5340页，39个victim，0065237被抢占两次。此组有意是观察诊断，不把0动作下的时延差称机制收益。选择及已有候选观察总CPU墙时186.228ms（含候选观察67.415ms，不能相加）；新flush observer开销未隔离、可改变异步重叠，不与旧无observer运行作加速对比。

| 完整请求描述量 | 结果 |
|---|---:|
| 到达/提交/完成；失败/未完成 | 320/320/320；0/0 |
| 输出；结束原因 | 282868；309 length / 11 stop |
| 持续 / 最后外部到达后排空 | 98.104 / 94.914 s |
| 输出吞吐 / 完成吞吐 | 2883.357 token/s / 3.262 req/s |
| 全请求平均flow / P50 / P95 | 68.091 / 74.502 / 93.563 s |
| TTFT P50 / P95 / max | 9.029 / 62.687 / 65.548 s |
| 每请求max-gap P50 / P95 / max | .161 / 20.272 / 32.370 s |
| 外部到达至客户端提交滞后 mean / P95 / max | .0124 / .0244 / .0299 s |

无显式拒绝/超时记录，独立计数器未采集；全部完成且未触发600s截止。无应用SLO，不选择frontier阈值制造goodput。输出量相同不证明序列/质量等价。测量JIT警告1，非实际编译次数。受损请求的绝对停顿仍在：victim决策→下一输出中位17.100s，决策→下一次原生准入中位16.844s，准入→下一输出中位.290s；阶段含调度/潜在load等待，不能将其全部称为队列或DMA成本，也不是增量因果损失。

**决定与边界。** 关闭“本mixed burst域下，按pending STORE数减少即时强制等待”的投入，不实施选择器、不扩重复/阈值矩阵。此前near-one-page完成机会与.2s无压力分支保持关闭；不将此单格负结果扩大到其他负载/硬件或整个victim问题。现有证据更指向长时间无法再次服务，而非这里的必要STORE同步。下一项先只用已有数据核对重准入改写尾序所造成的再次抢占，以及原始到达/least-generated强简单规则是否已被正确检验；未得到具体新可证伪原因前不提交新GPU组、不复活host-near/equal-held。

```sh
python3 -B analyze_native_flush_wait_probe.py --session session-native-flush-wait-probe-20261008-r01 --output NEW-analysis.json
```

### 新问题依据：原始到达顺序与恢复后队尾不一致

既有mixed raw中，0065237在step664抢占、668恢复，至681又成tail；其外部arrival3.00s、held84/output28，suffix里的0053843外部arrival3.01s、held150/output13，本轮residency输出两者都13。原始arrival和least-generated都只在这一次建议不同动作，可合并为一个强简单对照。0053843原轨迹step686即被抢，停顿31.935s；0065237第二次停顿32.370s。这支持一个尚未执行的有限问题：很短的顺序纠偏是否可避免反复挤出同一请求，同时只小幅提前另一victim？它也可能只延缓几步，完全不改善总体。替代多释放66页，无容量中性机会；不是host/pending或近完成阈值规则。旧equal-held_once是统一预算step600另一对请求，host-near按host改选且长期反复，未正确检验这个动作。原混合点near-one-page完成假说及.2s分支仍停止；这里不改预算/到达，也不复活已退休规则。

当前未知：第一次原始到达最晚者与当前tail不一致时，纠偏一次能否改善全请求完成结果，或只是短暂调换两次抢占顺序。
主要竞争解释：保持原始到达顺序可避开紧接恢复后的重复挤出；或原tail很快再次被抢、只转移停顿，若有收益也可能全部来自多释放容量。
最小实验：原mixed burst320、40/280预算、正常KV、固定全部机制，tail/arrival_once/arrival_once/tail；首次合法原始arrival偏离纠正一次，之后永久tail，两策略共享同一shadow/日志，不读历史成本、未来轨迹、ID或step。最多这一ABBA，不调动作位置。
不同结果将改变：零实际动作则检查入口一次并停止；动作但净结果不改善/翻转则停止本次纠偏，不靠重复到有利；两次完整净收益方向一致才补匹配释放容量的简单解释和独立到达序列，不能立即宣称新机制或等容量收益。

准备期资源：共同锁2304:15049831297由110171持有、111887在等，A无候卡。根盘不足旧四格4GiB入口，先只整理本线终态副本：spread-r02、flush-wait-r01各28个远端output/archive与完整tar核对后，移除output重复目录及archive中的raw.json/warmup-1.json展开副本；完整远端tar、本地完整raw/tar仍保留。另移除这两格134024836B可重建CUDA/vLLM缓存，全部Triton证据保留。旧远端大raw展开路径需从各自tar恢复，未删除原始证据；新组仍保留4GiB入口，不因候卡先启动测量。

新包[candidate_native_arrival_once_r01](candidate_native_arrival_once_r01/)仅2文件变化：staged helper与runner两处白名单；其余36payload字节相同，不继承flush observer。helper只在无phase/protection、合法suffix、有限原始arrival时提案，未知回tail；首次提案消费，后续tail，tail臂对称shadow。三项针对性CPU检查通过，原始burst40决策CPU重算仅step681提案；这不是GPU结果。manifest `44ba2f75a268662d962614f9a70beb84aadfa87a49abf12a1fbd01c4eb3c194f`；staged `c333fd0a5a797cfd78ce42b28f2ea776fa18d0df1bd866ea1eb60ffa217a3448`；runner `37cc1ccba78564cf714f3bfa552dc9baf85d9eb255a5d2f9611e40c78e5c3fd3`。

冻结[plan-native-arrival-once-20261008-r01.json](plan-native-arrival-once-20261008-r01.json) SHA `0bc2619a21f45fe951d12277cddcaa71ddc997f1f2203d629e4e0f153018551c`；部署tar SHA `c074d8872a54d19ae37247ad81a961b0d24944c1ba348e8d8f03ef403b903612`。两个新薄controller只改kind/规则白名单/import名称，复用同一锁、种子、4800s整组/1200s单格/600s捕获上限；全部预算/外部输入/预热/恢复不变。组内禁止改代码，数据单位为两对运行，不以事件当重复。

```sh
# 远端 /root/autodl-tmp/moe-a-victim-20261004，使用全新session路径
LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib /root/miniconda3/bin/python -B -u run_native_arrival_once_group_seeded.py --plan plan-native-arrival-once-20261008-r01.json
```

部署后实查：38payload及全部controller SHA通过，新session不存在；110171与111887进程均存活，前者持同一2304:15049831297锁、后者等待，GPU worker113035占88080MiB。根盘4124446720B低于4294967296B入口。A未提交新的GPU进程/候卡；准确断点为共享资源，不是CPU检查成功即实验完成。资源变更后先确认旧A PID109113仍终态/无新A任务，再用上述唯一计划进入；不要盲目重启或并列提交。

分析器[analyze_native_arrival_once_group.py](analyze_native_arrival_once_group.py)已就绪，SHA `5a69ae24abee8abf7fbe67f71dc4bccea486b9a25c5421881da16b03c057819e`。复用全请求、预算和真实preempt/free核对；仅补SHA匹配的once状态重算、实际动作前输出/抢占/原始arrival顺序差异及动作后原tail再次抢占，保留未知和容量差。两个定向CPU夹具通过，不是GPU结果。命令：`python3 -B analyze_native_arrival_once_group.py --session session-native-arrival-once-20261008-r01 --output NEW-analysis.json`。

**启动前资源修正，非科学调参：** 用完整tar条目计算seeded-host-r02、equal-held-r01的archive＋output双份、执行包与每格81MiB既有缓存规模，四格预计2.2804/2.2823GiB；最新mixed单格预计.5394GiB。剩余A root文件全部迁移也不足维持原4GiB门槛，不再为了入口反复搬迁。新[plan-native-arrival-once-20261008-r02.json](plan-native-arrival-once-20261008-r02.json)把尚未启动组的磁盘入口一次调整为3GiB，约.7GiB余量；不改GPU/Host容量、源代码、输入、顺序、量子、恢复、准入或种子，若仍不足不再下调。r01保留为UNRUN资源版本，不记GPU失败。上述运行与分析命令将plan/session换成r02。共享写入仍可能导致失败，此估计不是硬上界。

r02计划SHA `991a8f38640b230ec9856bda8da6ffd011adcb2521d833c663bfc47044363e53`；唯一实际提交controller114335 / SSH97687，采用既有最多3600s共同锁等待，尚未获得GPU结果。后续只接续此handle，等待结束前不创建另一个任务。

组启动后已确认114335取得同一共同锁、GPU入场EMPTY，首tail真实初始化。等待结果期间只用既有mixed/flush候选补核对：step681合法suffix index55..258，tail0065237为258/84页；arrival候选0053843为257/150页，恰好也是最近的非tail候选。150页层另有149、234，最近者仍257。因此该事件的arrival排名、next-tail和匹配150页最近尾序都选同一对象；即使本动作获益，也不能单凭它区分这三种局部选择解释。它不证明这些完整在线策略相同，也不提供新增信号贡献。

首对运行中间结果：cell00/cell01均320完成、exit0、archive VERIFIED，267.377/265.967s。tail一次shadow零请求；首候选一次提案/请求/实际改选，step681选择0053843替代0065237，raw中唯一同step+ID原preempt成功返回、实际释放150页。反序候选仍在运行，114335/SSH97687继续持锁；没有提前作净收益结论或改组设置。

等待本组时只读核对的备用域边界：原生block_pool.py:702–740按ref_cnt归零释放，因此真实共享引用可让同held对应不同实际释放；但当前safe_static.py:24,31–32、staged_store_rotation.py:385–397及rotation_native.py:153–169均要求NoPrefixCache。oldest commit的共享页拒绝和Q1 funding按len(owned)预算也需同步改契约；不能只删入口断言。该备用域不在本轮实施，不以未运行共享KV声称收益，保持既定恢复/传输边界。

## arrival-once真实结果：重复挤出未避免，微小正向计时未完成归因

[原始session](session-native-arrival-once-20261008-r02/) / [唯一完整分析](analysis-native-arrival-once-20261008-r02.json) / [分析器](analyze_native_arrival_once_group.py)。执行冻结r02及manifest44ba2f75…194f；四格267.377/265.967/267.088/267.626s，整组1084.140s。2026-10-08 05:27:41 CST四格exit0、CELLS_COMPLETE、GPU after EMPTY、控制器退出并释放共同锁。112输出/archive SHA远端及本地一致；完整tar95,595,267B，SHA `75d78acc5c1be780a164d670788c97c6b1e9ea87e8964869d8af3d5c462f6f96`。r01仍为未执行资源版本，不计科学重复。

**新增证据与实际执行。** 两候选各仅1次提案/请求/执行，均在step681改抢0053843（output13），实际释放150页；tail同位置抢0065237（output28），释放84页。原tail在候选中继续9个输出，step690、output37即再次被抢，距改选.693/.698s，释放85页。基线step686已抢0053843（output18）。四臂都40次成功抢占、39个victim，只有0065237被抢两次；funding0，无失败preempt调用。总实际释放5340/5341/5341/5340页，160次均等于当时held。纠偏没有减少本请求再次抢占或全组抢占数，不支持原先“避开紧接恢复后的重复挤出”解释。

从第四次抢占step695开始，后37次的请求/步号/输出进度/释放页数相同；这只是抢占后缀对齐。四臂2191次engine调用/返回；两对中除参与者0065237、0053843外，318个请求的输出步骤/累计计数/token ID相同。两参与者仍造成1022个输出call集合不同，延续至step2135，不能称全轨迹恢复。每臂输出282868、309 length/11 stop；仅0065237最终序列跨策略改变，1024位置中927不同。因此输出量相同，质量等价未检验；raw的recovery_count/recomputed_tokens为UNKNOWN而非0。

**完整服务结果。** 四格均320到达/提交/完成，失败/未完成0；从外部到达计时，最后到达3.19s，未触发600s截止。无独立穷尽拒绝/单请求超时计数器，保持UNKNOWN；无应用SLO，不选诊断frontier阈值宣称goodput。

| 臂（执行次序） | token/s | req/s | mean flow s | TTFT P50/P95/max s | flow P50/P95/max s | 请求max-gap P50/P95/max s | 持续/排空 s |
|---|---:|---:|---:|---|---|---|---|
| tail-first | 2882.378 | 3.261 | 68.0995 | 9.033/62.677/65.523 | 74.458/93.586/94.957 | .156/20.211/32.294 | 98.137/94.947 |
| arrival-once-first | 2890.896 | 3.270 | 67.8462 | 8.968/62.457/65.299 | 74.187/93.297/94.668 | .150/20.141/32.231 | 97.848/94.658 |
| arrival-once-reverse | 2883.289 | 3.262 | 68.0602 | 9.014/62.599/65.499 | 74.448/93.557/94.926 | .167/20.197/32.316 | 98.106/94.916 |
| tail-reverse | 2881.283 | 3.260 | 68.1283 | 9.008/62.735/65.593 | 74.515/93.626/94.994 | .164/20.247/32.401 | 98.174/94.984 |

两对candidate−tail mean flow为−.253324/−.068036s（−.372%/−.100%），输出率+.296%/+.070%。两运行对的差值均值−.160680s、样本SD .131018s；n=2，不能用640请求缩小运行级不确定性。全部请求flow更晚0/48个，TTFT更晚1/227个，max-gap更长75/204个。客户端发送滞后mean .01186/.01205/.01209/.01178s、P95 .02444/.02424/.02504/.02354s、max .02891/.02962/.02896/.02686s，未隐藏引擎外等待。

| 参与者，candidate−tail | flow差 R01/R02 s | max-gap差 R01/R02 s |
|---|---:|---:|
| 原tail0065237 | −.442605/−.213079 | −.910343/−.932530 |
| 替代victim0053843 | −.217071/+.007217 | +.371814/+.356029 |
| 分配失败请求0024355 | −.307642/−.095923 | −.006456/−.005619 |

**归因与开销限制。** 动作前两对99237条输出、一次真实抢占和原始arrival排序一致；387条匹配候选状态中各4行的host前缀/missing不同，涉及另2请求，故非严格同状态。候选在动作前已经快.158538/.050784s，不能把全程mean flow差都归因纠偏；仅按动作锚点相减也不是因果校正。替代victim动作后剩余完成时间差−.058533/+.058001s，方向翻转。两候选均比tail多释放66页，且该动作与next-tail、150页层最近tail建议相同，原始arrival信号的独立贡献没有被区分。

选择器+共同观察CPU墙时合计180.526/176.920/181.079/190.866ms，其中候选状态读取60.510/61.439/63.234/69.364ms已包含，不能相加。它们约占捕获时长.18–.19%，不是已隔离的新增规则成本，也不以正负差推断纯选择器收益。所有组内配置/输入/40-280预算核验一致；执行源与分析器源码随session保存。

动作前负对照已用现有raw完成，无新增GPU：两对各43请求在各自proposal前已完成，其mean flow差−.083499/+.004940s、TTFT差−.053244/+.005865s，不能由尚未执行的纠偏解释。同策略后次−前次，全请求mean flow为tail+.028752s、once+.214039s；capture duration为+.037292/+.258139s。这不是噪声分布的精确估计，也不作前缀相减校正，但说明较弱那对.068s收益没有脱离已观察到的运行波动。只读[诊断脚本](diagnose_native_arrival_once_timing.py)输出stdout、不生成第二结果JSON，SHA `7c77baf54f25c9bc47f49c58243f2c39f6b493311106484d0405d3670e027f51`。未用host输出接收间隔冒充纯engine.step或GPU耗时。

**决定。** 保留两对原始正向计时，但净效应归因仍不确定；重复抢占避免这一具体解释未出现。维持本组最多一次ABBA的预声明上限，不重扫动作位置、到达率或保护阈值，不将微小正值升级为稳定收益。当前最薄弱环节仍是“实际改选是否带来有解释的完整服务净改善”，而非动作入口。动作前负对照和同策略波动检查已完成；只有新证据能区分竞争解释，才安排新的运行域/探针，不自动扩模型矩阵。全文尚无可信中心贡献，不进入投稿完成状态。

```sh
# 已执行命令；复现先复制计划并设置全新session_dir，不重跑原目录。
LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib /root/miniconda3/bin/python -B -u run_native_arrival_once_group_seeded.py --plan plan-native-arrival-once-20261008-r02.json
# 本地D目录，分析器拒绝覆盖已有输出。
python3 -B analyze_native_arrival_once_group.py --session session-native-arrival-once-20261008-r02 --output NEW-analysis.json
python3 -B diagnose_native_arrival_once_timing.py --session session-native-arrival-once-20261008-r02 --analysis analysis-native-arrival-once-20261008-r02.json
```

## 新的覆盖缺口：已测压力均在外部到达结束之后

仅复用近期原始数据：mixed burst与arrival-once四格共5次运行，最后外部到达均3.19s，首次原生抢占31.580–31.744s、末次57.159–57.413s，首次压力之后新到达均0。40/40短预算请求中已有32个完成；40个决策仅6次同时有短/长预算合法候选（13个短候选观测、4个不同请求），不能用这些事件当独立重复。另一.2s spread持续到63.8s但原生抢占0。该证据只说明已测时间结构的缺口，不证明改变到达就会有victim收益。

同时核对固定原生队列：scheduler._preempt_request将victim prepend到FCFS waiting；FCFS先看skipped_waiting，再看waiting，分配失败停止该队列循环。arrival-once首对除664→668快速恢复外，其余39次重准入均严格按抢占逆序；该探针已经交换两参与者的恢复先后。仅以“恢复排序”再交换同两对象没有新判别用途，不开该分支。

当前未知：正常KV下，新外部请求与容量抢占同时出现时，原生合法victim集合是否呈现此前未覆盖的服务阶段/恢复代价差异。
主要竞争解释：此前近似封闭批次使选择主要交换既有长请求的停顿；或原生准入/调度也会吸收新增到达，使合法动作仍无有用差异。
最小实验：仅一格tail；同320内容/顺序及40×128+280×1024预算，前256仍i×.01s（既有knee前缀），后64为30+(i−256)×.2s；正常GPU/host、所有恢复/准入/传输/预热不变。30s由既有聚合压力起点附近取整，仅为显式合成时相诊断，非独立确认或应用流量代表性证明。
不同结果将改变：若无真实到达-抢占重叠或无新的合法异质性，停止此时相点，不移动起点/节拍扫描；若出现，只依据动作前新状态选择下一次有界干预，先测完整服务净效果，不自动重跑退休规则。本格本身没有策略替换或收益主张。

已冻结[candidate_native_mixed_overlap_r01](candidate_native_mixed_overlap_r01/)，manifest `bbcda5cda3a30111e678987893aed9d4ff630215d1fe50ec3effa4bd6eb1b5f8`；仅pro_high三份JSON变，37个pkg文件中的另34个及verifier不变。首256到达浮点原值保留，33短/223长；后64为7短/57长。arrival_gap_s为null，权威时刻为显式trace。构造器复现、既有verifier和真实measure_episode的CPU假引擎检查通过；不是GPU结果。部署tar7,510,967B，SHA `88fa7e7348371699db87429ee3493aac73b42b5940c0661ff493b90ab0fed486`。

[plan-native-mixed-overlap-20261008-r01.json](plan-native-mixed-overlap-20261008-r01.json) SHA `5197f376b9343023f30a68a9f81aeadc1c834d19ca151e34aa359e28076cb6d9`，沿用既有单格controller、正常71.8379GiB GPU/16GiB host、1800s整组/1200s单格/600s捕获及相同Triton种子/预热。磁盘入口复用单格2GiB（近期完整产物约.54GiB），无费用遥测。远端38payload/全部controllerSHA核验、新session不存在，唯一controller122268 / SSH77006真实启动；实查处于locks_lock_inode_wait，118932持有原共同锁，A没有第二任务。只接续此handle，不盲目重启。

[分析器](analyze_native_mixed_overlap.py)复用已有全请求分析，仅补实际到达一致性、抢占相对时间、新旧输入阶段候选及同held/computed整页对照；最终SHA `b669600790f3291eeaa02510bdb81b5313b3bdb95f248b9b1383327c8dfebc18`。三个时钟/身份边界夹具及旧tail原始负对照通过；不是GPU性能验证，不把外部到达跨越首末抢占区间当作持续KV压力。

等待中发现的具体入口解释：staged_store_rotation.py:674–682的qualified继承pure_decode＋RUNNING/持页/residency条件；任意suffix行失败就使整次fallback_unknown。partial-prefill可以进入原生suffix并被原tail抢占，却不满足该选择器条件。因此qualified不等同原生抢占合法性。分析器已在新数据产生前补入按输入prompt长度、当前computed/output重建的进度分类，区分“晚到者未进suffix”“进了但未产生输出/尚未满足decode关系”“满足关系却被其他资格条件挡住”。未产生输出不能精确区分partial与刚完成prefill；UNKNOWN保留。若新原生机会被这一旧选择器条件遮蔽，下一项应是对应入口诊断，而不是移动到达时间或声称问题没有机会。GPU冻结包与规则仍不改变。

```sh
# 已实际提交，接续122268/SSH77006；不要重复运行这个session。
LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib /root/miniconda3/bin/python -B -u run_native_equal_held_group_seeded.py --plan plan-native-mixed-overlap-20261008-r01.json
# 终态后本地D；输出必须尚不存在。
python3 -B analyze_native_mixed_overlap.py --session session-native-mixed-overlap-20261008-r01 --output analysis-native-mixed-overlap-20261008-r01.json
```

## overlap结果：时间覆盖成立，强简单排序仍与tail一致

[原始session](session-native-mixed-overlap-20261008-r01/) / [唯一分析](analysis-native-mixed-overlap-20261008-r01.json)。320到达/提交/完成，0失败/未完成，输入时间/预算/固定预热/实际KV均VERIFIED。单格345.285s、整组363.443s；06:09:22 CST exit0/CELLS_COMPLETE、GPU EMPTY后释放，A无任务/候卡。28个原始输出本地与远端SHA一致，tar23,767,918B，SHA `6c8998cc3bbd264d8b5e23bf078ebf80dcecd63b2d54f03f4682ae9040a6c236`；分析代码随session保存。

**新增证据。** 首次/末次成功原生抢占38.642/65.037s，最后外部到达42.6s；首抢占后还有20个到达，30–42.6s的外部stream区间内有5次抢占。35次决策中28次新旧cohort共存且未被资格/phase条件挡住，9次有与tail同held及同computed整页的跨阶段候选，共13个；不是独立重复或同状态反事实。实际原生释放35次均等于held，累计4615页；33个victim，0021202与0054622各两次。funding0、35次均原tail、策略替换0，这是有意的输入表征而非零动作性能对照。

**没有得到的新机会。** 34/35决策可完整排序，最大剩余声明预算、最少已输出、最晚原始arrival均与tail一致，影子改选0。首压力前已有33/40个短预算完成；短/长预算共存仅2次，唯一短候选0036565剩余28/8，均被tail自然保留，没有临近一页上限的tail。同held/computed的13个替代全部已输出更多、剩余预算更少，且当前host missing高于tail；仅同held的14个替代也无missing或pending改善。因此没有移除旧host-near容量混淆的机会，不复活near-one-page或扫描阈值。

唯一fallback在step722：suffix只有自身0062890（index254、held128、computed2036、output0、free5），native tail照常抢占。确有pure_decode资格失败，但没有第二个suffix候选，未遮蔽可替换动作；不得把它写成接口故障。其余3594个候选观测全部满足pure_decode/qualified，身份UNKNOWN0。

| 完整请求描述量 | 结果 |
|---|---:|
| 输出；结束原因 | 282847；309 length / 11 stop |
| 持续 / 最后外部到达后排空 | 108.366 / 65.766 s |
| 输出 / 完成吞吐 | 2610.112 token/s / 2.953 req/s |
| mean flow / P50 / P95 / max | 68.805 / 74.597 / 91.196 / 92.280 s |
| TTFT P50 / P95 / max | 5.751 / 33.835 / 34.296 s |
| 每请求max-gap P50 / P95 / max | .221 / 16.739 / 32.284 s |
| 客户端发送滞后 mean / P95 / max | .01799 / .06538 / .17963 s |

无捕获超时，独立拒绝/单请求timeout计数仍UNKNOWN。无应用SLO，不挑goodput阈值；与旧burst的外部输入时序及实际输出量不同，不能当作策略加速或质量等价。CPU选择+共同观察315.684ms，其中109.346ms候选观察已包含；非隔离新增机制开销。测量JIT警告1，非实际编译次数。

**谁承担停顿。** step731仍在到达时，晚到tail0049468（arrival37.6、held107/computed1707/output12、ready106/missing0）被抢，32.284s生成间隔、最终flow69.794s；同容量/同computed的初始请求0034261已有382输出、ready0/missing106，本次未被选。step751 tail0051693（held160/computed2555/output39）生成间隔30.084s、flow69.799s；同容量层0040628已有235输出。它们显示不同服务阶段下的真实停顿，未抢替代者的较短gap不是改选收益。

**决定。** 该单格补齐了真实到达-压力重叠；但目前可观察的强简单排序与host等容量对照没有提供新改选理由。停止本时相点的起点/节拍扩展，不重跑已收束规则。现有性能raw未采集实际offload lookup/load完成量（offload-events为NOT_MEASURED）；下一residency held小于抢占时host-ready，不足以判定未来命中或prefix丢失。先限定能从现有事件推出什么，不为一个尚不能改变动作选择的猜测自动追加GPU观察组。当前仍无已证明的完整服务净收益或论文中心贡献。

### 边界检查：不能从驻留页数倒推 host 命中；共享前缀已有近邻

固定源 `kv_cache_manager.py:411–427` 的 `full_sequence_must_fit` 只检查完整现有序列能否放入；实际分配在429–485，按外部命中加本次计算量进行。`scheduler.py:766–789,978–1011` 才决定外部命中、异步加载及进入running；现有residency记录不包含该命中量。因此0049468下一次进入running时held17不等于17页host命中，也不能单独判定何时/为何丢失抢占时ready106页。不追加仅为填这个字段、却不能区分行动价值的GPU观察组。

只读查新发现 [TOPAS §III、IV-E](https://arxiv.org/html/2608.25523v1#S3) 已显式将共享静态前缀计一次，并联合选择驻留前缀、运行请求与成组抢占/准入；共享前缀的组合容量效应不是空白。[vLLM原生scheduler](https://docs.vllm.ai/en/v0.30.0/api/vllm/v1/core/sched/scheduler/) 仍使用tail/priority，物理释放由[block_pool引用计数](https://docs.vllm.ai/en/v0.9.2/api/vllm/v1/core/block_pool.html)决定；若将来进入共享前缀运行域，最大即时可释放页数必须作为强简单对照，引用计数正确计费本身不算贡献。目前未开启prefix cache、未部署新包、未启动新GPU组；先定位现有private恢复合同是否允许这种固定运行域，不能仅删断言。

### 下一轮四行：原生共享前缀下的物理释放机会（准备中，尚未运行）
当前未知：现有私有KV域所有实际release==held；固定启用原生prefix cache后，原生合法suffix是否出现held相近而即时可释放页不同的选择，且正常容量仍有压力？
主要竞争解释：共享引用确实改变动作的容量效果；或cache复用消除了压力/候选都私有，所设运行域没有新动作价值。最大即时release只是强简单规则，尚非创新。
最小实验：一格tail、320请求；从每个既有长度层取前两请求组成同文章输入对（64对，其余192保持），只复制输入不看输出；原0.01s到达、40×128/280×1024预算、正常GPU/host与Q1全部固定。开启prefix cache所需启动资格适配保留准备/提交/保护期全部private gate；观测真实refcount及free增量。
不同结果将如何改变机制选择、下一步投入或论文主张：有共享容量差异且实际触发抢占，才设计有界真实动作探针及最大release强对照；无压力/无共享合法替代则收束此固定共享输入点，不搜索复制比例、负载速率或KV限额。该组合是显式合成机制诊断，不与旧输入时延作收益比较，不称同状态因果实验。

### 共享前缀包已冻结；部署连接阻塞，GPU未运行

实际安装路径为`staged_store_rotation.install`，并不调用`rotation_native.install`。Q1的`oldest_resource_gate`在准备和提交前均检查全部running页ref_cnt==1及物理唯一性；这些函数与保护期gate原样保留。只允许已核对的NoPrefix/Unitary单组FullAttention coordinator，prefix开时本包仅执行tail。每个原生suffix候选新增即时可释放页、shared页与UNKNOWN原因，最大release仅做影子（partial-prefill不被旧pure-decode资格误挡）。这是启动兼容和观测改动，不是新恢复机制或策略收益。

独立包[ candidate_native_shared_prefix_probe_r01 ](candidate_native_shared_prefix_probe_r01/)，parent为mixed_budget_probe，payload仅6文件改变（3输入JSON、runner、safe_static、staged），其余32 manifest项相同。输入builder按64个既有长度层复制第一篇给第二请求，64对/192singleton/256唯一文档，ID/到达/预算不变；理想prompt共享页量60.186GiB仅为输入估算，实测是否共享/是否有压力仍UNKNOWN。3项针对性CPU夹具通过，原Q1相关函数AST不变；输入重建check通过。准备过程中修正provenance后曾先运行check而未重建，得到预期input drift，随后重建并通过；没有GPU执行/结果。

manifest `a9c85333719019a858531201e8b6643353e6855e4dcd833cc5c445778703786c`；[plan](plan-native-shared-prefix-probe-20261008-r01.json) SHA `2cb1b4298a6a156980ac7800d93f5c5a119a84b948cdeabb71536d5cf2fa6690`；包tar6458954B SHA `31e3d07a07f7f4e498ef548a439c68f42d9b24fc45aebd43b8fc219f68cda8aa`。复用原单格控制器、共同锁、1800s整组/1200s单格/600s捕获、正常77135347712B GPU/16GiB host，未降低空间入口。

**状态UNRUN_CONNECTION_REFUSED**：上传exit255连接reset，旧SSH master消失；重连同一已授权westb:25495返回Connection refused。新controller从未提交，没有新增A候卡者；远端部分上传情况未知。上次成功只读看到其他PID125429占卡80186MiB，不能当作当前占用。保留[部署失败记录](shared-prefix-deployment-20261008-r01.log)。恢复连接后先确认新session不存在及远端包完整，再在本线source根执行一次：

```sh
LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib \
 /root/miniconda3/bin/python -B -u run_native_equal_held_group_seeded.py \
 --plan plan-native-shared-prefix-probe-20261008-r01.json
```

本次没有共享前缀GPU数据，不将CPU检查或连接失败记作负结果；既有overlap真实结果不受影响。下一决定点仍是共享时的实际释放机会；连接未恢复前只完成本地分析器和复现断点，不改共享比例/容量，也不创建重试候卡任务。

配套[分析器](analyze_native_shared_prefix_probe.py)已完成，SHA `6ebbc740485b137a3033ac4fa6c37e13ceef6460cf9e0ccff77962b55487532d`。复用完整请求指标/预算/预热口径，新增prefix/Unitary、64对输入和实际到达核验；实际释放与选中候选即时release逐事件核对，held仅描述，旧private资格不当作native合法性。针对性CPU检查覆盖同held不同release、UNKNOWN回退、旧prefix=false拒绝及未运行NO_MEASUREMENT；未生成新结果JSON。取回真实session后仅执行一次：

```sh
python3 -B analyze_native_shared_prefix_probe.py \
 --session session-native-shared-prefix-probe-20261008-r01 \
 --output analysis-native-shared-prefix-probe-20261008-r01.json
```

连接复核（2026-10-07T22:51:16.768673+00:00）：第二个连续goal轮次仍在westb:25495得到Connection refused，未能检查远端部分上传；新实验未提交，无新增科学数据或A候卡。代码、冻结计划及分析器已就绪，不再增加CPU审查/阈值方案来代替所需GPU实验。目标仍ACTIVE；该连接故障不构成机制负结果。

第三轮连接复核（2026-10-07T22:51:59.308464+00:00）：同一已授权westb:25495仍Connection refused（exit255）。本地冻结plan/tar/analyzer SHA与上一轮一致，全部运行准备已完成；无新controller/候卡、无GPU数据。下一有判别力的动作依赖连接恢复，不能用更多CPU重复检查代替。满足连续三轮同一真实资源阻塞条件，目标记BLOCKED，非完成、非机制失败。恢复时先核实远端部分上传和session/进程，再按已保存单格命令继续。

### 新授权westd:53005接续（2026-10-08，未改研究输入/选择规则）
新主机保留A工作区、全部原始结果、模型、私有overlay与Triton seed；GPU为PRO6000 `GPU-51b8e4bb-27b8-4b82-5254-7317aae7298c`/driver580.95.05，cgroup110GiB。相同公共锁路径现为2304:4312099778；首次现场GPU为空、无持锁者，该观测不预约资源。系统python3不在PATH，后续沿用/root/miniconda3/bin/python；源码及版本匹配，未改全局环境。
当前未知：冻结共享输入在新设备正常容量下，是否存在共享引用造成的物理释放异质性及实际抢占？
主要竞争解释：持页代理遗漏真实释放差异；或共享消除压力/候选无异质性，仍没有值得干预的空间。
最小实验：同冻结package/320输入/预算/到达/恢复/传输/准入，先当前GPU .90正常容量profile，再一格native tail；全组同主机公共锁，不用旧GPU时延或旧容量profile作本机结论。
不同结果将如何改变机制选择、下一步投入或论文主张：复用已冻结判别规则，有合法物理差异才做有界动作探针；无压力/机会则收束该共享输入点，不扫复制比例/限容/到达率。新主机不改变该判断。

新设备计划[plan-native-shared-prefix-probe-westd53005-20261008-r01.json](plan-native-shared-prefix-probe-westd53005-20261008-r01.json) SHA `be6c63b332079839cf71f9bfca653041258fc3211cb56a2b4537cb3ff16bdfcf`，旧未运行计划保留；payload仍a9c85333…786c。上传/38项payload及9控制器核验通过。2026-10-08T08:39:41.967902+00:00唯一controller2519/SSH41658实查存活并等待同一2304:4312099778锁（2066持有），session未创建、无A CUDA/第二候卡；目标已恢复ACTIVE。
配套新[profiled分析入口](analyze_native_shared_prefix_profiled.py) SHA `0516caaa6289915fdc689157ff72d0c4a0bf02b59a676b0b2ddc89ea84273191`复用旧指标/shared分析，额外核对本组profile、GPU UUID及实际字节预算；相关容量/UUID/未启动CPU夹具通过，旧代码未改。最终结果必须来自此新session，未出结果前不生成占位科学JSON。

## westd共享前缀实测完成：容量异质性不等于改选价值

[唯一分析](analysis-native-shared-prefix-probe-westd53005-20261008-r01.json) COMPLETE_CHARACTERIZATION；[原始session](session-native-shared-prefix-probe-westd53005-20261008-r01/)39输出均SHA核验，tar31,530,309B，SHA `ef158bf2eac1483c9736f690f5ad076b6ddb8f140c179b643fbb3fd54944191e`。当前设备正常.90 profile为77,076,627,456B（71.7832GiB，36753总页/36752可用）；测量精确固定该容量，host16GiB。2519/SSH41658在16:51:35 CST exit0/CELLS_COMPLETE，GPU EMPTY后释放共同锁，无A在途/候卡；整组412.805s，profile114.983s、测量进程277.367s，等待共享锁438.623s另计。不是旧卡时延对照。

**新增证据与执行。** 320到达/提交/完成，失败/未完成0；27次成功native抢占、20个victim、funding0、实际策略替换0。所有选中tail均shared0且实测release==即时release==held，共3854页。3318候选观测物理状态全已知，11次合法suffix含共享候选；6次同held不同release的替代全部少释放，没有同held更大释放机会。最大release影子提出26次改选，与全suffix最大held有11次不同，却27次都等于“私有候选内最大held”；不能将影子当执行或恢复成本优势。step837原tail139/139页，最大held候选238/50（188共享页），最大release候选228/228；共享容量代理确有差异，原tail已自然避开共享victim。停止据此直接开发共享评分器。

| 全部请求描述量 | 结果 |
|---|---:|
| 输出；结束原因 | 282776；308 length / 12 stop |
| 持续 / 最后到达后排空 | 88.206 / 85.016 s |
| 输出 / 完成吞吐 | 3205.863 token/s / 3.628 req/s |
| mean flow / P50 / P95 / max | 63.392 / 68.885 / 81.785 / 85.026 s |
| TTFT P50 / P95 / max | 4.030 / 20.743 / 23.621 s |
| 每请求max-gap P50 / P95 / max | .167 / 4.444 / 24.968 s |
| 客户端发送滞后 mean / P95 / max | .0121 / .0239 / .0436 s |

没有捕获截止，独立拒绝/请求timeout计数仍UNKNOWN；无应用SLO，不挑goodput阈值。输出/输入不同于旧非共享运行，不声称等工作量或质量等价。CPU选择+观察207.531ms（含140.766ms候选观察及.852ms最大release影子）；额外边界资源观察12.826ms，非隔离新增机制开销。测量期1条fused_moe JIT警告，不等于测得编译耗时。

**谁停顿。** 0064221最大生成间隔24.968s、最终flow85.026s。27次victim到下一输出等待中位8.238s，失败分配请求到下一输出中位.070s；都是真实绝对等待，不是反事实损害。0063744、0027616各被抢4次，0053843两次。

**新而窄的机会。** step1090原tail0063744已有402输出，但computed1592尚未恢复到现有序列末端；它不是此次分配失败对象。suffix54个请求，另53个均pure-decode，物理状态已知、无保护phase。旧选择器因这一个non-pure-decode整体fallback；原生路径仍允许改选。最近容量候选0070086即时释放101页、computed1613/output658，对tail100页；二者shared0、ready0、pending0，hostmissing100/99。没有严格同释放替代，1页混淆必须保留。另一partial事件step1121为self且suffix单例，不可改选。新问题是打断尚未接回已有历史的请求是否有完整损失；不是因为历史已花成本就保护，也不是重新复活旧self-currentguard/arrival排序。

当前未知：在非self tail尚未恢复已有输出历史的真实决策，避免这一次打断是否减少随后重复抢占和完整服务损失？
主要竞争解释：短暂保留能跨过恢复阶段并改善服务；或很快再次抢占、仅将停顿转移给替代者，容量/计时差足以解释表面收益。
最小实验：同共享320输入和本机正常容量，tail/partial_restore_once/partial_restore_once/tail一次ABBA；首次满足状态时只改选一次，从合法pure-decode私有候选挑不低于tail的最小即时release、同值近tail。不按ID/step，其他机制不变；所有请求排空，保留动作前状态/发送滞后/输出序列差。
不同结果将如何改变机制选择、下一步投入或论文主张：零执行只诊断入口/机会不扩矩阵；重复抢占未减或总体受损即收束此动作，不调保护阈值；有一致完整净信号再与同动作范围无恢复信号的简单替换做消融及独立到达确认。最多本次ABBA，不测到显著；当前尚无新探针GPU结果或投稿主张。

补充动作价值边界：step1090原tail在.222s后已有下一输出（重新准入.149s，准入后.073s）；不能把此前step937开始的11.947s整段等待归因于这次恢复中断。该探针可能只改变很小的暴露路径，或者反而让原本不停顿的替代者受损；这正是仅做一次ABBA、无信号不继续保护阈值搜索的原因。
[UniBoost §3.3](https://arxiv.org/html/2606.18431v1#S3.S3)已规定dispatch后到下一个几何decoded-token阈值前不可驱逐，覆盖更宽的最小服务保护；将其涵盖恢复后的无输出阶段是对原文的推论，文中未展开容量不足例外与SelectVictim细则。[FastServe §4.1–4.2](https://www.usenix.org/system/files/nsdi26-wu-bingyang.pdf#page=5)已有MLFQ量子/饥饿提升与ENST交换顺序。当前历史追平谓词、suffix非self及一次性范围不同，不足以主张恢复保护新颖；若动作真的有价值，必须面对这些强近邻/简单服务保护。
空间处理：1791451299.708（UTC epoch）先核验本地179份raw，再核验远端三个终态session的archive/output逐文件SHA；仅清理arrival-once-r02、mixed-overlap-r01、shared-prefix-westd-r01的重复output、runtime/preflight缓存。所有远端archive、SHA记录、日志、冻结/执行代码、tar、本地raw及全局Triton seed保留。根盘空闲2,853,089,280→4,797,575,168B；无A进程/候卡，不持GPU锁整理。沿用上一ABBA实际plan的3GiB入口，不降低门槛。

### partial_restore_once包与ABBA已冻结（尚未执行）
[candidate_native_partial_restore_once_r01](candidate_native_partial_restore_once_r01/)仅在共享包上改staged selector与runner规则白名单；输入、预热、33个其他嵌套生命周期/恢复函数及native free路径原样。新增prompt/status当前态，独立once状态区分trigger/proposal/returned/action_requested；实际执行以preempt/free关联确认。4项定向CPU检查通过，包含原生suffix pop与原_preempt_request调用；旧1090重放101对100页仅为接口诊断。
manifest `5d2f04a0e0007a0a8bc3a2e8f54c4efd3b3dd9f73a389fddb483bb7301e90a09`；staged `36406e001cfbdd151c11aab8a27e61728655c65a15f6bfd43df08d9d89c905d2`；tar7,517,024B SHA `3146cbe5a4fc692e21d06b6bf8801caa10993851455dfa2f35afe61a5a61cbe5`。
[计划](plan-native-partial-restore-once-westd53005-20261008-r01.json) SHA `8eb400686ac3d2d042aaa8aa746fbb8fd12ca970c620f2aae72aa2233517ca52`：四格同320输入、正常77,076,627,456B/host16GiB，tail/once/once/tail，最多一次ABBA；单格1200s/整组4800s/捕获600s、一次3600s共同锁等待，沿用3GiB空间入口。容量来自本机刚完成profile，SHA `89fc9e5d191225d3d543159f4c76fde13223248c9f9a3a32a3e518301b9011ec`；不重复profile。没有消费遥测/已知共享余额。
[分析器](analyze_native_partial_restore_once_group.py) SHA `13d95ceea9028002f97ab74b8d6337535dd3bddc90b89e07fa55473bd71692b7`复用全请求指标，补once执行、实际release、动作前差异/后续轨迹、来源capacity与输出差异；未启动/缺profile/旧规则/时序重放CPU检查通过，未生成占位结果。
```sh
# 远端A source根；只提交一次，session已存在时不得重跑。
LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib /root/miniconda3/bin/python -B -u run_native_partial_restore_once_group_seeded.py --plan plan-native-partial-restore-once-westd53005-20261008-r01.json
# 全组终态/释放后取回，在本地D唯一分析。
python3 -B analyze_native_partial_restore_once_group.py --session session-native-partial-restore-once-westd53005-20261008-r01 --output analysis-native-partial-restore-once-westd53005-20261008-r01.json
```

实际提交（2026-10-08T09:35:19.270419+00:00）：远端38payload/9controller/正常容量源SHA一致，新session此前不存在；唯一controller9990/SSH55292已进入同一公共锁2304:4312099778等待队列。6342持锁、7091在前，GPU9123占88288MiB；A session/receipt尚不存在，没有A CUDA或第二候卡。只接续现有handle，不能盲目重启；提交/候锁不是实验执行。

### 等待期间发现的贡献反证：stage与普通首输出保护同选
仅使用已完成shared-prefix-westd53005-r01原始状态：对每个决策取此前最近一次residency_admissions的output_count，得到“当前驻留期间尚无新输出”；其余合法suffix/private/phase/最小足量选择和once消费结构与partial probe一致。27个决策逐点及顺序once建议全同；3318候选的输出计数与决策时钟对齐，两条原residency_outputs=None按之前admission重建为0，并未漏partial。
首机会同为step1090：0063744在1088重新进入running时已有402输出，1090仍402，computed1592<1477+402−1=1878；二者都选0070086、101对100页。step1121两谓词也为真，但self/单候选，不存在替换；1093已新增1输出且恢复完成，两者都不再触发。这是同观测状态的动作等价诊断，不是另一策略执行结果。
**研究决策改变：** 本探针即使positive，也不再支持stage信号有增量贡献；先按普通最小服务保护的动作价值/工程适配归类。ABBA仍能区分净损失转移与有益动作，为是否保留此类保护作为强基线提供证据，不增第三个同动作GPU臂。候选实际轨迹产生后再核对其动作前是否仍等价；无信号则收束该动作，不追加保护阈值。没有新的、可区分强简单解释的证据，不自动开发所谓stage-aware完整方法。

可复现上述同选诊断：[diagnose_partial_restore_first_output.py](diagnose_partial_restore_first_output.py)，SHA `a3d0b8a40b6b4a4fa8e2d48fbedceab7880b374b803433faa1b9c52130fbfaee`；本地D执行`python3 -B diagnose_partial_restore_first_output.py`，只输出stdout、不生成第二结果JSON。支持`--session/--cell`，只在对应真实轨迹上比较动作建议，不输出另一策略的请求性能。
运行接续：09:49UTC实查9990已持同一公共锁，session实际RUNNING；09:51:32UTC tail-first320/320完成、exit0/archive VERIFIED，partial-restore-once-first初始化。未改冻结组/计划；整组结束前不作计时优劣判断。

## partial恢复单次ABBA完成：简单保护转移了重复抢占，未见完整净收益
[原始session](session-native-partial-restore-once-westd53005-20261008-r01/) / [唯一分析](analysis-native-partial-restore-once-westd53005-20261008-r01.json)：FOUR_CELLS_COMPLETE_COMPARABLE，四格全部320到达/提交/完成，失败/未完成0，输入/预热/容量/源码及原生动作核对通过。原normal容量profile来自同物理GPU，未用旧卡时延；funding均0。112raw本地与远端逐文件SHA相同，tar96,090,377B，SHA `3b132e61e7c923e16623ea26c6bdf487f4c84958e3d1db3480d7091d0d0d7080`。整组1027.143s，finished epoch1791453745.4362972；9990/SSH55292终态0，锁已交出，A无在途/候卡。导出97862、下载88394、分析95984均已终态0，原始/冻结包/压缩副本保留。

**实际执行。** 顺序tail/once/once/tail，真实改选0/1/1/0；两候选均step1090选0070086替0063744，原生preempt成功，实测释放101页，原tail当时即时可释放100页。不是严格同容量。原tail在候选中1090后不再被抢；它的下一输出从step1092提前到1090，决策后等待.221336→.071722s、.222249→.072522s。但替代者在1090、1093、1097被抢三次，下一输出等待约.224/.218/.514s（反序.226/.221/.512s）。总抢占27/26/26/27，只少一次；不是净消除三次恢复。

| 全请求描述量 | tail正序 | once正序 | once反序 | tail反序 |
|---|---:|---:|---:|---:|
| mean flow (s) | 62.739 | 63.149 | 63.094 | 63.091 |
| flow P50 / P95 (s) | 68.215 / 80.957 | 68.644 / 81.468 | 68.568 / 81.424 | 68.743 / 81.195 |
| TTFT P50 / P95 (s) | 3.840 / 20.467 | 3.956 / 20.718 | 3.940 / 20.664 | 3.927 / 20.618 |
| per-request max-gap P50 / P95 (s) | .178 / 4.404 | .170 / 4.425 | .172 / 4.422 | .190 / 4.481 |
| 最大全请求max-gap (s) | 24.721 | 24.842 | 24.777 | 25.035 |
| 输出吞吐 (token/s) | 3237.104 | 3218.556 | 3220.625 | 3228.990 |
| 完成吞吐 (req/s) | 3.663 | 3.642 | 3.645 | 3.654 |
| 持续 / 最后外部到达后排空 (s) | 87.355 / 84.164 | 87.858 / 84.668 | 87.802 / 84.611 | 87.574 / 84.384 |

两对candidate−tail的mean flow为 **+.410510 / +.003197s**，输出吞吐 **−.573% / −.259%**。未观察到整体净改善；仅两个运行级对照，不给显著性/置信区间，也不按1280请求当重复。无应用SLO，不事后挑诊断goodput阈值。每格282776输出，308 length/12 stop；但0049468、0061042两请求token序列均不同，不能声称质量等价。捕获未达截止；独立拒绝/单请求timeout计数仍UNKNOWN。发送滞后mean .01217/.01273/.01271/.01262s，最大均<.045s。

**损失与计时边界。** 原tail最终flow反而+.405434/+.128226s；替代者flow+.722470/+.429452s、最大生成间隔+.338030/+.328928s，是两对恶化最大的请求。正序320请求flow均更晚；反序171更晚/149更早。动作前21次抢占、231000条输出事件、2847条候选状态、host/物理/进度/原始arrival次序均匹配，无UNKNOWN；但动作墙钟差已为+.425628/−.121131s。两臂动作前都完成的80请求，mean flow差+.273164/−.019070s，证明全部墙钟差不能归因于改选。不能用前缀相减作因果校正。此组支持局部输出提前和重复抢占转移、未观测净收益，不精确估计全局机制造成的损失。

**开销与简单解释。** 外层选择+共同观察耗时.210921/.206050/.198825/.204859s；其中候选观察已包含.142945/.138707/.133540/.138480s，partial helper合计各约.00062–.00066s亦包含其中，不能相加。额外资源边界观察约.0121–.0124s，非隔离生产机制开销。四格各有1条测量JIT警告，非编译耗时。当前四轨迹的27/26/26/27个决策上，stage与普通首输出保护共106/106同选、UNKNOWN0；两个实际改选前21决策及1090提案也完全相同。因此不追加相同动作的GPU臂，撤下stage信号增量主张。完成一次预声明ABBA即收束本次简单保护动作，不追加阈值/动作位置/重复矩阵来找正值。

## 下一问题：相近释放量后的近期容量需求（仅已有数据诊断）
当前未知：合法近容量选择能否显著改变剩余运行集合的近期新增页需求，且差异能否持续到原生恢复重新占用之后？
主要竞争解释：共同running成员抵消，纯解码页相位差仅1–2页；或partial历史补齐/恢复重新准入带来可观测多页差；已知输出cap导致的完成归还属于强简单解释，不能混入新信号。
最小实验：不用GPU，用现有四条实际轨迹，在当前native suffix上比较等实际释放/最近释放层；固定k=1/4/8/16/32/64条件解码推进，核对当前持页冗余、cap/context边界，再连接真实preempt和residency时间线。不是策略回放性能或同状态分叉。
不同结果将如何改变机制选择、下一步投入或论文主张：若仅≤2页或快速恢复消耗余量，停止纯页相位评分；只有出现尚未被现有动作覆盖、决策前可分辨的多页持续差异，才设计有界真实干预。必要缺项仅补能区分该解释的原生调用差值，不直接建设预测器/追踪系统。

**已得到的量级。** 当前single_type_kv_cache_manager.py（SHA bcb27e38…ba8）169–191对已有running执行`max(ceil(num_tokens/16)-held,0)`；固定KV manager429–435中无spec lookahead，本域FullAttention无滑窗。条件为每请求共同推进k个新token、尚未完成/退出或新增准入、未越context时，`g_i(k)=max(0,ceil((computed_i+k)/16)-held_i)`；两个victim选择的共同成员需求抵消。13220条pure-decode候选观测全部`held=ceil(computed/16)`，无额外预留。因此纯解码两选择需求差最多1页；这不是实际能运行k个scheduler step的保证。虽然本组max running277低于token budget1024，决策日志仅有未处理suffix，缺此前scheduled prefix的阶段与逐请求scheduled-token映射，不能证明共同成员在真实每步都获得1个decode token；恢复/预填充可能消耗预算。这个条件界不能外推下一次抢占的实际时间，也不能排除其他调度条件下1页引发阈值效应。
每格等实际释放/private/pure层19对、12个决策；k=1的Δ(-1,0,+1)页计数(1,18,0)，k=4为(3,13,3)，k=8为(2,12,5)，k=16/32/64全部0。最近释放层42对、25决策也≤1页，最近足量策略的额外释放最多11页，容量差反而更大。例step855均释放188页、computed2998/3005，4/8步仅差1页。所有比较未越声明cap/context，UNKNOWN0；未将未来EOS当已知。
唯一近容量多页对仍是已测1090：保留partial0063744、改抢0070086，即时仅多释放1页，却留下相对[18,17,17,18,18,18]页的条件历史补齐需求。1121还有31页历史差但singleton，无合法替代。same-held共享层9对/6决策，替代实际释放少72–150页，不属于等真实释放。此处不重跑退休大小排序，也不为≤1页信号开发复杂预测。
复现：[diagnose_native_forward_page_demand.py](diagnose_native_forward_page_demand.py)，SHA `604c5a513ffab4e4e7a517ac18e03691278c5a30209d0b0977d7796460f204d5`；D下`python3 -B diagnose_native_forward_page_demand.py`，只输出stdout，无第二结果JSON。

**恢复重占的真实时间线。** tail臂1090 free0→100；1092恢复进入running时持118页；1093再抢时free0→118、只新增1输出；1094重入118页；1097再次被抢；1102重入109页。候选臂1090 free0→101；1092重入101页；1093再抢free0→101、同样仅新增1输出；1095重入101页；1097再次被抢；1102重入92页。两次重复的step/持页/进度一致。1090→1093约.2274/.2280s对.2301/.2311s，重入后约79ms再抢，原余量已净耗尽。
但1091、1093也有共同请求完成归还；native新准入还可能触碰free cached blocks及partial prefix-hit CoW（manager217–220），不能把“重入持页”当freepool新增扣减，更不能将一臂未来轨迹借给另一臂。现有日志只有schedule-return全程极值，无逐步free/running/queued；精确分拆恢复重占、其他增长、完成归还仍UNKNOWN。若以后有具体多页差需要验证，缺的最小字段是allocation前后free（分running/resume）与完成free前后差值。当前不因这个统计缺口自动追加GPU观察组；已足以收束纯解码页相位预测，并指出恢复重新占用是后续原则必须包含的项。整个victim问题尚未被否定，论文中心贡献仍未成立。


## 新机会：严格同释放量的host差异，先检验是否实际兑现
当前未知：step994新发现的205对205页、相同204个computed整页、host连续ready16对166页，是否在真实恢复中仍形成更少重算/停顿，还是GPU前缀重叠或host淘汰让差异失效？
主要竞争解释：host可用差异降低必要恢复工作；或差异未实际使用、较快重新占用吸收局部收益、改抢更早/更接近声明cap的请求只转移损失。原native抢占prepend waiting，恢复append running，原arrival不单独决定下一次抢占。
最小实验：冻结原320共享输入和本机正常容量，tail/strict-equal-release-host-once/once/tail一组ABBA；首个private pure-decode、pending0、相同实释/相同computed整页且host missing严格改善的合法动作只替换一次，其他机制不变。两臂共同记录原生lookup的local/external返回和成功分配后的external请求/对应load完成，避免把host前缀或held当实际命中。
不同结果将如何改变机制选择、下一步投入或论文主张：命中未兑现即收束该预测量；兑现但无完整净收益即收束本次动作，不调host阈值；一致净信号才做结构匹配无host信号、最大实际释放量及独立输入。最多本次ABBA，仍为动作价值开发探针，非独立确认或新颖性证据。

该新证据存在于刚完成的四条轨迹中：原tail0062890/index252、alt0035191/index210，均held/release205、shared0、pending0、phase off；computed3270/3272、host ready16/166、missing188/38、已输出485/592、剩余声明cap539/432。每格19个等释放pair中只有这一对替代missing更低，16对更高、2对相同。旧private域严格同容量host规则没有动作，旧host-near则多释放量/多次替换明显；本次新机会解除该混淆，不能据此复活其完整策略或抹去旧负结果。规则不用请求ID、step或未来EOS。

tail实际在994释放205，1033以held22重新进入running，1037才有下一输出；但已有offload-events明确NOT_MEASURED，不能把held22解释成host命中、GPU命中或新增分配。原生host lookup从GPU local prefix之后开始（offloading_scheduler.py539–618、720–773），因此独立host prefix并非实际可用增量。本组新增标量观察仅用于这个具体可信性风险，不打开全量memory/worker tracing。

另纠正范围：max-remaining与tail同选是overlap域34/34，不是全部旧mixed域；旧mixed有3/40分歧，其中已测arrival681，另706/712为短tail剩66/58对等held长请求剩490/895且host代价不同。预算按ID hash，未按arrival排序；overlap晚到7短中6个首次进入running晚于最后压力。当前不再改到达率/预算来制造机会，也不复活≤16到64的完成保护阈值搜索。


### 严格同释放量探针已冻结部署，唯一候锁
新包 [candidate_native_equal_release_host_once_r01](candidate_native_equal_release_host_once_r01/) 只改变3/38 payload：staged选择、runner及两臂共有的恢复标量观察。原native preempt/free、Q1/target/trigger/传输/准入不变；3项选择/once/native调用及3项观察透明性检查通过。既有四条raw重算均唯一提案step994，不是新GPU执行。manifest `1288e3844bf17b024e4c48d17160bc905784995a0c6c179753005a3e11899a49`；tar7,521,280B SHA `011ac82a140caba5b17f9db6efc30f91a5911904390134e403a047e1cff6d25c`；[计划](plan-native-equal-release-host-once-westd53005-20261008-r01.json) SHA `d77fb68965871c7422a2f299aa48b0c5710080c735c68bafefd359b31ac2e739`。
同320共享输入/正常77,076,627,456B/host16GiB/预热及私有Triton种子，tail/once/once/tail；3GiB入口、1200s/cell、4800s整组、一次3600s共同锁等待不变。lookup offer、allocation请求的external量及worker acknowledgement分开记录，不当作复制暴露时间。无费用遥测，不声称共享余额。
空间处理只涉及旧partial终态组：本地112raw及remote output/archive逐文件核验后，删除重复output与runtime/preflight缓存，root1,545,261,056→2,801,848,320B；再次核验完整remote tgz及其8个raw/warmup-1条目后删除对应展开副本，2,573,791,232→3,449,593,856B（间隔有共享写入）。所有localraw及完整remote tgz保留，其他remote事件/指标/receipt/hash/源码仍在；展开raw可从tgz恢复。部署后root3,362,263,040B。不删除原始证据，不修改他线或降低门槛。
```sh
# 远端A source目录，唯一任务已提交；不得重复启动。
LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib /root/miniconda3/bin/python -B -u run_native_equal_release_host_once_group_seeded.py --plan plan-native-equal-release-host-once-westd53005-20261008-r01.json
# 本地D针对性CPU检查，不是GPU结果。
PYTHONDONTWRITEBYTECODE=1 python3 candidate_native_equal_release_host_once_r01/test_equal_release_host_once_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -q test_native_recovery_scalars_cpu
```
38payload/9controller/来源profile均部署核验；epoch1791456906.7426965唯一19113/SSH69680存活，wchan=locks_lock_inode_wait，同一2304:4312099778锁由17172持有、19038在前。新session不存在，A无CUDA/第二候卡；只接续该handle，不自动重试。

[唯一分析入口](analyze_native_equal_release_host_once_group.py) SHA `25c1bf8af35bd6c834a8e17b2420fedc3556ac41731688d79a667e0d79ed6004` 已准备：复用全部请求/ABBA、容量与原生free关联，新增按request+preemption ordinal对齐的lookup/allocated/load acknowledgement。两个新schema/once fixture通过；缺失保留UNKNOWN，尚未生成科学结果JSON。全组终态且锁释放后，本地D执行：
```sh
python3 -B analyze_native_equal_release_host_once_group.py --session session-native-equal-release-host-once-westd53005-20261008-r01 --output analysis-native-equal-release-host-once-westd53005-20261008-r01.json
```
查新边界：[CacheOPT §3.4–3.5](https://arxiv.org/html/2503.13773v1)已有SLO/预测剩余量/KV分层victim和profiling选择swap/recompute；[FastSwitch §3.3](https://arxiv.org/html/2411.18424v1)已有有效host副本片段及增量交换；[TOPAS §IV-B–E](https://arxiv.org/html/2608.25523v1)在联合容量决策中计移动/重建/撤销成本。核对这些主文未见当前等实际释放+native GPU后host增量复用的完整局部规则，但不足以宣称新颖；普通成本感知和有效副本复用本身不是贡献。若有净信号，必须先面对同资格/动作结构的无host基线、最大真实释放量及计入GPU与host重叠的重建工作代理；代理仍须用真实恢复验证。


### r01资源失败已保留；r02在资源恢复后启动
r01在epoch1791457883.0265722（19:11:23 CST）真实终态ABORTED，持锁后5.336s因`Insufficient new-host disk space`在CUDA前退出，cells=[]；19113消失、SSH69680退出1并释放锁。随后实查root2,567,602,176B。完整plan/receipt/gpu-start/controller日志已本地保留在[r01失败目录](session-native-equal-release-host-once-westd53005-20261008-r01/)，这是资源失败，0到达/0策略执行，不是机制负结果。
仅A旧arrival-once-r02的8个expanded raw/warmup-1，经localraw、remote archive与remote完整tar逐文件SHA一致后回收，root3,219,148,800→4,091,887,616B（epoch1791458166.11513）；local所有raw及remote完整压缩原始、其他事件/指标/receipt/source保留。没有再调整门槛或清理他线。
独立[r02计划](plan-native-equal-release-host-once-westd53005-20261008-r02.json) SHA `8ae5acc69b391e82ec3c94d96ac6382dd63aefe9574e997a7338f9b4d17a6100` 只改session_dir与resource_revision_reason；代码manifest1288e384…99a49、输入/容量/预热/3GiB/时间上限及controller均原样。epoch1791458395.3885152只提交一次，实查root3,999,600,640B，唯一22623/SSH94126；等锁19.526s后获同一2304:4312099778锁。epoch1791458456.6542153，session RUNNING、首tail-first启动，无第二A任务。r02命令仍原entry，将plan/session/output后缀改r02；绝不重跑已有r01目录。r01无科学单元，r02仍为预声明唯一科学ABBA。

### 正式渠道核对，不代表已可投稿
官方[CCF第七版目录发布页](https://www.ccf.org.cn/Academic_Evaluation/By_category/)及[正式PDF](https://www.ccf.org.cn/ccf/contentcore/resource/download?ID=112CF3BF7E1140ACEB271ADAED12A67ADFABB8FF099E40C2759502A85C8A281F)将The Journal of Supercomputing列入C类。[期刊范围](https://link.springer.com/journal/11227)包含系统、算法及性能度量，故可作为本线系统研究的正式期刊候选；这是范围匹配判断，中心贡献与证据仍未成立。[官方投稿指南](https://link.springer.com/journal/11227/submission-guidelines)提供普通稿件在线入口，未列统一截止日，可按持续投稿安排；需要可编辑源文件及数据可用性声明。未投稿、不承诺录用，也不因有候选渠道而提前写可投稿结论。

## 严格同释放量host单次ABBA：有真实复用，完整收益方向翻转
[原始session](session-native-equal-release-host-once-westd53005-20261008-r02/) / [唯一分析](analysis-native-equal-release-host-once-westd53005-20261008-r02.json)已完成，状态FOUR_CELLS_COMPLETE_COMPARABLE。四格均320到达/提交/完成，失败与未完成0，funding0；模型、精度、正常容量、外部输入、预热、源码和私有Triton种子可比。112原始文件逐一SHA核验，完整tar96,189,990B，SHA `a4328cf2caa4e23447243ac6d879cb2c88fffd284b283009d1272aefa94cbdae`。finished epoch1791459409.733563；控制器22623已退出、共同锁已释放，无A在途/候卡。原观察SSH中断后先读远端终态，没有重跑。

**执行与容量。** tail/once/once/tail真实改选0/1/1/0，原生抢占27/26/26/27。两候选均step994以0035191替0062890，原生preempt成功且free0→205；基线同点也是0→205，容量混淆消除。原tail在候选中多输出13 tokens后于1007仍被抢；两参与者各自被抢后未再次被抢。原tail的停顿从3.431/3.405s变为2.435/2.465s，但替代者新增停顿3.820/3.876s。全局只少一次抢占，不能把保护原tail等同消除其恢复。

**host解释被实际调用收窄。** 0035191动作前ready166页，首次lookup GPU local2784 tokens=174页，external offer0，说明host与GPU覆盖重叠。后续多次offer没有成功allocation，不能累加；最终成功allocation为local0、external640 tokens=40页，job14237的worker ack1/pending1并完成移除。tail臂0062890最终local0、external0，没有load job；候选臂稍后被抢的0062890最终external256 tokens=16页，job14223确认完成。因此记录支持40页真实host请求与完成，**不支持150页增量节省或166页全复用**。确认时间不等于暴露传输时延，held46等持页状态也不是命中/新增分配量。

| 全请求描述量 | tail正序 | once正序 | once反序 | tail反序 |
|---|---:|---:|---:|---:|
| mean flow (s) | 62.995 | 62.945 | 63.059 | 62.854 |
| flow P50 / P95 (s) | 68.464 / 81.275 | 68.412 / 81.049 | 68.650 / 81.114 | 68.336 / 81.055 |
| TTFT P50 / P95 (s) | 3.939 / 20.665 | 3.972 / 20.653 | 3.934 / 20.559 | 3.952 / 20.648 |
| per-request max-gap P50 / P95 (s) | .170 / 4.406 | .177 / 3.913 | .180 / 3.971 | .178 / 4.374 |
| 最大全请求max-gap (s) | 24.727 | 24.695 | 25.135 | 24.643 |
| 输出吞吐 (token/s) | 3225.107 | 3227.849 | 3225.763 | 3232.937 |
| 完成吞吐 (req/s) | 3.650 | 3.653 | 3.650 | 3.659 |
| 持续 / 最后外部到达后排空 (s) | 87.680 / 84.489 | 87.605 / 84.415 | 87.662 / 84.472 | 87.467 / 84.277 |

两对mean flow差 **−.049757/+.204894s**、吞吐 **+.085%/−.222%**，运行级n=2，不估显著性或把请求当重复。max-gap P95改善.493/.403s，但268/299请求的各自max-gap增加；flow更晚29/268请求。0035191两次都是最大个体损失：flow+1.130003/+1.426079s、max-gap+3.650392/+3.698289s；0062890 flow−.255742/+.027924s、max-gap−.996442/−.940270s。全部请求的原始分布与损失保留，不用P95局部改善代替总体收益。

动作前17次实际抢占、207713条输出事件、2399条候选状态与18份arrival排序快照匹配，physical/progress UNKNOWN0；仍不是完整隐藏状态同一或墙钟同一。动作时墙钟差−.021441/+.067249s；此前双方已完成的50请求mean flow差+.008141/−.049229s，不能相减作因果校正。所有请求首次输出都早于26.736s，动作约57.4s，故本组全部TTFT差异发生在干预前。四臂均282776输出、308 length/12 stop，但0024485与0061042的token序列均改变；不声称恢复工作量或质量等价。未触达capture截止；独立拒绝/单请求timeout仍NOT_SEPARATELY_RECORDED，0失败/未完成来自完整请求记录。发送滞后mean .01219/.01281/.01261/.01275s，max均<.048s。

外层选择和共同观察耗时.199511/.204530/.213493/.208262s，已包含候选观察.131218/.135842/.142435/.137823s及helper .002896/.002982/.003164/.003048s，不能相加。原生恢复标量observer body .005637/.005618/.005871/.005742s，排除原调用但未隔离wrapper成本；不是生产机制单独开销。四格各有1条测量JIT警告，不等于已测编译时间。原始lookup/allocated/ack分别保存，不把重复offer或重叠工作当节省。

**决定：** 按冻结判别规则收束此动作与当前host前缀代理：实际复用只部分兑现，平均完成时间/吞吐符号翻转，同时稳定转移个体停顿。不给旧host-near改名，不追加本规则阈值/模型/重复矩阵；保留等释放量会改变损失分配的证据，暂不升级为优化原则或测量论文。

```sh
# 已完成，复现时只用全新session_dir，不能覆盖或重复提交本组。
LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib /root/miniconda3/bin/python -B -u run_native_equal_release_host_once_group_seeded.py --plan plan-native-equal-release-host-once-westd53005-20261008-r02.json
# 已生成的唯一分析命令；重分析必须另给新output。
python3 -B analyze_native_equal_release_host_once_group.py --session session-native-equal-release-host-once-westd53005-20261008-r02 --output analysis-native-equal-release-host-once-westd53005-20261008-r02.json
```

当前未知：相同实际释放后，已声明生成预算所限定的较早完成，能否让剩余集合获得持续容量，而非再次被原生恢复吸收？
主要竞争解释：原tail已等价于最晚完成优先被抢；或有短tail被恢复顺序推到尾部，保留它比host复用更重要；后者首先属于强简单剩余量基线，并非新贡献。
最小实验：先仅用已有旧mixed真实step706/712及生命周期核对两个已发现但未执行的等容量短/长候选，确认是否已被旧动作覆盖、短请求在原生恢复中是否真能结束。无GPU、无新阈值、无到达/预算改造。
不同结果将如何改变机制选择、下一步投入或论文主张：若已有同动作失败或只有未兑现的未来完成假设，停止该入口；若存在未测的合法同容量动作与明确完成期限差，优先以完整简单规则检验动作价值，若它足够则撤销额外信号创新，而不先设计预测器。

**已有数据已回答机会问题。** 旧mixed tail与arrival ABBA共5条真实轨迹各40个决策，完整等容量预算影子均只在706/712改选：0049936→0064011，100页/99computed整页，remaining66→490、host missing4→99；0052099→0026292，136页/135computed整页，remaining58→895、missing4→9。两处均private/pure/qualified/pending0、phase关闭；838/847有pending STORE而回tail，不遮蔽这两个机会。arrival-once真实只改681且多释放66页，706/712从未执行替换。两短tail基线只抢一次，随后约30.003/29.420s最大停顿，1183/1173步到cap完成；不是预知未来EOS，也不能把30s都当反事实收益。
旧字段来源限定：这是合同补齐的影子诊断，不是旧rows原样的新字段完整重放。prompt取同一raw请求元数据；RUNNING由旧源码qualified条件推出；release=held/shared=0依赖已记录的prefix-off、单组FullAttention合同，旧行没有未执行候选的refcount快照。冻结新helper对旧rows原样均UNKNOWN_PHYSICAL_RELEASE，仅补物理合同仍UNKNOWN_PROGRESS；按上述来源补齐后五轨迹仍首706、仅706/712改变。原策略40/40实际释放等于held也不能替代未选候选的物理观测。新GPU组已共用逐块观察和实际free关联，不需要为旧字段另占GPU。

### 本阶段选择：完整等容量预算基线（已运行；设计记录保留）
当前未知：保持同即时容量的完整简单预算排序，能否通过保留较早完成请求降低全部请求mean flow，而非把同量停顿转移到长victim？
主要竞争解释：完成归还能持续缓解容量压力；或仍很快被恢复/新准入吸收，并以长victim更高恢复代价抵消。706同时增加95页host missing，不跳过它事后只挑712。
最小实验：原320独立文章mixed输入原样、当前卡正常物理KV预算，tail/budget/budget/tail一次ABBA；在线每次在同actual-release+computed整页层选最大声明remaining cap、同分tail，保护和未知回退固定，所有后继轨迹继续到排空。没有近完成阈值、动作位置触发或只保护一次。
不同结果将如何改变决定：若减少整体等待且副作用可接受，先认定简单基线价值并查剩余空间，不宣称新贡献；若只转移损失/无实际动作/主效应不确定，分别按执行证据限定收束，不扫阈值/输入找正值。冻结主目标是全到达mean flow，报告吞吐/gap/个体代价取舍，不要求全指标同时改善。

资源上限：仅这一组4×320，预计整卡约17–22分钟（按近期同栈4格耗时估计），硬上限1200s/格、4800s/组、一次最多3600s公共锁等待，捕获每格最多600s；不自动追加组或直到显著。已有GPU证据足够回答的阶段/host规则不再运行。继续使用同主机锁2304:4312099778，完成即释放；无费用遥测，未知共享余额。为恢复磁盘，仅在本地raw与远端完整tgz/112输出核对后回收已完成exact-host组重复output、runtime/preflight缓存及8份expanded raw/warmup-1，所有本地原始和remote tgz/其他事件/receipt/source保留；实际free2,507,640,832→4,640,563,200B，无他线变更。

新包[candidate_native_equal_release_budget_r01](candidate_native_equal_release_budget_r01/)已冻结：38 payload中只改2份规则/runner代码，3份high输入恢复为已存在mixed包的原字节；其他原生生命周期、Q1、传输/准入及共有观察不改。3项针对性CPU检查通过（重复选择、未知/保护/容量回退、native suffix pop/preempt生命周期），不是GPU结果。manifest `e80bdcb3c7fa52697f0751c85169568f0dcdef4059957bda3e805bf3b505b72a`，staged SHA `f057bac8e53544f2a0c76d626aecfddcb6a7e7f6d4c2578a5c439856454d7fb5`；[计划](plan-native-equal-release-budget-westd53005-20261008-r01.json) SHA `b38fc8a9c44dfe386c4e3c2b1827b3f4aca103d787a0c9896ddd1cf3ae87195a`。无可调排序参数，不给此基线额外搜索预算。
本机来源profile的prefix=True，此组原mixed输入prefix=False；显式复用同物理KV字节预算，并非声称重新测得prefix-off正常profile。组内两臂相同；分析仅允许这个声明的engine差异，仍核对布局/native源码/UUID/其余参数。部署核验38payload、9controller及来源profile通过，root4,234,731,520B，共同锁身份不变。
分析入口[analyze_native_equal_release_budget_group.py](analyze_native_equal_release_budget_group.py)，SHA `9bc16a9d3c4c0280e6348c560c016b06b01375b373c515e92fed3d55caf90c47`；两项schema检查通过，区分多次实际执行与首动作前缀、缺失字段不记0。没有新增科学JSON。epoch1791464778.093184唯一提交29913/SSH99005，实查root4,160,475,136B；等待23.160s后取得同一公共锁，已有handoff确认上一context退出，21:06:49 CST实查session RUNNING。只接续该任务，未有完整策略结果。

```sh
# 远端A source；唯一新组入口，不覆盖历史目录。
LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib /root/miniconda3/bin/python -B -u run_native_equal_release_budget_group_seeded.py --plan plan-native-equal-release-budget-westd53005-20261008-r01.json
# 全组终态后本地D唯一分析；已有output不能覆盖。
python3 -B analyze_native_equal_release_budget_group.py --session session-native-equal-release-budget-westd53005-20261008-r01 --output analysis-native-equal-release-budget-westd53005-20261008-r01.json
```

### 完成：严格等容量预算规则零干预，当前运行域收束

新证据来自[本组原始目录](session-native-equal-release-budget-westd53005-20261008-r01/)及[纠错后唯一有效分析](analysis-native-equal-release-budget-westd53005-20261008-r01-r02.json)。四格均320/320完成、失败/未完成0，均282868 tokens、309 length/11 stop；2191 engine calls及全部282868条无时钟输出事件（step/request/cumulative/new tokens）相同，所有请求token序列和结束原因相同。自然EOS只表示允许提前结束，不能外推文章任务质量。每格41次实际抢占、0 proposal/0请求改选/0实际改选，预测释放与真实free增量41/41匹配；没有被候选救下的请求。

**零动作已定位。** 每格24次无剩余预算改善、15次无等释放/整页候选、2次tail pending STORE，非UNKNOWN或phase覆盖。当前可用36752页，比旧36780少28页，压力轨迹已变：新704的短tail0049936释放100页、剩68，旧替代0064011已不在未处理suffix；新710的0052099释放136页、剩60，旧替代0026292释放135页且pending STORE=1。不能把旧706/712影子称本组动作。两短请求实际最大停顿仍约30.4–30.5/29.9–30.0s，均只抢一次，随后到128 cap完成；本组没有反事实提前归还容量证据。

| 全请求描述量 | tail正序 | budget正序 | budget反序 | tail反序 |
|---|---:|---:|---:|---:|
| mean flow (s) | 68.966193 | 69.001822 | 68.954677 | 69.150748 |
| flow P95 (s) | 94.648080 | 94.678730 | 94.568068 | 94.691705 |
| TTFT P95 (s) | 63.513492 | 63.644188 | 63.551309 | 63.850153 |
| per-request max-gap P95 (s) | 20.751141 | 20.702268 | 20.634689 | 20.691508 |
| 最大max-gap (s) | 33.067649 | 32.980713 | 32.876408 | 32.820996 |
| 输出吞吐 (token/s) | 2852.890364 | 2852.066478 | 2855.128076 | 2851.939520 |
| 持续 / 最后到达后排空 (s) | 99.151374 / 95.961243 | 99.180016 / 95.989890 | 99.073664 / 95.883541 | 99.184432 / 95.994294 |

两对mean flow差+.035629/−.196071s、吞吐−.028879%/+.111803%；晚完成319/23请求，早完成1/297。0053423正序晚149.3ms、反序早391.4ms，不能归为稳定受损者；tail自身反序mean flow漂移+184.6ms。**全部只是时钟差异，不能归因为victim选择收益、代价转移或模型失败。** 统计单位两个运行对，不用1280请求当重复，也不对继承的goodput网格挑阈值。

计费与口径：外层selector/共同观察总墙时.321351/.324823/.328238/.316730s，内含观察.211896/.215241/.216569/.207932s、预算helper .005262/.005284/.005398/.004844s，不能相加；两臂均有相同诊断开销，不声称生产单独开销。发送滞后均值.01214/.01202/.01215/.01248s、最大<.030s；所有请求进入引擎且完成，未触截止；独立拒绝/单请求timeout仍未单独计数。四格各1条测量JIT警告，不将警告数换算编译时间。

整组1041.878s，21:24:03 CST释放公共锁，无A新任务/候卡。112原始文件本地核验；完整归档96,146,837B，SHA `e9ab6dbb256a66fcdf671afdf7263a5e098bb8bc8580187854eb3b393151e586`，远端同名tgz及本地raw保留。原[分析r01](analysis-native-equal-release-budget-westd53005-20261008-r01.json)误把prefix-off的KVCacheCoordinatorNoPrefixCache要求为Unitary类型，因此状态为PARTIAL_FAILED_OR_NONCOMPARABLE；它是分析器错误，不是GPU无效。只纠正该类型和增加纠错说明，r02 SHA `4dd8c2d0132b2354ed62524261cd8d740f1245f549d4173f1aa6a98457fe33dd`，其余检查/指标/冻结代码与raw未改，两份均保留。

```sh
# 本组已完成；此命令记录纠错分析来源，已有输出不能覆盖。
python3 -B analyze_native_equal_release_budget_group_r02.py --session session-native-equal-release-budget-westd53005-20261008-r01 --output analysis-native-equal-release-budget-westd53005-20261008-r01-r02.json
```

**判决：** 收束严格等释放＋computed整页匹配的预算规则在当前运行域的尝试。完成归还假设未被实际检验到；不能把零动作当其负结果。不开±1页阈值搜索、不恢复旧28页制造动作、不增加同类模型/重复，也不把严格匹配永久当完整系统方法的限制。

当前未知：完整native合法集合中的普通剩余量/最大实际释放基线是否仍有未测试的有意义动作，还是已被历史失败覆盖？
主要竞争解释：严格匹配人为排除了简单方法的主要机会；或原tail已近似这些规则，剩余差异只是已失败size排序。
最小行动：只用本组41个tail前态计算两条无阈值影子并定向核对旧size/oldest规则，CPU分析上限本轮，不建新日志或GPU任务。
不同结果将如何改变决定：若无新动作或已被相同规则覆盖，停止当前域的选择器投入；若确有不同的简单基线机会，先完整执行该基线检验可控空间，收益先归已有原则，不发展额外预测器。

**CPU判别已完成，存在一个未测基线。** 完整native suffix的最大remaining影子4/41异选、UNKNOWN0：673/679为0065237→0053843，释放84→149/150、remaining1007→1020/1005→1014；704为0049936→0061042，释放100→67、remaining68→959；710为0052099→0061042，释放136→67、remaining60→953。四处pending均0；最后两处host missing均4→4，是真正预算异质性，非旧arrival-once覆盖。39/41建议与latest-arrival/least-generated同选，仅704/710不同。这是当前前态的建议，不保证在线仍4次，也不估计反事实收益。最大release则41/41改选，全部与max-held/max-computed同序，额外释放14–177页，无独立信号。

历史覆盖需准确限定：[旧size规则](../20260929_commit_recheck/candidate_native_victim_size_r01/pkg/staged_store_rotation.py)只允许非current、私有pure-decode且单请求足量的victim；5090/4096页/maxseq32/ordinary backfill下min/max真实47/28次改选，mean flow37.785/37.876s对tail37.375s、最大gap10.723/10.244s对8.490s。它不是当前正常容量完整suffix的最大实际释放评估。故不把maxrelease改名当创新，也不声称它已被本机强对照击败；若普通预算策略获得支持，它仍是必须面对的强简单基线。oldest强基线改变恢复目标优先级及funding，不是上述普通victim排序，其旧gap主目标收益不能替代当前mean-flow证据。

### 下一最小实验：完整native suffix的普通剩余预算基线
当前未知：去掉非原生的等容量/纯decode等限制后，普通剩余预算排序能否减少全请求完成等待？
主要竞争解释：保留短预算请求能较早完成归还KV；或少释放33/69页会触发更多抢占，长victim恢复和批次改变抵消收益。
最小实验：同原mixed320、当前正常物理KV预算和所有其他机制，tail/remaining/remaining/tail一次ABBA；每次合法suffix取最大cap−output，同分latest，保留原phase/protected及预算UNKNOWN回退。没有新阈值、等容量要求、动作位置或未来EOS。
不同结果将如何改变决定：若主目标支持改善，先归普通剩余量原则，再测最大实际释放等最强简单解释与剩余损失；若执行而无支持，停止本域的预算信号，不放宽/组合预测器；若在预算内不确定就如实暂存，不自动重复到显著。0动作仅允许入口诊断，不扩矩阵。

阶段上限仍为1组4×320、预计整卡17–22分钟，1200s/格、4800s/组、600s捕获、最多一次3600s公共锁等待；无费用遥测/已知共享余额。新原型只改native victim排序，允许其后继多次抢占和不同轨迹，完整计入全部请求；明确这是已有调度原则的简单组件适配，不是新的同容量机制或CacheOPT完整复现。共享锁忙时仅准备代码，不启动并行计时进程。

已冻结并部署[candidate_native_remaining_budget_r01](candidate_native_remaining_budget_r01/)：manifest `45c5fd771017851428202f6b435e3ef6b5cde38cc854514aee5b78e754f37cd4`、staged `46607b53e7605833a766350b635fbe1483dc4f8bea690ad981ec8c21f9748c6a`；相对parent仅规则/runner两份payload改变，mixed输入全部原字节。单项综合CPU检查涵盖current/nonself、旧qualified未知、partial/pending/少释放候选及原生pop/free/preempt，已通过；不称GPU结果。新[计划](plan-native-remaining-budget-westd53005-20261008-r01.json) SHA `b10cc59d48d30b3e0f1c0c945d4d2342fc8511d881e2aa1675c2f995404af206`，entry SHA `ff6556e057247f1821bbe1c09e1b0009a483ec3062ccc4d1fd4703913c9d100a`。分析使用薄适配[analyze_native_remaining_budget_group.py](analyze_native_remaining_budget_group.py)，复用既有全请求统计与prefix-off容量合同，仅替换规则资格和执行重放。

资源准备仅回收刚完成strict-budget的重复output、缓存及8份expanded raw/warmup-1，逐份核对本地原始与保留的完整远端tgz后执行；全部本地raw/远端压缩归档及其他事件/receipt/source保留。root free1,686,089,728→3,821,350,912B，未动他线数据、未降低3GiB门槛；部署时3,799,678,976B，38payload/9controller/来源profile/公共锁2304:4312099778核验通过。

```sh
# 远端A source；只提交一次，session已存在则必须先查现有任务，不能重启。
LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib /root/miniconda3/bin/python -B -u run_native_remaining_budget_group_seeded.py --plan plan-native-remaining-budget-westd53005-20261008-r01.json
# 整组终态后；输出使用全新文件名，不能覆盖。
python3 -B analyze_native_remaining_budget_group.py --session session-native-remaining-budget-westd53005-20261008-r01 --output analysis-native-remaining-budget-westd53005-20261008-r01.json
```

资源失败保留：r01唯一38116/SSH50235候锁684.138s后取得公共锁，22:09:24 CST在GPU交接确认后、模型初始化前因根盘低于原3GiB门槛ABORTED，cells=[]；进程/SSH已终态退出1并释放锁。receipt、plan、GPU边界和完整错误日志已回收本地，**0实验单元，未检验策略，不是负结果**。没有重复候卡或自动重启。

修复仅涉及存储：六个A已终态session的36份重复archive/运行副本，在与完整本地raw/source及远端tar逐份一致后回收；全部原始、远端完整tgz、原冻结包、来源profile archive、plan/receipt/hash/log仍保留。他线未改。root实际free2,754,224,128→3,994,120,192B；不降低门槛。新[资源重试计划r02](plan-native-remaining-budget-westd53005-20261008-r02.json) SHA `52b1b6ad622aab8c2a930424fb3bdd97f7669ecfeff1d24285e570d46b49e360` 仅改变session_dir并增加资源说明，代码/输入/容量/口径/顺序均原冻结值，仍最多一组真正ABBA。

**等待中的条件模型检查（CPU，非服务实验）。** 在既有strict-budget tail-first的704/710前态，假设改选0061042释放67页、保留其余已记录suffix，并让这些pure/private请求共同推进k token、无自然EOS/准入/恢复/额外抢占，触及声明cap即归还全部KV。两快照分别有48/93个保留请求，均满足held=ceil(computed/16)、UNKNOWN0。令r=cap−output，g(k)=ceil((computed+min(k,r))/16)−held；该子集合的累计净页增长为Σg(k)−Σ[r≤k](held+g(r))。这只是在明确条件下的容量记账，k不是scheduler步或墙钟，两个快照未拼成候选策略轨迹。

| 原tail快照 | k=1净增长 | k=16净增长 | 到短tail cap时：新增−归还=净增长 | 首次超过释放67页 | 完成归还后的前缀需求峰值 |
|---|---:|---:|---:|---|---|
| 704：0049936剩68 | 8 | 48 | k68：206−244=−38 | k22：68页（k21为65） | k65：200页 |
| 710：0052099剩60 | 8 | 93 | k60：350−140=210 | k12：71页（k11为66） | k59：345页 |

704的首个归还来自0052099于k66释放140页，随后0049936于k68释放104页；若先分配再处理该轮完成，途中峰值202页。710的完成前需求350页，完成后才归还140页。**终点净增长为负也不证明途中可达**：在这些条件下，单次67页连已记录子集合都不足以持续到短tail完成。但未记录running前缀的完成归还、非均匀调度、自然EOS会改变实际路径，不能把差额换算为真实必须牺牲多少长请求或预测全服务收益；原生恢复重占还需真实轨迹。此结果只指导本组解释：若短请求被保留，必须同时报告为其完成前发生的全部追加抢占及全请求代价，不能以其最终释放量宣称容量已持续获益。

### 完成：普通预算保护有稳定个体收益，完整服务净收益不确定

原始数据：[remaining-budget r02](session-native-remaining-budget-westd53005-20261008-r02/)；唯一分析：[analysis-native-remaining-budget-westd53005-20261008-r02.json](analysis-native-remaining-budget-westd53005-20261008-r02.json)，状态FOUR_CELLS_COMPLETE_COMPARABLE。四格均320/320完成、失败/未完成0、282868 tokens、309 length/11 stop；逐请求长度与终止原因完全一致，但两对均有同一批19个请求token序列改变，同策略两次序列相同。三个重点请求0049936/0052099/0061042序列未变。只能排除总输出量变化解释，不能声称计算工作量或任务质量等价。

| 全请求描述量 | tail正序 | remaining正序 | remaining反序 | tail反序 |
|---|---:|---:|---:|---:|
| mean flow (s，预声明主目标) | 68.994035 | 68.629547 | 68.833842 | 68.772398 |
| flow P95 (s) | 94.643890 | 94.669307 | 94.970743 | 94.336394 |
| TTFT mean / P95 (s) | 13.370485 / 63.602153 | 13.283827 / 63.101654 | 13.316923 / 63.269748 | 13.314590 / 63.369962 |
| per-request max-gap mean (s) | 2.259460 | 2.099272 | 2.104201 | 2.256571 |
| per-request max-gap P95 / max (s) | 20.620352 / 32.954829 | 18.949304 / 32.597544 | 19.017543 / 32.704420 | 20.620675 / 32.847928 |
| 输出吞吐 (token/s) | 2852.936654 | 2858.888342 | 2850.440688 | 2861.550093 |
| 持续 / 最后到达后排空 (s) | 99.149765 / 95.959642 | 98.943354 / 95.753238 | 99.236585 / 96.046483 | 98.851319 / 95.661184 |

两对mean flow差**−.364487 / +.061444s**，吞吐**+.209% / −.388%**。同策略mean flow自身变化为tail−.221637s、remaining+.204294s；没有独立确认或统计显著性结论。gap均值−.160189/−.152369s、P95−1.671048/−1.603132s，但中位数+.016236/+.011019s，234/292个请求max-gap增加。允许报告这种取舍，不将gap替换为事后主目标，也不从继承goodput网格选成功阈值。

**动作确已改变，而且是持续保护。** 两候选各41次原生抢占、10次真实改选，41/41规则重算匹配、改选均唯一关联native preempt/free，实际释放与预测一致；两tail各0实际改选。首改选673：0065237→0053843，释放84→149页。704：短tail0049936→0061042，100→67页。704–767共九次改选持续绕开同一短tail，选择九个不同长请求，累计周转释放1037页；不是净增容量，也不能称相对tail“额外九次抢占”，因为四臂总抢占均41。两短请求随后在769/771步到128 cap完成、未被抢占，最大gap约.130–.133s。0052099没有直接充当改选决策的tail，是后继轨迹的间接受益者。

| 稳定个体效应，candidate−tail | 正序对 | 反序对 |
|---|---:|---:|
| 0049936完成差 (s) | −30.5783 | −30.3886 |
| 0052099完成差 (s) | −30.1268 | −29.9220 |
| 0061042完成差 / max-gap差 (s) | +.0381 / +1.0332 | +.6471 / +1.2411 |

两对均晚完成的七个请求是0041130、0049468、0051693、0055053、0061042、0062890、0066337，它们max-gap均增加约1.03–1.33s；前五完成损失排序相同。只有18个请求稳定提前、7个稳定推迟，295个完成差换号。两短请求对全体mean flow的算术贡献为−.1897/−.1885s，其余318个请求平均差−.1759/+.2515s；这是损益分解，不是因果校正。

恢复代价仍存在：0061042在704被抢、1111重新准入、1113输出，约30.139/30.245s无输出，实际恢复local/external均0；0021202在767被抢、772输出、777再次被抢，其后gap23.532/23.621s，恢复external720 tokens有ACK；0039450在757被抢、783输出、790再次被抢，后gap22.211/22.291s，首次external1200 tokens有ACK。候选有4个请求各被抢两次，tail只有0065237被抢三次。恢复后的held不是新增占用；短请求完成当刻独立free增量未记录。真实轨迹符合“保留短请求需要持续供给”的方向，但旧704/710条件模型不是本次精确预测，更不能由两个短请求完成证明全局容量获得净收益。

前态与开销限定：首动作前各对1次抢占与97159条无时钟输出事件相同；首动作墙钟候选已经快.143582s / 慢.037649s。不是严格同状态快照，也不从最终指标扣除该偏移。两tail测量期各1条JIT警告、两候选0；警告不是编译耗时，既有种子不保证零编译，不据此事后修正服务时间或再开同规则诊断矩阵。外层共同selector墙时依次.331215/.319261/.372227/.327368s，其中观察.218408/.209458/.260405/.214861s，预算helper .001512/.001626/.001566/.001510s均内含，不相加；两臂共同观测不等于生产增量开销。发送滞后均值约12ms、最大30.388ms，全部到达进入引擎并完成，未触600s截止；独立拒绝/单请求timeout未单独记录，不能将missing字段伪造为计数0。

**判决与贡献归类。** 普通已知预算保护实现了可重复的个体取舍，属于强简单基线适配；本组主目标净收益在预设上限内不确定，停止本域预算信号调参/复杂组合及自动重复。两个短请求不作为两个独立系统重复，不能用局部改善证明H_method或同容量全局原则。当前仅支持局部H_model，强简单方案后的H_problem仍未建立；不把它扩大为一般victim选择问题无价值。

资源与复现：整组1039.720s，22:38:16 CST四格exit0/archive VERIFIED，39549/SSH94131终态0、公共锁已释放；无A runner/候卡/新提交。112原始文件远端及本地核验，完整归档96,046,518B，SHA `0989cb2cb81964212c6733d8931923fdbfb9057fde3710e15847b3ce8509f867`，远端tgz和完整本地raw保留；r01零单元磁盘失败不覆盖。实际使用计划r02 SHA `52b1b6ad622aab8c2a930424fb3bdd97f7669ecfeff1d24285e570d46b49e360`、原冻结包manifest `45c5fd771017851428202f6b435e3ef6b5cde38cc854514aee5b78e754f37cd4`，分析器SHA `55107f88be6d3efe4df0736887c89d834644ba7c51bf292841e6d60acda4380a`。

```sh
# 实际已完成的命令；既有session/output禁止重启或覆盖。
LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib /root/miniconda3/bin/python -B -u run_native_remaining_budget_group_seeded.py --plan plan-native-remaining-budget-westd53005-20261008-r02.json
python3 -B analyze_native_remaining_budget_group.py --session session-native-remaining-budget-westd53005-20261008-r02 --output analysis-native-remaining-budget-westd53005-20261008-r02.json
```

当前未知：普通保护之后的严重等待是否仍有不同强简单victim动作可控制，值得继续本域系统方法研究？
主要竞争解释：大部分可控收益已由短作业保护覆盖；或即刻释放更多容量能减少牺牲/恢复队列压力，比完成预算更重要。
最小行动：仅用本组原始轨迹关联剩余max-gap与实际抢占，并核对现成完整native max-release影子及旧size覆盖；本轮CPU读取，不新增观测平台或重复GPU组。
不同结果将如何改变决定：若剩余大停顿与合法victim无关，或替代动作已被同条件测试，则收束此域；若有明确不同容量端点且未被旧实验覆盖，再为一个有上限的强简单端点对照冻结方案，收益先归简单基线，不承诺新贡献。

**CPU判别完成。** 两个remaining臂各37个请求的最大生成间隔均跨越真实抢占，其余请求最大仅.180/.177s；最长0053843为32.598/32.704s、0065237为31.675/31.826s。不是TTFT混入。完整max-release影子在两臂均41/41不同于实际budget选择、每次多释放14–176页；704可选200页替67页，767可选232页替56页，但41/41也等价maxheld/maxcomputed，没有新的物理异质信号。2次影子选择current、4次pending>0；既有原生suffix允许这些动作，旧size资格恰排除了它们。当前数据只能确认未测容量端点，不能预测其会减少总抢占或全请求损失。

### 两个完整强简单端点直接对照（已完成，以下保留冻结设计）
当前未知：已有预算保护之后的主目标损失，是否由每次释放较少容量导致；最大即时释放是否足以得到更好的完整服务结果？
主要竞争解释：较大释放减少后续牺牲/恢复竞争；或保护更接近完成的请求更有价值，大victim的恢复与队列损失反而更高。
最小实验：同正常KV、原mixed320、模型/精度/输入/预热及所有其他机制，remaining_budget/max_release/max_release/remaining_budget一次ABBA；两臂均为完整native合法suffix在线规则，latest tie，没有新阈值。max-release直接使用已存在的refcount物理释放shadow，不加观测或改变生命周期。
不同结果将如何改变决定：max-release若一致更好，先接受普通容量基线并撤销“预算保护优先”的投入；预算若一致更好，只支持此域下该简单原则及取舍，仍不视为新方法；若换号或收益不清，在此组上限内暂存并停止本域相邻排序搜索。任何结果都先定位简单基线后的剩余空间，没有明确新预测就不开发组合评分器。

阶段上限仅1组4×320，预计整卡17–22分钟，硬上限1200s/格、4800s/组、600s捕获、一次最多3600s公共锁等待；主目标仍全外部到达mean flow，全部副作用/输出/失败保留，无SLO或费用遥测，不改指标追求正值。本组新增的是一个历史未覆盖的完整容量端点，不是预算规则的追加重复、旧size改名或近邻完整复现。实现、冻结与磁盘准备完成后已按下述唯一入口提交；尚无新完整策略结果。

已冻结并部署[candidate_native_max_release_r01](candidate_native_max_release_r01/)：manifest `bd72e54183f552c7f2997030adbd03b9817b827acd7b91d23b8fe826802d8f1a`、staged `19934a03468d311d9994de1feeefa1026e0a08ae88d684eac39b9cc29becd457`；仅规则/runner两份payload改变，38文件manifest内其余36份原字节。单项CPU回归通过共享引用计数、current/partial/pending资格、保护/未知回退及原生pop/free/preempt；初次仅fixture异常类型预期修正为ValueError，无运行时代码修复，不称GPU结果。新[计划](plan-native-max-release-westd53005-20261008-r01.json) SHA `938c05792aac7c13acd0cb6e0d756d11d35d7b151a4debe0554085ae80f9b362`，entry SHA `b140074c87231bfe252256e1746a9e51cdc068fb366ccaa351f68429a4942a7e`；controller沿用原锁/交接/种子，配置检查通过。原生未知ownership的view异常行为两臂不变，不悄悄改为另一回退。

存储准备仅回收刚完成remaining r02的17份重复展开目录与缓存；删除前核对本地全部原件和远端完整tar中的268文件，完整本地raw/source、远端tgz、原冻结包及plan/receipt/hash/log保留。root余量1,067,716,608→3,386,888,192B；原3GiB门槛不降低，他线未动。SSH控制连接过期导致首次上传未完成，凭原授权重连后上传成功，无实验重启。部署后root3,373,019,136B，公共锁2304:4312099778不变，session尚不存在。

```sh
# 远端A目录；新session仅提交一次，已有同名目录必须先查实际进程。
LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib /root/miniconda3/bin/python -B -u run_native_max_release_group_seeded.py --plan plan-native-max-release-westd53005-20261008-r01.json
# 整组终态后本地分析；禁止覆盖已有output。
python3 -B analyze_native_max_release_group.py --session session-native-max-release-westd53005-20261008-r01 --output analysis-native-max-release-westd53005-20261008-r01.json
```

执行断点：23:29:16.978 CST唯一48782/SSH26921提交并立即取得公共锁2304:4312099778，交接.181s；23:46:06 CST四臂exit0/archive VERIFIED整组完成，耗时1009.624s，48782消失、SSH26921终态0，公共锁释放。无A runner/候卡/新提交。112原始文件远端核验，完整归档95,701,887B、SHA9227161570c934c4353cc6a77be6b8e59bce809640eac141e9375f3ce04f7e35正在限速取回，尚无完整科学分析。新[分析器](analyze_native_max_release_group.py) SHA `74e6ea0df3ec51961408b3bfd6f12694d7127e2009ae24b4b5e7806207b7be40`复用全请求统计，单独核验两种主动策略及其实际preempt/free；reference自身改选不记为0。两项针对双主动字段/物理ACTIVE与缺失状态的CPU检查通过，没有生成新科学结果。整组完成释放后才回收分析。

**等待期的近邻核对，限制当前对照的覆盖声明。** [CacheOPT §3.4](https://arxiv.org/html/2503.13773v2#S3.SS4)先按TBT SLO桶降序、同SLO内按预测剩余输出桶降序、同剩余桶内按已占用KV桶升序抢占；因此精确remaining与max-release两个端点并未覆盖其“相近剩余量中优先小KV”的组件。论文给出128-token跨度的剩余/占用桶作为示例，未明确边界归属、最终同桶tie-break，也未定位到官方代码，不能写作已验证默认值。[§3.2](https://arxiv.org/html/2503.13773v2#S3.SS2)使用微调OPT-13B预测长度/偏差方向，不等于声明cap。无SLO可将第一维设同级，但用cap并限制native suffix只能称组件代理适配；§3.3预分配/借用/预留及§3.5交换或重算不属于本组。当前GPU两端点对照不因此扩大、不改变已冻结判断规则，也不能声称已超过CacheOPT完整系统或其victim组件。

### 完成：更多即时释放减少抢占，却使平均完成更差（2026-10-09分析）

数据来自[本组原始目录](session-native-max-release-westd53005-20261008-r01/)及[唯一分析](analysis-native-max-release-westd53005-20261008-r01.json)，状态FOUR_CELLS_COMPLETE_COMPARABLE；112文件已完整本地核验。四格均320/320完成、失败/未完成0，均282868 tokens、309 length/11 stop；逐请求长度与终止原因相同，但两对同26个请求token序列改变，不能声称质量等价。同策略两次的无时钟输出事件、逐请求token序列和实际抢占step/request/output/release轨迹完全相同；时钟仍有波动。

| 全请求指标 | budget正序 | max-release正序 | max-release反序 | budget反序 |
|---|---:|---:|---:|---:|
| mean flow (s，主目标) | 68.000075 | 68.785905 | 68.399064 | 67.822490 |
| flow median / P95 (s) | 74.602221 / 93.615622 | 75.964105 / 90.164306 | 75.552406 / 89.679386 | 74.408719 / 93.552133 |
| TTFT mean / P95 (s) | 13.177796 / 62.755269 | 13.268478 / 62.944436 | 13.129548 / 62.595118 | 13.137726 / 62.363477 |
| per-request max-gap mean (s) | 2.089481 | 1.377764 | 1.367432 | 2.066615 |
| max-gap P95 / max (s) | 18.929905 / 32.589804 | 10.218649 / 33.078771 | 10.206513 / 32.967723 | 18.718938 / 32.209356 |
| 实际输出 token/s | 2889.977438 | 2894.140824 | 2908.755772 | 2891.744415 |
| 持续 / 最后到达后排空 (s) | 97.878965 / 94.688834 | 97.738160 / 94.548063 | 97.247078 / 94.056963 | 97.819157 / 94.629026 |

max-release−budget的主mean flow差**+.785830/+.576574s**，两对均更差；吞吐+.144%/+.588%、flow P95−3.451/−3.873s、max-gap P95−8.711/−8.512s，属于明确取舍，不能用更少抢占或尾部改善替代预声明目标。候选219/167请求晚完成、101/153早完成，稳定受损167、受益101、换号52。两个运行对是重复单位，没有置信区间或独立确认结论。

**执行与容量。** budget各41次原生抢占、10次非tail实际改选，37个unique victims、4个各抢两次；max-release各27次、24次实际改选，24个unique victims，0054622/0064312/0054011各抢两次。每个实际决定规则重算MATCH，全部成功preempt/free唯一关联，实际释放等于预测物理释放。max实际释放192–238页、中位214；budget56–205、中位135。相邻抢占engine-call间隔中位14对9.5，压力段仍662→1020对662→1018；这是实际调度间隔，不保证所有保留请求每轮都输出。两策略压力段都只完成相同5个请求，包括769/771到cap的两个短请求；没有“更多即时释放促成额外完成归还”的证据。

**谁获益、谁受损。** max中0053843未被抢占，step1692完成；budget被抢两次、2138完成。其完成提前5.975/6.399s，max-gap从32.590/32.209降至.169/.162s；0061042完成提前5.566/5.985s。0049936和0052099两短请求只额外提前约.05/.15s；全部40短cap的mean flow仍增加.1425/.0098s，280长cap增加.8777/.6575s。

| 主要稳定受损请求 | 完成差：正序 / 反序 (s) | max-gap差：正序 / 反序 (s) |
|---|---:|---:|
| 0042067 | +21.773 / +21.652 | +31.195 / +31.091 |
| 0051355 | +21.668 / +21.668 | +26.007 / +25.919 |
| 0066860 | +18.204 / +17.984 | +28.523 / +28.433 |
| 0045217 | +17.209 / +17.033 | +23.506 / +23.422 |

这些受损者的输出序列未变。0042067到达.63s、prompt3062；max在692抢占释放227页，1117重新准入、1122才输出，gap31.360/31.252s，完成89.110/88.626s；budget未抢，完成67.337/66.974s。较大KV的早到请求承担了新增等待，而非少数输出长度变化造成指标差异。抢占数和受害请求数确实减少，不能只称“相同数量等待换人”，但净平均完成仍更差。

原生恢复重占的实际反例仍在：0064312于673释放203页，675以held48重新准入，679尚无新输出时再次释放192页，当时computed3064落后既有3236-token序列；到1126才从out206→207，形成一段33.079/32.968s连续停顿。不得将两次重叠等待相加；恢复后held也不全是新增分配。max于692释放后保留的0054622确实输出66→80，706才再次抢占，但局部可推进更久没有转为全请求mean-flow收益。

**波动和成本。** 首分歧662前94306条输出事件完全相同、均无先前抢占，动作前max已慢.120784s / 快.102555s；不扣除前态时间、不称严格同隐藏状态。budget两次mean flow自身变化−.177585s、max−.386841s。本组四格测量期JIT警告均0，这不证明所有编译耗时为0。外层共同selector总墙时依次.373241/.264988/.257650/.362011s，内含观察.265599/.196811/.189674/.258766s；budget helper .001551/.001140/.001162/.001448s、release helper .001303/.000878/.000823/.001234s也内含，不能相加。恢复观察体约6ms/臂，不等于恢复链路耗时；native allocated callback47/30/30/47、load ACK6/3/3/6，不等于省下同等数量的传输或服务时间。客户端发送滞后均值约12ms、最大<30ms，全部进入引擎并完成、未触截止；独立reject/timeout未另记，保留UNKNOWN字段。

**判决。** 按冻结主目标，普通remaining-budget在本组两个运行对优于完整max-release；最大释放仅在吞吐和部分尾部分布上更好，不将其作为本域mean-flow优先原则，不调size阈值补救。收益首先属于已有简单调度原则；预算相对原tail的前一组净收益仍不确定，本组不能用跨组时钟替它补证。H_model只支持“释放更多可拉长两次抢占间隔，但服务损失取决于牺牲谁及后续恢复”，这不是独立新颖性证明；H_problem在强近邻之后是否还有可控剩余空间仍未建立，H_method未成立。当前不扩大GPU矩阵、不发展组合预测器或开始论文包装。

资源已终态：23:46:06 CST结束，1009.624s，48782/SSH26921退出0、公共锁释放，无A任务/候卡。完整95,701,887B归档SHA `9227161570c934c4353cc6a77be6b8e59bce809640eac141e9375f3ce04f7e35`保留远端与本地，原始112文件及10份分析来源本地可追溯。冻结代码、计划与上述复现命令不变，未追加运行或覆盖失败。

当前未知：CacheOPT式分桶在这两个端点之外是否真的提供不同合法动作，还是当前状态下仍与已测普通预算等价？
主要竞争解释：精确remaining已覆盖近邻组件；或同剩余桶中优先小KV会以不同即时释放/恢复代价产生独立选择，需要先作为强基线覆盖。
最小行动：本组四条既有轨迹CPU影子；同SLO，remaining取声明cap−output，剩余桶ceil(remaining/128)、占用桶ceil(held×block_size/128)，依次剩余降序/占用升序/latest index。128来自论文示例，边界与tie是显式适配选择，不称作者默认，不扫描其他值。
不同结果将如何改变决定：无差异则不占GPU测相同动作；存在差异只说明近邻组件未覆盖，先核对完整在线适配价值，不据影子宣称收益或新颖性。当前只做CPU机会判断，没有提交下一GPU组。

**CPU近邻机会判断已完成，未运行在线组件。** [固定脚本](shadow_cacheopt_cap_bucket.py)在四格现成决策上相对实际online选择异选37/41、27/27、27/27、37/41；相对remaining影子37/41、25/27、25/27、37/41，相对max-release影子41/41、27/27、27/27、41/41。UNKNOWN及保护回退均0，所有异选的同实际释放量数均0。首662代理选0021202：50页/remaining982/桶(8,7)，remaining为0065237：83页/1011/(8,11)，max-release为0054622：192页/981/(8,24)。代理相对remaining影子在两条budget轨迹的释放差中位−81页、remaining差−27 token，在两条max轨迹则−99页/−49 token；只是各自当前状态的差值分布，不能拼接成新策略累计容量或服务轨迹。

由此确认尚缺一个有不同动作的强近邻组件，不能写作“两个简单端点已覆盖CacheOPT”。若继续这个运行域，下一项有判别力的服务实验应是固定cap分桶代理的完整在线适配对普通预算基线；其价值首先归已有原则，仍不支持同容量新模型。不开桶宽/边界/tie扫描，不把论文示例当作者默认，不因CPU异选多就预称有效。当前没有新GPU提交，先保留完整本组取舍及此明确缺口。

```sh
# CPU机会诊断，只stdout；没有新科学JSON或服务收益结论。
python3 -B shadow_cacheopt_cap_bucket.py --session session-native-max-release-westd53005-20261008-r01
```

### 固定近邻组件对照：在线cap分桶适配（准备中，未运行）
当前未知：相近剩余量中优先小KV的固定近邻组件，能否比精确剩余预算排序降低完整平均完成等待，还是少释放导致更多恢复与抢占？
主要竞争解释：分桶允许较便宜的victim而收益归已有CacheOPT式原则；或粗化剩余量/更少释放损害近期完成并使整体更差。
最小实验：同原mixed320、正常KV、模型/精度/输入/预热与其他机制，remaining_budget/cacheopt_cap_bucket/cacheopt_cap_bucket/remaining_budget一次ABBA。固定128跨度、正桶((k−1)128,k128]、零桶0、latest tie，沿已运行CPU代理；声明cap代替预测器、SLO同级、仅native suffix，不称原系统或作者默认配置。
不同结果将如何改变决定：代理一致更好则先接受已知组件为强基线，再检查剩余损失，不记新贡献；预算更好则保留预算并收束这个代理；换号或无清楚主效应就在本组上限内暂存。均不扫桶宽/边界/tie、不追加测到显著、不发展组合预测器。只有具体的新剩余问题和判别预测才支持继续方法投入。

预算仅本组4×320，预计整卡17–22分钟，1200s/格、4800s/组、600s捕获、最多一次3600s公共锁等待，无自动重试/第二候卡。主目标仍全部外部到达mean flow，TTFT、gap、吞吐、排空、个体损害、全部失败与输出差异完整报告；无应用SLO、费用遥测或已知共享余额。两臂均执行全部相同影子观测；只改变在线victim排序，原生命周期、Q1、恢复目标/触发、传输与准入不改。当前仍在CPU实现/分析准备，不存在新GPU结果。


**实现与配置已冻结，GPU尚未运行。** 已重新完整读取工作区新版AGENTS；研究卡沿用全部外部到达mean-flow服务目标和三层假设，不重写历史判决。新包[candidate_native_cacheopt_cap_bucket_r01](candidate_native_cacheopt_cap_bucket_r01/)仅改staged/runner两份payload，其余36份与已完成max-release包相同；原remaining/max helper保持原字节。新增规则无EOS预测、无额外合法性缩窄；全部三份影子两臂共有。单项选择/生命周期CPU回归和38份部署hash通过，只说明实现可运行，不是GPU或科学证据。

冻结[plan](plan-native-cacheopt-cap-bucket-westd53005-20261009-r01.json) SHA `9b768caf4689685762cc2d3bd27eba2011a2cd42ebcbc901b52048c5babaed87`；manifest `3006a941bc3a5292b16442edbcb18035c6c5a250aa06f4a2ec10807a85f5e922`；staged `18f0e3f17e56b8a6b1e8516a9b47f33c2f0ca025e9ab90caa67522b5a56f6784`；runner `1ba10f4678c70e3ee4be81c24bd81db9ad3ef5f4b4284947faaf34f5aedb4a2c`。controller `bb964b8e11f84b32a395c1d0e116cc1f3a5a7e201325881241efba9d8e329f4e`、seeded `c5584f1a59ded3e7212a30398aa27f126a4bc5d08dcd779b43ae0b112c53a403`；部署tar6,448,948B SHA `2440626d6e88d2ae23be14f016717915e581b34d4cad96acbc97eb373da28b6a`。唯一新分析入口[analyze_native_cacheopt_cap_bucket_group.py](analyze_native_cacheopt_cap_bucket_group.py) SHA `a8c7d8a08ee90769107a64f70757abc2929ef738ca660af5bca5bc8475ad85cd`，逐动作重算两实际规则和max影子；前缀对齐截止两实际规则首次建议分歧，不以较晚cap非tail提案冒充共同前缀。

```sh
cd /root/autodl-tmp/moe-a-victim-20261004
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
/root/miniconda3/bin/python -B -u run_native_cacheopt_cap_bucket_group_seeded.py --plan plan-native-cacheopt-cap-bucket-westd53005-20261009-r01.json
# 只能提交一次；若已有进程，先接续它，不重启。取回终态完整session后：
python3 -B analyze_native_cacheopt_cap_bucket_group.py --session session-native-cacheopt-cap-bucket-westd53005-20261009-r01 --timeout-s 1200 --output analysis-native-cacheopt-cap-bucket-westd53005-20261009-r01.json
```


**已真实启动，尚无完整结果。** 00:34:07 CST（epoch1791477247.3728504）唯一controller57669/SSH exec4613提交，公共锁即时取得，handoff0.188s确认GPU清空；无第二A任务。此前仅回收已完成max-release r01的17个重复展开/私有缓存目录，289份本地与远端完整tar文件、112份output逐字节核验；根盘1,644,830,720→3,956,420,608B，释放2,311,589,888B。完整本地raw/source、local+remote全tar、计划/receipt/hash/log与原冻结包均保留，未动其他线。回收脚本SHA dd7655f6…0842，原3GiB门槛未降低。当前只接续57669/4613，不重启；最终性能及失败以receipt/raw为准。


### 固定cap分桶ABBA完成：主目标换号，停顿尾部更差（2026-10-09）

[完整原始session](session-native-cacheopt-cap-bucket-westd53005-20261009-r01/)与[唯一分析](analysis-native-cacheopt-cap-bucket-westd53005-20261009-r01.json)均已保存，状态FOUR_CELLS_COMPLETE_COMPARABLE。四格各320外部到达/320完成，失败0、未完成0、无缺失请求；无捕获超时，所有到达均进入引擎并排空。独立拒绝/超时计数未单列，保持NOT_SEPARATELY_RECORDED/null，不虚填0。主目标与模型/精度/输入/预算/预热未改；两运行对是重复单位，不将1280请求当独立重复。

| 指标 | budget正 | bucket正 | bucket反 | budget反 |
|---|---:|---:|---:|---:|
| 平均完成flow s | 67.748570 | 67.909489 | 67.859320 | 67.902392 |
| 完成flow P95 s | 93.461345 | 93.725896 | 93.646058 | 93.660032 |
| 平均TTFT s | 13.121332 | 13.220304 | 13.123266 | 13.194396 |
| TTFT P95 s | 62.273806 | 62.074230 | 62.055263 | 62.428640 |
| 每请求max-gap均值 s | 2.071157 | 2.416740 | 2.439352 | 2.068498 |
| max-gap P95 s | 18.729332 | 21.290628 | 21.509136 | 18.717814 |
| 最长max-gap s | 32.187543 | 31.503002 | 31.883910 | 32.189390 |
| 输出tokens/s | 2894.727575 | 2883.095146 | 2885.416073 | 2888.592918 |
| 完整测量至结束 s | 97.718349 | 97.930518 | 97.851746 | 97.925879 |
| 最后外部到达后排空 s | 94.528225 | 94.740402 | 94.661628 | 94.735772 |
| 输出tokens | 282868 | 282343 | 282343 | 282868 |
| native抢占 / 非tail实际改选 | 41 / 10 | 43 / 36 | 43 / 36 | 41 / 10 |

**完整收益与代价。** bucket−budget mean flow为+.160918/−.043072s；实际输出率−.40185%/−.10998%，gap P95+2.561296/+2.791322s、平均max-gap+.345583/+.370854s，最长gap−.684541/−.305480s。前向96请求提前/224推迟，反向292提前/28推迟；96稳定提前、28稳定推迟、196换号。0066842完成+6.527/+6.256s、gap+21.453/+21.691s；0058251完成+6.271/+6.001s、gap+20.669/+20.890s，两者token序列未变。0054622完成−5.011/−5.269s、gap−20.001/−19.950s，0064254完成−4.198/−4.447s，均序列未变。分组损益不替代全体主指标。

**输出差异与波动。** 两对都同样25个序列改变，2个长度改变，1个结束原因改变：0066337由1024/length变453/stop，0046501由45/stop变91/stop；总少525 token，短cap40组输出同为5003。0066337提前11.189/11.422s伴随少571输出，不能记作等工作量收益；两长度变化请求对全体mean差的算术贡献−.026438/−.027754s，不是可扣除的因果校正，其他请求也可能受其提前结束影响。同策略两次逐请求长度/结束原因/token序列及完整无时钟输出schedule一致；budget自身mean漂移+.153821s，bucket−.050169s。首次实际规则分歧都在662，此前94306无时钟输出、0次抢占一致，但bucket已分别慢.497203s/快.066290s；不是严格同隐藏状态，不能相减后宣称纯策略效应。测量JIT warning为0/1/1/0，不将warning当编译时长或追加重复的理由。

**真实执行与机制。** 两bucket各43实际抢占、36非tail改选、40 unique victims（0021202三次、0053843两次）；budget各41/10/37，4请求各两次。两实际规则在各自状态的建议异选37/41、38/43、38/43、37/41，只是诊断计数；所有执行规则及preempt/free连接均MATCH，实际释放等于记录物理页，UNKNOWN0。bucket释放中位109页（50–213），budget135（56–205）；相邻抢占调用间隔中位8对9.5，不能译为所有请求连续前进的保证。更小KV未形成更低全体停顿代价。

**控制与外部计时。** 外层selector总墙时.351457/.280139/.299817/.358202s；其中观测.247313/.182538/.193099/.254945s，cap helper仅.001379/.001347/.001370/.001402s，全部嵌套，不相加。两臂共有三份影子，额外工作计入实测。恢复观察body约6ms/臂，不是恢复耗时；native lookup347/355/355/347，allocated回调47/49/49/47，load ACK均6，次数不直接等于传输/服务节省。发送滞后均值约12ms、最大不足30ms，外部到达计时完整保留。无应用SLO，不选择有利goodput阈值。

**判决与资源。** 固定cap代理未获稳定主目标净收益支持，且停顿P95更差、存在输出工作量变化；收束该固定组件本域调参，不称完整CacheOPT失败或被击败。普通预算也尚未被证明相对tail有稳定全体净收益，不能跨组时钟补证。下一投入按新用户指令转向最小释放窗口模型，首先复用既有轨迹判断是否有增量信息，不加全局优化器或新GPU矩阵。

整组1008.864223s，00:50:56 CST结束，57669不存在、SSH4613退出0，锁已释放，无A runner/候卡。112raw全部本地SHA核验，11份分析源随组保存；完整local+remote tar96,108,188B，SHA `4a55cda5bb19195e613d8b2e147bf79afbc03d5026ff52334f9ec7d61c5c9f3a`。export60772、download72376、extract95836、analysis59249均终态0。冻结代码/配置/命令见上段，历史失败保留，未改旧包或推送。当前没有中心贡献、独立确认或可投稿结论。

### 用户提出的下一判别：释放窗口服务于谁（CPU既有证据，不新增GPU组）
当前未知：在remaining、max-release和固定cap分桶已提出的合法候选中，“首次重新分配之前的保留进展/下一完成”是否提供即时释放量与remaining之外的可预测差异？
主要竞争解释：native等待/恢复次序迅速吸收释放，候选之间窗口几乎相同；或较长窗口由后续多次抢占共同造成，并非当前动作独立价值。
最小行动：复用已完成轨迹，对首个规则分歧和同victim快速重复抢占，输出实际首次allocation、readmission、next-output、next-preempt/next-completion及保留suffix进展；列出当时三个规则建议的候选状态，未执行候选的后果保持UNKNOWN。只做一个可运行CPU诊断，不新建追踪平台或占卡。
不同结果将如何改变决定：若可见状态能区分有意义且不被立即恢复吃掉的窗口，再做一次有界真实动作对照；若窗口同质、关键状态缺失或只由后续动作决定，先收束该局部模型或明确最小必要状态，不写复杂评分器。真实持有量不当增量分配，host request必须区分实际ACK；离线未来事件不得进入online规则。


**有界窗口诊断已完成（事后观测，非候选反事实）。** [diagnose_capacity_window.py](diagnose_capacity_window.py)，SHA `503796b692832092c053f87848268be46134fcac797e041a416a4e48964b243e`，只stdout，实际运行本组budget首662、bucket的0021202三次抢占，以及上一max-release正序首662。列出remaining/max-release/cap组件当时建议和可见状态，未执行候选后果保留UNKNOWN；旧max包没有cap影子字段，明确NOT_RECORDED，不回填未来建议。窗口截止首次allocated callback或下一全局完成的较早者，同时标注完成是否在已记录保留suffix、期间后续抢占及host请求/ACK。没有新增GPU实验、结果JSON或追踪字段。

首662三个实际端点释放分别50（bucket）、83（budget）、192（旧max）页，但首次allocated callback距释放都约.211s；原victim均669 readmission并next-output、下一抢占均673。首次全局完成都是0050842在664，先于callback约3ms；已记录保留suffix均36请求各多3 token、0 suffix完成、0介入抢占。三条实际轨迹在这个短窗口未显示基于释放量可获得的额外进展，不支持在首事件就增加容量窗口评分。旧max与新组仅作有边界的事件描述，不构成新的三臂同组性能比较或严格同状态因果。

首次恢复callback：budget held83/requested external1328、随后ACK；bucket held45/external720、随后ACK；max held187/external2992、随后ACK。这些都是持有快照与原生请求/ACK，精确新增占用量UNKNOWN，不能把held×页尺寸写成新占用或把host可见前缀写成兑现服务节省。callback未记录engine_call，保留UNKNOWN，不从wallclock猜精确call。

bucket的0021202在673再抢后，suffix中的0036565于674先完成，105个保留请求各多2 token、无额外抢占介入；679第三次抢后，下一完成是0052099的769，早于该victim1116 readmission/1117 next-output，但到下一完成已有14次其他抢占。长间隔因而不能归功于单次释放；第三次后的连续gap31.503/31.884s也不能把多段重叠等待相加。另0025181在682仅释放56页却直到1115/1116才readmit/output，gap31.214/31.590s，budget为8.820s；长窗口伴39次其他抢占，不支持“小victim天然代价低”。

**对下一事件模型的具体限制。** 首662前最后返回call661，prefix请求0050842已出158/cap1024（remaining866），随后161时stop，不能从声明cap预测这一完成，也未仅凭stop认定EOS token；0024322已出124/cap128（remaining4），665的length完成才是cap可见的近期归还。两者均不在该事件已记录可选suffix。模型必须区分完整运行集合与合法victim后缀，且区分已声明cap完成与未知自然终止；不能把事后首次完成时间塞进online打分。

当前新原则仍是**未获支持的模型假设**：评价候选的释放能否在原生恢复重新进入前帮助保留集合跨过近期完成或压力阶段。首事件窗口无区分度不否定后续状态；下一项值得做的是针对已见快速部分恢复（旧max的673/679）核对原生重新分配门控，能否只用动作前状态给三个简单规则候选不同的事件顺序判断。若不能区分、或推荐始终被remaining/固定组件覆盖，就不占GPU跑同选策略；只有新排序及有依据的动作价值预测，才做一次有界真实干预。此处没有开发完成的预测器、同状态rollout或新方法收益。

```sh
# 仅CPU事后事件诊断；保持原始文件，stdout无新汇总文件。
python3 -B diagnose_capacity_window.py --session session-native-cacheopt-cap-bucket-westd53005-20261009-r01 --cell cell-00-remaining-budget-first --steps 662
python3 -B diagnose_capacity_window.py --session session-native-cacheopt-cap-bucket-westd53005-20261009-r01 --cell cell-01-cap-bucket-first --request 0021202
python3 -B diagnose_capacity_window.py --session session-native-max-release-westd53005-20261008-r01 --cell cell-01-max-release-first --steps 662
```


### 重新分配门控的可判别性（源码＋四个现成事件，本轮CPU）
当前未知：native full-ISL fit是否使完整KV候选的“释放页数”和“重入所需页数”近似抵消，从而解释三个首事件窗口相同；部分恢复为什么会在新输出前被再抢？
主要竞争解释：重入门槛/恢复实现本身主导窗口，新增horizon仅重述既有size/remaining；或动作前可见的部分恢复、有效复用及保留集增长提供真正不同的动作判断。
最小行动：读实际运行hash匹配的native scheduler/allocator/connector，仅结合旧max与新cap的673/679四个事件，推导门槛与首次实际分配之间的区别，不调用有副作用lookup、不新增GPU组。
不同结果将如何改变决定：若释放与fit门槛抵消，停止纯decode窗口评分投入，明确例外条件；若有稳健不同门槛，先看是否被remaining/分桶建议覆盖，再考虑一次有界真实动作探针。Host当前ready只作可失效的上界，不能偷用未来ACK或未来完成。

**新增证据：必要fit门槛与真实重占必须分开。** 已读取本机native源码，scheduler及connector SHA与冻结实验记录一致；最少四份源码保存于[native_reentry_source](native_reentry_source/)。allocator两份是当前主机读取版本，未冒称有历史hash记录。等待/被抢占请求的[full-sequence检查](native_reentry_source/kv_cache_manager.py#L411)采用已有prompt+output序列，不是声明max-output，也不是当前computed；随后的[真实分配](native_reentry_source/kv_cache_manager.py#L429)只给本次computed external+new tokens所需页。当前单FullAttention、prefix-OFF、lookahead0、抢占后owned0时，必要门槛为Q(v)=ceil(min(P+O,4096)/16)，与host命中多少无关，因为host KV也需要GPU目的页。watermark和后续reserved-block检查可以进一步限制准入，此处没有省略它们后宣称充分条件。

令J(v)为动作前引用计数推导的可释放页，D(v)=Q(v)−J(v)。仅已执行victim的J通过raw free join验证为真实释放，未执行建议不能写成已发生的free。若computed=P+O−1、held=ceil(computed/16)、无共享且全部可释放，则D只能为0或1；这项抵消只说明必要容量门槛几乎不因victim大小而变。在固定其他预留/队列条件时它是静态容量偏移，**不是等待、首次重占、持续推进或收益的预测**，也不证明不同候选窗口相等。实际free还受保留请求增长、完成归还和其他准入/恢复改变。

[native_reentry_fit.py](native_reentry_fit.py)已实现可运行纯函数及stdout诊断，SHA `18980e3fb87cb1fd91a80ab0095d7ed126e12c092dc2800e61768d50b0f3ea90`。域条件由现成config/engine/store/profile检查，未观测的源代码前提单列，不补旧日志缺失的cap建议。一次针对0/1、partial11、host不降低Q、未知/不匹配拒绝的CPU检查通过；没有新GPU运行或结果文件。

| 真实轨迹中的已记录建议 | D=0 / D=1 / D>1 | 未知 | 合法候选中出现partial的决策 |
|---|---:|---:|---:|
| budget正序41次：remaining | 37 / 4 / 0 | 0 | 0 / 41 |
| 同一轨迹：max-release | 37 / 4 / 0 | 0 | 同上 |
| 同一轨迹：cap-bucket | 36 / 5 / 0 | 0 | 同上 |
| 旧max正序27次：remaining | 25 / 2 / 0 | 0 | 1 / 27 |
| 同一轨迹：max-release | 24 / 2 / 1 | 0 | 同上 |

旧max包未记录cap建议，27次全部UNKNOWN，不事后回填。以上是各自策略到达的状态上的建议诊断，不是这些建议都已执行，更不是独立运行数。已执行的GPU动作计数及完整服务结果仍以上一组原分析为准；本轮只增加CPU解释，实际新干预数0。

**203→48→192页的机制含义已缩小。** 旧max在673选0064312，P3031/O206/computed3236，Q203/J203/D0；其首次lookup和实际allocation external均0、无LOAD job，675 readmit时held48。因此48是冷补算过程的持有量，不是兑现了48页host复用；日志没有单次scheduled-token标量，不把48页进一步写成实测768-token工作量。native full-fit通过后不一次持有整段Q，running续算也不持续执行这一full-fit检查；in-flight reservation只在async waiting allocation中扣减，不能保护全部冷补算增长。679同victim尚未新输出，computed3064/held192，而完整已有序列仍3237：Q203/J192/D11；随后又被抢，直到1126才输出207。说明反复补算确实存在，但该事件的remaining建议已是0053843（Q150/J150/D0），预算强基线全部41决策也没有partial候选。它不足以建立一个超过普通预算的新保护器。

**Host只按实际兑现计。** 新cap的0021202在673动作前ready49页，allocation接受784 external tokens并有LOAD ACK，677新输出；679再抢前同样ready49、首次lookup也offer784，但最终实际allocation external0/无LOAD job，1116 readmit时held30。可见早期offer不等于将来兑现；现有字段不证明offer丢失的确切原因。当前prefix-OFF无GPU本地命中，不能搬用此前prefix-ON重叠解释。pending STORE/LOAD、队列时序、host变化及保护阶段仍是边界；不额外调用会touch缓存的lookup来伪造动作前预知。

**研究决定。** 强预算轨迹没有大于1页的必要门槛差，也没有部分恢复候选；已测三端点首窗口同质，较差max轨迹的11页例外又被普通预算建议覆盖。当前“纯decode＋首次重入门槛”不支持继续开发评分器或占卡重复，不把微小门槛差译成时间优势。此前[10月2日prefix-finish deferral真实失败](../RESULT_LEDGER.md#L1092)已测试过“让持有容量足够的请求先完成”：276实际deferral，mean flow37.362对tail36.441s、max-gap9.026对8.729s，输出不同；故不把等待完成作为未经测试的新配套机制，也不原样复跑。那是不同历史运行域的失败，不冒充本机当前负结果。

本轮最薄弱环节因此仍是**强简单策略后的可控剩余损失与新增状态的决策价值**。所获是有明确源码域的必要容量解释，不是全局horizon模型、新online方法或论文中心贡献。收束本候选；重新投入需要具体的新证据，能说明普通预算/引用计数/首输出保护/近邻组件遗漏了哪种合法动作后果。资源不是当前阻塞理由：本轮GPU占用0、没有提交新任务/候卡，没有新增性能或独立确认数据。

```sh
# 本线目录；只读既有结果，stdout，不产生GPU工作。
python3 -B native_reentry_fit.py --session session-native-cacheopt-cap-bucket-westd53005-20261009-r01 --cell cell-00-remaining-budget-first --steps 673,679
python3 -B native_reentry_fit.py --session session-native-max-release-westd53005-20261008-r01 --cell cell-01-max-release-first --steps 673,679
```

### 动作范围是否仍留下未覆盖空间（已有代码与raw，CPU）
当前未知：当前remaining预算规则只看未处理suffix，是否因此遗漏可合法撤销本次计划的prefix候选，影响对强简单基线和释放窗口的判断？
主要竞争解释：旧full-running BidKV已覆盖相关机会；或当前规模/排序下有不同候选，但现成日志只保存suffix，不能把“没记录”判为“不存在”。
最小行动：只核对现成full-running rollback代码及最新预算臂41次决策的动作前可还原状态，最多一次CPU诊断；不改GPU代码、不重新运行旧full-running或新增追踪平台，预计CPU准备不超过一轮、GPU0。
不同结果将如何改变决定：若完整状态支持新合法异选，再判断它是否仅补强简单基线以及是否影响当前主损失；若无差异则关闭范围解释；若状态不足则明确缺口，不据缺失状态宣布机会为0，也不为补字段自动占GPU。

当前未知：预算代理未建收益，是否因为把未知自然stop都当作声明cap，误选本来很快结束的victim？
主要竞争解释：缺长度预测是关键；或已有轨迹中所选victim本来就达到cap，纠正终点也不改变该建议。
最小行动：同一预算参考轨迹，在原合法suffix以最终实际输出量代替cap做一次事后建议诊断，所有已到达/失败先保留；仅作固定参考轨迹的已知终点敏感性，不称反事实、可实现online、理论上界或CacheOPT完整结果，GPU0。
不同结果将如何改变决定：若建议未变，不投资预测器救现有排序；若建议变化，则先查看是否由很少提前stop驱动，承认近邻预测组件尚未覆盖，不把变化本身算新贡献或服务收益。

**两个替代解释已用既有数据排除到本参考轨迹。** [shadow_remaining_scope.py](shadow_remaining_scope.py) SHA `5881d57d9bd14c7a8e8c5ddc0c809392413a53ca9b4cfb6e41a2535bbe41677b`，按真实running.append hook记录的residency_admissions、先前已发生的preempt/finish，以及严格早于决策的已返回输出还原成员/顺序。41/41重建suffix顺序吻合，output差异0、生命周期/时间歧义0；计算出的suffix赢家与已记录proposal 41/41一致。扩到全running仍41/41同选：662时suffix remaining1011对prefix最大922；673为1020对726；704为959对836；1018为566对105。这里把所有running作为可撤销prefix的超集；既然超集也不改变赢家，不需要假装已知每个prefix的本步scheduled状态。即时held/freeable/refcount/host/pending仍UNKNOWN，不生成full-running max-release反事实。

现有full-running rollback本来就能撤销scheduled prefix的本步token/block/spec/encoder计划并返预算，然后native preempt；当前remaining只用suffix是实验动作范围，不是生命周期绝对限制。旧BidKV组件还受pure-decode/residency资格限制，旧失败不能等同当前完整预算；不过本次41/41同选已直接排除本轨迹的预算动作范围解释。扩大动作集合本身也不算新机制。

同一脚本独立打印HINDSIGHT_ENDPOINT_DIAGNOSTIC：原轨迹320/320完成，41次合法建议全部仍同选，41个winner最终都到声明cap；**256个不同suffix候选全部最终达到cap，提前结束0**。最终长度仅用于独立的事后敏感性诊断，没有进入前述在线状态还原。这不是真实预测器、策略反事实或性能上界；若改变动作，终点也可能改变。它说明当前参考轨迹中，给现有remaining排序配上已知最终长度也不会提供新动作，不值得为此开发EOS预测器。

**运行域限制。** 本组全体仍有11个stop，但它们没有进入上述256个合法候选集。全局“允许EOS”不等于受抢占候选包含自然终止异质性。当前吞吐/完成时间取舍、实际损失及失败结论全部保留；不能把此以cap结束为主的续写压力域当作自然摘要服务证据。本轮新增在线干预0、GPU0，没有新的服务性能结果；新增的是范围与终点信息都不足以改变当前预算动作的证据。

```sh
python3 -B shadow_remaining_scope.py --session session-native-cacheopt-cap-bucket-westd53005-20261009-r01 --cell cell-00-remaining-budget-first
```

### 新部署假设的可行性准备：文章摘要与自然停止（未运行GPU）
当前未知：对一个有实际完成语义的摘要批处理服务，在本机正常KV容量下，是否仍有抢占损失及可区分的候选，而不是当前全部候选达cap的人工预算竞争？
主要竞争解释：自然完成及时归还容量，正常域根本无需victim优化；或真实终止长度与重占存在不同关系，当前续写域的同选结论不足以外推。
最小行动：复用同一320篇完整WikiText文章、同到达序列及已缓存OLMoE-Instruct/BF16，统一摘要指令，真实chat template，不截文章、不强制输出长度；先CPU构建与固定质量抽样，后续只允许一格原生强简单参考的可行性测量（预计5–8分钟整卡、上限1200秒），不扫并发/容量/指令寻找抢占。
不同结果将如何改变决定：若摘要不合格、仍主要截断或无实际选择空间，停止本域投入并报告边界；若有合格自然输出及强简单方案后的具体可控损失，再设计有界动作干预。原已退休评分器不自动恢复，换权重/任务/预算的跨域差异不叫策略收益，也不叫第二模型确认。

已只读确认现有Instruct snapshot `7f1c97f440f06ce36705e4f2b843edb5925f4498`，3个权重shard齐全、13,838,721,960B（不是重新校验的完整权重哈希），BF16/OlmoeForCausalLM/4096；config和generation_config均EOS50279。官方[模型卡](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct#use)要求自己的apply_chat_template，摘要指令是本线适配，不是官方摘要质量保证。已有runner能接收config模型身份和token IDs；新controller只需适配base硬编码身份校验，KV/传输/准入无需改写。现有raw保存token IDs/finish_reason，可离线解码；行级stop_reason实际是finish_reason，具体停止token尚未单列，不能仅凭字符串stop认定EOS。当前只有CPU准备，无新GPU包/任务/候卡，也未下载权重。

**CPU输入已实际完成。** [build_natural_summary_inputs.py](build_natural_summary_inputs.py) SHA `80e9f64b5d2ce7adba6902929636a56de001a9b4e432eace59096563a1fdf14f`，仅用固定Instruct tokenizer，关闭torch/TF/Flax和网络；320篇原文章/顺序/外部到达保留。官方chat模板后prompt493–3115，统一安全cap由min(1024,4096−最大prompt)得到981，不按输出结果调参；不截原文、min_tokens0、ignore_eosFalse。token化与模板文本往返一致，原始文章hash保留。输入[inputs_natural_summary_r01](inputs_natural_summary_r01/)已本地取回，workload SHA `0254f76708b4c3cf5873d4fb3454b6ca09ab59df87646ab065afb4f645680b85`；预选8个长度分层质量样本写在task.quality_check_ids。没有输出/质量/性能结果。

后续已实读现成模型全部9文件SHA供冻结计划使用，不是下载或载入权重。共享锁仍是2304:4312099778；本次读资源时GPU0MiB/0%、无进程/持锁者，但这不替代启动时重新检查。根盘不足原3GiB门槛；02:13:38 CST仅回收已完成cap-bucket r01的17个重复展开/私有缓存目录，289份本地与完整tar、112份output校验，1260私有cache文件再查无变；1,172,860,928→3,489,927,168B。完整local raw/source、local+remote全tar、冻结包/receipt/hash/log保留，未动其他线。复用回收脚本SHA `1dcd703cfbf07392380a3564e676cf816a2ed96c13d563df2fc05383a5a999fd`，未降低磁盘门槛。当前没有GPU实验进程。

```sh
# 已执行过的CPU准备；输出目录存在时脚本拒绝覆盖。
USE_TORCH=0 USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python -B build_natural_summary_inputs.py \
  --source candidate_native_cacheopt_cap_bucket_r01/pkg/inputs/pro_high \
  --snapshot /root/autodl-tmp/moe-a-20261002/hf/hub/models--allenai--OLMoE-1B-7B-0924-Instruct/snapshots/7f1c97f440f06ce36705e4f2b843edb5925f4498 \
  --output inputs_natural_summary_r01
```

**单组探针已冻结并部署，尚未执行GPU。** 新包[candidate_native_natural_summary_probe_r01](candidate_native_natural_summary_probe_r01/) manifest `7142430b4bc582ebfc79184e79ce92107f3be46fbec13e7ffd3fd3c1951cfdfd`，28 payload；选择器、runner、传输保持原字节，仅测量记录增加native_stop_reason，输入与预热模型身份更新。新[controller](run_native_natural_summary_probe.py) SHA `81aa90860da1f4edb73f44908bf68424dfc41eced5360c3e4a03d27a33eaa0f3`，复用同锁/私有种子/handoff/归档；固定profile→remaining-budget两格，只有后一格320测量请求，绝不把profile当重复。profile必须来自本次Instruct正常.90，随后按实际字节pin，拒绝沿用旧base容量。原32/384/2预热程序保留并同步模型身份，不冒称新旧域性能可比。

CPU准备发现workload_sha256接口不符：旧builder用了文件字节SHA，现成runner要求json.dumps(workload,sort_keys=True)的语义SHA。旧CPU r01文件保留，新包config修为 `60f2db86e8f9807e2932e2be5680589544fe3a8d88b47e4b31d29c95c21d7553`，另存file SHA0254…；workload字节不变。builder已修复，当前SHA `2ef17d591c95d426f719b07358470051688fc742cd444748784bc8bee9e9eb18`。这是输入接口修复，非机制或GPU负结果。定向输入/停止字段/profile→pin检查通过。

[唯一计划](plan-native-natural-summary-westd53005-20261009-r01.json) SHA `3549d427271ef7b715827371e302d3a1e1e07bccc956abfb3d28fcf57d397c0c`；38文件部署包3,199,659B，SHA `7025a68b4fad5eca9bea29d5f94d144d992028e54edbfd4fb8fe389b853e0382`。总1200秒、每格600秒，沿用一次最多3600秒公共锁等待/300秒GPU交接，不自动重试或创建第二候卡。预期5–8分钟整卡，无费用遥测或已知共享余额。研究决策规则保持上段：无抢占/自然质量不合格也是有效收束结果，不提高并发或缩KV救场。

```sh
cd /root/autodl-tmp/moe-a-victim-20261004
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
/root/miniconda3/bin/python -B -u run_native_natural_summary_probe.py --plan plan-native-natural-summary-westd53005-20261009-r01.json
# 仅首次提交；若进程/结果已存在，接续原任务而非重启。
```

**已唯一提交，GPU尚未轮到本线。** controller74306/SSH exec12296真实存在；02:25:07 CST（epoch1791483907.363）实查wchan=locks_lock_inode_wait，同一公共锁2304:4312099778由73467持有，73874在前，74306是WRITE*等待者，实际GPU进程73472。本线session尚未创建、未CUDA，无第二候卡；只接续原handle，不将等待写成实验结果。分析入口[analyze_natural_summary_probe.py](analyze_natural_summary_probe.py)已准备，SHA `7d8bb2eb8c0ea354f5b44ea54f9af634ac985fa46da34275d0b7d57fd6e433ab`；EOS区分实际token/配置契约与未分类stop，8篇质量仍未评估，不产生自动quality pass。

CPU分析依赖闭包15文件已部署到本线analysis_natural_summary_r01（69KB压缩），没有修改运行包或执行分析。任务终态、GPU锁释放后才运行下面命令；输出使用独占创建，避免覆盖失败。解码只加载已核对tokenizer，torch/网络关闭，质量仍须对照预选文章逐例审读。

```sh
USE_TORCH=0 USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python -B analysis_natural_summary_r01/analyze_natural_summary_probe.py \
  --session /root/moe-a-victim-20261007/session-native-natural-summary-westd53005-20261009-r01 \
  --policy-source candidate_native_natural_summary_probe_r01/pkg/staged_store_rotation.py \
  --input-config candidate_native_natural_summary_probe_r01/pkg/inputs/config.json \
  --tokenizer /root/autodl-tmp/moe-a-20261002/hf/hub/models--allenai--OLMoE-1B-7B-0924-Instruct/snapshots/7f1c97f440f06ce36705e4f2b843edb5925f4498 \
  --output /root/moe-a-victim-20261007/session-native-natural-summary-westd53005-20261009-r01/analysis-natural-summary-r01.json
```
