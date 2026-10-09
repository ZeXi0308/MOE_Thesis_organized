# A latest — 2026-10-09 续写候选全达cap；自然摘要域唯一候锁
已复用现成raw排除两个解释：预算41决策的running成员/顺序重建与suffix全匹配，扩全running仍41/41同选；256个不同合法suffix候选最终全达声明cap，换成参考轨迹最终长度仍41/41同选。这是已有轨迹诊断，不是全局反事实或在线预测器。当前续写域不支持继续扩大候选集或增加长度预测，原窗口评分方向保持收束。
新投入只检验有完整服务语义的自然摘要域H_problem。复用现成Instruct rev7f1c…4498与同320文章/外部到达，官方chat模板，无文章截断；prompt493–3115、统一cap981由上下文几何决定，预选8篇质量样本。CPU输入/独立包/分析与唯一计划已冻结；profile必须来自本次Instruct正常.90，随后1格remaining预算参考。runtime只补native_stop_reason，选择器/恢复/准入不变。预计5–8min整卡，1200s组/600s格，若无抢占或摘要不合格就停止此域，不扫压力救场。
02:25:07 CST实查唯一controller74306/SSH12296存活、locks_lock_inode_wait，同2304:4312099778公共锁73467持有、73874在前；A session未创建，无CUDA/第二候卡，后续只接续原任务。先前根盘不足通过仅回收A已归档cap-bucket重复副本恢复至3.49GB，全部local raw/source和两端全tar保留。准确版本/命令/限制见[RESULTS文末](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)。没有新GPU结果或质量结论，goal ACTIVE，中心贡献仍未成立。

# A latest — 2026-10-09 重入门槛诊断完成，收束纯decode窗口评分
已重读新版AGENTS并更新[研究卡与三层假设](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)。核对原生full-ISL-fit与实际allocation：prefix-OFF完整私有decode的必要重入页Q与引用计数可释放页J只差0–1页；预算正序41个决策的remaining/max/cap三份建议全符合，合法候选中partial事件0。旧max轨迹只有679的0064312例外D11，普通remaining已建议另一完整请求；不能把较差策略造成的病理当新方法优于强基线的空间。
203→48→192页链路的首次external allocation为0、无LOAD，48是冷补算持有量，不是host复用。full-fit门槛通过不等于持续为运行中的补算预留整段容量。cap victim先前ready49也会从offer784变为最终external0，原因未完全观测，不用旧prefix-ON的GPU重叠解释当前prefix-OFF。Q−J只作必要门槛偏移，不能预测重占时刻或服务收益。
新增可运行CPU入口native_reentry_fit.py，代码/域与复现命令见RESULTS文末；本轮GPU0、新执行干预0，不生成另一套科学结果。当前强基线后的可控剩余损失及新增信号价值仍未成立，停止纯decode首次重入评分的同域投入；已失败的prefix-finish deferral也不改名重启。一般victim问题未被否定，重新投入需具体的新损失/动作预测。没有新runner/候卡，资源不是本次未跑GPU的原因；goal ACTIVE，中心贡献和可投稿证据尚未成立。

# A latest — 2026-10-09 固定CacheOPT式组件完成，转向释放窗口的最小诊断
[研究卡、完整数据与复现入口](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)已更新。cap-bucket r01四格各320/320完成、0失败/未完成，112raw核验、唯一分析FOUR_CELLS_COMPLETE_COMPARABLE，11份分析来源随组保留。两bucket各43抢占/36非tail实际改选/40 unique victims，预算41/10/37；释放中位109页对135，实际preempt/free及规则重算全MATCH，未发生零干预。
按预声明全外部到达mean flow，bucket−budget+.160918/−.043072s换号，输出率−.402%/−.110%，gap P95+2.561/+2.791s。更小KV不是更低受害代价：0066842晚完成6.53/6.26s、gap增加21.45/21.69s。输出少525 tokens；0066337从1024/length到453/stop，0046501从45到91，25序列/2长度/1终止不同，不能称等工作量或质量等价。首次分歧前已有+.497/−.066s漂移，不相减校正；仅两个运行对，未独立确认。停止此固定代理桶宽/tie/阈值搜索，不能据此否定完整CacheOPT；当前中心贡献仍未成立。
用户最新要求以释放窗口评价动作。已用raw核实首次662两选择释放50/83页，但都约.211s后首次allocation；已记录36个保留suffix请求各推进3token、0完成，均669 readmit，分桶victim又两次被抢。allocated/readmission held是快照、实际增量UNKNOWN，不把host request当ACK。可复现CPU脚本diagnose_capacity_window.py已完成并运行：加上旧max首662释放192页，三个实际端点首窗口仍相同。首次global完成664来自prefix自然stop（remaining866，cap不可预测），另prefix请求remaining4在665达cap；完整运行集合与合法suffix不能混用。后续679长窗口伴14次追加抢占，非单动作收益。下一步仅据原生重新分配门控检查快速部分恢复状态能否对三个现有候选给不同的动作前事件顺序预测；不开发复杂优化器、不新增GPU组。
00:50:56 CST整组1008.864223s结束，57669/SSH4613终态0、公共锁释放，无A runner/候卡。完整tar96,108,188B SHA4a55cda5…c9f3a保留local+remote，所有原始与失败完整保留。goal ACTIVE，尚不可投稿。

# A latest — 2026-10-09 新科研基准已重读，固定近邻组件已冻结部署
[一页研究卡与当前判决](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)继续区分H_problem/H_model/H_method：强简单方法后的重要可控损失尚未成立，局部容量取舍已测，新增方法未获支持。复用max-release四格与CPU分桶证据，不重做已答问题。下一项只检验未覆盖的CacheOPT式固定cap分桶组件对普通预算：同mixed320/正常KV，remaining/bucket/bucket/remaining，128-token桶及tie预先冻结，不称完整复现或新贡献。
新包manifest3006a941…e922、plan9b768caf…ed87已部署；38 payload中仅staged/runner改变，单项CPU回归通过。00:34:07 CST唯一controller57669/SSH4613实际提交并即时取得同一2304:4312099778公共锁，handoff0.188s后启动；尚无完整GPU结果。核验289份完整本地/remote tar文件后，只回收上一max-release终态重复副本，根盘达3.956GB，原3GiB线不变。无第二A任务，后续只接续原进程。确切命令、版本、资源上限及结果分支见RESULTS文末。目标ACTIVE，尚无中心贡献或可投稿结论。

# A latest — 2026-10-09 最大释放减少抢占，却使全请求平均完成更差
[最新研究卡、原始与复现入口](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)已更新。max-release r01四格各320/320完成、0失败未完成、282868 tokens；逐请求长度/终止原因相同，26序列改变，质量未验证。唯一分析FOUR_CELLS_COMPLETE_COMPARABLE，112原始文件全部本地核验，10份分析来源随组保存。
两个预算臂各41抢占/10非tail实际改选/37 unique victims；两个max臂各27/24/24，全部native preempt/free唯一关联、实际释放匹配。max中位释放214页对135，抢占间隔中位14步对9.5；但mean flow增加+.785830/+.576574s，最坏请求晚完成约21.8s。吞吐+.144%/+.588%，gap P95−8.711/−8.512s，最长gap反而+.489/+.758s，是明确取舍。按主目标先保留已知预算原则，不将少抢占当全局收益，不称新增方法贡献。两个运行对、非独立负载确认；预算相对tail的旧换号结论不被跨组时钟覆盖。
整组1009.624s，于10月8日23:46:06 CST结束，48782/SSH26921终态0、公共锁释放，无A runner/候卡/新提交。完整归档95,701,887B、SHA92271615…7e35保留本地及远端；模型/输入/资源/冻结代码均未改，失败记录保留。
查新明确CacheOPT在同剩余长度桶内优先小KV，当前两端点未覆盖。固定128-token cap分桶CPU代理在两条budget轨迹各37/41不同于已测预算、两条max轨迹各25/27同时不同于两端点；UNKNOWN0，异选同释放0，伴随少释放81/99页的中位差。只是未测近邻组件机会，不是服务收益或作者默认实现。下一项有判别力的在线工作应先覆盖该固定组件对普通预算，不发展复杂评分器/桶宽搜索。当前仍无中心贡献、独立确认或可投稿结论，goal ACTIVE。

# 上一完成结果 — 普通预算保护有个体价值，主目标净收益不确定
已重新读取工作区AGENTS，并据此更新[一页研究卡与完整结果](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)。remaining-budget r02四格均320/320完成、0失败未完成、282868 tokens；两个候选各真实改选10/41，两短请求提前约30s完成，但七个长请求稳定推迟、max-gap增加约1s。主mean flow差−.364487/+.061444s、吞吐+.209%/−.388%，不能宣称稳定全请求净收益。逐请求长度/终止原因相同，19个序列改变，质量等价未验证。普通预算规则归强简单基线适配，不归新贡献；停止它的同域调参/自动重复。
112原始文件全部本地回收核验，唯一分析FOUR_CELLS_COMPLETE_COMPARABLE；整组1039.720s，于22:38:16 CST退出0并释放公共锁，39549/SSH94131终态0。r01磁盘0cell失败保留。该完成组无遗留runner/候卡；下述新组独立提交。目标ACTIVE，中心贡献及可投稿证据尚未成立。
现成轨迹的新判别：预算臂各37个请求的max-gap跨越真实抢占，其他请求最大不足.19s；剩余长停顿不是TTFT混入。完整native最大实际释放影子41/41异选、额外14–176页，同时等价maxheld/maxcomputed；它未被旧5090/受限pure-decode size实验覆盖，也没有新物理信号。已冻结普通预算与完整max-release一次直接端点ABBA（manifestbd72e541…d8f1a、plan938c0579…b362）；23:29:16唯一48782/SSH26921提交并取得同一公共锁，23:31:22实查首remaining-budget臂RUNNING。新组没有完整结果，不把影子当服务收益；只接续原进程，不扩展预测器或重复候卡。

# 上一完成结果 — 严格预算ABBA零干预
已重新读取工作区AGENTS，RESULTS开头维护服务目标/部署域/容量模型/近邻及三层假设。当前最薄弱仍是强简单方案后是否存在重要可控损失，不把候选失败等同问题不存在；主目标继续全部外部到达mean flow，允许并报告取舍。
session-native-equal-release-budget-westd53005-20261008-r01四格1280/1280完成、0失败未完成，各41抢占且0 proposal/0实际改选。四臂输出量282868、逐请求token/结束原因及无时钟输出事件完全相同；mean flow差+.035629/−.196071s不可归因策略。旧706/712机会在当前704/710分别因替代者不在suffix、少1释放页并pending而不合格；停止严格匹配规则的本域投入，不扫页数阈值。一般剩余量简单基线和完成归还假设未被零动作否定，下一步只用已有数据核对它们是否有未测试的合法动作及旧size失败覆盖范围。
21:24:03 CST整组1041.878s结束，29913/SSH99005退出0、公共锁释放。112raw本地SHA核验；原分析错用prefix-on coordinator检查，原件保留，定向纠错r02为FOUR_CELLS_COMPLETE_COMPARABLE，raw/GPU代码/指标不变。
后续CPU判别：完整suffix最大remaining影子4/41异选，其中704/710短预算机会不被旧arrival-once覆盖，但少释放33/69页；不能称同容量或预估服务收益。旧max-size只测过5090/小KV/非current pure-decode候选，不永久替代本机完整maxrelease强基线。已冻结完整普通remaining_budget（manifest45c5fd77…7cd4，planb10cc59d…f206），单项生命周期检查通过；r01于22:09:24因根盘低于3GiB在模型初始化前0cell失败，38116/SSH50235终态退出1，原记录本地保留。核对并回收仅A已归档重复展开副本约1.24GB后，r02仅session/resource说明变化（plan52b1b6ad…e360）；唯一39549/SSH94131于22:20:56提交，22:38:16 CST四格exit0/archive VERIFIED整组完成，39549/SSH94131终态0、公共锁释放，无A runner/候卡；112raw远端核验后限速回收，科学分析待完成。一次ABBA检验简单原则价值，不称新贡献，不自动重复。完整原始、判断及命令见[RESULTS](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)文末；goal ACTIVE，中心贡献/可投稿结论未成立。

# 上一完成结果 — strict-equal-host：局部复用未形成稳定净收益
r02四臂1280/1280完成、失败/未完成0，112份raw核验，唯一analysis为FOUR_CELLS_COMPLETE_COMPARABLE。19:36:49 CST远端22623已完成退出、公共锁释放，无A runner/候卡；SSH94126断线255不是实验失败，恢复后核实终态，未重启。冻结manifest1288e384…99a49、plan8ae5acc6…a6100；r01 CUDA前0cell磁盘失败保留。
两候选均step994实际改选1次，释放严格205对205页；host16/166页差未全兑现，GPU首次local174页与host重叠，替代者最终只请求并确认40页host加载。两对mean flow−.049757/+.204894s、吞吐+.085%/−.222%；替代者flow+1.130/+1.426s、max-gap+3.650/+3.698s。P95 gap有所改善，但完整净收益符号翻转、存在动作前漂移，收束该host动作，不追加调参/重复。原stage同首输出、纯decode条件需求≤1页结论不变。[RESULTS](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)保存完整指标、命令、原始与下一判别问题；goal ACTIVE，无中心贡献/可投稿结论。

以下为历史状态。

# A latest — 2026-10-08 贡献说明已更新，partial恢复ABBA唯一候锁
本线RESULTS开头现维护中心问题/近邻具体差异/候选原则/已有证据/未成立主张；无完整净收益或可投稿结论。基于刚完成共享诊断的新nonself partial入口，包manifest5d2f04a0…90a09、计划8eb40068…17ca52已冻结部署，同320输入/当前GPU正常容量/恢复/准入固定，tail/partial_once/partial_once/tail一次ABBA。09:35:19UTC唯一9990/SSH55292等同一2304:4312099778锁（6342持有、7091在前），session不存在，无A CUDA/第二候卡。新probe未出GPU结果；原共享诊断320/320、27native/0实际改选仍有效。只接续现有handle，完整断点见[RESULTS](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)文末，goal ACTIVE。

# A latest — 2026-10-08 新设备共享前缀实测完成，下一步partial恢复单次干预
当前GPU正常容量71.7832GiB，320/320完成、27native/0funding/0策略替换；11次suffix含共享候选，6次同held差release均不优于原tail。最大release影子26次改选，不能当收益。mean flow63.392s、max-gap P95 4.444s、282776输出。39原始输出本地SHA核验，16:51:35 CST 2519/SSH41658退出0并释放公共锁，无A在途/候卡。新证据step1090：非self tail已有402输出但computed1592尚在恢复；旧pure-decode全局fallback掩盖53个合法替代，最近即时释放101对100页。准备单次state-trigger替换ABBA，尚未运行，不复活旧self-currentguard/arrival/共享评分器。完整原始/版本/命令/判别分支见本线[RESULTS](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)文末；goal ACTIVE，未证明完整净收益或中心贡献。

# A latest — 2026-10-08 新授权主机接续，唯一控制器等待共同锁
westd:53005连通，PRO6000 UUID51b8e4bb、driver580.95.05、110GiB cgroup；旧A目录/模型/运行时/种子保留，新增共享前缀包已部署。2519/SSH41658唯一提交，实查locks_lock_inode_wait，公共锁2304:4312099778由2066持有，A未CUDA/session未创建。新plan be6c63b3…6bdfcf先当前设备正常容量profile再单tail诊断，不作旧GPU时延对照；输入/选择规则/Q1/传输/准入仍冻结。完整断点见[RESULTS](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)文末。goal ACTIVE，原连接BLOCKED是已解除的历史状态；暂无新GPU结果。

# A latest — 2026-10-08 共享前缀诊断就绪，连续三轮SSH拒绝，BLOCKED
新包`candidate_native_shared_prefix_probe_r01`/计划`plan-native-shared-prefix-probe-20261008-r01.json`已本地冻结：64重复文章对＋192singleton、原320 ID/0.01s到达/40×128+280×1024预算、正常KV；只做tail物理释放机会诊断。startup资格允许已核实的Unitary coordinator，Q1准备/提交/保护期private gate保持原样；max-release仅影子，不执行策略替换。manifest`a9c85333…786c`，3项定向CPU检查通过。上传连接reset，旧master消失，重连westb:25495返回Connection refused；新controller未提交、无新增A候卡，远端部分上传情况UNKNOWN，GPU结果UNRUN。完整版本、部署失败、恢复命令及判别规则见[RESULTS](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)文末。上一真实overlap实验仍已完成；第三个连续goal轮次复核仍Connection refused，下一真实实验依赖外部连接恢复，目标BLOCKED、未完成/未证明净收益/未可投稿。

# A latest — 2026-10-08 新时相诊断完成，排序机会仍不足
mixed-overlap-r01固定256初始突发＋64个30–42.6s后续到达，320/320完成、0失败未完成；35native/0funding/0策略替换。38.642s首抢占后仍20个新到达，外部stream区间5次抢占，28次新旧阶段候选共存。9次同held/computed整页机会共13替代，全部host missing更高；最大剩余预算/最少输出/原始arrival均0影子改选。唯一pure_decode回退只有自身候选，没有被挡掉的替换空间。mean flow68.805s、max-gap P95 16.739s、282847输出；不同到达时序，非策略收益。06:09:22 CST 122268/SSH77006退出0并释放GPU/共同锁，28原始输出本地核验，A无任务/候卡。停止此点的起点/节拍搜索，不重启退休规则；实际恢复host命中未采集，不能由next held推断。原始/唯一分析/完整分布/实现/命令见[RESULTS](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)文末；目标ACTIVE，尚无论文主结果。

# A — arrival-once ABBA完成，净收益归因仍弱
四格1280/1280完成、失败/未完成0，112原始输出核验；05:27:41 CST controller114335/SSH97687退出0并释放GPU/共同锁，A无在途/候卡。两候选各真实改选1次（step681，0053843替0065237，释放150对84页），全请求mean flow −.253/−.068s、吞吐+.296%/+.070%；动作前候选已快.159/.051s，不能将全部差异归因纠偏。原tail只多输出9个token，约.69s后再次被抢，四臂仍40次；替代victim最大停顿+.372/+.356s。抢占后缀从step695对齐，但两参与者输出调度仍不同，原tail token序列改变；非严格同状态、非质量等价。弱正向计时/容量混淆未解，无独立贡献或可投稿结论。r02资源版本、完整指标/原始/代码/命令见[RESULTS](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)文末；不追加原规则阈值/动作位置扫描，goal仍ACTIVE。

以下是上一阶段记录，目标仍未完成。
r03四格1280/1280完成，112输出SHA已本地核验；controller66669/SSH95726退出0、GPU与共同锁已释放，A无在途/候卡。候选实际改选34/49、33/48，mean flow +0.630/+0.712s，吞吐−0.546%/−0.641%；反复绕开同一tail，容量差中位扩大至52.5/59页，并有明确victim损失转移。完整分布、原始/图和唯一分析见本线[RESULTS](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)。
两host测量期各有一次fused_moe JIT，对齐约0.70s共同输出停顿；两个tail无此警告。当前最薄弱环节是启动编译成本与持续服务损失的归因。统一四臂私有Triton种子诊断已单次提交：16:37UTC实查唯一controller73861/SSH95639持原共同锁，前三格exit0、最后tail运行；其余规则/输入/预算/请求预热原样，不继续调host阈值。原负结果保留，目标未完成/未可投稿。下方旧运行/候锁/BLOCKED均为历史现场。

# A latest — 2026-10-07 15:32UTC 正常容量host-near r03唯一等待
唯一controller66669/SSH exec95726已实际提交并实查为共同flock2304:15049831297的WRITE*等待者，66402持锁；A尚无CUDA，r03 session未创建。新入口仅在首次取得原锁后增加最长300秒的GPU上下文交接等待，计入原4800秒组上限；无进程信号/自动重试。tail/host_near/host_near/tail每格320及冻结包/输入/预算/控制机制均不变。原r01/r02入场失败保留，r02不再提交，后续使用独立r03。准确命令和计划hash见本线[RESULTS](experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md)文末；当前无新策略结果，目标未达成。下方旧BLOCKED/无候卡状态仅为历史。

# A latest — 2026-10-07 新规则GPU仍未启动，失败断点已保存
2026-10-07 14:36:19UTC恢复后第三个连续goal轮次实查：58138与GPU worker58293均存活，共同锁2304:15049831297由58138持有，GPU88230MiB/89%；r02未创建，A53702消失、无A进程/候卡。没有新科学数据或策略执行。代码/输入/分析与非阻塞命令均已准备，无需继续离线准备；下一有信息动作依赖整卡释放，目标再次BLOCKED，不是研究完成。
14:22:37UTC实查56960已退出，57657已接力持同一锁2304:15049831297/GPU88188MiB，进程存活；A53702消失、无A进程/候卡，r02仍不存在。本轮无新策略执行或科学结果；代码/命令/分析已准备，等待真实资源释放，不重复候卡。
14:24:25UTC第三个连续goal轮次核实同一共享GPU阻塞：57657处于running并持共同锁2304:15049831297，GPU88188MiB/93%；A53702消失、r02 session不存在，无A任务/候卡。实现、输入、分析及命令已完成准备，下一有信息的步骤需要真实GPU资源释放；目标已记BLOCKED，不是研究完成或机制负结果。

r01 controller53702候锁1083.987秒后14:08UTC取得共同锁，但PID51255仍14464MiB，CUDA前ABORTED、0cells、立即释放，SSH29836退出1且PID已消失。随后GPU/锁确实空闲时提交r02非阻塞启动，56960已接力持锁导致EAGAIN；r02 session不存在。两次失败原始文件已本地保存；当前无A进程/候卡，不反复排队。可运行r02计划37f13077…d381141与命令见RESULTS文末，冻结源码manifest5226b31a…40d2a2未变。

本轮最薄弱环节仍是host驱动victim变化的完整请求净收益。原探针50次已有事件中43次ready>0，38次随后准入held更小；恢复前等待中位20.600s、准入→输出.225s，都含调度而非隔离copy成本。新分析补选择容量混淆、准入分解、客户端发送滞后与完整排空，不增加GPU观察或改规则。还没有执行新规则，不能把资源失败写成科学负结果。后续按实际动作/总体损失区分收益、成本估错、损失转移或零干预；不靠阈值搜索救场。没有费用遥测，未知共享余额。目标ACTIVE，未达论文投稿证据。

以下旧候锁状态保留为历史，不覆盖以上事实。

# A latest — 2026-10-07 正常容量 host-near ABBA 唯一候锁
14:04:49UTC实查controller53702仍存活，wchan=locks_lock_inode_wait，同一共同锁由49617持有、51255在前；A session尚未创建，无CUDA初始化、无第二A等待者。唯一SSH29836仍待原进程结束，不重启。上一goal轮有实质进展（探针证据、冻结补丁/部署、实际提交），当前阶段为已验证等待，目标ACTIVE。


新包candidate_native_host_near_r01已冻结/部署，manifest5226b31a…40d2a2，plan479f159d…65da8。仅普通allocation-failure suffix优先host missing严格改善且held≥tail、额外held最少的候选；无阈值，保护/prepare/未知回tail，其余恢复/准入/传输保持不变。15 CPU检查通过。探针20/50影子动作、额外held中位4.5页；最少输出与arrival均等tail，合并对照。尚无新规则GPU收益。

tail/host_near/host_near/tail每格320正常容量high；唯一controller53702/SSH29836已提交并实际等待共同锁2304:15049831297，3600秒上限，不自动重试。目前49617持锁、51255在前，A未初始化CUDA/session。新输出/root/moe-a-victim-20261007，避免数据盘不足，原4GiB余量门槛不降；模型/锁/overlay不改。确切命令、全部配置与限制在本线RESULTS文末。

# A latest — 2026-10-07 原生 host 探针完成

r03真实GPU完成，controller43175/SSH75811退出0，13:10:07UTC释放共同锁，A无在途/候卡。320/320完成、失败0；50次原生抢占实测释放全部等于held。正常KV71.8379GiB、host16GiB与固定high输入。唯一分析 `experiments/admission_capacity/victim_selection_20261004_A/analysis-native-host-probe-20261007-r03.json`，28文件SHA通过；本线RESULTS记录指标/例子/命令，原始失败保留。

host在41/50决策的某个同held+computed整页层内有真实差异，但tail自身同容量层中missing已最少，严格同容量host规则0动作。不限容量min-missing影子22/50改变、释放量混淆明显；下一步准备仅普通allocation-failure suffix中的最小额外容量host改善规则，尚未冻结/运行。5807/5807候选pending计数恰等于整页边界，暂不发展此信号；arrival仍与tail重合。不放宽funding触发、不改Q1/传输/准入。当前无策略净收益或独立贡献结论，目标继续ACTIVE。

以下保留旧现场记录，不覆盖以上完成状态。

# A latest — 2026-10-07 正常容量实测完成

最新：13:05:04UTC A controller43175候锁1324.373秒后已取得同一整组锁；r03 receipt=RUNNING，入场GPU0MiB/0%/无进程，模型/磁盘校验通过。high-tail-host-probe真实初始化中，仍固定71.8379GiB/host16GiB/320 high/384预热与全部原控制机制。等待数据后再选择信号，当前还不是完成结果；仅SSH exec75811这一A作业。

最新恢复现场覆盖下方旧BLOCKED：11:54UTC GPU/共同锁空闲，r02实际提交但在CUDA前因磁盘317MiB退出；原始失败已保存。A完成私有缓存清理、重复输出hash去重和74个raw/warmup无损压缩，余量4.63GB；两模型权重不同，均保留。相同科学包/输入/预算的r03仅换结果目录，唯一controller43175/SSH exec75811已在共同锁1800秒有限等待，12:44UTC确认存活/内核等待；GPU测量仍UNRUN。当前无其他A等待者，不修改冻结探针，不把资源整理称性能结果。另r01最终receipt已为ABORTED，早期RUNNING只是观察到的中间状态，并非runner未记录异常。

当前目标 **BLOCKED（共享GPU）**：2026-10-07 08:55:16UTC第三个连续goal轮次验证同一资源阻塞。13431已退出，但共同锁2304:15049831297已由存活的13862取得，GPU worker14278占13928MiB；A r02仍不存在、无A进程/候卡。准备、必要分析及紧邻文献已完成，下一有效步骤依赖实际整卡空闲，不能以再加离线方案代替GPU证据。恢复后执行既有r02非阻塞命令，不重跑正常容量表征。

用户新入口westb:25495、实际GPU-94203fc3-1021-3a9c-a367-cff792479616。正常容量profile+128/256/320 tail已全部完成并释放共同锁，704/704请求、0失败；PID4153消失、SSH exec76258退出0。实际KV71.8379GiB，三格固定相同物理字节，宽预热均达到384纯decode。原始95输出文件本地校验通过，完整证据见本线RESULTS及analysis-pro-capacity-20261007-r01.json。旧入口历史阻塞不代表当前状态。

128低压0抢占；256/320用尽36780可用页，31/50原生抢占，最大gap P95为16.392/27.216秒，但funding选择均0：366/531次过年龄检查后被全局pending-transfer条件拦住。原生合法候选最多219/252，arrival影子与tail也全重合，故不空跑该对照。下一步仅补native allocation-failure入口host前缀/准确释放观测，在相同high负载做单格tail探针，再按可区分信号决定策略。不放宽触发、Q1、准入或传输规则；未宣称策略净收益或独立贡献。

探针已实现/CPU检查/部署（candidate_native_host_probe_r01，manifest8568fa3e…23103），仍GPU_UNRUN：唯一候锁13301取得同一共同锁后，入场检查发现前一作业11853仍占14464MiB，初始化CUDA前exit1并释放。r01失败log/入场记录保留；receipt旧RUNNING为异常前遗留，不代表还在运行。A无进程/候卡者；当前13431持锁/GPU80052MiB。r02新结果目录计划及analyze_native_probe.py已准备，实际空卡后按RESULTS非阻塞命令执行，不重复排队。

08:52:47UTC续轮实查13431仍存活并持同一锁/GPU80052MiB，r02不存在。CPU分析补同held且同computed整页机会10/31、18/50及分析器严格层内口径，排除页数差混淆；FastSwitch CPU副本复用/依赖同步为已有工作，本线不能宣称这些基础机制新颖，具体决策区别见RESULTS。冻结GPU包未改。

# A current — 2026-10-04 victim-only

最新资源状态：A唯一900秒共同锁等待PID15254/SSH exec40366已在flock处超时退出（exit1），09:02:28UTC确认进程消失且session-pro-capacity-r01未创建。B controller13233/worker18195仍占卡；A无GPU进程或候卡，不再重复排队。原始wait-timeout-pro-capacity-launch.log已保存；正常容量GPU实验仍UNRUN。09:07:37UTC最新实查B已退出，C PID15300接力占80186MiB；A session不存在。恢复后连续三轮同一资源阻塞，目标再次BLOCKED。此状态覆盖下文较早候锁/blocked现场记录。

独立本线目录 `experiments/admission_capacity/victim_selection_20261004_A/RESULTS.md`。只改变 recovery funding victim，固定原生Q1/触发/准入/传输。tail/host_missing/host_missing/tail 四格512/512请求完成、0失败，108输出文件校验通过；共同锁已释放。200次决策全部候选missing=0，host规则改变0次，失败类别为“信号退化/无动作差异”；mean flow+1.89%/+2.39%、rate−1.30%/−1.10%，不支持净收益或novelty。所有输出差异和个体损失保留，唯一完整分析 analysis-host-r01.json。

下一最小组为已有原始arrival简单规则，避免恢复append重写尾序；只改funding排序，其余原样，tail/arrival/arrival/tail各128，candidate_arrival_r01及plan-arrival-r01.json已冻结/CPU检查通过。首tail影子有19/53不同选择，多数也改释放容量，非同容量因果结论。实际提交遇B组controller8302共同锁忙，GPU0、未创建session-arrival-r01，无A在途/候卡任务。远端 `/root/autodl-tmp/moe-a-victim-20261004`，同一锁 `/root/autodl-tmp/moe-research-gpu.lock`。27CPU检查与私有torch RECORD overlay修复、原失败日志保留；全局环境不改、无push。下方20261002量子结论与全部历史保留。

PRO 6000补充：上述4096页首组明确标为受控机制实验。当前优先以.90实测正常KV并固定预算，在BF16/4096上下文下增加统一并发384及嵌套128/256/320长短混合请求，先做tail容量表征。小容量arrival对照已准备但暂缓；未把预计容量或CPU检查当GPU结果。新包candidate_pro6000_r01已冻结并远端校验（manifest ffdbab66…7fa64），27CPU检查通过；真实提交在共同flock返回EAGAIN，session-pro-capacity-r01未创建。其他会话PID9792占用88210MiB，无A候卡者；仍未测正常容量。可运行命令、完整输入与分析入口见本线RESULTS。 续轮实查PID9792仍存活/占卡，连续三轮共享资源阻塞；上一轮目标BLOCKED；本轮恢复现场9792已退出但12017占卡，13142等同一锁。随后08:41UTC确认12017/13142已退出，B controller13233/worker13413占卡。A未排候卡；本轮补齐双tail到达顺序影子和host-vs-size混淆检查，准备就绪等待真实GPU与共同锁释放，未缩小论文目标。 08:43UTC恢复后第三轮仍实查13233/13413占卡，目标再次BLOCKED；正常容量GPU结果仍为UNRUN。

---
# A progress — 2026-10-02T10:02:01.322235+00:00

本轮目标完成于“形成足以决定换方向的具体证据”这一允许的终点。决定：停止当前 h/d 自适应量子规则的开发，保留已实现的原生资源/执行机制，下一研究问题转向 victim/peer 完成代价。不是整个恢复量子家族 NO-GO，也未宣称达到 CCF-C 投稿标准。

## 主要实测结论
- 强对照普通回填/原生/原有fund均完成128请求。native/ordinary/fund P95输出间隔6.965/3.885/1.575秒；fund比ordinary实际速率低1.52%、平均flow高1.85%，86/128请求完成更晚。原有funding/priority/protection并非新增创新。
- 新原型跨全量子物理KV、运行槽和执行token预留，并选择性接纳peer。26包内与6纯模型CPU检查已通过；六格真实lease动作均通过核对，受保护目标无实际抢占，peer/victim最终均输出并完成。条件进展不等于无条件q输出或墙钟SLO。
- 两轮q1/fixed4/adaptive比较按相反顺序执行、各128请求。adaptive相对fixed4的P95变化+14.44%/-10.80%，实际速率-1.00%/+0.74%，平均flow+0.35%/+1.62%。已知成本desiredq始终3（103/103、75/75次）。排序不稳定，不继续靠阈值搜索包装独立优势。
- 健康dev16低压三臂均16自然EOS、7/16正确；512页五臂均16自然EOS、8/16正确，块内序列一致。独立16题稳态512页五臂均自然EOS，但native/ordinary/q1/fixed4/adaptive正确数10/8/8/9/8，非native每臂有10/16序列不同。全部健康域均无恢复量子动作；不能主张质量等价或将差异归因于自适应动作。
- 两个健康512页组有完整原生传输完成计数、分开capture/drain。copy-work可重叠，不是暴露请求延迟。文章组完整controller CPU、传输开销及活跃自然任务质量仍未测；上游完整相关系统和结构消融未完成。

## 交付与资源
统一索引：`experiments/admission_capacity/20260929_commit_recheck/recovery_quantum_20261002/README.md`。其中含模型、冻结候选、可复用入口、全部raw和失败、结果JSON及图。决定记录`output/RESEARCH_DECISION_20261002.json`；最终工作稿`paper/draft.tex`与`output/pdf/draft.pdf`，6页、3图、1表，双次编译无警告，最新全部页面已视觉检查。

22个原生实验单元共1360次请求执行（不是1360独立任务）完成。最终controller35189在1790934663.1462655结束、GPU边界为空并释放共同锁；现场后查确认controller消失、无A作业或排队实验。A自有SSH控制连接已关闭。全部权限/上传授权已解决，无待确认事项；无push或公开发布。

唯一未来研究问题：什么当前可观察的peer/victim代价与剩余服务能解释完整请求损失？重新打开本规则需新的异质成本/实际服务headroom证据和有恢复动作的自然任务对照，不以改名或阈值调整复活。历史记录保留如下，旧RUNNING/UNRUN/BLOCKED字段不覆盖此状态。

---
## Historical checkpoint retained below
# A progress — 2026-10-02T01:25:08.952855+00:00

Independent A line; goal BLOCKED on SSH access, NOT READY. No A GPU job; all controllers/monitors/export/download complete.

## Latest: repeated oldest recovery R02 positive exploratory block
All128perarm complete; all20frozencriteriaPASS. Native/queue/fund p95gap7.938750/9.581802/1.552190s, globalmax12.066355/12.883049/2.187850s; rate1507.069617/1507.384391/1497.005644; flow37.055118/37.590503/37.487052s. Fundvsnative gap~80.45%lower,rate-.668%,flow+1.166%. Fund58episodes/57actualevictions/1directresume/21sources/0cancel; fullnative/Q1/retire/rawjoinsallclose. Preempts46/48/103, so in this block morepreemptions accompaniedshorterstalls. Fullrequestmaxgap57better71worse,median.041560→.055986;goodput14higher6lower versusboth. Naturaloutputs127401/128154/127286,stops4/3/4;noequalworkspeedup.
Canonical A_NATIVE_OLDEST_REPEAT_TRIPLET_RESULT_R02_20261002.json SHAb1cc9b0487e3ed372f277606fd763c750d9c3aecc1948aa06dbd58da4482f5da. Candidatebfe6858307624b09dcac0d4e57a47ee965c9279e643b4d98c002ae3fde0a4946; analyzer7e954006f8a7028f8a92461d14bdb7360a4e9b04ffe0c77952015d4b799a2d33. Runtime436.408824s,allGPUEMPTY; raw130files/188411114B,81hashesverified; tar0cbe2ff369943c43bada213deece807006874ba9b289d4b3aabb3f38c230b47c. Remote/localfullraw andtar retained. R01zero-cell25.071s diskabort included. R02dedup1326immutablepayloadpaths across63completedownpackages,allpaths retained; root2145824768Bbeforecontroller. Neverrepeatpreps/controllers/exports. Warmseedimmutable.

## One next direct experiment: frozen, SSH unavailable
Strong ordinary / unchanged repeated fund / native is FROZEN and UNRUN. Candidate manifest7c5221fb80402843e10c459471045baf7c2ea50541d0eb4f461c26ce46046c01; analyzer90418d39bdb9843f86cb96ccef6d657ac5666a43f6942d10d1a4a1844b5c9d22; plan81840eb19c2378958b31f624c92e8c0ed09e7a50b5dcc63c796eb409c46077a4; staged tar39de3699d2ed17f8cd657a6ce20340361743bc636c52d2b4ad7b1d6afa49249c (5150387B). Primary22937 and backup45495 both return SSH connection refused. Upload failed before launch; prepare/controller NEVEREXECUTED. Current endpoint requested; existing broad resource authorization remains. Three consecutive goalturns confirm bothports refuse; blocked status recorded, no automaticretry pending. Resume frozen upload + prepared commonlock script, then all3cells/export/readback/frozenanalysis. No remote job currently submitted by A.
Primary criteria: fund p95gap below both,rate>=97%,meanflow<=105%; all128/all20frontiers; ordinary must actually admit freefit requests. CacheOPT/UniBoost overlappriority/funding/protection, so no novelty/freshconfirmation/READY claim.
Local residual analysis: all58 anchor→nextoutput63–79.5ms (median67.7ms); worst2.18785s gap includes2.11708s beforeanchor and70.77ms after. Aggregate pendingtransfergate546 rejects cannot attribute specific delay. Of71worsened requestmaxima, medianincrease4.35ms but7increase>1s. Full gap segments and pairs retained in A_NATIVE_OLDEST_REPEAT_GAP_SEGMENTS_R02_20261002.json.

## Latest direct experiment: actual oldest recovery funding R03
Three arms native / queue_only / queue_fund complete, each128 requests, no failure. One actual victim preemption, native target restoration and Q1 output protection executed; all frozen one-shot criteria pass. Target source0042224 trigger-to-output9.155/8.930/.0673s, preceding output gap10.508/10.256/1.713s. Different physical states (6/7/71 prior outputs), so not an identical-state causal effect. Displaced victim pause7.238s absolute; its native/queue counterpart pauses7.553/7.611s, not evidence of a7.24s incremental penalty. Target later full-trajectory maxgap4.480s remains in metrics.
Output rate1513.008/1517.654/1509.396, meanflow37.471/37.308/36.135s, globalmaxgap11.827/12.756/11.643s. Outputs128148/128154/126304, naturalstops3/3/5. Full128 effects/20goodputfrontier and all sequence differences retained; no equal-work acceleration claim.
Canonical A_NATIVE_OLDEST_ADMISSION_TRIPLET_RESULT_R03_20261002.json SHAf76147d85e97cd2885554bee50767987696d6cee6ba6711e9af48c8e35ad9940. CandidateR02 manifestf29d62e904355cc1c2261ffe7352a791ec69a2ec772fb37f8afdfe0de8348928. Corrected commit gate explicitly permits only private known disjoint running STORE jobs, preserving native victim flush. R01 cancelled with0actualforced; R02 abortedbeforeanycell on diskreserve. All retained.

## Completed failure boundaries
Size-only min/max victim, completion deferral, full-running BidKV, density continuation and both fresh current-guard blocks fail their service criteria; no unchanged expansions. Native shorter host-prefix restore does not reduce full-history GPU need for the first new output. Earlier artifacts and exact claims remain in RESULT_LEDGER.md.

## Resources / retention
All completed raw results remain verified locally; repeated R02 remote compressed and uncompressed raw last observed intact. Strong prepare has NOT compacted any repeatedR02raw or changed remote paths. One-shotR03 remote uncompressed raw was compacted earlier; verified remote tar and local full raw retained. Never rerun completed prep/controllers/exports; warmseed immutable. Primary commonlock2304:29005388732, GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36. No open A PTYs. Paper30pages includes repeated main result/action followup; full128 request-gap ECDF added, buildPASS; Figure10 atp25. Full followup scripts/results retained locally.
