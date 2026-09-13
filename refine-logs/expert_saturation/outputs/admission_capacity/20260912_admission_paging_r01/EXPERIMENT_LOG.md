# 连续实验记录

主文档：[REPORT.md](REPORT.md)。所有原始失败与成功运行保留。

## 2026-09-12：完整模型 paging 接入

- 判断：6项GPU primitive测试不能证明完整OLMoE服务；WiSP支持的0.11.2是当前最短执行路径。
- 不变：同一OLMoE快照、BF16、top-k、单GPU；固定专家cap16、固定KV512MiB、LRU；关闭预取、动态分池和prefix caching。
- 改变：独立环境中的 vanilla / WiSP paged执行；2个短请求、每个输出8token、eager、maxseq2、tokenbudget32。此轮是接入和成本诊断，资源不同的resident对照不计为方法收益。
- 采集：真实请求tokenIDs与返回时间、实际KV块/bytes、16层scratch和pinned权重、warmup与measurement分开、miss/eviction；后续再补load-section事件。
- 命令与raw：独立远端目录 `/root/autodl-tmp/wisp-runtime-0112-r01`；安装为 `python -m pip install --no-cache-dir vllm==0.11.2 transformers==4.57.1`。本地回传将位于 `../20260912_wisp_olmoe_r01/`。
- 开始状态：HEAD `7059fc98e0d98a127ff4edda86d44f7f634d26f4`，已有未提交改动保留；GPU空闲，主机cgroup92GiB，数据盘空闲29GiB。没有覆盖vLLM0.26环境。
- 允许解释：成功则推进真实paging下的少量动作成本面；失败先定位模型加载、内核、pager语义或环境问题。该实验不终止主问题，也不产生调度收益结论。
- 接入结果：vanilla、paged、paged-traced均完成，各2条测量请求×8输出，token序列全部相同；实际KV512MiB/256blocks，GPU expert scratch3GiB、pinned master12GiB。测量wall分别0.065251/0.581240/0.597461s。原始数据为 `../20260912_wisp_olmoe_r01/integration/`；独立核算 `integration_analysis.json`。
- 搬运结果：paged与trace均1750miss、22,020,096,000bytes实际copy payload；trace唯一下界15,992,881,152bytes，extra6,027,214,848bytes全部在13-row prefill调用。load-section421.386ms包含map/launch，不当纯带宽；trace开销一次配对约2.8%，未形成稳定估计。

接入后的最小动作诊断：复用WikiText文章token输入，两条已有请求进入decode后加入第三条prefill，对照chunk8/32；固定engine最大能力、实际KV池和专家cap/LRU。首先用事件触发到达对齐共同pre-action状态；这是受控干预，不冒充正常时间到达压测。每arm独立重演并保留全部后续状态，先检查共同前缀与驻留集合是否一致。观察旧请求max-ITL、第三请求TTFT、三者完成时间与真实加载；未预定统一SLO，所以先展示完整权衡。具体旧版scheduler接口正在核对，尚未执行。

- 动作实验已启动：远端 `wisp-runtime-0112-r01/injection/run_campaign.py`，8cells顺序与配置固定在本地 `../20260912_wisp_olmoe_r01/injection_frozen/protocol.json`。两个旧请求32输入/16输出，第三请求128输入/8输出；short负控截其前8token；两旧请求输出恰4token时注入。所有cell同cap16、KV512MiB、maxseq3、engine tokenbudget64，动作只改实际`long_prefill_token_threshold`。每cell保留原始warmup、请求、step、pager事件，失败即停下诊断。

- 0.11动作8cells、plain4cells、CPU0–7 affinity4cells全部完成：16episode、46测量请求执行、32warmups、3篇重复文章。小chunk保护旧请求max-ITL但新请求TTFT与整批wall更差；plain出现排序翻转。原始数据与分析分别为`injection`、`plain`、`affinity`，详见主报告。
- 0.26 transfer：复用已验证适配，default/prefill32 ABBA四cells完成，共16测量请求执行；结果`../20260912_wisp_olmoe_r01/native_transfer/results`。两策略wall3.6359/3.3590与3.3573/3.6634s，排序翻转；可识别同层重复copy分别74,956,406,784与76,919,341,056bytes。旧raw未采入口LRU，明确未验证，不由新源码补证。
- 当前实验：新`expert_baseline_frozen/protocol.json`，保持0.26/cap24/KV1GiB/四篇64输入8输出/CPU0–7，比较token分组与resident-first expert互斥分组。新增小模块复用原生partial kernel与WiSP LRU，先独立校验再ABBA；这是强执行基线，尚未运行，不作novelty/调度GO。

- Expert分组资格与ABBA均完成：资格16层64row有限、最大relative-L2 0.0047597，输出与token暖机相同；性能token3.204/3.366s、expert1.483/1.232s，实际payload124910567424→37685821440bytes。相同入口完整cache/LRU、真实KV1GiB，额外allocator峰值4,241,408bytes。晚到两请求max-ITL变差，跨执行方式34/160层调用route union不同，保留边界。
- 原始数据：`expert_baseline/{qualification,performance}`；归档SHA分别`a0c1a256fb299537cd7120fdcfd2e7a664c530011b4cdfe7d8b83865d76263f3`与`132b49bd5d52ce3c8520ebab091a6c21a12f2981e55c5f02fbbdb54a51bd8c75`。上传最初遇PID20532，占用检查退出且未启动CUDA；其退出后再启动新资格/性能进程，记录于launch_notes.json。
- 下一动作：复用当前native_capture增加可选事件到达和分请求输出上限，测expert强底座下两旧请求4输出后的chunk8/128/no-new干扰。不新建Controller；当前仅实现中，未启动GPU。

- 0.26 expert事件六cells完成：hold/8/128及反序，共16请求执行/224输出，完整前态一致；chunk8全程78.6306GB、wall3.076/2.933s，chunk128 36.3394GB、wall1.587/1.078s，但后者首边界ITL更长。独立分析native_injection/analysis.json，全部raw保留，archive SHA82c83284ab453f3cce544f10ec6a0070d783673301a8fd4a9b5fe1cd2e348960。
- 冻结uniform模型首步copy误差(actual/predicted−1)为hold−52.67%、8−54.63%、128−7.50%；不能将其失败等同专家特征决策增量成立。旧请求0跨chunk从token12起不同，repeat内部一致。
- 当前后续：固定chunk128同engine四次，清空expert cache并共同暖机，加CPU/rusage/runqueue诊断；代码准备中。远端原SSH复用过期，误用11155旧helper的重连失败；改用本会话23478既有凭据后恢复，GPU空闲。无新资源购买。
- 同实例诊断协议已预写same_engine_frozen/protocol.json，仅expert/chunk128×4；reset不改变任何专家/KV tensor allocations或KV free queue。CPU schedstat字段依据Linux官方文档 https://docs.kernel.org/scheduler/sched-stats.html ，只取计数差且不与CUDA/load spans重复相加。接口已对实际安装0.26/WiSP源码只读核验。
- 同实例诊断已启动：remote same-engine-20260912-root-r01，driverPID30208；source archive SHA98984a0d98d89baf97b7eb9c325ce40959611eae999dc0b15dcba5fa6e968fee。启动前GPU空闲，真实结果尚待回传。
- 同实例chunk128四轮完成，wall1.089326/1.058170/1.070879/1.068858s，差异2.944%；每轮copy36,352,032,768bytes，cache/alloc/logic/记录的route与输出相同，物理KV block不同。原始same_engine，archive ec46b79048ede7da477bbfedc3c6288e34fefd442e8674034d07b356c57c9539。cgroup无新增throttle，CPU观察包络计入wall；不追认旧漂移根因。
- 下一实验已预写release_baseline_frozen/protocol.json：同实例fixed8/release8 ABBA，旧全部完成才解除cap，保持native预算；共同暖机两策略。只补普通phase基线，GPU未运行。

## 2026-09-13：普通phase基线清算

- release_baseline ABBA在同engine完成，共12请求执行/160输出。fixed wall2.175743/2.206423s，release1.979471/2.006894s；净copy77.674316→70.111986GB，后续decode额外0.364904GB已计入。call0–15已记录前缀与旧输出相同，call16 old各16且completed、新96/128时仅8→0一次；新max-ITL略升。原始release_baseline，archive5861bbca22b0fc938836f4bf863c3dfe167c7d386d8a0bf385c09baf92e81934；复算analyze_native_phases.py。
- 当前唯一假说转到同release8基线上的专家分组终态：保留当前mixed batch的decode expert集合，维持当前层每miss只load一次，但计入额外groups和未来真实状态。先做最小CPU partition/helper与同容量同resident交集的hash负控设计，未运行GPU，不主张专家信息增量或新颖性。

- 用户要求共享结论台账、竞争机制互为基线、GPU跑前检查留痕，已落实到experiments/admission_capacity/RESULT_LEDGER.md并链接两个活动入口。逐来源核对而未重算旧数：96.4%属于09-12 budget90；轮转0.39秒是缺席投影，其2.6%分母为合成时长；最新headroom_fast已实测1.386秒、吞吐−3.25%/−2.86%。长上下文旧三臂计划更新为四臂，GPU仍UNRUN。
- retention生产接入完成，12项CPU检查通过；四模式均继承release8，正反序共八次。资格范围改为每个独立qualification运行的首个actually applied mixed层调用；不增加未实现的逐层覆盖。GPU启动先补占用检查留痕，未改旧raw/frozen。

- retention冻结source archive SHA379a38384c200203532cb7660aa11a9159702bf93ba512eafc22e1f5a1de3d5f。首个launcher因远端脚本字符串换行语法错误在解析前退出，未启动GPU，launch原记录保留；修正launcher后同包两项资格完成。各测3请求/40输出，layer0/step4/10rows的首次实际跨组保护调用均finite/allclose(.01)，非bitwise，maxabs0.0009765625；relativeL2为decode0.002739、hash0.003070。仅各1层调用，不声称16层或质量覆盖。qual archive52b58a5652e44e0e3616da88776f6da9017150b4701b6d53fd707b037b581040。
- 已启动预定八次同engine性能矩阵，driver36940；全部继承release8、共同五段暖机，none/frequency/decode/hash/hash/decode/frequency/none，尚未回传结果。边界检查写gpu_checks.jsonl，禁止把反复ensure触碰作为独立hit收益。

- retention八次全部完成并回传，24测量请求/320输出，40份预热。analysis.json无issues，资源与逻辑动作前状态一致，各策略独立执行；decode payload+0.41%、groups+30.08%、wall+2.28%/+8.97%，frequency payload−1.70%但wall+1.35%/+0.24%。时间尖峰保留，未做扣除。performance archive a167eb2bce1e5392793bb4e3d2c56d1c6577a0b7a73a9d3d0370edf82e26445d。停止该完整decode集合保护formulation，完整性复核进行中。

- 新的相邻调用诊断protection_reuse.json完成，无新增GPU：8×192对、缓存连续性全部匹配。旧仍活跃时decode保护79.33%复用、高于同缓存未保护70.15%，但下一miss总量未降；跨完成边界保护复用87.39%、未保护95.68%。是各策略自身轨迹观察，不作因果或Oracle。用户随后指出噪声校准问题：本轮两次none的0.73%只是观察差值，不是噪声界；所有小幅耗时变化继续仅作描述。

- fresh GPT-5.6-Sol限定复核完成：WARN、无P0；运行与数值、四项核心检查通过，保留缺少per-row route/frequency重建的P1证据边界。审查补读既有environment后确认运行源6/6 SHA匹配，撤回执行源未绑定疑虑。分析helper归档缺项已由执行者补齐两份分析源及provenance，并在临时目录逐字节复现原analysis.json；未修改raw或预注册协议，未新增GPU或审计轮次。

- 本轮继续原问题，发现具体实现税的重开条件：当前保护先加载再长期占槽并非最少分组。在线当前层下界 g>=ceil((|A|-|P∩C|)/(K-|P∩C|))；延迟加载缺失保护专家至末组的构造待全调用核对。只优化执行顺序，保护身份/容量/当前miss-once不变。retention_order_frozen/protocol.json已预写两项独立数值资格与两engine×8episode对照；新增同域none A/A重复仍只作描述，不设噪声倍数门槛，尚未执行GPU。

- retention_order的3072调用构造全部达到最少组数且无额外load，10项CPU检查通过。冻结source archive1dda467b8e29c74a8ecfedddd0f8c89d2792324ef6200c38d833993d2fd9fe2a。首个远端launcher因PATH无python3而未执行，attempt01保留；使用既有绝对venv后两项资格完成：frequency/decode各layer0 step4 10rows、3/2groups、allclose(.01)、maxabs0.0009765625、relativeL2 0.002865/0.002424；非位级一致，仅单层调用资格。资格archive5fd991f917937d5cb45912ef654433d0cbe0e6078cb0093743d106b38ec64e8d已回传。随后启动两engine×8测量矩阵，结果待回传。

- retention_order性能16次完成，48请求执行/640输出，80份预热保留；archive65df53d24ed1880b15c05959c6987a90db434883be00734c35b6ad3122dabf6a。6144层calls均miss-once，late实际达到自身当前A/C/P下界；frequency810→714组、bytes69.306679→68.702700GB，none为636组/70.388810GB。late-frequency相对none四块wall−0.19%/−0.44%/+0.19%/+51.56%；late-decode+6.03%/+7.82%/+0.86%/+45.79%，不宣布稳定收益。
- 第二engine末块两late策略同模式bytes/groups/输出不变但全程变慢，动作前4calls已增加85/134ms；cgroup无throttle、GPU边界38项PASS。timing_addendum.json保留重叠范围，不追认根因、不删repeat。已查共享rotation四臂执行时间：其最后一格结束1789233581.454，早于本轮性能开始1789233995.265，不能将本次变化归于那个已完成GPU任务。下一实验为同程序none A/A，加GC与CPU核心/频率观测，保持当前策略和预热，定位动作前漂移。

- 本轮fresh GPT-5.6-Sol限定复核完成：P0=0、P1=0，四项核心完整性通过，same-family/provisional WARN（覆盖范围与动作前运行变异）。未确认稳定请求收益。独立规范化每种策略4次的当前row top-k、active/entry/final cache、分组/loads/victims签名均一致；不因此追认时间异常根因。代码、完整结果、台账和主报告已更新，GPU任务已结束。

2026-09-13 runtime_variance：同程序none/early/release8完成8测量+40预热，24测量请求/320输出，3072层row-topk/分组/搬运/输出一致，7源码SHA匹配、19GPU边界PASS。repeat3/5 call11耗时198.723/256.328ms，与gen2包络77.824/135.313ms对齐；其它call11为119.066–121.095ms。此前持续动作前漂移未复现，不宣称已解释。观测本身5.047–24.694ms计入wall。原始归档5bd2bf1f…e67d80完整回传，源与分析见../20260912_wisp_olmoe_r01/runtime_variance_{frozen,performance}/。下一步仅对照trace保留生命周期，保留全部原始记录并计入写出/回收/最终归档和完整执行周期。
本轮复核：reused reviewer / same-family/provisional WARN，P0=0/P1=0，四项必要检查通过；fresh线程受数量上限限制，未冒充fresh审查。计时范围ADDENDUM已限定capture内观测与外围写出的差别。结束查询GPU空闲，当前无本轮后台任务。

2026-09-13 trace_lifecycle：冻结memory/episode/episode/memory四新engine，每格8次none/early/release8及40份共同预热，仅改排空后trace/CUDA-event引用写出释放位置。3项flush测试及10项分组测试通过。四格完成0，96测量请求/1280输出、160预热/416请求/5184输出全部保留；全局call ID连续，每次636组/70,388,809,728bytes、row/top-k/输出相同，源码7份一致、76GPU边界PASS。结果见../20260912_wisp_olmoe_r01/trace_lifecycle_performance/analysis.json，code archive837e4df1…c821446c，result archive356dce27…d2b805。
episode相对memory的cycle变化−1.2825%/−0.2005%，capture合计−0.7798%/−0.5471%；已观测GC交集减少0.6718/0.6453s，写出包络反增0.0850/0.2732s。首对process−14.84%主要位于cycle外，不算trace收益。memory旧请求max-ITL出现218–249ms，episode范围120–136ms；两engine/模式、重复三篇文章，不构成总体噪声界或方法GO。ADDENDUM限定GC观测尾部、进程/归档边界和reset字段的继承措辞；原raw/frozen未改。下一步固定episode写出，复测已有none/early与frequency/late，不再增加selector。
四格子进程均已退出，无本轮后台任务。结果回传后的02:43:54（远端显示时间）查询见其它PID47546占29492MiB；后续GPU实验须重新通过空闲检查，本轮完成记录不回溯改为共享失败。
本轮限定复核完成：reused reviewer / same-family / provisional WARN，P0=0/P1=0；独立重建全部请求与calls、flush实际字节、动作时点、源码/分析哈希、76边界及GC并集。首对cycle外差异、GC观测尾部、reset措辞均已限定，停止追加审计。原始数据及源码归档哈希再次匹配；未修改current权威结论、未push。

2026-09-13 retention_lifecycle准备：共同episode写出下，两engine各8次none/early与frequency/late（ABBA BAAB及交换），参数/五段预热不变。runner仅增加矩阵和观测接入；新159行分析器复用实际请求/阶段账，旧真实fixture适用部分通过，两个CLI拒绝路径在修正测试命令缺失expert-cap后通过。source archive67c8670a006cd1d1867ddb2b0f9900992c0b74df9c8a6093d1f5d5d2648565a1，预写协议见../20260912_wisp_olmoe_r01/retention_lifecycle_frozen/protocol.json。
attempt01/02分别在GPU初始化前因PID54498/56066 ABORT/91，均无请求执行，全部回传且SHA为c0b16d3a…136964 / 5beb26f1…39a283。再次检查定位到已在执行的20格rotation holdout campaign，最后一格cohort0-block0-native_aa/PID57701仍存活；此前空闲是格间空档。后续只在该campaign全部终止且GPU重新空闲后，使用相同冻结包的新目录恢复，不覆盖两次ABORT、不根据请求结果重选或重跑。
该20格已全部COMPLETE、末格PID消失后，又观察到短暂WiSP进程58240；其退出且nvidia查询为空后，原包在retention-lifecycle-20260913-root-r03启动，driver58471实际存活。0_forward已完成0，1_reverse初始化中；当前是RUNNING，完整比较尚未形成。两次退出仅发生在初始化前，不存在删去请求结果或更换实验参数。
retention_lifecycle两格随后全部完成0且driver已消失；16测量/48请求/640输出，80预热/208请求/2592输出全部回传，归档f5dc1a7534360e51a383ee3de636e8cd6f2c012a39c5a120ba8b1d3d53aa530b。相同模式8次row/top-k/分组/cache一致，全部输出一致；每格15648条ID连续，6144测量层、38GPU边界通过。frequency始终少搬2.395%、多12.264%分组；capture八对四升四降，engine内均值+0.417%/−0.564%；repeat cycle均值+0.069%/+0.494%，不宣称净收益。首ITL八对增加0.099%–20.275%，最大者与约16.1ms GC区间对齐，timing_addendum.json保留全部16次边界，不删除或扣除。
随后86行CPU结构探针检查当前状态上“不增组才保护”动作：每次none状态126/384、frequency状态112/384调用保留非平凡动作，各8次一致；所有6144调用的下界/覆盖/容量及两种合法淘汰顺序下的entry-miss-once通过。结果group_budget_opportunity.json为POST_HOC_STRUCTURAL_OPPORTUNITY_ONLY；源码与冻结planner身份留存，未模拟guard的未来bytes/时间。下一项为同频率身份的gP=g0 guard接入与none/完整frequency/guard实际三臂对照。
本轮独立限定复核reused reviewer / same-family / provisional WARN，P0=0/P1=0；8配对全部16指标、逐row/frequency重建、flush/GC/资源通过；尾部合同同步项已修正。03:10:55远端GPU查询为空，本轮无后台任务；主问题OPEN，完整frequency formulation净收益未成立，guard仍未接入runtime或实跑。

2026-09-13 group_guard：13项分组检查与runner边界检查后，冻结同frequency候选的gP=g0约束；一独立数值资格和两engine×6个三臂episode全部完成。资格3请求/40输出，call1664拒绝、1665实际applied且allclose但非位级一致；性能36请求/480输出、60预热/156请求/1944输出、4608测量层、30GPU边界全部保留。每次none/full/guard为636/714/637组、70.388810/68.702700/69.797413GB；guard119 applied、71拒绝，逐调用等于自身g0。多1组发生old完成后step16/layer5的实际专家并集48→49，无保护动作。guard少47次加载=实际active少2+entry hit多45，保留full节省35.075%；这是实际轨迹账，不是固定未来反事实。guard/none capture四块+0.226/−0.306/−0.774/−1.195%，repeat cycle−1.649/−0.918/+1.820/+1.872%，净收益未确认。代码/资格/性能archive SHA分别be9492ea…9010a2 / 039b2588…111dc8 / a19d35ce…17f0fd，详见group_guard_{frozen,qualification,performance}；三份完整包与原raw保留。唯一下一步为已有实际轨迹的保护/驱逐后复用寿命诊断，不新增selector。全部进程已结束，结束查询GPU空闲/磁盘余4.1GiB。
group_guard限定复核完成：reused reviewer / same-family / provisional WARN，P0=0/P1=0；12配对全部逐请求/聚合差值、真实guard和后续cache、flush实物字节/GC交集、post-hoc实际轨迹账及主报告合同一致，无必要修正，停止追加审计。

2026-09-13 reuse/layer_budget：实际12episode/8配对复用分析完成；full/guard保护2859/1792项，下一同层复用且原驻留存活2424/1514(84.785%/84.487%)；共同需求saved/added=1278/1153及973/931，含独有需求后Δmiss−134/−47。排除“主要保护对象迟迟不用”，不等于纯cache因果收益。cap24从reset含五预热精确复现1952calls/3575groups/32anchors，固定trace分配384槽，少miss5513/636groups vs uniform5594/636，预热2939→3017groups；少groups为5554/624。两份有界只读复核P0=0/P1=0(reused/same-family/provisional)。源码/结果位于../20260912_wisp_olmoe_r01/group_guard_performance，运行命令为对应analyze_retention_reuse.py/analyze_layer_budget.py --input-dir该目录 --out新路径，exclusive保存。
静态层预算57行净实现通过CPU检查；phase reader改读实际layer caps，在旧0_forward上逐字段完全复现。三映射固定后取校准外文档316/480/507，六engine×8episode U/M/G/G/M/U性能方案冻结，先启动三个非注入数值资格；代码包SHA cf48dd4c0b7c32fb19c6d576319b7026de9d7ecd67aa7d9a88fc41f3202939c7。资格尚在执行，不能记性能结果。
远端重复输出清理完成：36个本地归档与远端archive SHA一致，仍存在的展开成员逐文件内容核对后删3,807,848,134bytes；24个归档成员在此次检查前已不存在。双端archive、本地展开原件和远端同级日志保留；磁盘余17,530,527,744bytes。前两次仅验证停止未删除，回执和脚本见../20260912_wisp_olmoe_r01/remote_cleanup_20260913_r01。
layer_budget资格三格均完成0，共12请求/96输出，48逐层finite/allclose(.01,.01)通过，maxabs0.03125/0.03125/0.0625，非位级相等；实际384槽/scratch4,831,838,208B/KV1GiB一致，资格archive92953bfc…abab38完整回传。性能包SHA f367e3b0c6a3c7dfd8a47e79f7af9b203b25615dbf3a63ec896ac3ed8d1bb7f6 首格初始化期间被新外部PID71634占25.81GiB导致OOM，raw请求文件0，archive0685086e…dda8cd留于attempt01。后续定位/root/het-r01完整八格campaign仍在最后block1-hom-native/PID73122，父69456；等待完整终态和GPU空闲后原包新目录恢复，不把边界检查当连续独占。

layer_budget接续：attempt02首次完整完成0_uniform/1_selected，各8测量；2_min_groups初始化因新外部PID74507占24.88GiB而OOM，零raw。完整archive e4eeea2ac54e7e8858fc586d952cf436ab62a6036db43c5b5d497271148f0d6e保留。在读取两格性能之前，EXECUTION_ADDENDUM.md规定首次完整合格格为canonical，只补剩余四格，保留中断与时间漂移。第二个外部campaign父74123及全部worker退出、GPU查询空后启动attempt03；其仍在初始化前ABORT_BUSY（PID77465，5268MiB），0.242s退出、零模型/请求执行，并非该次OOM。archive afecac7de8de07598dd3c07132b7ca47253bd5420595be29e729dffa22fb0588（96,908B）双端与31展开文件匹配。
再次核实GPU无compute进程、无存活campaign后，用同一remaining包（2bac040ab44070853fd82d98d1d388dbbf527b0b472308489dedcf182076888c）在/root/autodl-tmp/layer-budget-performance-20260913-root-r04继续原2/3/4/5顺序。driver78010、首worker78012实在执行；2_min_groups已完成、3_min_groups在跑，最终比较未齐全。组合索引将保留所有attempt与原始进程起止，失败启动成本另列，不计入成功格均值或从总执行账消失。

2026-09-13 layer_budget完成：attempt04四格全0，完整归档8c4036be…32fa4c4（31,619,639B）；与attempt02首两格组成原U/M/G/G/M/U，索引保留4attempt/3失败。48测量144请求1920输出、240预热624请求7776输出、93888calls/18432测量层、114GPU边界与源/资源全匹配。实际U/M/G loads5224/5195/5209、groups631/627/627；M少搬0.555%，capture对U−1.477%/−0.811%，cycle−1.047%/−0.017%。G/M capture变号，不确认稳定净收益或校准排序。三个慢episode全部保留，第三请求注入前已慢（静态层预算此时已生效），不用GC/负控最大差扣除。成功process694.627993s、失败36.185328s分开保留。下一唯一项为384物理槽expert_map接口的同输入数值资格，暂存已有prior art，不先实现完整共享pager。
静态层预算同一次限定复核结束：P0=0/P1=0(reused/same-family/provisional)，原值/独立性/失败保留均通过；已澄清注入前不等于静态预算干预前，停止追加审计。

2026-09-13 shared_pool interface：r01按原合同FAILED，48正向按位一致但错误map仍allclose（maxabs0.003174、relL2 9.332%）；进程36.706551s保留。revision02在新资格文档480/507/628/799上冻结BITWISE_INDEXING_IDENTITY，r02进程36.575670s成功：48正向全按位、固定错误map[0,3]非按位且非allclose，relL2 18.077%；4请求32输出，额外pool/reference资格峰值13.597GB，不计同预算性能。r02 archive bfc4d9bb…681d0c1c（242123B）完整回传；r01不重判。下一唯一项为私有21×16+共享48的真实生命周期资格，保持普通cap21 LRU终态、明确D2D和容量缩小成本。

共享池生命周期资格启动：CPU符号复跑8080 calls、30初态、11非法输入PASS；错误顺序负例5114次损坏。新planner89行+runtime wrapper约230行，复用原7份底座，读写次序/逐调用参考/唯一storage经有界只读复核P0=0/P1=0。冻结源码包33add9eb504169d7de0e749f793e3d06a27d534ab5c90a8e0394af54dc07b9c4（79268B），源文档索引8…11，即976/1107/1157/1244。remote shared-pool-lifecycle-qualification-20260913-root-r01按split21→oneshot21串行，每格4请求64prompt8output，KV1GiB，总384权重槽；所有measurement层调用数值/权重校验，额外reference开销不计性能。当前实跑未读回结果。

生命周期attempt01：split21固定stream-ID在vLLM dummy warmup阶段失败，初始化16calls完成、measurement0、raw0，oneshotUNRUN；26.052556462s，archive4ba80dce…920bc3（51108B）完整留存。revision02以每调用event+切流wait及host非阻塞锁保持读写顺序；CPU流顺序/异常回归PASS，targeted复核P0/P1=0。数值/权重与输入合同不变，代码包3db4f36cf4c4aad75dd70c2d8e8af9894657ee2288665bc7d908fed3c7edaf4d（79690B）已在remote shared-pool-lifecycle-qualification-20260913-root-r02启动，当前未读回资格结论。

生命周期r02两格完成：各224calls/160测量验证、4请求32输出，oneshot160bit_equal/640weightchecks；split160allclose/46bit_equal/320weightchecks。切流event路径通过，384总槽/KV1GiB实际一致；跨臂cache14/16相同、outputs3/4相同，不称同未来状态。archive8cdc3ee0…d9ce93（602798B）完整回传；成功process32.565799167/32.077871118s，失败原记录不变。性能四臂U/M/S/X/X/S/M/U在两组新文档上每格8episode冻结；共同受保护warmup仅用ordinary21，测量none/oneshot分支CPU核对通过。代码archive88aac002…d5faf8（129151B）已在remote shared-pool-performance-20260913-root-r01启动。

2026-09-13 shared_pool_performance八格完成，失败0：[analysis.json](../20260912_wisp_olmoe_r01/shared_pool_performance/analysis.json)保留64测量/192请求/2560输出、320预热/832请求/10368输出、125184全阶段层调用/24576测量层调用及152个通过的GPU边界。X/U两block capture−5.542%/−5.931%、repeat cycle+2.715%/+2.871%；X/M为−6.729%/−6.893%、+2.246%/+1.396%；X/S为−12.701%/−12.517%、−4.426%/−4.641%。完整逐请求表见REPORT新增共享池四臂性能段，所有repeat和慢行计入，旧资格失败不改判。
实际唯一384槽/1GiB KV一致；X相对U的H2D增加11.449%/10.099%，另有D2D，planner/map/event/观察/写盘/flush和parent process全计费。五次warmup加一次measurement的cycle较慢不是持续服务摊销性能测量；请求改善不等于net GO。各臂实际route/cache独立，X/U输出16/16配对同，X/M、X/S各8/16同，跨臂route均不同；同engine相关重复与文档/顺序混淆不作显著性或质量主张。reused/same-family/provisional限定复核重算raw、配对和资源，无P0/P1。唯一下一步为普通20×16私有+64共享full-stage GPU资格及同预算性能；[CPU符号检查](../20260912_wisp_olmoe_r01/full_stage_qualification/cpu_check.json)8096调用/30特殊初态/11非法输入通过，GPU UNRUN。

- 2026-09-13：普通 full-stage 已接入同一共享池 wrapper，20×16 私有+64 stage；8096 连续 CPU 状态检查通过，限定源码复核无 P0/P1。4 请求 GPU 资格输入2038/2270/2378/2555与先前资格/性能不交叠，包已上传核验；跑前发现 PID85558 正执行另一项 Qwen 原生资格，占3650MiB，状态 `BLOCKED_BEFORE_GPU_INITIALIZATION`，本实验0 GPU请求。记录 full_stage_qualification/preflight01.json，冻结协议与所有原产物保留。后续 X/F/F/X 两组新文本性能已准备，必须在本资格通过后执行。

- 2026-09-13 06:56:41 CST：连续第三个goal turn核验同一单卡阻塞。nvidia-smi与/proc/85558及ps确认Qwen资格PID85558仍存活（运行14:16，占3650MiB）；full-stage资格和性能远端均无execution.json，保持UNRUN。CPU实现/验证、冻结输入/命令/分析及上传均已完成；下一步依赖该GPU任务结束，goal标记资源BLOCKED，不改变主问题OPEN/MEASUREMENT_ONLY。接续点仍为full_stage_qualification/execution_status.json。

- 2026-09-13 新主机接续：用户提供新SSH主机，实测空闲RTX5090 32GB/driver595.71.05、cgroup内存90GiB/CPU配额25核，旧端点关闭。同11源资格包、X/F/F/X性能包和逐字节匹配旧底座的WiSP4文件已上传核验。复用现有PID2080/2081安装vLLM0.26+Transformers5.15.1并下载固定OLMoE revision；不重复下载。HOST_ADDENDUM声明换机后先qualify fullstage20与oneshot21两格，再全部同机性能；旧主机性能不与新主机直接配对。模型下载未完成，GPU实验仍UNRUN。

- 2026-09-13 新主机准备闭合：三份OLMoE正式权重共13,838,721,960B逐文件LFS SHA通过，10篇token IDs匹配。PID3326退出后PID5127新Qwen加载再次占3650MiB。继续进程PID5178/exec69695真实WAITING，未创建资格/性能execution.json。6小时资源等待后依次权重复核→F/X资格→逐调用CPU检查→X/F/F/X→原冻结分析，任一失败停止并保留；不终止其他任务。准备包r01保留，r02仅将就绪条件改为正式权重存在并随后完整哈希，不要求无关临时下载消失；实际启动前临时文件也已全部完成。研究目标未完成，GPU两项仍UNRUN。

- 用户要求持续接续后：续跑PID5178一直WAITING，Qwen PID5127已依次消费第1–3片、正在第4片；非进程失联。新增旧P五预热成本定位：三protected普通分组路径贡献+0.496/+0.540s，两个共享执行预热抵消−0.127/−0.144s；这是观察定位，未将差值全部归于cap。另用同本地固定Arrow/tokenizer准备eligible第33–64篇自然文章输入池（32篇），与原short32/本机制9份准备输入无交叠，仅input pool，尚无arrival/输出长度/GPU矩阵。原F/X及X/F/F/X冻结计划未改。

- 用户再次换机至weste:23478：旧westc取消连接失败，旧SSH退出255，末次1789292405仍WAITING。新端点原FQ/FP archive SHA、22份源检查、三权重LFS SHA、十篇token IDs、WiSP导入均通过；16核/92GiB/driver595.71.05，软件版本同前。独立weste-r02 Q/P/C目录与控制包a7aa35d9…5816c8（9753B）已核验，续跑PID4534 detached启动。GPU中途被同仓库rotation_first_swap六格占用，当前排队；不直接使用旧主机计时配对，不停止其他任务。

- weste r02控制预检退出archive3cc08e68…5942b4（22978B/16文件）已回传。r03将模型校验前移、1s查询后成功启动fullstage20，但它在8.761357917s因并发模型加载导致启动free17.34/31.36GiB、requested28.22GiB失败；09:57:06.779/11.969 UTC两个边界原本PASS，说明检查到CUDA初始化仍有竞态。archive4121ce16…e33cdf（78025B/31文件）全校验取回。0pager calls/0validation/0raw请求，oneshot与性能UNRUN；不调低gpu_memory_utilization、不把环境失败判成机制NO-GO。所有我方控制进程已退出，已请求用户协调另一会话在当前格后暂缓新提交约10分钟，继续只读监控。

- 接续阻塞复核：1789294392.894，经/proc/13275、ps和nvidia-smi确认Qwen仍存活/3650MiB，1/16分片已消费；本轮verified wait，未再次启动GPU。等价共享GPU阻塞连续三轮复现，CPU准备和失败定位无剩余可独立执行项，按goal规则标资源BLOCKED，研究问题仍OPEN。准确恢复点weste_resume_20260913/execution_status.json；需协调独占窗口后在新目录执行原两格资格，再跑原四格性能，所有旧失败保留。

2026-09-13T18:54:50.558641+08:00：用户查询后恢复的连续三轮均为verified wait；/proc、ps及nvidia-smi再次确认Qwen PID13275存活、占3650MiB，已消费5/16分片，正在第6片。weste我方控制进程已退出，F/X新资格及X/F/F/X性能未启动。现有CPU准备和离线分析已完成；本次按同一资源阻塞收敛，待GPU独占窗口恢复原冻结实验。主问题OPEN/MEASUREMENT_ONLY，不记作科学负结论。

2026-09-14：westc-r02 F/X资格两格完成，8请求64输出；43文件回读SHA c51ee819…，冻结checker issues=[]，各160参考按位一致/640权重检查通过。原rotation会话恢复声明与本次启动交叉，controller4540在资格末期SIGSTOP；资格driver自然收尾，GPU已释放，原组继续。性能UNRUN，恢复时只SIGCONT原controller并保留协调等待成本，不重跑Q。详见主报告同日附录及GPU_COORDINATION.md。

- 2026-09-14 F/X性能整组完成并释放GPU给A-review：四格X/F/F/X、32测量/96请求/1280输出，160预热/416请求/5184输出；归档4824480d…e6b、40,713,785B/547文件已回读。X/F两block capture−6.244%/−5.514%，五预热实验cycle−4.330%/−4.079%，新TTFT−6.845%/−6.357%；旧maxITL有5/16配对恶化，保留。资格参考关闭，双方384槽/KV1GiB；限定性能审计在进行。下一最小项为同新16请求时钟到达序列、单次初始预热的U/M/F/X及反序，CPU准备后遵守A→rotation队列。旧X/U、X/M实验cycle负结果不变。

- 2026-09-14 finite-arrival：原F/X限定审计数据PASS、唯一旧hash自动检查缺口由新包直接比较补上。31项封存输入/4份WiSP/6版本在westc核验，包272026B SHA340039f8…430f。A结束且rotation明确不保留窗口后，1789318148.591现场GPU空闲，首次启动新controller13642。U/M/F/X/X/F/M/U，各16同源新文档P128/O32、0.25s到达、prefill32，单warmup、无测量reset，完整8格成本全部保留。当前RUNNING，未给性能裁决。

- 2026-09-14 finite-arrival COMPLETE：八格128请求/4096输出、162文件回读SHA490f5c48…fbe4；X/F capture +3.394%/−0.948%，完整进程+1.771%/+0.636%，D2D−52.061%/−53.862%。两次U漂移−20.900%，跨臂route均不同，不能称稳定收益或总体噪声底。限定审计WARN/P0=0/P1=1：外部PID13921仅直接重叠首M初始化，warmup/capture前已退出，仍不能断言无后续影响。GPU已交接且无存活driver。层0长调用定位已完成，下一唯一项为普通U的逐次编译事件定位，CPU准备，需重新排队。详见finite_arrival_r01/REPORT.md和EXPERIMENT_AUDIT.md。

- 2026-09-14 JIT source localization STARTED：D6八格终态并明确交接后，1789320955.319现场GPU为空，controller22999首次启动普通U两格。39项封存输入，archive29984b65…65ecbb/302212B；188行observer区分compiler pipeline、disk-cache load、launcher/driver load，原runtime未改。两新引擎使用专属Triton磁盘cache空→保留；32测量请求计划、同16文档P128/O32/0.25s，完整冷/暖成本保留。共同flock整组持有，之后交给APC-most；当前RUNNING，无归因结论。

- 2026-09-14 JIT source localization COMPLETE：32请求/1024输出、5088positions、1600测量/1824全层调用，6GPU边界PASS且整组持锁。archive c798b3c7…b4844/2824639B，90成员/89 payload SHA全回读。冷cache测量12次真实pipeline、3.755144s；保留cache12次disk hit且仍2次新pipeline、0.624777s。7个长layer0包络97.40–98.25%被实测编译/加载覆盖，缺失组合为tile32/128/64。两次capture11.646/9.232s但轨迹49/51步、输出9/16一致，不能扣编译重算机制收益。PID22999退出后已交接APC；下一仅准备按预算域固定specialization覆盖基线，计入启动成本。

## 2026-09-14 B compile-domain r01 execution failure

The first F engine initialized, then the new snapshot failed on vars(WispMoEState), because the actual pinned state uses __slots__. Full process 42.951954032 s; 0/320 preparation calls, no request warmup/measurement, other three cells UNRUN. [Retained report](../20260912_wisp_olmoe_r01/compile_domain_r01/REPORT.md). This is INVALID_EXPERIMENT due to an interface bug, with no F/X scientific verdict. The GPU was released to the queued LTR group. New r02 preserves runtime, resources and documents while fixing six stats reads and the CPU state-layout fixture.

## 2026-09-14 B compile-domain r02: qualified coverage, no stable X/F completion gain

[Full report](../20260912_wisp_olmoe_r01/compile_domain_r02/REPORT.md). New r02 fixes only the r01 slotted-state snapshot; the original runtime, four-cell F/X/X/F sequence, resource budgets and new16 documents remain unchanged. Four fresh private Triton caches; all 320×4 compile/handle setup calls completed, CPU pager/KV snapshots stayed equal, and measurement had zero actual compiler calls. Setup 7.019/5.460/5.929/6.841 seconds is retained in process time.

64 requests/2048 output tokens/10176 scheduled positions completed. X/F capture −3.873%/+0.086%, mean completion −5.705%/−0.873%, maximum ITL −7.864%/−6.368%, full process −7.745%/+0.804%. D2D bytes −54.040%/−54.623%, H2D −1.369%/−2.253%. X same-arm capture +2.999% and mean completion +4.627%; only two correlated engines per mode, not a noise bound. Outputs and route traces differ. No statistical improvement, noninferiority, SLO, quality, all-runtime steady-state or method GO is established.

The original failure stays INVALID_EXPERIMENT before measurement; r02 is MEASUREMENT_ONLY with coverage fulfilled. Strongest paired baseline is ordinary F at the same 384 expert slots and actual1GiB KV; it is a joint layout/execution comparison, not an isolated writeback treatment. Original archives are preserved: r01 62 members; r02 134 members/SHA acf66ad7688296778dfdcdc969d67b190f956b8eb2fe2153caa77b3cf6754cf1. Local frozen analysis matches remote exactly; limited result review is in progress. GPU released 1789324505.834, no additional GPU matrix. Next only CPU localization of the residual X same-arm time, respecting nested host/CUDA clocks and action-dependent arrival trajectories.

### 2026-09-14 r02限定结果审计闭合

[审计](../20260912_wisp_olmoe_r01/compile_domain_r02/EXPERIMENT_AUDIT.md) PASS/P0/P1=0，复用同族reviewer、provisional。独立133payload/40输入/11runtime、64请求2048输出、10176位置/4032全调用核对通过；1280准备调用与108真实编译/加载仅在初始化/准备、测量事件本身为0。r01失败独立保留。无稳定F/X收益主张。新[CPU定位](../20260912_wisp_olmoe_r01/compile_domain_r02/OBSERVED_COST_LOCALIZATION.md)将下一问题限定为可观察前缀先出现的host成本；host_cost_r01四项观察开关仅CPU准备/GPU UNRUN。
