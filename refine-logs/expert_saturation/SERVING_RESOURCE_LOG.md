# 实验记录

## 2026-09-12：接续固定主问题，完整 pager 接入

判断：KV 四项已完成，WiSP 人工专家 6/6 已通过；下一不确定性是实际 OLMoE/vLLM0.26 接口与加载计费，不重复旧实验。

已完成：核对实际 HEAD/dirty、最新原始结果；只读取回已安装 vLLM 的 OLMoE、MoE method/kernel/loader、scheduler/model_runner/input_batch；适配原生已选 top-k，复用 WiSP state；建立原生请求账本接线。CPU 签名/分组检查通过，不能替代 GPU。

运行方案与保留规则：[20260912_native_pager_r01/config.json](outputs/admission_capacity/20260912_native_pager_r01/config.json)。执行入口为该目录 `run_attempt.py`，每次上传到独立远端目录；失败后停止该序列，修复放新 attempt。

结果：attempt02两臂均完成，cap24 worst mixed step的1个decode+127prefill对应1.3318s生成间隔及55.1016GiB专家tensor-copy payload；16层真实row→request join闭合。attempt03独立检查16层相同hidden/top-k，分页与full-weight kernel逐位一致。见主文档链接与原始归档，不能升级为方法收益。

完整性复核：fresh same-family provisional review为`WARN`，A–F=`PASS/WARN/PASS/PASS/WARN/PASS`，P0=0/P1=0。冻结源码/readback哈希、调用路径、重排后row join、4/4请求各8token、cache transition与字节和均闭合；WARN来自单模型/单卡/单episode、非充分暖机/JIT及copy payload非PCIe wire计数。relative-L2只作数值诊断，不是任务准确率；16层检查只覆盖各层首次调用前16行。

执行记录：attempt01的数据上传被自动审批拒绝，未传/未运行；改为纯代码包、复用远端已有且哈希匹配输入后attempt02/03成功。attempt04的0/32同资源ABBA在模型加载前因其它GPU任务退出，原始结果保留，没有测量。

0.26窄transfer补证完成：cap24、实际KV 1GiB等资源在4/4 cells一致，每cell 4/4请求、32个输出完成。prefill32相对default在r0/r1使已有请求max-ITL分别−21.60%/−9.62%，全请求mean TTFT +8.57%/+40.28%，episode wall −7.66%/+9.06%，weight-copy payload均+1.007%（116.3203125→117.4921875GiB）。它显示ITL→TTFT权衡且wall符号翻转；实际admission轨迹随step返回而变化，不是matched-prestate counterfactual，也不并入0.11.2冻结action证据。

承接：共享目录新回传的0.11.2同资源注入矩阵已证明chunk8/32的停顿—TTFT—总copy权衡，避免重复同一存在性实验。0.26仍是已跑通的原生适配；两个runtime耗时分开。

plain与affinity补证均已完成。无trace plain的chunk8 wall为5.703/4.363s，chunk32为5.323/5.350s，排序翻转。随后共同绑定CPU0–7的affinity重复得到：chunk8旧请求max-ITL 308.7/307.3ms、新请求TTFT 4.0064/4.0401s、wall 5.8161/5.7843s、完整episode payload 163.08984GiB；chunk32依次为822.9/810.1ms、3.0674/3.0296s、5.3961/5.2683s、159.01172GiB。固定CPU后权衡方向一致，但只有`taskset`，没有显式membind；不把plain漂移归因为唯一NUMA原因，也不升级为scheduler收益。

模型修正：原顺序LRU不足以表达WiSP整组保护和slot tie；在同一resource_transition_model新增grouped-slot模型。22项结构检查通过。命令：`python3 experiments/admission_capacity/calibrate_grouped_pager.py --results outputs/admission_capacity/20260912_wisp_olmoe_r01/injection --output <fresh.json>`（本目录执行）。8条实际轨迹/5,924组的loads、evictions、bytes与最终slot/tick/clock全部复现；输入是各自post-router轨迹，不是在线反事实。

原始带trace的chunk16预测为首步copy 10.211GiB、旧decode首间隔0.324/0.282s；对应执行仍在 `outputs/admission_capacity/20260912_native_pager_r01/midpoint_prediction/` 保持`BLOCKED_BEFORE_GPU_INITIALIZATION`，没有测量，不覆盖或改写。

affinity chunk16两cell已于12:34:16.507–12:35:23.818 UTC完成。冻结first-ITL 354.854/362.022ms，实测338.834/339.241ms，误差−4.515%/−6.293%；已有请求max-ITL 489.066/490.712ms，新请求TTFT 3.476334/3.459523s，wall 5.557003/5.537933s，完整episode copy均162.90234375GiB。两cell各3/3请求、40个输出且prestate/全部检查为true；每cell 29个NVML样本只见本worker PID并匹配CPU0–7，但不是持续隔离证明，且无显式membind。无trace协议仍不能测首动作payload；完整raw留在远端，local `safe_readback/`只保存派生结果、执行记录和哈希/源码。

static8/phase8 ABBA现已4/4 cells完成；[safe analysis](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/safe_readback/analysis.json) 显示每cell 3/3请求、40个输出，全部paired prestate检查为true，0 decision issue、0抢占，四臂的三个输出hash相同。相同逻辑工作持续到call index 15；之后phase8用一次32-token prefill替代static8的4×8，engine calls从27降至24，完整episode copy从163.08984375降至162.94921875GiB（−144MiB）。动作及会计有效。

性能因果未闭合：r0 wall从static8 5.808687s降至phase8 4.312848s（−25.75%），r1却从4.305069s升至5.733669s（+33.18%）。已有请求在策略分叉前就分别于4.525592→3.518542s和3.480656→4.550808s完成，共同prefix首步也形成约1.22s对约0.97s的跨策略快慢组。采样GPU clocks、CPU0–7 affinity和零throttle不能解释该差异；无显式membind，不归因为NUMA。状态为`ACTION_AND_ACCOUNTING_VALID / CAUSAL_PERFORMANCE_UNRESOLVED / MEASUREMENT_ONLY`，无method GO。

同进程、同engine ABBA也已4/4完成；[safe analysis](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/same_engine/safe_readback/analysis.json) 显示四臂均为PID23757、allocation hash `bbcaa057…`、相同KV顺序和`7d07d456…`完整prestate，0 decision issue且资源有效。动作差值仍为27→24 calls及−144MiB copy，但wall在r0为5.771153→5.634268s（−2.37%），r1为4.868633→5.618388s（+15.40%），继续符号翻转。

第四臂static8-r1首步仍为1.219s慢带，之后已有请求max-ITL却为0.242s而其余约0.307s，old-done后尾段0.821s而static8-r0为1.284s；相同PID和持久allocation仍不能消除episode内变化，allocation-only解释不足。采样GPU clocks、CPU0–7 affinity和零throttle也未解释该差异，不归因为NUMA。状态保持`ACTION_AND_ACCOUNTING_VALID / CAUSAL_PERFORMANCE_UNRESOLVED / MEASUREMENT_ONLY`；无method GO，也不判方法或问题NO-GO。原4个独立进程cells和新4个同engine cells均保留，remote raw与local safe readback不互相覆盖。

[independent_input.json](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/independent_input.json) 保留原运行前`PREPARED_UNRUN`记录；其中选择的文档已用于后述completed holdout，该静态字段不是当前状态。

已有trace只读分解完成：long8相同schedule/subgroup/load-evict/bytes的action后23 calls wall差1202.059ms，仅111.604ms来自host-read union差，1090.455ms（90.72%）在host-read spans之外；hold同向为90.10%。CUDA load span差124.101ms且重叠host spans，不是纯H2D。

同engine static8 profiler attempt02完成4次诊断；[summary](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/profile/attempt02/profile_summary.json) / [safe analysis](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/profile/attempt02/safe_readback/analysis.json) 中，GPU activity均约3.423s，request wall却为11.456/8.667/5.009/6.285s。p0–p2 capture-byte lower bound闭合；p3为`INCOMPLETE_ACTIVITY`并保留。后续addendum在原trace的call13/14 marker间找到缺记的8MiB copy，global大copy计数闭合，但不改写原p3或升级边界归因。每cell约27,870次4-byte H2D及34,285次stream sync使逐scalar expert-map更新成为工程税候选；profiler插桩开销使它只能用于诊断，不能主张CPU根因、加速或请求收益。

首次profile launcher因导入路径错误在GPU初始化前失败；attempt02首次analyzer因CPU/GPU marker schema混淆失败，均保留。修正只重析原raw、未重跑GPU；本地只采用safe summary/readback，远端原始目录未覆盖。

batched expert-map同engine static8 ABBA已4/4完成。两臂prestate/schedule/output/final pager/专家weight bytes相同；5个toy CUDA cases及四cell×16层map检查通过。source scalar writes 27,834→full-map copies 1,672，metadata 108.7→418KiB、两臂pinned staging均4KiB，专家payload均163.08984375GiB。r0/r1 wall为5.540343→3.911191s（−29.41%）和4.340916→3.808304s（−12.27%）；已有请求max-ITL为303.400→211.271ms和239.177→210.392ms，新请求TTFT为3.920823→2.664095s和3.026612→2.655714s。original自身仍有时变，效应量未稳定、根因未完全解析；分类`ENGINEERING_TAX_REMOVAL_POSITIVE / MEASUREMENT_ONLY`，无scheduler GO。

optimized common-pager holdout ABBA现已4/4完成；[summary](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/summary.json) / [safe analysis](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/safe_readback/analysis.json) / [completion](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/completion.json) 显示prestate均为`134afb54…`且全部资源、身份、决策、map检查通过。static8→phase8的r0/r1 wall为3.850803→3.788532s（−1.617%）和3.841899→3.802419s（−1.028%）；已有请求max-ITL为202.461→203.340ms和203.164→206.023ms，完成时刻仅差+0.289/+8.154ms；post-old tail为−8.530%/−6.615%，新请求TTFT为2.699880→2.657274s和2.688242→2.666425s。27→24 calls，copy为166.0078125→164.70703125GiB。

已有两个请求输出hash相同；新请求hash跨策略不同、策略内两次相同，尚无质量验证，copy下降也不能全部归因于同一continuation。分类`PHASE_RESIDUAL_POSITIVE / MEASUREMENT_ONLY`：保留小幅重复信号，不任意判死，不给scheduler GO或稳定效应量。

首次metadata上传被自动审批拒绝，原`BLOCKED_BEFORE_UPLOAD`记录保持0上传/0 GPU。随后`public_source_check.json`以官方Hugging Face `Salesforce/wikitext`完整正文hash确认三文档公开，`approval_addendum.json`记录相同archive获批并已完成campaign；当前不再是blocked。

受控continuation ABBA已4/4完成；[summary](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/controlled_tokens/summary.json) / [safe analysis](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/controlled_tokens/safe_readback/analysis.json) / [completion](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/controlled_tokens/completion.json) 显示四臂prestate均为`a2af02be…`，native sampler的新请求8个单例token约束有效，三请求输出hash全部一致；5个toy CUDA cases与四cell×16层map检查通过。GPU/CPU mask各150,912B且四cell指针分别一致，CPU未pinned；109.1/41.9/39.6/41.9ms primer均在episode wall外，之后仍跑原warmup。

static8→phase8的r0/r1 wall为3.777829→3.743439s（−0.9103%）和3.784312→3.719282s（−1.7184%）；已有请求max-ITL仅+0.624/+0.339ms，完成时刻−8.624/−11.272ms，old-done后tail −25.766/−53.758ms；新请求TTFT为2.705281→2.665377s和2.723734→2.658067s。27→24 calls，copy为160.81640625→160.7578125GiB，恰少60MiB/5次专家加载。自然holdout的新请求首处分叉为output index 3，TTFT收益先于该分叉；恒定continuation下的小幅重复信号仍存在。

受控primer使prestate与自然campaign不同，continuation也不同，不能把自然−1.30078125GiB和受控−60MiB相减后归因于route。当前强简单基线是optimized static8；分类仍为`PHASE_RESIDUAL_POSITIVE / MEASUREMENT_ONLY`，无exact Oracle、质量/SLO或scheduler GO。成本模型的候选动作枚举尚未接入，native phase-prefill wrapper已经实现；group/LRU/字节会计仍有效，scalar map-write税与必需专家payload分开，旧未优化时间查表已失效。下一唯一实验是同prestate重校准optimized fixed-prefill 8/16/32，校准前不扩大episode；当前无GPU任务，不再追查token numerics，也不加predictor/controller。

- 2026-09-13 跨会话协作修正：复用 admission_capacity/RESULT_LEDGER.md，登记已有 map/phase 自由与受控输出结果及最新 rotation 分母修正；AGENTS 与主文档链接同一台账。长上下文四臂使用 fast-headroom，轮转 actuator 未接入仍 UNRUN。当前优化后 static8/32→预报16 校准尚未上传/执行，先补初始化与 episode 边界 GPU 检查落盘；不为输入共享新建 capture v2。

- 同轮接续：校准包 SHA `367b4827d4338e5ba00d19bfdfc2420e224d942d4189040a4c5e7b247f067ba0` 已上传。首次审批拒绝在找到本会话原始用户 goal 中同一 SSH 地址授权后解除；保留 execution.json 与 authorization_evidence.json。attempt01 父监控解释器缺 pynvml，在 GPU 前失败；改用既有0.26监控父环境（0.11.2模型子环境不变）后 attempt02 因 PID36942 占用 ABORT/91，检查原始 JSON 已回读，GPU cells仍0。下一动作使用 attempt02.json 的同包命令，通过新占用检查后执行已定6cells；原始失败不覆盖。

- 接续完成：attempt03 新preflight通过，optimized-static-calibration 六格完成并安全回读，最终 readback_completion.json 为当前状态（旧attempt保留）。同状态/任务/资源与map检查通过；16首step预测299.833ms、实测305.031/305.745ms，低估1.734%/1.972%。chunk32相对8 wall−5.70%/−4.09%，已有maxITL约203→562ms；不同cap自由输出有差异。无SLO/method GO。下一唯一实验：固定本轮曲线，对新独立文档做同三动作迁移验证。当前没有本会话GPU进程或SSH会话。

- 2026-09-13 optimized_static_transfer：代码包2d69c682…、安全回读4e2c0fcb…，6/6 COMPLETE，worker37992已退出。固定旧8/16/32预测175/300/549ms，新3文档实测150/268/481ms，排序保持、实际比预测低10.7%–14.3%。各域同资源/同前状态与冻结截止检查通过；跨域不同进程，不能归因。下一唯一实验固定cap16同engine old/new ABBA及首步加载计数。初始execv相对argv[0]导入失败保留，完整sys.executable启动成功；未安装依赖。

- 2026-09-13 document_state_abba：r01已实际启动，在首mixed step后被错误的 ensure_calls==16 断言中止，0完整cell，INVALID_EXPERIMENT；原始远端结果保留。ensure_calls实际统计subgroup，已修为16 layer条目及计数守恒，4个针对性CPU用例通过。修正包SHA1868f80c…位于 document_state_abba/attempt02，计划新远端r02。两次自动审批拒绝代码/配置上传，且不接受历史会话恢复的授权作为当前批准；修正包0上传0GPU。唯一下一动作仍是获准后执行同一四格，不由测量错误推导科学结论。

- 解释器说明修正：先前将 pynvml 导入错误写为父环境缺依赖不准确；既有0.26环境直接导入成功，使用完整 sys.executable 作为 execv 的 argv[0] 后启动成功，没有安装依赖。旧失败记录保留，未单独复现 Python 路径解析原因。

- 2026-09-13 自动续跑：上轮为实质进展（定位首step计数断言错误并完成修正包），本轮重新核对同一上传审批阻塞；未重试被拒绝动作。共享台账出现rotation-fourarm 8/8新结果，已只读继承REPORT/analysis并修正主文档旧“未接入”状态。模型新增两条边界：交换间隔不是单请求停顿上界；完整成本须同时计缺席队列/恢复与策略改变的小batch尾部，不能固定原轨迹再加重算税。轮转相对headroom的候选改善及相对native平均完成损害一起保留；没有重复计算同一raw或新开轮转实验。paging下一唯一实验仍为已准备的同engine old/new/new/old chunk16诊断，等待上一条明确批准。

- 2026-09-13 阻塞审查第三轮：重新读取目标、READY及审批拒绝记录，两个待执行archive哈希一致，修正四格仍0上传/0GPU；共享台账没有比上一轮新增的可用结果。上一轮已完成新轮转证据对主模型的修正，本轮无新的安全实质工作可推进。相同“当前可信上传/执行批准缺失”阻塞已连续三轮，目标现标记BLOCKED，等待用户对已准备包及指定SSH目的地明确批准；科学状态仍OPEN/MEASUREMENT_ONLY，不是问题级NO-GO。

- 2026-09-13 明确批准后：document-state-abba-r02 4/4完成并回读，safe SHA2f60f7cb…，worker45315退出。first-step old/new稳定为134/119 subgroup、1164/1012加载（13.640625/11.859375GiB）；old/new前状态不同，各组内重复相同。首step新较旧短10.56%/4.83%，同文档时间漂移仍达同量级；wall差−4.98%/+0.11%，不称稳定性能收益。host计数读取税低于0.222ms。下一唯一实验为共同旧pair/prestate只替换incoming第三文档的fixed16 A/B/B/A；审批历史保留，当前已不阻塞于上一条授权。

- incoming-document-abba-r01 因果对照已上传（SHA3085169b…）：A=[4,5,6]、B=[4,5,9]，相同旧pair/共同prestate134afb54…，fixed16四格。两次尝试均在模型初始化前ABORT/91，0 GPU cells；首次预检记录显示外部PID45993占用，已安全回读。原始失败保留，同包可在GPU空闲时直接重跑staging/run_remote.py，无需重新上传或改输入。

- 2026-09-13 incoming-only第三次预检通过，4/4完成并安全回读（SHA83a0c8e4…），worker46985已退出。共同旧pair/prestate/资源及旧输出一致，仅第三输入变化；首步1164→994 loads、13.640625→11.6484375GiB，旧请求跨步间隔约308→266ms。不同incoming工作负载不称scheduler收益。已有raw的8个chunk时序只读导出到post_chunk_timing.json：首步普通延迟预测第二步误差约+1.8%或−6%，后续chunk成本排序会交叉。下一动作是首chunk16后的单次8/16/32分叉与统一恢复16。

- next_chunk_branch最小模块与12格随机block已准备；CPU因果/前态/预KV动作/decode及两步计数检查通过。包SHA1ec47414…上传后，首次attempt01因SSH shell无python3在上传前失败，改既有完整解释器路径解决。attempt02的父/子预检与5项CUDA小检查通过，但第一cell模型加载前发现外部PID56066，ABORT/91、0完整cell；两份预检回读保留。attempt03复用相同代码/动作次序，只将结果改为next-chunk-branch-r02，现已启动，最终完成状态待回读。

- next-chunk-branch-r02已12/12完成并回读（SHAc5cf5216…），worker58240退出。每doc六个分叉前态一致，三请求输出跨动作均相同；next8使calls19→20却减少payload，B相对next16 wall−43.45/−40.50ms、oldmax−9.36/−8.05ms、新TTFT+6.14/+5.66ms，A oldmax反而+5.75/+6.82ms。B→32 wall符号翻转保留。first-mixed比例预测B-next8低估约36ms，对B-next32只差5–7ms，统一文档系数失配。状态仍MEASUREMENT_ONLY，不能升级方法GO或总体噪声界；无需另跑恒定token对照，因为本轮所有臂自然输出已相同。

- one8_static_holdout：选既有公开source索引10/11（与本paging链已用0–9分离），20cells=2docs×static8/16/32/phase8/one8×2随机block；20入口、phase8语义与本地/远端输入构造hash检查通过。包SHAb4e7f336…已上传并通过初始化前预检，当前已启动。输入从远端既有32-doc源按冻结hash重建，未上传原文/token；不修改规则或次序，结果待回读。

- one8-static-holdout-r01已20/20完成并回读（SHA95fc57b7…），worker59241退出。c11 one8在wall/oldmax/TTFT被static16两次全面覆盖；c10仅权衡。one8/static16全部输出相同，故c11失败不依赖输出差异解释。停止该固定位置规则，不判问题家族NO-GO；下一先复用共享已测expert-group强执行基线，避免重复另一线group-budget工作。共享fourarm-noise-floor行已修正为描述性跨序差异，原报告保留；不把样本最大差作为界或非劣证据。

- 强执行基线源核查收敛：retention_lifecycle_performance/source/run_native_pager.py --execution expert 已实跑，但为0.26/cap24/KV1GiB/token160/输入0–2，与本线0.11.2/cap16/KV512MiB/token64/输入4,5,10或11不等资源。下一接续现成0.26 runner开放相同输入与配置后对比token/expert执行，不依据70GB vs150GiB排名，不移植新controller。该对照UNRUN；当前无本线GPU进程/SSH会话，固定one8规则停止，研究问题仍OPEN。

- 2026-09-13 execution-baseline-v026-r01：同预算token/expert ABBA4/4完成，12请求/160输出，完整预热另40请求/488输出。代码SHA2f98f8d0…、原始回读SHA19d0c493…，worker60757已退出。首次审批拒绝，核验本会话原始目的地/代码执行明确批准后同命令获准；授权与拒绝保留。expert wall−26.15%/−25.95%、oldmax−50.11%/−49.61%、新TTFT−45.28%/−44.71%，payload155.84766→101.58984GiB；全部输出相同，逻辑前态/前缀一致，物理KV IDs跨臂不同。既有底座收益不归scheduler，状态MEASUREMENT_ONLY。唯一下一项在expert底座重测static8/16/32动作曲面；旧one8仍停止，不新增controller。

- 2026-09-13 expert-chunk-curve-v026-r01：6/6完成，18请求/240输出，worker65288退出；代码SHA7a67b44d…，回读SHAc43e0385…，全输出相同、共同前态/分配一致。8/16/32三点各有权衡；32少搬却更慢，真实三请求decode由4→7次抵消prefill节省。阶段重构wall残差0，动作前prefix漂移与32自身100.9ms漂移单列，不造噪声界。下一四臂检验C16/k8对应纯decode RR2，仍为实际同问题因果动作，不新增预测器。

- 2026-09-13 decode-width-rr-v026：r01模型加载前检测外部PID69467退出，完整保留；r02连续3次空闲预检后8/8完成，worker72254退出，24请求/320输出、预热152请求/1936输出，回读SHAb83dada6…。RR每层单组/held状态条件通过，却calls16→20、payload+1.74%、新生成等待增加，未取得优于static32证据。RR自身505.74ms漂移及动作前差单列，互斥阶段残差0，不造噪声界。停止该RR2机制推进；下一补三请求不丢工作的admission cap2/cap2+旧完成释放64基线。存储清理已与共享回执对齐：40远端冗余raw由共享线删除，我们只清理pip HTTP缓存，原始备份/模型/环境保留。

- 2026-09-13 admission-width-baseline-v026-r01：六格I/D/R/R/D/I完成，18请求/240输出、预热96请求/1212输出，worker77465退出，回读SHA2364c5a2…。D/R真入队等待12步，old全推进；D新prefill4×32、R2×64，恢复检查全通过。延后准入降低旧暂停但增加新TTFT/整批；R−D wall −17.73%/+17.82%翻号，D自身−19.68%、R+15.02%漂移。相同old阶段结构下CPU计费时间变化主导，observer/GC/load span变化更小，未定位Python/launch/driver。下一固定D做plain/profile×4/plain的CPU函数诊断；持续到达接口只完成CPU准备，不扩GPU策略矩阵。

- 2026-09-13 admission-cpu-profile-v026-r01：6/6同D完成，18请求240输出、预热96请求1212输出，worker79239退出，archive6bc6b036…。所有逻辑轨迹/输出同一，plain wall2.409815→2.344809s；4份cProfile全部出现计时会计异常，函数归因INVALID，保留全部signed原值。0.46s不能当已证成本。下一依据实际Torch递归展开源码，做无profile flat/nested/nested/flat等价观测四格，暂不扩调度矩阵。

- 2026-09-13 admission-memory-observer-v026-r01：4/4全D、12请求160输出完成，worker79915退出，archive e505ca53…。flat/nested/nested/flat不带profile，两对完整wall−78.171/−74.112ms；每格1104个allocator值与全部工作量/逐行post-event route/输出一致，新maxITL+6.303/−3.058ms保留。不解释旧578ms漂移，不是scheduler收益。采用nested为共同观测路径，下一回到8新评价文档/异构长度/clock arrival的cap3-static32、cap2-static32、cap2-phase32六格。

- 2026-09-13 continuous-admission-v026-r01：6独立engine A/B/C/C/B/A完成，48请求1584输出，每格928位置，archive97129e00…。A cap3/static32 wall7.862/7.817s，B cap2/static32 8.437/8.421s，C cap2/phase32 8.304/8.349s。C仅step0真实64，后续持续decode期间无动作；较B小幅wall改善却TTFT更差、mean完成翻号，仍落后A。停止本域phase32，保留普通准入的暂停/等待权衡。下一真实超显存Qwen3串行加载器CPU生命周期/完整资源资格化，不继续阈值或seed救援。

- 2026-09-13 Qwen3-30B-A3B BF16资格化：串行分片及实际installed loader契约CPU通过，4条公开输入用冻结Qwen tokenizer重分词；固定cap48/KV512MiB/token64/maxseq4/prefill32、48层same-precall参考，包SHA0ebb2422…。已上传既有5090独立目录，首次预检见外部PID84182，尚未模型初始化；等待空闲，不影响对方实验。参考计算含在wall内，本轮仅加载/数值/请求完整性，不作性能结果。

- Qwen资格化r01结束：0分片/0请求，loader误拒vLLM原生model_weights空串默认，非OOM；回读1e4b9c51…。已用实际config/model.py的默认字段和本地early-return复现并修正，r02仅此guard变化，同配置/输入；包e01106fa…已重新上传并等待连续空闲预检，原r01证据保留。

- Qwen r02结束于huggingface.co网络不可达，首分片120秒后0字节，0tensor/0请求，回读a2930e44…。同机CPU验证hf-mirror冻结revision的16片均HTTP200、1MiB前缀与全Content-Length响应成功；仅替换匿名transport hostname，完整SHA/逐片消费不变。r03包0f48de22…已上传启动空闲预检；48层参考及完整资源峰值仍未完成。

- Qwen r03已完整退出/回读（archive e94c772e…，worker85558）：16片/18,867源张量/435目标、4请求32输出、284位置/624层调用闭合。真实KV为341block/511.5MiB；原分析器把512MiB预算当分配值而误拒，新增v2精确取整分析和针对性回归，原失败/raw保留。48层参考finite，47/48原allclose诊断通过，layer47 maxabs0.25/relative L2 0.00221848，未改阈值。GPU总峰26.19GiB、cgroup总峰88.39GiB（包含原文件缓存/内核分配），11,174采样无外部PID/OOM增量。状态RUNTIME_COMPLETE / NUMERICAL_DIAGNOSTIC_DIFFERENCE，参考/JIT计时不作性能；下一只定位该数值差异，再推进已冻结static32/16，不能以资格化或两静态点宣称新版总目标完成。

- Qwen localized-static r01：新包 f1137a7ecbc8dbe9f545397afd7e978097229ce23258b236dcb04f7130f2befc 已通过连续三次 GPU/launcher 空闲预检并启动 worker89182。先沿原资格化执行并保存 layer47 同输入、同分组 full-weight partial 比对和 CPU 快照，全部有限且 partial/final 逐位归因通过后才进入未改静态32/16/16/32；同模型分配仅切换记录范围。总父进程和 session 计费保留加载/资格化/切换/四格/关闭，cycle wall 也含资格化而其 repeats/flushes 列表仅性能。当前 RUNNING_LOADING，无新性能结果。

- Qwen localized-static r01 连接中断：本地SSH命令255，最后已确认读数为第9/16片、约57.3分钟；两次只读重连均在认证前被远端关闭。远端worker89182与父监控状态未知，尚未回读任何新资格化/性能结果；状态REMOTE_STATE_UNVERIFIED，不能写实验FAILED/COMPLETE。未重启/终止远端任务，先恢复只读状态核对。CONNECTION_INTERRUPTION.json与原execution_attempt01保留。

- 断连后本地最小修正完成：transport_recovery/detached_launch.py（87行）通过caller退出/独立session/文件日志/子任务完成/venv解释器保留检查。未改冻结包、未上传、不能接管原运行或证明此次断连根因。四次间隔重连均认证前关闭；原任务状态仍未知，GPU后续受连接阻塞，唯一下一步是只读核对原运行。

- 断连接续核对1：第五次只读重连仍在认证前关闭；最新RESULT_LEDGER没有新增可用结果。上一轮属于实质进展（启动实验及完成最小本地修正），本轮仅重新确认同一连接阻塞，非verified live wait，远端状态仍未知。无新的安全GPU动作或必要CPU实现可推进；目标保持active，等待连接/入口恢复后先核对原进程与结果。

- 断连接续核对2：第六次只读重连仍认证前关闭，台账无新增结果。连接不可达/原进程状态未知这一阻塞已跨首次断连轮及两次自动接续，共连续三轮；可独立完成的快照检查与最小启动修正已完成，无新数据可继续科学判断。任务标记BLOCKED，研究问题仍OPEN/MEASUREMENT_ONLY，原运行仍REMOTE_STATE_UNVERIFIED，不认定停止或失败。恢复同一主机或获知有效入口后，第一步只读核对worker89182及launch/hardware/loader/status，保留原结果，再决定继续或新尝试；不以观察中断直接重启。

- 用户提供新GPU后接续：connect.westc:53036已连通，RTX5090 UUID70fa1c0a…/32607MiB、cgroup90GiB、Xeon8470Q。复用现有安装进程完成的vLLM0.26/Torch2.11cu130/Transformers5.15.1，关键源码SHA均匹配；WiSP恢复到独立目录。新包af3a807e…保持17运行源码及输入不变，仅父机绑定/路径/预算与detached启动变化。monitor3313/parent3314/worker3326实际存活、空闲检查通过，当前第1/16片加载；旧机仍状态未知，不合并新旧机器计时，不自动判旧实验终止。

- 新机r01加载时限修订：第1片约774s、第2片408s、第3片回落3–4MB/s。已核验并仅停止本worker3326，父正常回收−15；2完整分片/2325源张量，0数值/性能cell，1428.133s，2820采样无foreign/OOM增量，部分第3片752877568B远端保留。回读327d646c…/54文件。r02包06cd9efc…只改总时限7200→21600及路径，17运行源码/输入/600s单episode不变；monitor5090/parent5092/worker5127已实际存活，第1片加载。全部尝试与成本分别保留，不记科学负结果。

- 用户切回weste:23478：原worker89182已不存在，8完整SHA分片/0数值0性能，48文件完整回读f4017668…，外部标INTERRUPTED_DURING_LOADING_NO_TERMINAL_RECEIPT而不改旧launch。新r02包7f54419f…在独立目录启动monitor13262/parent13263/worker13275；17源码输入/600s episode不变，整体21600s，临时分片系统盘，两盘>6GiB。已见第1片加载；westc r02断连后状态未知，保留最后4完整分片/第5片下载记录。

- 返回weste r02本轮接续：只读observer34278确认parent13263/worker13275存活并进入第2/16片，采样无foreign，数值/性能尚未开始。CPU准备qwen3_continuous_inputs_v026的16篇候选文章及三档前缀，排除已检查96篇，SHA49cd8abe…；8×(512+64)静态KV上界288块、剩52，未冻结arrival/策略，不能作为实测可行或方法结果。

- 源码核对既有engine复用与WiSP resize_cap：下一版可预声明更多drained episode而不重复加载CPU master；cap48→56须同步runtime.cap/资源记录，既有resize应复制20.25GiB D2D旧scratch且有暂态内存峰。仅源码结论，无GPU热调或新controller；本轮无追加接口、原运行不改。

- weste加载期间CPU单次限量Range probe完成：总64MiB，串行32MiB 11.0307s、四路8MiB 19.4919s，逐8MiB SHA全等；并发最慢请求header等待12.8216s。与当前串行加载并存，非隔离带宽或4GB全分片基准；没有加速依据，不改/重启本轮loader，不扩大并行下载实现。原parent13263/worker13275在probe前后均实际存活并下载第3片。证据return_host_20260913/attempt02/transport_probe01.json。

- 本轮接续最后核对：observer34278返回实际parent13263/worker13275存活，已进入第4/16片；数值/性能均尚未开始。限量并发Range探测没有加速依据，原loader/协议未改，后续只等待并回读本轮结果。本轮为传输因果诊断进展加verified wait，不属于连接阻塞，不标BLOCKED或目标完成。

- verified wait接续：本轮多次读取同一observer34278及其远端/proc检查，parent13263/worker13275持续存活，加载从第4片进入第5片。无新数值/性能产物、无代码或协议变化；不是连接阻塞。保持原问题OPEN，继续当前任务后回读，不追加准备或审计。

- 2026-09-13观察中断补记：observer34278已rc255，[中断记录](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913/attempt02/observation_interruption01.json)与watch末行一致。最后实际live为UTC11:21:00.364910（北京时间19:21:00.364910），parent13263/worker13275存活，第7片1,806,696,448/3,999,975,472B，qualification为空、结果INITIALIZING；当前远端worker/结果UNKNOWN。reconnect_after_observer01/02均255；[handshake_diagnostic02](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913/attempt02/handshake_diagnostic02.json)实见TCP建立、SSH认证前HTTP/1.1 502 Bad Gateway。本地只读路由显示116.172.94.204经utun4，尚不能据此确定根因，也不能推断机器关机或实验终止。未修改raw/冻结输入；本次补记仅改文档，未另行连接或启停，修正此前verified wait的当前状态。

- 控制台补记：UTC11:42:20.570的[控制台只读观察](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913/attempt02/browser_console_observation01.json)显示当前可见5条AutoDL实例均已关机；F04（ddbfwkg1fy-f70e677e）的RTX5090/Gold6459C/92GB与系统30GB、数据50GB配置相符，但页面没有SSH端口，targetSshMappingVerified=false，尚未核验其就是weste:23478。因此“控制台5实例全off”与“目标远端终态回执UNKNOWN”分列；无现成live网页终端，未开机或付费，SSH502/utun4不构成VPN根因结论。
