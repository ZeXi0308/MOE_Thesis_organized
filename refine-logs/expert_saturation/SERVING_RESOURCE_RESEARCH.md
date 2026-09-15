# 固定资源下新工作对已有生成的影响

当前主问题：**单卡固定显存、计算与传输预算下，怎样安排新请求入场与 prefill，减少对已有 decode 的不成比例停顿，同时提高有效服务量？** 研究问题保持不变；可修改模型、机制和实现。当前paging执行链控制admission与prefill；共享执行链已完成resident域的恢复轮转实测，纳入同一资源模型与结论台账，分别保留适用域。

## 适用范围（2026-09-13 由异构输出长度实验钉定）

**上述停顿现象需要同构长输出。** 42 请求 / 1024 输入 / 均值 2048 输出、两臂预留块数（8064）与总输出 token 完全相同、仅输出长度 CV 由 0 变 0.500 的同域对照显示：长度异构使峰值 KV 由 102.0% 降到 68.7%、抢占由 1 次降到 0 次，轮转动作因此无对象，两个 `het-rotate` 格由实现自报 `rotation never applied a forced preemption`，按冻结设计记 `INVALID_NO_ACTION`。原因是 `full_sequence_must_fit` 在准入时只按 prompt 预留（1024→64 块），短请求先退场，长请求的峰值互相错开。详见 [异构长度报告](outputs/admission_capacity/20260913_heterogeneous_length_r01/REPORT.md)。

因此 resident 长上下文四臂结论（轮转把 max-ITL 从 4.46 秒降到 1.01 秒，吞吐差小于已观测重复差）**只在完成时刻聚集的域内成立**；未被推翻，但范围已钉住。同构长输出仍是真实场景（批量摘要、固定长度生成、等长采样）。该实验也说明：只对齐预留总和不足以对齐时间峰值，后续异构域实验须按"短请求退场时刻的全体占用"反解请求数。

同一对照另给出一条与策略无关的量：**异构负载在同等 token 下 wall +23.6%、吞吐 −19.1%**，且由步数完全解释——均宽降 30.5% 而步均成本只降 13.6%，故步数涨 44.2%（2189→3156 步，同为约 84,200 纯 decode token）。这是步成本对宽度强次线性的第三个实例（前两个为 safe29 的 941 步跑在宽度 3、轮转靠维持高宽度换来近乎免费）。**保留无拟合的「步数 × 步均成本」口径**；三项物理分解 `t=γ+α·w+β·C` 域内留出良好（同构 +0.67%、异构 +3.94%，且 γ×Δ步数解释两臂 decode 差 95.9%）但跨域高估 11.8%–24.8%、系数摆动 2–76 倍——serving trace 中 w 与 C 强共变而不可分别辨识，故不主张物理系数。

## 当前判断与证据

- 跨会话共享 [结论台账与四臂结果](experiments/admission_capacity/RESULT_LEDGER.md)。长上下文 native32 / safe29 / headroom32-fast / rotate32-c20 已8/8完成、256/256请求完成，轮转进入独立确认候选；复用其已登记分析。当前 WiSP cap16 短 mixed-batch 校准按自己的相同资源基线比较，不能与 resident 长上下文结果直接排名。下一次加载及每个 episode 边界都检查 GPU 并记录 ABORT/通过状态。
- 代码基线：`agent/publish-current-moe-code`，本轮起点 `7059fc98e0d98a127ff4edda86d44f7f634d26f4`；已有未提交文件保留，不自动提交、推送或修改科学总入口。
- 权威：`docs/current/README.md`、`docs/ideas/README.md`、本目录 `EXPERIMENT_TRACKER.md`、最新 [资产与相关工作报告](outputs/admission_capacity/20260912_admission_paging_r01/REPORT.md) 和 [KV 原始结果](outputs/admission_capacity/20260912_kv_budget_r01/)。旧方向裁决保持原适用范围。
- 四项 KV 对照已经完成。增加实际 KV 池消除该 cohort 的约 4.6 秒停顿，批吞吐提高约 4.0%/4.7%；平均完成延迟略增。它是资源配置结果，没有证明同预算新策略收益。普通 cap 扫描及 `per_request512` 负结果不重复。
- upstream WiSP 固定缓存人工专家 GPU 测试 6 项通过。原始证据在 [pager_smoke](outputs/admission_capacity/20260912_admission_paging_r01/pager_smoke/readback/execution.json)。它不证明完整 OLMoE、原生 runtime 接入或性能。
- **当前状态：OPEN / MEASUREMENT_ONLY。** 完整 OLMoE/vLLM0.26 pager 接入和同输入16层 kernel 检查已完成；batched expert-map工程税移除为正，更强optimized static8基线上的自然与恒定continuation对照都保留了小幅、重复的phase信号。优化后静态曲线能预测同状态未见cap16的首步成本，但独立文档迁移出现10.7%–14.3%偏差；修正后的同engine文档组四格与仅替换incoming的四格均已完成。后者共同旧pair/KV/pager前态一致，首步加载1164→994次，旧请求跨步间隔约308→266ms；这定位了新请求内容的成本影响；随后next-chunk十二格已完成，单次16→8动作对两份文档产生不同完整请求权衡，普通首chunk耗时比例不能统一预测全部候选动作，随后两新incoming的20格强基线对照完成：一份文档的固定one8被static16全面覆盖，另一份仅为权衡，已停止固定位置规则。请求内容影响动作成本的测量结论保留，尚无online selector。resident轮转已出现完整请求权衡的候选正信号，仍无任务质量、SLO或method GO。

- 最新同预算执行对照已4/4完成：expert相对token wall约−26%、旧maxITL约−50%、新TTFT约−45%、payload−34.8%，所有输出相同。详见末节与[报告](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/execution_baseline_v026/REPORT.md)。当前必须在这个已有expert执行基线上重新判断调度空间；没有新增方法GO。

- 最新RR2四臂8/8完成：单层单组与held下一步服务条件成立，但完整calls16→20、payload+1.74%，未取得优于同轮static32证据。RR自身wall漂移22.83%，动作前TTFT/混合阶段差单列；[报告](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/decode_width_rr_v026/REPORT.md)。停止当前RR2机制推进。随后[完整准入强基线](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/admission_width_baseline_v026/REPORT.md)已6/6完成：18测量请求/240输出。延后第三请求保持旧decode每步推进，旧maxITL约173ms降到66–101ms，但新TTFT与完整wall均增加。释放到prefill64虽少搬12.57%，相对固定32的wall差却−17.73%/+17.82%翻号。同臂结构/输出重复而主线程CPU计费明显漂移；随后同D的plain/profile×4/plain已6/6完成，但四份函数profile计时均INVALID，未定位漂移；原始轨迹/输出相同，plain wall仍差65ms。[诊断报告](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/admission_cpu_profile_v026/REPORT.md)。[无profile显存统计等价读取ABBA](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/admission_memory_observer_v026/REPORT.md)随后4/4完成，同工作量/输出和每格1104个allocator值全等，两对wall少约78/74ms；采用nested作为共同观测实现，不能算scheduler贡献或旧漂移归因。[8请求clock arrival六格](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/continuous_admission_v026/REPORT.md)现已完成48请求/1584输出：phase仅启动step0放宽一次，较静态cap2的微小wall节省伴随TTFT变差，仍比cap3/static32慢5.63%/6.80%。停止该二元phase规则在本域的推进；真实超显存Qwen3 BF16的原生运行资格化现已4/4请求完成：16片/18,867源张量/435目标闭合，GPU峰26.19GiB，实际KV511.5MiB；48层finite但layer47未通过原allclose诊断，尚待定向解释。[Qwen报告](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_native_qualification_v026/attempt03/REPORT.md)。参考/JIT计时不作性能，static32/16对照仍UNRUN，研究问题保持OPEN。

## 最小模型与可控制边界

以 [resource_transition_model.py](experiments/admission_capacity/resource_transition_model.py) 的不可变 Request/State/Event 为请求状态接口；包含到达、等待、prefill、decode、恢复、完成，真实 KV 增长/释放，独立策略状态以及 shared miss/LRU 重载分析。新增 grouped-slot 模型表示整组 needed 保护、同 tick 按 slot 淘汰和实际空槽分配顺序；相关结构测试 22/22 通过。旧 [paging_cost_model.py](experiments/admission_capacity/paging_cost_model.py) 只复用成本 helper。group/LRU/专家字节会计仍有效；逐scalar map-write是独立于必需专家payload的实现税，改成batched map后旧时间查表不再代表当前后端，必须重新校准。

资源约束：`nonexpert + expert scratch + actual KV + activation + staging + workspace <= allowed GPU memory`；CPU master/pinned storage 单独计。初始化的 utilization 不代表 KV 字节或运行中 resize。

目标先观察 TTFT、每请求 max-ITL、完成时间与吞吐的权衡；独立校准后冻结请求级 SLO-goodput 约束。每个 shared expert load 只计一次，分组导致的重载另计。时间区分计算、暴露搬运、排队/恢复与观察/调度开销，不能直接相加重叠 span。

2026-09-13共享[轮转实测](outputs/admission_capacity/20260913_rotation_fourarm_r01/REPORT.md)对该模型补充了两项已观察约束。第一，单请求恢复等待取决于整个缺席队列、最短缺席时间、全局交换冷却、victim资格与可释放KV，不能由交换间隔单独给上界。最长事件在931被抢占、976开始恢复、979返回新token，实际max-ITL约1.006s，旧0.39s估算已被反例修正。[absence_rotation](experiments/admission_capacity/absence_rotation.py)与[原生adapter](experiments/admission_capacity/rotation_native.py)已维护并执行这些状态；通用CPU动作枚举尚未纳入该恢复动作。

第二，完整成本必须随各策略的实际batch轨迹求和：`cohort wall = sum(scheduler inclusive + engine non-schedule) + outside engine`。重算调用可能同时产生新decode位置，不能把整个调用计为纯重算再另加decode时间。轮转每轮比native多重算30,876位置，但总调用1348→1257、纯decode宽度1调用132→15；[互斥成本账本](outputs/admission_capacity/20260913_rotation_fourarm_r01/analysis/rotation_accounting.json)显示新增恢复成本与减少的小batch尾部部分抵销。当前是执行后可复核的会计关系，尚未成为在线完整请求预测器；动作模型必须重新生成后续batch/完成轨迹，不能固定旧尾部再额外加重算税。

同轮轮转相对headroom-fast的吞吐读数高2.435%/2.242%，最大ITL低约0.376/0.364s；相对native吞吐差−0.548%/+0.055%，平均完成却增加2.244%/1.600%，另有两条请求新增约0.95s停顿。因此它支持恢复顺序存在可作用空间，未证明原生吞吐非劣或逐请求支配。`held=0`也不支持把收益归因于余量保护。上述结论均继承现有四臂分析，不重新消费同一raw生成重复结论；它们不能直接证明paging域的轮转收益。

尚未接入原生 scheduler的是成本模型的候选动作枚举；本轮用于真实执行的native phase-prefill wrapper已经实现。模型的 token 预算没有纳入本 step 原生恢复 work，不能直接执行为完整 schedule。原生 `max_num_running_reqs` 也会挡住恢复，不等于只控制新请求；合法的新准入动作需识别 never-started WAITING，且在 KV 分配/状态推进前决定。已有正常 decode 必须继续，必要抢占由原生引擎处理并记录。

## 已完成的当前运行链

最弱链路：既有 pager 能否通过当前 vLLM0.26 真正执行 OLMoE，同时准确记录加载、驱逐及请求身份。

- 复用官方 WiSP `86f69720f0de0647c51728ea2e27e4289bf3dea2` 的 `WispMoEState`，适配已取回的实际安装源码，不注册全局插件或更换环境。
- 同 BF16/TRITON/eager wrapper，cap64 与 cap24；显式 KV 1 GiB；4 个既有自然文本前缀，每个 64 输入/8 输出，50 ms 到达间隔，token budget128。
- cap64 是能容纳全部专家的容量对照，专家 scratch 更多；其测量期仍有142次首加载，没有提前填满全部专家，不能当充分暖机的稳态全驻留速度基线。两臂比较不能写成同资源方法收益。后续机制臂固定相同 cache/KV、pager 和后端。
- 复用 [run_native_pager.py](experiments/admission_capacity/run_native_pager.py) 和现有 native_capture；模型重排后建立 row→request/position，记录初始化与测量阶段，保留所有失败。
- CUDA load span 包含 ensure_resident 内部复制、映射及可能的主机供给间隙；不是纯 memcpy 时间，也不等于完整请求暴露等待。
- 证据上限：完整模型接入及真实专家权重加载测量。受控 OLMoE 缓存限制不是必须 offload 的部署证据；后续超显存主张需真实超显存模型。
- [attempt02](outputs/admission_capacity/20260912_native_pager_r01/attempt02/readback/results/execution.json) 两臂各4/4请求、32个输出完成；16层和全部真实输入行均与请求/位置对齐。两臂输出token相同，但各自批次/后续状态独立演进，不能当同轨迹速度比较。
- cap24 的最坏混合步骤包含1个 decode +127个 prefill输入；已获调度的旧请求生成间隔1.3318秒，无KV抢占。该步提交55.1016GiB专家权重 tensor-copy payload，一次缺页并集下界6.9258GiB，额外加载48.1758GiB，放大7.956倍。纯decode步骤约49–82ms。数据与会计见 [analysis.json](outputs/admission_capacity/20260912_native_pager_r01/attempt02/analysis.json)。这些字节来自实际copy_调用的两矩阵行大小，没有硬件PCIe wire计数。
- [attempt03](outputs/admission_capacity/20260912_native_pager_r01/attempt03/readback/results/cap24_validation/pager_summary.json) 在独立验证臂中，对每层首次请求调用的相同前16行 hidden/top-k，比较分页输出与直接完整专家权重kernel：16/16逐位一致，最大误差0。验证时间不计为性能；不是任务质量测评或所有shape保证。
- fresh same-family 完整性复核为 **provisional WARN，P0=0/P1=0**。A–F 为 `PASS/WARN/PASS/PASS/WARN/PASS`：冻结源码与readback哈希、调用路径、重排后row join、请求完整性、cache transition及字节求和均闭合；WARN只保留单模型/单卡/单episode范围、非充分暖机/JIT边界，以及copy payload不是PCIe wire计数。支持上限仍是当前0.26原生接入、实现内加载会计和16层首次前16行数值诊断，不升级为性能、任务质量或普遍正确性。

同一0.26 bridge又完成了 [cap24静态prefill transfer ABBA](outputs/admission_capacity/20260912_wisp_olmoe_r01/native_transfer/results/execution.json)：4/4 cells成功，每cell 4/4请求、32个输出，均为实际KV 1GiB、512 blocks、cap24、4.5GiB expert scratch、CPU0–7；无抢占且已有decode均继续。prefill32相对default的结果如下，copy是实现内weight-copy payload。

| 0.26 transfer | r0：default→prefill32 | r1：default→prefill32 |
|---|---:|---:|
| 已有请求max-ITL | 1.602864→1.256633s（−21.60%） | 1.407840→1.272461s（−9.62%） |
| 全请求mean TTFT | 2.116391→2.297682s（+8.57%） | 1.852705→2.598888s（+40.28%） |
| episode wall | 3.635877→3.357278s（−7.66%） | 3.359002→3.663429s（+9.06%） |
| weight-copy payload | 116.3203125→117.4921875GiB（+1.007%） | 116.3203125→117.4921875GiB（+1.007%） |

该重复只支持vLLM0.26下静态prefill的ITL→TTFT权衡，完整请求wall符号翻转。两臂资源相同，但step返回会改变后续到达提交和实际admission轨迹，不能称为matched-prestate action counterfactual，也不与0.11.2冻结action证据合并。

## 新回传的同资源动作证据与模型校准

共享工作区另一执行链完成了 [vLLM0.11.2 注入矩阵](outputs/admission_capacity/20260912_wisp_olmoe_r01/injection/execution.json)。这是沿用官方WiSP的校准证据，不与0.26的绝对时间混合；主原生适配仍保留0.26。8项覆盖hold、short8、long8、long32，正反序重复；固定cap16/BF16/LRU、512MiB KV、maxbudget64、maxseq3。两个旧请求各输出4token后注入新请求，全部臂在该位置的tokens、scheduler/KV、完整pager slot/LRU状态一致。

| 实测量 | long8，两次 | long32，两次 |
|---|---|---|
| 旧请求动作后最大ITL | 321/247ms | 664/675ms |
| 新请求TTFT | 4.193/3.141s | 2.468/2.498s |
| 整批wall | 6.049/4.492s | 4.229/4.300s |
| 动作后权重copy payload | 120.070GiB，两次一致 | 115.992GiB，两次一致 |

**结论是请求级权衡：较小chunk减轻单次生成停顿，同时增加新请求等待与整体成本。** 两个旧请求在动作后的自然输出随后分歧；各策略独立演进，不共用未来route。重复存在明显时间波动与冷shape/观测开销边界，不能据此给出生产尾延迟或新方法GO。hold没有第三请求，只作负控，不作同任务吞吐基线。

随后完成的4个无trace plain cells按32/8/8/32运行：chunk8 wall为5.703/4.363s，chunk32为5.323/5.350s，排序翻转，原始不利结果保留。再把两臂进程共同限制到CPU0–7、保持OMP8与GPU/模型/专家/KV不变，完成4个affinity cells；这只是CPU放置控制，`Mems_allowed_list`仍为0–1，没有显式membind。

| affinity静态动作 | 旧请求max-ITL，r4/r5 | 新请求TTFT，r4/r5 | episode wall，r4/r5 | 完整episode copy payload |
|---|---:|---:|---:|---:|
| chunk8 | 308.7/307.3ms | 4.0064/4.0401s | 5.8161/5.7843s | 163.08984GiB |
| chunk32 | 822.9/810.1ms | 3.0674/3.0296s | 5.3961/5.2683s | 159.01172GiB |

固定CPU后重复方向一致：chunk8保护旧请求生成间隔，但新请求更晚、整批更慢且完整搬运更多。它没有证明NUMA是plain漂移的唯一原因，也没有形成scheduler收益；完整证据分别保留在 [plain](outputs/admission_capacity/20260912_wisp_olmoe_r01/plain/) 与 [affinity](outputs/admission_capacity/20260912_wisp_olmoe_r01/affinity/)。

成本守恒修正为 `episode copy = sum(each layer-call entry-miss bytes + that call extra-load bytes)`。从long8改long32，entry-miss累计减少32.355GiB，而call内extra增加28.277GiB，净总copy减少4.078GiB。只降低局部extra不能预测总服务收益。

[calibrate_grouped_pager.py](experiments/admission_capacity/calibrate_grouped_pager.py) 从每个策略自身action前slot/tick/clock出发，输入其真实post-router分组；[校准结果](outputs/admission_capacity/20260912_native_pager_r01/grouped_model_calibration.json) 在8条轨迹、5,924组上逐项复现load/evict字节和最终完整状态。这验证成本会计，不是预测未知动作或未来路由。

## 唯一下一实验与资源状态

原始带trace midpoint协议曾用chunk8/32两端点冻结最小线性模型，给出chunk16首步权重payload 10.211GiB和首个旧decode间隔0.324/0.282s；该协议的运行仍是下述 `BLOCKED`，没有产生测量。

[冻结预测和配置](outputs/admission_capacity/20260912_native_pager_r01/midpoint_prediction/config.json)、[执行入口](outputs/admission_capacity/20260912_native_pager_r01/midpoint_prediction/run_remote.py) 已保存。GPU被另一任务占用，当前 [execution.json](outputs/admission_capacity/20260912_native_pager_r01/midpoint_prediction/execution.json) 为 `BLOCKED_BEFORE_GPU_INITIALIZATION`，0次GPU测量；预测数值不是结果。此前0/32 ABBA也在加载前退出，其原始attempt04保留；已回传更直接的注入矩阵后不重复该存在性对照。

基于affinity端点冻结的 [affinity_config.json](outputs/admission_capacity/20260912_native_pager_r01/midpoint_prediction/affinity_config.json) 和 [run_affinity_remote.py](outputs/admission_capacity/20260912_native_pager_r01/midpoint_prediction/run_affinity_remote.py) 已在远端新目录 `/root/autodl-tmp/wisp-runtime-0112-r01/affinity-midpoint16-r01` 完成两个chunk16 cells，实际执行时间为12:34:16.507–12:35:23.818 UTC。安全回读的 [analysis.json](outputs/admission_capacity/20260912_native_pager_r01/midpoint_prediction/safe_readback/analysis.json)、[execution.json](outputs/admission_capacity/20260912_native_pager_r01/midpoint_prediction/safe_readback/execution.json) 和 [raw_manifest.json](outputs/admission_capacity/20260912_native_pager_r01/midpoint_prediction/safe_readback/raw_manifest.json) 保留派生数值、执行记录及哈希/源码；完整raw因含workload未获自动审批回传，仍只在上述远端目录。

| affinity chunk16 | repeat 0 | repeat 1 |
|---|---:|---:|
| 冻结first-ITL预测 | 354.854ms | 362.022ms |
| 实测first-ITL | 338.834ms（误差−4.515%） | 339.241ms（误差−6.293%） |
| 已有请求max-ITL | 489.066ms | 490.712ms |
| 新请求TTFT | 3.476334s | 3.459523s |
| episode wall | 5.557003s | 5.537933s |
| 完整episode copy payload | 162.90234375GiB | 162.90234375GiB |

两cell均为3/3请求、每cell 40个输出，prestate及全部检查为true。每cell的29个NVML样本只看到本cell worker PID且CPU allowlist为0–7；这是离散采样，不证明采样间持续隔离，也没有显式membind。无inline trace，所以首动作payload仍是`NOT_MEASURABLE`。这是同三个文档上的未见action，不是文档holdout、在线scheduler证据或阈值方法结论；原始traced `BLOCKED`记录不变。

[static8/phase8 ABBA](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/safe_readback/analysis.json) 已完成4/4 cells。每cell 3/3请求和40个输出完成，paired scheduler/pager/slot/LRU prestate检查全为true，decision issue和抢占均为0，三个请求的输出hash在四臂完全相同。两策略到call index 15为止执行相同逻辑工作；index 16起，static8用4个call各做8-token prefill，phase8一次做32-token prefill并解除限制。

| 已验证的动作/会计 | static8 | phase8 |
|---|---:|---:|
| 总engine calls | 27 | 24 |
| 完整episode copy payload | 163.08984375GiB | 162.94921875GiB |
| phase8差值 | — | −144MiB |

| 性能诊断：static8→phase8 | repeat 0 | repeat 1 |
|---|---:|---:|
| episode wall | 5.808687→4.312848s（−25.75%） | 4.305069→5.733669s（+33.18%） |
| 已有请求完成时刻 | 4.525592→3.518542s | 3.480656→4.550808s |

约1秒的差异在策略尚未分叉时已经出现：共同prefix首步中static8-r0/phase8-r1约1.22s，而phase8-r0/static8-r1约0.97s。采样到的GPU clocks、CPU0–7 affinity和零throttle没有解释该分组；没有显式membind，也不能把NUMA写成根因。因此当前结论是 **`ACTION_AND_ACCOUNTING_VALID / CAUSAL_PERFORMANCE_UNRESOLVED / MEASUREMENT_ONLY`**，没有method GO。

[同进程、同engine ABBA](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/same_engine/safe_readback/analysis.json) 随后完成4/4 cells。四臂均为PID 23757，持久allocation hash均为`bbcaa057…`，KV free queue顺序相同，完整prestate均为`7d07d456…`；资源检查有效，decision issue为0，三个请求输出hash相同。

| 同engine性能诊断：static8→phase8 | repeat 0 | repeat 1 |
|---|---:|---:|
| episode wall | 5.771153→5.634268s（−2.37%） | 4.868633→5.618388s（+15.40%） |

27→24 engine calls和163.08984375→162.94921875GiB（−144MiB）的动作/会计差值继续复现，但wall仍符号翻转。第四臂static8-r1首步为1.219s，仍处于约1.2s慢带；其后已有请求max-ITL却降至0.242s，而其余三臂约0.307s，old-done到wall尾段也只有0.821s，对比static8-r0的1.284s。这说明相同PID和持久allocation下仍有episode内变化，allocation-only假说不足；采样GPU clocks、CPU0–7 affinity和零throttle也未给出解释，不把NUMA写成根因。当前仍是 **`ACTION_AND_ACCOUNTING_VALID / CAUSAL_PERFORMANCE_UNRESOLVED / MEASUREMENT_ONLY`**，既没有method GO，也不构成方法或科学问题NO-GO。

原4个独立进程cells与新4个同engine cells均在各自目录保留，未覆盖；同engine完整raw仍在远端，本地只保存safe readback。[独立输入配置](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/independent_input.json) 保留为运行前冻结记录，其中选择的文档已用于下述completed optimized holdout，不能再把该文件内的`PREPARED_UNRUN`当作当前campaign状态。

既有traced injection的只读成本桶分解显示：相同schedule、subgroup needed/load-evict/bytes的long8 r0→r1中，action后23 calls wall相差1202.059ms，host-read union只相差111.604ms，余下1090.455ms（90.72%）位于host-read spans之外；hold的区间外占比同为90.10%。CUDA load span差124.101ms且与host spans重叠，不能称作纯H2D时间。

[profiler attempt02摘要](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/profile/attempt02/profile_summary.json) 与 [safe analysis](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/profile/attempt02/safe_readback/analysis.json) 对同engine static8做了4次诊断：

| profiler cell | p0 | p1 | p2 | p3 |
|---|---:|---:|---:|---:|
| request wall | 11.456s | 8.667s | 5.009s | 6.285s |
| GPU activity union | 3.423s | 3.423s | 3.423s | 3.423s |
| wall减GPU activity | 8.027s | 5.236s | 1.583s | 2.858s |
| capture-byte状态 | lower bound closed | lower bound closed | lower bound closed | `INCOMPLETE_ACTIVITY` |

p0–p2的capture byte lower bound闭合，p3的活动字段/边界不完整并原样保留。[boundary addendum](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/profile/attempt02/boundary_addendum.json) 随后在原trace中找到该8MiB copy：它从call13/14 CPU markers之间开始，global大copy计数与专家counter闭合；原`INCOMPLETE_ACTIVITY`文件不改写，这仍不证明精确CPU/GPU边界。每cell约27,869–27,870次4-byte H2D和34,285次`cudaStreamSynchronize`与逐scalar expert-map更新路径一致，是可验证的工程税候选；profiler插桩本身显著放大wall，这些数值只用于活动归类，不支持CPU根因、性能加速或请求收益主张。

失败证据不覆盖：[首次launcher](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/profile/execution.json) 因导入路径错误在GPU初始化前失败；attempt02首次 [analysis](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/profile/attempt02/analysis_attempt01.json) 因CPU/GPU marker schema混淆失败，随后只重析同一raw，没有GPU重跑。当前本地结论限于上述summary与safe readback，远端原始目录保留且未覆盖。

[batched-map summary](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/batched_map/summary.json)、[safe analysis](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/batched_map/safe_readback/analysis.json) 和 [completion](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/batched_map/completion.json) 显示该同engine static8 ABBA已4/4完成。两臂prestate、schedule、三个输出hash、final pager state和专家weight-copy bytes相同；5个BF16 toy CUDA all-hit/fill/eviction cases通过，四cell的16层host inverse/full device map检查均通过。

| map更新：original→batched | repeat 0 | repeat 1 |
|---|---:|---:|
| episode wall | 5.540343→3.911191s（−29.41%） | 4.340916→3.808304s（−12.27%） |
| 已有请求max-ITL | 303.400→211.271ms | 239.177→210.392ms |
| 新请求TTFT | 3.920823→2.664095s | 3.026612→2.655714s |

实现把27,834次source-level 4-byte scalar device writes改为1,672次整map copies；metadata payload由约108.7KiB增至418KiB，两臂都预留相同4KiB pinned staging，专家weight-copy payload均保持163.08984375GiB。因此正信号来自减少细粒度更新/同步机会，不是减少专家权重字节。original自身仍从5.540s漂到4.341s，效应量不能视为稳定估计，既有wall根因也未完全解析。当前分类为 **`ENGINEERING_TAX_REMOVAL_POSITIVE / MEASUREMENT_ONLY`**；这是pager实现改进证据，不是scheduler GO。

[optimized holdout summary](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/summary.json)、[safe analysis](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/safe_readback/analysis.json) 和 [completion](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/completion.json) 显示该optimized common-pager static8/phase8 ABBA已4/4完成。三个文档相对当前校准输入不重叠，shape仍为两个32-token已有请求加一个128-token新请求；四臂prestate同为`134afb54…`，资源、请求身份、决策和16层map检查均通过。

| optimized pager：static8→phase8 | repeat 0 | repeat 1 |
|---|---:|---:|
| episode wall | 3.850803→3.788532s（−1.617%） | 3.841899→3.802419s（−1.028%） |
| 已有请求max-ITL | 202.461→203.340ms | 203.164→206.023ms |
| 已有请求完成时刻差 | +0.289ms | +8.154ms |
| old-done后tail | −8.530% | −6.615% |
| 新请求TTFT | 2.699880→2.657274s | 2.688242→2.666425s |
| engine calls | 27→24 | 27→24 |
| 完整episode copy payload | 166.0078125→164.70703125GiB | 166.0078125→164.70703125GiB |

这形成了更强公共基线上的小幅、重复phase信号：wall/new-request TTFT/post-old tail同向改善，同时已有请求max-ITL略升。已有两个请求的输出hash跨策略相同；新请求hash在static8与phase8之间不同，但各策略内部两次一致，且尚未做质量验证。由于continuation已经分叉，观测到的copy下降不能全部归因于同一未来route。结论仍为 **`PHASE_RESIDUAL_POSITIVE / MEASUREMENT_ONLY`**：不因幅度小任意判死，也不升级为scheduler GO、稳定效应量或任务质量结论。

首次上传metadata被自动审批拒绝，[原记录](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/execution.json) 保持0上传/0 GPU；随后 [public source check](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/public_source_check.json) 用官方Hugging Face `Salesforce/wikitext`完整正文hash确认三份文档均为公开数据，[approval addendum](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/approval_addendum.json) 记录未变archive获批，实际campaign已完成。该历史拒绝不是当前阻塞状态。

[controlled-token summary](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/controlled_tokens/summary.json)、[safe analysis](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/controlled_tokens/safe_readback/analysis.json) 和 [completion](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/controlled_tokens/completion.json) 显示native `allowed_token_ids`对照已4/4完成。四臂完整prestate同为`a2af02be…`，三请求输出hash全部相同，新请求8个实际生成位置的单例token约束均有效；5个BF16 toy CUDA cases与四cell的16层map检查均通过。GPU/CPU native mask各150,912B，四cell分别复用相同指针，CPU mask未pinned；primer wall依次为109.1/41.9/39.6/41.9ms，位于episode wall之外，随后仍执行原warmup。

| 恒定continuation：static8→phase8 | repeat 0 | repeat 1 |
|---|---:|---:|
| episode wall | 3.777829→3.743439s（−0.9103%） | 3.784312→3.719282s（−1.7184%） |
| 已有请求max-ITL | 203.717→204.341ms（+0.624ms） | 203.113→203.452ms（+0.339ms） |
| 已有请求完成时刻差 | −8.624ms | −11.272ms |
| old-done后tail差 | −25.766ms | −53.758ms |
| 新请求TTFT | 2.705281→2.665377s | 2.723734→2.658067s |
| engine calls | 27→24 | 27→24 |
| 完整episode copy payload | 160.81640625→160.7578125GiB（−60MiB） | 160.81640625→160.7578125GiB（−60MiB） |

60MiB恰对应5次专家加载。自然生成holdout的新请求首处分叉在输出index 3，TTFT改善发生在该分叉之前；恒定continuation下的小幅wall/TTFT信号仍同向重复。自然与受控campaign分别减少1.30078125GiB和60MiB，但受控primer改变了prestate，后续continuation也不同，不能跨campaign相减并把差额归因于route。以optimized static8为当前强简单基线，结论仍为 **`PHASE_RESIDUAL_POSITIVE / MEASUREMENT_ONLY`**；没有exact Oracle、任务质量/SLO或scheduler GO。

下一唯一实验是同prestate重校准optimized fixed-prefill 8/16/32权衡；只有新基线校准后才扩大episode。当前无GPU任务，不再做token数值追查，也不引入新predictor或controller。

## 相邻工作与贡献边界

最新版本/代码范围已核查并列于上面的资产报告。WiSP 已公开动态控制器；FluxMoE 已覆盖在线评测；FreeToken 有公开 runtime，不能沿用旧 README 的缺功能描述。SLOs-Serve/Niyama 已覆盖性能模型、准入和生成 deadline 下的 prefill 控制。因此“成本模型 + admission/prefill”本身没有新颖性。

待验证的剩余问题是：固定 pager 下真实共享搬运、重载或恢复成本，是否改变普通长度/KV/队列/slack 控制的动作排序并影响完整请求。历史专家信息不是必选贡献；必须用同底座消融分离其增量。当前不存在 method GO 或问题级 NO-GO。

## 接续

每轮假说、命令、结果和下一动作只追加到 [SERVING_RESOURCE_LOG.md](SERVING_RESOURCE_LOG.md)。原始尝试永久保留；没有新数据时不追加重复审计。完成标准仍是一个最小有效机制、可解释模型、真实运行路径、强基线、重复与必要消融/失败边界，或足够的问题级否证。

## 2026-09-13：优化后静态动作曲线与未运行动作预测

本轮问题：batched-map 后端下，普通 prefill 工作量能否解释固定状态的首个 mixed step 成本？复用 cap16/KV512MiB、同一引擎与三个已观察公开文档，按 `8→32→32→8→16→16` 实际完成六格。首四格结束后、两个16 cell开始前冻结线性插值；不使用16结果或未来路由。已通过请求身份、实际首步shape、每个ready decode推进1 token、同前状态/任务/资源、零抢占重算与map正确性检查。全部测量窗口的GPU进程采样只见同一worker PID37306；采样不能排除间隔内短暂竞争。

| prefill cap | wall 秒，两次 | 已有请求最大ITL 秒 | 新请求TTFT 秒 | 专家 payload GiB / engine calls |
|---|---|---|---|---|
| 8 | 3.9156 / 3.8307 | .2030 / .2027 | 2.7621 / 2.6794 | 166.0078 / 27 |
| 16 | 3.7583 / 3.7579 | .3213 / .3219 | 2.4316 / 2.4313 | 165.6797 / 19 |
| 32 | 3.6924 / 3.6740 | .5622 / .5624 | 2.1926 / 2.1838 | 162.4453 / 16 |

首mixed step预测299.833 ms，16实测305.031/305.745 ms，残差+5.198/+5.911 ms（+1.734%/+1.972%）；对应已有请求跨动作间隔预测300.737 ms、实测305.982/306.659 ms。冻结时序和四条端点raw哈希均通过。该模型在此状态能预测未见cap的局部成本；没有验证完整请求预测或专家历史增量。32相对8的整批wall减少5.70%/4.09%，代价是已有请求max-ITL约2.77倍，不能称无损加速。各cap两次输出hash一致；16的一条旧请求输出跨cap不同，新请求在32下也不同，因此完整结果是各策略自由演进的服务轨迹，字节差不能全归因于固定token执行节省。

证据：[summary](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_static_calibration/summary.json)、[冻结预测及完整派生检查](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_static_calibration/safe_readback/analysis.json)、[完成记录](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_static_calibration/readback_completion.json)。原始result留远端；只回读派生指标、哈希、配置和执行元数据。首次审批拒绝已凭本会话原始用户SSH授权解除；一次监控解释器导入失败和一次GPU_BUSY/91均保留，随后同包六格完成。

本轮Verdict：MEASUREMENT_ONLY；Evidence：原生vLLM0.11.2上的三请求自定义注入与真实paging；强基线：同后端固定cap8/16/32；Oracle：未运行跨策略Oracle；失败类别：未发现本轮测量失效，外推未验证。主问题仍OPEN。唯一下一实验：固定本轮曲线，在未参与校准的新文档episode上独立执行8/16/32，检验动作成本排序与预测误差是否迁移。若普通工作量已足够解释，不引入专家预测器；若失败，先定位状态/工作集变化造成的残差。

## 2026-09-13：冻结曲线的文档迁移

固定上一轮8/32端点均值及原16插值，在运行前冻结175.236/299.833/549.029ms首步预测。确定性选择原数据索引7/8/9的三篇新文档，与校准4/5/6的文档ID/完整原文SHA不重合；公开原文逐篇SHA复核通过。保持原资源、代码后端与六格顺序，各策略独立执行，全部完成并回读。[summary](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_static_transfer/summary.json)；[完成状态](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_static_transfer/readback_completion.json)。

| cap | 新文档首步ms，两次 | 实测相对冻结预测残差 | wall秒，两次 | old最大ITL秒 |
|---|---|---|---|---|
| 8 | 150.120 / 150.098 | −14.333% / −14.345% | 3.7258 / 3.7051 | .1995 / .1987 |
| 16 | 267.734 / 267.333 | −10.706% / −10.839% | 3.5988 / 3.5979 | .3109 / .3110 |
| 32 | 481.542 / 481.248 | −12.292% / −12.346% | 3.5309 / 3.5304 | .5609 / .5596 |

残差分母均为预测值；负数表示预测偏高。排序保持8<16<32，但旧曲线不是跨文档精确时间模型。同动作首步两次变化约0.01%/0.15%/0.06%，仅是本轮观测差异，不是噪声上界或显著性。各cap两次输出hash一致；cap16一条旧请求的输出与8/32不同，新请求三种cap输出相同。

当前不能把偏差直接归因文档：新旧campaign是不同进程、不同前状态，相隔约21分钟。全部cell资源hash相同、采样GPU只有自身worker、CPU affinity0–7、采样CPU throttling增量0、显存时钟相同，仍不能排除采样间隙/分配/上下文差异。全episode专家tensor payload少3.9%–5.0%，但它含preaction与后续轨迹，不能拿来解释首个mixed step的百分比或计算首步带宽。

本轮仍MEASUREMENT_ONLY，局部动作排序迁移成立于所测文档；精确成本迁移未成立，未证明专家历史增量，也无SLO/方法GO。唯一下一实验改为固定cap16、同engine旧文档/新文档/新文档/旧文档四格，验证文档条件差异是否在共同运行过程中保留，并用首动作前后host加载计数定位实际tensor搬运。不同文档允许不同前状态，要求同文档重复前状态一致；不再重复同形状阈值扫描。

该四格诊断的首次尝试在首个mixed step后被新增断言中止，0个完整cell，状态为 **INVALID_EXPERIMENT**：[失败记录](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/document_state_abba/completion.json)。错误把 `ensure_calls` 当作16层数；WiSP每层可执行多个token subgroup，每个subgroup独立调用ensure。该失败不支持文档效应存在或消失。原始result、代码与日志保留在远端 `document-state-abba-r01`，不修饰旧结果。

修正包位于 [attempt02](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/document_state_abba/attempt02/config.json)，只要求计数覆盖16个layer条目、subgroup调用为正、delta非负且map copies不超过subgroup调用。真实加载payload按首step的miss增量乘每专家12MiB计费；它覆盖重复加载，不是唯一工作集字节、PCIe wire bytes或纯copy耗时。针对该bug的4个CPU用例通过，尚未完成修正后的GPU验证。两次自动审批拒绝了代码/配置上传；第二次明确不接受从历史会话恢复的授权作为当前可信批准。[拒绝记录](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/document_state_abba/attempt02/approval_rejection02.json)保留；修正包0上传、0GPU。当前唯一下一步仍是获准后在新 `document-state-abba-r02` 目录执行同一四格，不改模型、机制或研究问题。


## 2026-09-13：同engine文档状态对照完成

用户明确批准后，原SHA不变的修正包在新远端 `document-state-abba-r02` 完成4/4，PID45315。当前状态以[readback completion](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/document_state_abba/attempt02/readback_completion.json)为准；旧INVALID与两次审批拒绝均保留。四格同engine/资源，各文档组内部prestate、输入与输出hash一致；old/new prestate不同，因此仍同时改变了旧decode请求与incoming内容。[summary](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/document_state_abba/attempt02/summary.json)。

| 固定chunk16 | 首mixed step ms | 首step subgroup / expert loads | 首step tensor payload GiB | episode wall s |
|---|---:|---:|---:|---:|
| old r0 | 339.949 | 134 / 1164 | 13.640625 | 4.293169 |
| new r0 | 304.066 | 119 / 1012 | 11.859375 | 4.079421 |
| new r1 | 291.120 | 119 / 1012 | 11.859375 | 3.762162 |
| old r1 | 305.900 | 134 / 1164 | 13.640625 | 3.758127 |

每个文档组的加载和分组数均逐次相同；new分别少13.06%加载与11.19%分组。首step为同18-token shape、16层，说明普通token数量不能唯一确定实际paging工作量。计数是动作后的诊断，尚非在线可用的未知动作特征，也不代表纯copy时间或PCIe wire流量。两次host计数读取合计0.0875–0.2214ms；首step计时排除读取，完整episode包含其成本。

new首step配对短10.56%/4.83%，但old自身首step漂移−10.02%、wall漂移−12.46%，new也有漂移。跨文档wall差−4.98%/+0.11%，不能主张稳定速度收益。动作前纯decode三步中位数在r0为old51.10/new46.27ms，在r1为old37.86/new45.86ms，后一对排序与mixed相反；近期decode单值在本例不充分，不能据此判死普通反馈。采样GPU仅自身PID，SM/显存clock相同、CPU throttling增量0；采样间瞬态仍未排除。

本轮Verdict为MEASUREMENT_ONLY：确认相同形状下文档状态改变真实执行工作量，尚未确认稳定性能效应或新调度收益。强基线仍为同底座固定chunk及普通阶段/延迟反馈；exact Oracle未测。唯一下一实验固定旧请求source4/5与共同prestate，只把第三请求source6换为9，保持chunk16并做A/B/B/A，区分incoming内容与先前decode/cache状态的贡献。该对照使用已观察文档，是因果定位而非新holdout。

该incoming-only对照的[配置与入口](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/incoming_document_abba/config.json)已准备并上传，代码包SHA3085169b…；原warmup/reset/资源不变，四格额外强制共同prestate134afb54…。两次启动均在GPU初始化前ABORT/91，0个实测cell，首次检测到外部PID45993占用。失败记录保留；下一动作是在新的空闲检查通过后执行同一staging入口，不重新选输入或修改机制。


## incoming-only 因果对照与下一动作（2026-09-13）

[incoming_document_abba/summary.json](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/incoming_document_abba/summary.json) 已4/4完成，最终状态见同目录readback_completion.json；此前两次GPU_BUSY保留。A=[4,5,6]、B=[4,5,9]，四格共同prestate为134afb54…，实际资源、engine及旧请求最终输出一致。由本地公开源按冻结算法重建的两份完整workload哈希与远端记录一致，支持只有第三条incoming输入改变。首步旧→新为134→119 subgroup、1164→994次专家加载、13.640625→11.6484375GiB tensor payload；两次计数相同。首步时间306.906/306.789→265.790/263.906ms，旧请求跨该步间隔307.996/307.718→266.793/265.014ms。计数观察税0.091–0.098ms。

这是固定shape和共同旧请求前状态下的incoming内容因果诊断，仍为MEASUREMENT_ONLY。不同incoming输入及自然continuation的wall差−9.90%/−8.04%不是调度加速；payload不是PCIe wire字节。四格内重复漂移只是本次观测，不是总体噪声上界。

同目录[post_chunk_timing.json](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/incoming_document_abba/post_chunk_timing.json)只读导出已执行的8个fixed16混合调用。首chunk预测下一chunk时，A高估1.75%/1.91%，B低估5.64%/6.14%；更后的部分chunk中B反而比A稍慢。因此不能给每文档一个恒定成本系数。下一唯一实验共同完成chunk16，再只将下一chunk分叉为8/16/32，随后恢复16；比较完整请求权衡，并先用冻结token/shape曲线和已完成mixed耗时的普通反馈解释动作，专家计数不得使用未来信息。当前还没有该动作的GPU结果。


## 共同首chunk后的单次动作分叉（2026-09-13，12/12 COMPLETE）

[next_chunk_branch/summary.json](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/next_chunk_branch/summary.json) 汇总两个随机六格block：每份文档共同首chunk16，然后下一chunk为8/16/32，余下恢复16。每cell三请求、40输出完成，共36请求/480输出；同文档六个prebranch哈希一致，同engine/资源及所有动作、计数检查通过，三个请求输出在各文档的所有动作中完全相同。状态哈希覆盖请求token、逻辑KV块/进度及pager slot/LRU，未比较KV tensor字节。共同snapshot约1.06–1.79ms，保留在wall与请求间隔中。

| 下一chunk相对16 | wall差，两个block | 旧请求最大ITL差 | 新请求TTFT差 | 完整payload差 |
|---|---:|---:|---:|---:|
| A→8 | −11.030/−3.698ms | +5.747/+6.816ms | +31.722/+35.627ms | −0.59765625GiB |
| B→8 | −43.449/−40.499ms | −9.363/−8.050ms | +6.141/+5.660ms | −2.1328125GiB |
| A→32 | −49.517/−22.900ms | +234.046/+236.778ms | −89.915/−73.257ms | −1.1953125GiB |
| B→32 | +46.878/−34.601ms | +163.715/+166.288ms | −75.036/−67.701ms | −1.0546875GiB |

8动作使calls19→20，32使19→18；额外一次call与更少专家payload可以同时发生。8相对16的B−A interaction为wall−32.419/−36.801ms、旧最大ITL−15.109/−14.865ms，支持本doc对存在不同动作响应；B的较好wall/旧间隔仍以新TTFT增加为代价。A/B各自baseline两次wall漂移−9.368/−4.103ms只是观测，不是总体噪声界；B→32的wall符号翻转未解释，首格完整保留。每cell15–17个GPU样本只见worker58240、CPU0–7；这是采样证据，不是持续独占保证。

冻结static曲线和first-mixed耗时比例模型均在动作前算出预测。B下一8实测187.700/186.936ms，而比例模型预测150.800/150.863ms；同一模型预测B下一32只高估6.945/4.945ms。不能用统一文档比例将首chunk成本外推到全部后续chunk大小。专家counter当前仅为已完成步骤的可见诊断，尚未形成有增量的online selector。

模型增加两点：`T(chunk, current state)`不应预设为`static_curve(chunk) × one_document_scale`；current state需保留prompt offset及旧decode进度。恢复cap16不会恢复后续chunk切分边界、pager状态或batch，因此一次动作的完整后果须继续真实执行，不能只按少/多一次call估算。证据仍为MEASUREMENT_ONLY；本轮baseline为共同first16后的next16，尚未证明优于同工作负载下全程static8/16/32的强简单包络，无SLO/质量或method GO。

下一唯一实验已冻结并启动：[one8_static_holdout/config.json](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/one8_static_holdout/config.json)。按源顺序选择两个未参与本paging链动作选择的incoming文档10/11，保留旧pair4/5；比较one8、全程static8/16/32和已知更强的phase8，两个随机block共20格。主比较固定为wall—旧maxITL—新TTFT；全体从同一pre-injection状态各自推进，仅one8校验同文档after-first16状态并独自承担额外snapshot成本。不得根据新文档post-action counters选择策略。该轮关闭“是否超过强简单配置”的缺口；若有可重复residual，下一步直接连续多次到达，不再扩三请求阈值扫描；若无，则停止这个固定位置规则，保留问题与此前因果发现。最终完成状态待本轮回读。


## 固定规则的独立文档与强基线结果（2026-09-13，20/20 COMPLETE）

[one8_static_holdout/summary.json](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/one8_static_holdout/summary.json) 与readback_completion.json为当前状态。20格/60请求/800输出全部完成，资源、同engine、共同入场前态、实际动作和计数检查通过；worker59241已退出。本轮one8与static16在两文档/两block的全部请求输出相同；static8/32/phase8的第二个旧请求输出与二者不同，策略内重复相同。跨这些其它基线只比较各自自然完整执行，不主张同token轨迹加速或质量等价。

| one8相对static16 | c10，两个block | c11，两个block |
|---|---:|---:|
| wall | −5.326/−6.576ms | +74.919/+45.736ms |
| 旧最大ITL | −20.387/−22.898ms | +24.028/+23.214ms |
| 新TTFT | +45.239/+40.588ms | +104.592/+92.192ms |
| 完整专家payload | −0.66796875GiB | +2.203125GiB |

c11被static16在预定三指标全部覆盖，两次方向一致；one8自身wall重复漂移约29.35ms也保留，不扣除或筛掉首格。c10仍是旧间隔改善/新TTFT变差的权衡，相对phase8 wall还慢18.081/12.127ms、旧maxITL高118.543/114.894ms，但新TTFT早100.022/105.242ms。它未形成一致优于强简单配置的服务收益。

裁决仅为 **STOP_FIXED_ONE8_RULE_IN_TESTED_SHORT_PAGED_REGIME / MEASUREMENT_ONLY**：停止一律first16-next8-then16和基于这几个文档构造的内容阈值selector。不是prefill/paging问题死亡，也不是候选动作上界。恢复条件须来自新的可执行信息、状态/动作关系或执行底座变化，不能只换偏移/文档/阈值继续抢救。

模型保留已测的非单调后果：source9首chunk约994 loads/264ms，next8收益为正；fresh source11约980 loads/262ms，却三项皆负。近似相等的首chunk标量成本不足以直接确定下一动作的完整后果；这不证明任何专家ID特征无用，也不构成新预测器的成功。

下一唯一动作先核对并复用共享台账中已经实测的expert-group执行底座，避免在较弱的token-group底座上继续优化scheduler。另一执行线已有retention/group-budget实验，本线不重复其raw分析或并行实现同一controller；先确认版本、资源、调用路径与最小复用障碍。

CPU复用核查已完成：已实跑入口是[retention_lifecycle/source/run_native_pager.py](outputs/admission_capacity/20260912_wisp_olmoe_r01/retention_lifecycle_performance/source/run_native_pager.py)的`--execution expert`，调用同目录`wisp_expert_groups.py::install_expert_groups(runtime)`。它在当前层router完成后按expert分组、先驻留后缺失，每个missing expert加载一次；本线batched_expert_map只批量写map，仍沿用token分组。另一线实际为vLLM0.26/cap24/KV1GiB/token160/source[0,1,2]，本线为0.11.2/cap16/KV0.5GiB/token64/source[4,5,10或11]；不能由约70GB与150GiB直接推性能优势。

下一可执行接续优先沿用现有0.26的expert/none/early runner，只开放同一输入选择与资源配置，再做同预算token/expert执行对照并分别真实演进状态；不移植新controller或重复group-budget试验。若以后移植到0.11.2，必须适配router后循环、每组masked expert map与独立partial output累加，不能直接复用完整resident map而重复贡献；跨实现BF16分组累加不承诺位一致。当前该统一配置对照仍UNRUN，当前所有本线GPU worker与SSH会话均已退出。


## 2026-09-13：同预算专家执行基线已确立

[execution_baseline_v026/REPORT.md](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/execution_baseline_v026/REPORT.md) 与summary为本轮实测。统一0.26/cap16/scratch3GiB/实际KV512MiB/token64/maxseq3/source[4,5,10]，相同batched map与三段预热，四格token/expert/expert/token从共同token前缀在入场时切换。12测量请求/160输出完成，另40预热请求/488输出保留。四格完整输出相同；共同逻辑前状态、记录前缀及分配相同，物理KV块跨臂不同，KV内容未做位比较。

expert相对token：wall3.30749/3.30851→2.44242/2.44984s（−26.15%/−25.95%），旧maxITL306.16/305.75→152.76/154.05ms，新TTFT2.07861/2.08279→1.13751/1.15167s。完整payload155.84766→101.58984GiB，groups1626→1032，两个repeat计数相同。首mixed实际加载1157→462，groups130→48；在线观测成本保留在完整时钟中。完整allocator峰值均4,735,429,120 bytes，但expert仍有部分输出和masked map开销；相同整段峰值不等于内部工作区无增量。

这是已有执行组织的强基线结果，不能归为本线scheduler贡献。状态MEASUREMENT_ONLY，未测生产SLO、质量、真实超显存配置或独立到达过程。两个同臂repeat漂移不作为噪声界。成本模型应写T(chunk,state,executor)，原token执行的动作曲线/文档响应需在expert底座重新资格化。 首mixed跨臂active union仅12/16层相同，尽管最终输出相同；每臂仍须沿自己的router/KV独立推进，不能固定未来route计算执行收益。固定one8规则保持停止；唯一下一实验重测expert底座static8/16/32曲面，确定剩余prefill权衡，当前UNRUN。worker60757已退出，原始readback保留；研究问题OPEN。


## 2026-09-13：专家执行后的prefill权衡与decode阶段转移

[expert_chunk_curve_v026/REPORT.md](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/expert_chunk_curve_v026/REPORT.md)六格完成，18次请求/240输出，跨动作全部输出相同且共同前缀/逻辑态/分配一致。chunk8/16/32 wall约2.92/2.44/2.56–2.66s，旧maxITL约127/154/211–226ms，新TTFT约1.77/1.14/0.83–0.87s。16是本校准最佳wall，8和32各有不同延迟优势；不存在单点全面占优，无method GO。

32比16少搬约12.46GiB且少115组，仍更慢完成。互斥阶段账表明：prefill节省317/275ms，三请求decode从4→7次却增加403/453ms；另有旧/新单独decode及call外gap变化。共同前缀慢33/34ms发生在动作前，不归因于chunk。32的100.9ms重复漂移保留，少量GC重叠没有被扣除或包装成已定位全部原因。

模型增加prefill完成→decode集合加入的状态转移：总bytes和首chunk成本均不足以排序完整后果。下一最小因果动作是纯decode三请求时轮转每步两条，利用C16/k8保证当前每层并集≤16、expert none单组；这只是结构条件，更多step及所有请求等待可能抵消。四臂比较rr32/static32/static16/static8，动作改变decode cadence并保留KV，不能伪装成只改prefill。代码准备中，GPU UNRUN；固定one8仍停止，研究问题OPEN。


## 2026-09-13：decode轮转的单组条件与完整代价

[decode_width_rr_v026/REPORT.md](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/decode_width_rr_v026/REPORT.md)八格完成：24测量请求/320输出，另152预热请求/1936输出。相同前缀、逻辑前态与分配；RR每格11次激活均每层单组、held KV不变且下一非空步服务。完整调用16→20，groups917→874，payload89.13281→90.67969GiB，说明单层少分组不等于全轨迹少搬运或少等待。RR相对匹配static32 wall读数高4.25%/28.36%，新maxITL高38.54%/103.32%；一条旧请求输出跨臂不同，不称固定token轨迹或质量等价。

RR自身wall漂移505.74ms，TTFT与混合prefill差发生在首次RR动作之前，不归于该动作。实际阶段会计显示7次三请求decode变成11次两行轮转；字节16.04297→18.53906GiB，阶段时间482.66/475.91→571.68/848.52ms。各臂独立执行，阶段差不是固定future-route反事实。没有把重复最大漂移设成噪声界或声称稳定负效应大小。停止当前RR2作为有益机制推进，保留结构与成本发现，问题仍OPEN。

下一唯一实验补强简单准入基线：三请求同到达、同工作量的immediate32/admit2_32/admit2_release64六格。第三条在旧4token时照常入WAITING并从该时刻计TTFT，cap2只等名额；两个旧请求本例同时完成，R再将prefill限制释放到实际budget64。旧hold少跑一请求，不能替代此对照。该六格代码准备中，GPU UNRUN。

真实超显存模型的只读准备：官方[Qwen3-30B-A3B权重索引](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/model.safetensors.index.json)共61,064,245,248B（56.87GiB），现有完整snapshot存储不足。vLLM0.26默认loader先完整下载，但[注册接口](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/model_executor/model_loader/__init__.py)与Qwen的Iterable加载支持单分片消费实现。WiSP逐层CPU→pinned替换，不强制双份完整expert master；按模型配置推算master54GiB、每层转换额外1.125GiB。逐片清理必须处理真实blob、mmap引用与页缓存，现有runner还硬编码OLMoE层数并强制离线。加载器尚未实现，92GiB cgroup下真实峰值未测；没有下载、扩容或删除现有模型。此为后续代表性验证准备，不是新并行机制。


## Qwen真实超显存验证的当前接续（2026-09-13）

[原资格化r03报告](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_native_qualification_v026/attempt03/REPORT.md)已证明16分片全SHA及源/目标参数覆盖、4请求32输出与原生KV块取整会计闭合；48层参考有限，但layer47保留数值诊断差异。因此代表性模型的加载和请求路径已经实测，性能与质量不能从该带参考/JIT的资格化计时推导。此前0.26普通准入/phase32和固定one8/RR2的局部停止结论保留，不代表问题死亡。

旧机数值定位与条件静态对照在第9片后失联，远端状态未知。用户提供新RTX5090后，在[新机独立目录](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_new_gpu_20260913/README.md)恢复同一冻结运行：先对layer47实际partial与同输入同分组参考进行逐位归因，通过才做static32/16/16/32。相同模型、17个runtime源码和输入，cgroup90GiB/新CPU与GPU UUID另记，不跨机器合并计时。新机r01完成2/16分片后，第3片带宽降到3–4MB/s，7200秒总启动时限不足；已仅停止本worker并完整回读，属于行政中断，0数值/性能结果。相同实验的r02只把总启动时限改为21600秒并换独立路径，已重新通过空闲检查并开始加载。研究问题仍OPEN，当前唯一实验证据缺口仍是该真实超显存底座上的数值归因与完整请求prefill权衡，不实现新controller。


## 返回 weste:23478 后的执行接续

用户重新指定 weste:23478。只读核对及稳定回读已确认旧 r01 的 worker89182 不存在，原目录没有生产进程；日志停在第9片，8片全SHA/9,889源张量，0数值结果、0性能格。原 launch 的 RUNNING 原样保留，外部 reconciliation 标为 INTERRUPTED_DURING_LOADING_NO_TERMINAL_RECEIPT；退出原因未知。旧回读48文件、4,568,801B，SHA f401766874b1e5888b105742c713b4b0c40c95ca3389d0e3fe199eb406870474。

此时 vLLM/Torch/WiSP 与11个安装源码SHA均匹配原冻结环境，GPU UUID0a66cc34…/32607MiB、cgroup92GiB。新 attempt02 包7f54419f…保留17运行源码、模型与请求输入；总行政时限21600s、每episode600s。临时分片独占系统盘 `/root/qwen3-localized-static-v026-r02-shard-workspace`，结果仍在数据盘独立r02目录；两盘启动前各>6GiB，不删除旧分片或其它会话数据。monitor13262/parent13263/worker13275已实际存活，三次空闲检查通过，目前第1/16片加载，数值与性能仍未测。记录在 [return_host_20260913](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913/README.md)。

**观察状态补记（2026-09-13）：returned weste r02 当前远端 worker/结果为 UNKNOWN。** [observer 中断记录](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913/attempt02/observation_interruption01.json)确认 session34278 已 rc255；[最后实际观测](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913/attempt02/watch_attempt01.jsonl)为 UTC 11:21:00.364910（北京时间19:21:00.364910），当时 parent13263/worker13275 存活，第7片为1,806,696,448/3,999,975,472B，qualification为空、结果仍INITIALIZING。这些是最后观测，不是当前存活确认。随后两次只读重连均255；[握手诊断](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913/attempt02/handshake_diagnostic02.json)显示 TCP 已建立，但 SSH 认证前收到 `HTTP/1.1 502 Bad Gateway`。本地只读路由核对显示目标116.172.94.204经utun4；仅记录路径，尚未定位502产生端或断连根因，不能推断机器关机、worker退出或实验终止。原raw/协议未改，无启动或停止操作。

UTC11:42:20.570的[控制台只读观察](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913/attempt02/browser_console_observation01.json)显示当前可见5条AutoDL实例均已关机；F04（ddbfwkg1fy-f70e677e）的RTX5090/Gold6459C/92GB与系统30GB、数据50GB配置相符，但页面没有SSH端口，targetSshMappingVerified=false，尚未核验其就是weste:23478。因此“控制台5实例全off”与“目标远端终态回执UNKNOWN”分列；无现成live网页终端，未开机或付费，SSH502/utun4不构成VPN根因结论。

**执行接续状态（2026-09-13 UTC11:49）：`BLOCKED_RESOURCE_ACCESS`，研究目标未完成。** 连续三轮同一接入阻塞；最新 `reconnect_after_observer04.json` 仍为认证前连接关闭、rc255，控制台刷新仍5条全关机。没有本轮数值资格或性能终态可取回，未将本地准备或中断记为科学负结果。F04实例与weste:23478的身份映射仍待确认；无卡开机入口已找到但未执行，以避免操作其他会话实例。恢复后先检查原r02进程与终态并回收原始记录，再决定下一次GPU执行；已准备的读取工具要求producer停止及有效终态回执，不能以失联替代这两个条件。阻塞复查保留在 `return_host_20260913/attempt02/continuation_events.jsonl`。

西侧 westc 的 r02 最后可见 worker5127 仍在加载第5片，4片完整；随后观察连接255。其当前远端状态未知，不记停止或失败，不合并两机结果。CPU压缩缓存只是本地准备，未确认远端部署。本轮唯一科学动作仍是冻结的layer47归因资格→通过后static32/16/16/32；不从换机或加载完成推导方法成立。


Qwen 加载等待期间已完成下一阶段的[16篇候选输入](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_continuous_inputs_v026/input_checks.json)：复用本地固定WikiText Arrow与文章生成器，排除已核对的原32篇及rotation64篇，选row12128至13079范围内首16篇满足长度的文章。每篇保留原文、模型/tokenizer身份及128/256/512三档精确前缀；16篇的候选版本不视为48个独立样本，也不声称全仓库未使用。CPU重新生成的token前缀与旧8条Qwen输入一致。输入SHA49cd8abe…，95行准备代码；无GPU执行或arrival/策略/次序冻结。按maxseq8、每条512+64、BF16 KV/block16静态估计288请求块/432MiB，低于340可用块；不包含新shape工作区资格，亦不证明动态运行或服务收益。当前唯一GPU实验仍为已启动的原layer47资格与静态ABBA。


后续减少重复加载的源码核对：当前[Qwen runner](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/source/run_native_pager.py)本身是一份engine/CPU master上的预声明episode循环，最后主动shutdown；本轮没有追加任务接口，不在运行中修改。已有WiSP `WispMoEState.resize_cap` / `resize_layer_caps` 可在drain及CUDA同步后调整各层GPU scratch；下一版还必须同步更新adapter的 `runtime.cap`（expert分组确实读取它）与资源会计，不能重新install或只改一个变量。resize不复制CPU master，但cap48→56的既有实现会复制全部旧scratch，即48×48×9,437,184B=20.25GiB D2D tensor-copy payload；不是H2D/PCIe字节，也不另乘读写2。scratch最终20.25→23.625GiB，还有旧/新缓冲并存的瞬时峰值。源码可行性不等于实际Qwen热调资格；待当前无reference性能范围的峰值后才选一个可行静态资源挑战点。下一版可预声明多个episode以摊掉加载，不制造第二份54GiB CPU master；新并发上限须在engine初始化前准备，不把本轮maxseq4当作已支持8。完整session仍计加载、调整、同步、预热、reset、capture、flush和shutdown。
