当前主张：OLMoE BF16/在线FP8均已完成七B九臂性能矩阵，小到中batch存在实测TPOT/能耗改善；幅度随格式改变，任务质量代价与独立重复仍须评估。
关键证据：两格式各63/63格；B8 high的TPOT/J在BF16低16.14%/20.41%、FP8低8.25%/13.73%；十臂free200/任务pilot完成，但GSM16题mid/high各4题、native6题，不能称同质量加速。
最强替代解释：每格式仅一次运行，host背景/时间漂移未由独立重复隔离；量化可改变生成内容与路由，跨格式不是固定轨迹干预；KL/PPL不等同任务准确率。
下一动作：SSH恢复后先核实原DeepSeek BF16→FP8链，再继续两模型、全量任务/滚动PPL和必要独立重复；Qwen准备串行复用已完成DS工作的临时RAM。FP8完整结果已纳入14页论文。
资源状态：PRO6000/vLLM0.26.0；已确认结束作业累计持卡11003.413秒（3.057小时/约15小时）。最后观测epoch1792115499.654：DS controller73768实活WAITING、桥70776等待；此后SSH50328拒绝连接，后续状态及新增持卡时长未知。Qwen失败后已仅回收K克隆恢复持久盘约49.91GB；C未改。无5090。

# 当前任务

按 [TASK.txt](TASK.txt) 持续建设 `code/`、`configs/`、`runs/`、`figures/`、`paper/` 和 `reproduce.md`。当前接入服务器为 westb:50328，GPU UUID `GPU-12433ef7-c83f-e383-fb23-85550b63b957`。用户已确认没有第二台 5090；设备范围按现有 PRO 6000 推进 BF16/FP8 主线，5090复测标记资源未提供。

下文按实验推进保留阶段记录；其中旧“当前/最新”、PID和未运行说明属于各自记录时点。最新成果以顶部五行及“论文与最近邻”为准，远端状态以文末最后成功观测为准。

## 实现注意点

vLLM 0.26 的真实后路由接口为 `FusedMoERouter.select_experts`；`MoERunner` 持有 router 和 routed_experts。原生 `moe_align_block_size` 描述的 `-1` 主要是 expert_map 映射后的块编号。不能由该描述直接推断负 topk_ids 可送入原生 histogram；本实现将提供安全对齐路径，并验证被跳槽清零与真实 GEMM 跳过。

单步字节分开保存：(1) 依据实际路由计数及实际每专家 tensor 大小得到的 distinct 权重足迹；(2) 如可用，硬件 DRAM 字节。前者是实测路由的逻辑字节，不冒充硬件读取流量。独占槽解析模型只用于解释和对照，不替代计数。

## 数据与复用

当前本地历史 `capture_routes_v2.py` 捕获整段文本位置，不自动等价于多会话自然 decode batch。F1 将区分位置回放、概率模型与真实 decode 实测。

`code/power.py` 复用同工作区 C 课题已有 PowerSampler，保留其 July 原始出处；`code/logits_capture.py` 复用 B 课题的完整分布 KL 捕获器。仅复制到本目录，不改原件。PowerSampler 同时保留 pynvml 功率积分和累积能量计数器；正式短窗测量前进行持续负载预热，避免已知功率平均窗口滞后。

## 首批真机结果（2026-10-16）

- `initial_mechanism_r01/mechanism.json`：15/15机制点通过，B=1/2/8/32/64，global/gate/marginal。相对独立 FP32 专家参考 L2 为0.00364–0.00418；将所有未使用专家权重置NaN后，15点输出均有限且与原始输出完全一致。该结果配合原生kernel的 `off_experts == -1` 早退/清零源码，支持本接线安全跳过专家计算；不是硬件DRAM计数或实际服务收益。
- 首组完整模型捕获在两请求跨arm分组不一致时中断；当前效果是KL配对测量失败，下一步改用单prompt身份检查，并以request/position配对新质量运行器。原始失败保留。
- `olmoe_model_smoke_r02/model_smoke.json`：完整OLMoE原生与dense身份臂的输出和full-logit KL检查通过，16层在B=1/8短decode都执行策略。B=8、7个decode步的marginal诊断累计跳过453槽、移除453个逐层逐步distinct专家出现；同组global4跳过3520槽、移除1084个distinct出现。不同策略生成轨迹已不同，这只是动作/计数诊断，不能由该对比计算性能或质量优势。
- 累计GPU持锁43.874+34.353=78.227秒。`fused_policy_r01` 未获锁，未运行GPU；接续组另用新目录。

核心实现现有torch参考和单CTA Triton小batch路径、设备计数器、安全负ID对齐，以及可开关worker。原生减k强基线、融合kernel、质量与性能采集的接续版本已提交验证，尚未报告结果。

`figures/native_smoke/` 从保留的dense原生路由矩阵抽取实际batch，保存来源SHA256及逐层CSV。请求B8的短运行中实际出现B7/B8，不能全部标成B8；仅实际B8的6,144槽中独占11.442%，拟合边际频率后iid模型为16.196%，均匀iid为39.270%。这是相似短提示的单次诊断，不区分相关性与有限样本误差，不支持代表性负载或服务收益。

`configs/quality_pareto.json` 与性能默认九臂共享参数和原生减k执行，另增dense身份/计数臂。质量的字节分母用同group dense路由，原生减k的before已经是减k后的路由，不能充当原top8反事实。`configs/drop_rank_gate.json` 单rank×gate区间干预已备好；r06已完成这组探索性GPU测量。GraphWorker固定进程策略、使用GPU真实token标量排除padding决策/计数；保留padding路由意味着逻辑独占不一定等于物理独占，主powers-of-two矩阵需确认无padding。

## 论文与最近邻

`paper/main.pdf` 当前为14页阶段稿，含八节、BF16/FP8长轨迹F1、r06真实T1、两格式三B F3、两格式完整F2/F4、F6 CPU/GPU开销与动作表T2，以及r08b真实生成、任务pilot与GSM逐题配对。最新PDF SHA256为`f4710351712584f4b09ae73ebf84ee8d2233dd0ad0d9aa2f90c6da6c012f81b3`，编译成功、14页逐页渲染目视核验通过；F2/F4同在第11页，保留所有策略曲线与不利值。此版论文及新增FP8图表目前仅在本地，SSH拒绝连接后尚未同步远端。全量任务、独立性能重复和另外两模型仍待完成。AdapMoE包含Fisher层敏感度及缓存/预取，纯gate基线仅为组件式对照；SERE已研究batch expert-union与替代路由，具体区别应落在独占成本约束的直接drop和质量预算，不主张首次batch-aware专家跳过。原始来源及版本状态在 `paper/literature_notes.txt`。

## 当前可运行交付与评测准备

- `dist/marginal_expert_skip-0.1.3-py3-none-any.whl`：最新可安装包包含七个CLI及单源配置；干净venv安装与真实rolling输入读取检查通过，记录在 `runs/package_install_r04/verification.json`，新rolling/FP8 tuner GPU执行仍待完成。旧版本与验证记录保留。
- `bench_graph.py`：每臂每重复独立进程，性能/能耗关闭路由计数，另用同策略计数进程；逐格核对输出token哈希、所有层/步/token计数以及FULL且无padding。r06九格短pilot和r07完整63格长窗矩阵均完成上述检查；r07仍只有每臂一次独立性能运行。
- `eval_tasks.py` / `free_generate.py`：四任务官方固定版本数据已在远端 `datasets/task_eval_r02` 备齐，333MB，60个子任务离线fingerprint验证通过；标准多选loglikelihood是prefill身份对照，另标生成式多选适配；自然生成默认200条并保留EOS/达到上限/失败及重复指标。尚无任务准确率或自由生成结果。
- 2026年3月31日发布的CCF第七版正式目录已下载核对：ICPADS/HPCC/ICA3PP为C，ICPP/CLUSTER为B；全目录未找到IISWC及其全名，不将其列为已确认CCF-C目标。原件与来源哈希在 `paper/sources/CCF_catalogue_2026.{pdf,json}`。这只修正投稿范围，不改变研究路线。

## 融合版本接续（r04）

`runs/fused_quality_graph_r04` 原始文件全部SHA256校验一致。融合策略CUDA单测、原生减k、真实decode教师强制B1/B8均通过；dense身份臂full-logit KL为0。当前效果是性能采集在清零inference tensor时异常，下一步在正确inference上下文中原地清零，保留图捕获地址并接续性能与graph检查。尚不能以这个短质量样本评价模型质量。

离线SM120编译96/96组合、160/160独立产物通过，范围B1/8/32/64、E/k=64/8、64/6、128/8，marginal/rank_gate及统计/活动行开关；RENORM=False、router fp32。`runs/offline_compile_r01/summary.json`保留源码及产物哈希，CUDA驱动未初始化。这只证明可编译。

`figures/f1_native_r04/` 使用不同文章前缀的无padding原生cohort，B1为512/512独占、B8为1271/4096（31.0303%），各4步；B8样本内边际模型25.4534%、均匀模型39.2696%。该短trace与旧相似提示诊断负载不同，不合并成一条普适曲线。

`figures/results_r04_microbench/` 的融合策略+计数器 captured device 时间在B1/8/32/64为2.457/2.891/4.714/6.551微秒；eager wall为21.473/21.129/25.774/24.068微秒，包含host/launch/device，不能相减估CPU开销，也不是服务TPOT。

r04教师强制质量只用B1的4 token和B8的32 token。B8 marginal05平均KL=0.004102684，PPL=9.299030→9.290262，但8篇中5篇NLL上升，首篇约占75%总KL；这是测量接通证据，不支持质量无损或提升。global1是post-drop诊断，正式强基线采用global_native。

FP8源码检查表明普通BF16 checkpoint的在线FP8可走显式TritonExperts+NoDPEP，现worker按真实参数统计权重与scale。预期OLMoE每专家为6,291,456权重字节+8 scale字节=6,291,464；这是源码预期，待真机确认。

新语料 `datasets/wikitext2_serving_r01/manifest.json` 为READY：基于原有WikiText-2 validation/test parquet，保存60/61篇合格文章；探索与留出各选32篇独立文章，token长度均覆盖prefix128+decode32+1。B64性能输入是来自60篇validation文章的64个不同非重叠窗口，64个128-token前缀全部不同；不把64窗口当64独立文章。r05仍候锁时已切换为此数据并恢复七batch检查，来源字段随请求保存。

## r05：七batch短测与接续

`runs/graph_resume_r05` 的198个原始文件SHA一致，26项测试通过（包含真实CUDA policy/graph replay与inference counter reset）。bench完成21格，native/global_j4/marginal_mid，B=1/2/4/8/16/32/64；每格8输出、7 decode，仅单次、原生未调优，marginal计数仍进入热路径。图在 `figures/results_r05_diagnostic/`，F4不输出，因为全部能耗窗只有0.074–0.110秒。

当前效果：marginal_mid在B8/32/64相对native的逻辑字节足迹减少23.43%/8.06%/4.06%，TPOT却增加25.50%/25.37%/23.62%。下一步：原生kernel共同调优，再比较关闭计数的CUDA Graph执行；当前结果不支持完整服务收益，也不否定尚未完成的执行路径。global_j4对应字节减少39.86%/23.92%/14.93%，TPOT变化+3.38%/+0.48%/−2.96%，同受短测限制。

`figures/f1_native_r05/` 另用trace pass：每B 1 prefill+8 decode、每请求9输出，共128 layer-steps；不同于性能pass。64窗口来自60篇WikiText validation文章，不算64独立文档，边际概率为样本内估计。

当前效果：graph检查的eager子进程采用默认多进程提交，出现不完整B3 cohort（224而非240 token-layer），尚未执行到graph。下一步：smoke使用与bench一致的V1单进程同步提交，并要求恰好1 prefill、5 decode、0 mixed。r06把质量/FP8与graph检查分别运行，保留独立结果。

200条自然生成输入已准备在 `datasets/free_generation_validation_r01/prompts.jsonl`：来自28篇未用于quality32的validation文章，整篇排除quality32和所有test；200个128-token前缀唯一。用途为探索性自然终止/长度/重复率，尚未生成，不代表200独立文章。

DeepSeek持久盘续传在新的共享预算检查中被阻止；6,928,990,208字节自己的partial已逐文件复制、SHA核对后迁入 `/dev/shm/research_K_expert_skip/deepseek_v2_lite_604d5664`，只移除已核验的原partial，其他项目未改。RAM文件易失，原始source、迁移记录、代码与结果保存在持久目录；仍保留20GiB目标文件系统余量。Qwen BF16 controller21895/child21898在本轮检查时真实活跃，39.37秒写入330MiB，短时速率不作为下载完成承诺。

## r06：真实质量表、graph与FP8

`runs/characterize_r06` 八阶段全部完成，完整1079文件SHA校验一致；持锁715.924秒，累计1003.347秒。七B原生调优采用同模型首MoE层真实权重、真实native路由和合成BF16激活，stock到selected的核内计时减少0.331–2.355%，只说明本次小网格的选优结果，不作为系统加速。所有BF16策略共用这份配置；逐事件/轮原值见 `figures/native_tuning_r06/summary.json`。

Graph dense/native身份与marginal一致性均通过：实际B3→bucket4、B5→bucket8，各1 prefill+5 decode；eager与graph输出和aggregate计数完全相同，真实token-layer为240/400，padding的80/240行未进入策略计数。正式七B矩阵采用无padding bucket，保留单步运行记录。

`figures/t1_r06/` 为实际16格drop表；每格32独立文章、1024教师强制预测、16384 token-layer。rank使用论文一基5–8，gate按干预前selected mass归一化，OLMoE实际renormalize=False。gate<.025只触发9/21/48/118槽，KL小不能当作普遍安全；[.05,.1)触发6781/12056/14529/14343槽，平均KL为.034685/.053181/.044293/.026772。每格是全模型反复执行该bin的干预，不是一个事件的独立质量成本，不能相加。

`figures/quality_r06/` 为B1/B8/B32共同九臂F3，同32篇validation文章，每篇32个真实decode预测。native与dense每B均相同；跨B的浮点与batch执行差异保留，不合并native PPL。当前效果：B8 marginal_mid的逻辑字节减少21.650%、KL=.045950、PPL增量=.282419；global_j2为17.727%、.073844、.474567，本样本中前者更省字节且较小质量扰动。B1对应mid为26.735%、.077403、.568457，而global_j2为25%、.072588、.468880，存在明确取舍。下一步：在留出文档与任务质量上检查这一差异，并关联实际长窗性能；未宣称任务无损。

FP8真实安装16层全部为TritonExperts + MoEPrepareAndFinalizeNoDPEPModular，renormalize=False；每专家6,291,456权重字节+8 scale字节，合计6,291,464。B1/B8短bench及native/dense质量身份通过。该32预测小样本只证明测量接通，不与BF16正式32篇质量表混合。

`figures/graph_pilot_r06/` 为关闭计数器的9格短graph诊断：128前缀、128输出、1次运行、不采能耗；性能与独立计数进程逐格输出哈希完全一致。marginal_high在B1/8/32的TPOT为2.581/6.373/10.550ms，native为2.998/7.731/10.952ms；global_j4为2.417/5.756/9.496ms。这些策略质量不同，不能直接称为同质量性能胜出。当前效果：graph完整执行出现了相对native的短测改善，下一步r07长窗九臂主矩阵与独立重复检验，不把这一次pilot当作确认。

r07首轮63格每臂每B采用2048输出（2047纯decode步），统一总显存比例.35、max_model_len2304；依据pilot最快2.417ms推算约4.95秒能耗窗，每格仍检查实际>=3秒。每臂一个全新性能进程和独立计数进程、随机臂顺序；单轮原值不足以估运行级不确定性，重复和留出属于后续独立运行。具体参数保存在 `configs/olmoe_bf16_matrix_r07.json`。


后续可运行入口已在本地补齐，尚未同步到运行中的r07代码：`tune_moe.py --quantization fp8`直接调用vLLM0.26在线FP8转换与TRITON量化配置，CPU plan/fixture通过，CUDA调优待运行；`smoke_graph.py`已可传格式与每模型显存比例；`run_task_quality.py`按独立任务family运行共享native的固定五臂，并提供显式小样本pilot与完整评估，失败保留而继续后续family。r07全部结束后才同步这些源文件。


r07进度快照（远端epoch1792099126，非最终结果）：已配对7/63格，均为global_j1，性能/独立计数输出身份一致，实测能耗窗6.54–36.52秒；native及其余策略尚未齐，不能据此给相对收益。DeepSeek本课题RAM目录文件累计约19.64GB、PID30370仍写入；Qwen他课题已有6/16完整分片、PID21898仍活跃，只读复用。`configs/task_quality_pilot_r08.json`已准备下一轮十臂200条自然生成及有限任务pilot，尚未提交。


完整WikiText rolling输入已在 `datasets/wikitext2_rolling_olmoe_r01/{validation,test}` 准备：validation排除首128上下文后覆盖252,353计分token，test覆盖288,602；最后65/90 token短窗及不足整批的窗口均保留，无评分区间重叠/遗漏。首计分位置采用127-token prompt+1枚不计分prefill bridge，首预测具有128-token上下文；窗口不是独立文章。当前只完成CPU数据构造与入口检查，PPL/KL未GPU执行，也不声称等同其他上下文长度的标准PPL。新代码须待r07终态后再统一部署。

两模型输入接续已完成：`datasets/wikitext2_rolling_deepseek_v2_lite_604d5664/{validation,test}` 为2,133/2,430窗、272,943/310,997计分token；`datasets/wikitext2_rolling_qwen3_bf16_ad44e777/{validation,test}` 为2,049/2,336窗、262,209/298,950计分token，均完整覆盖首128上下文之后的token并保留短尾窗。两模型的bench64/free200均有64/200个唯一128-token前缀，validation/test quality32全部满足257-token长度；详情在 `datasets/model_input_compatibility_r01.json`。`ref_july/tokenizers/` 下对应两目录的small assets已按官方HF固定revision核实原始字节，`source.json`记录来源及SHA；本地与远端实际分词器版本的semantic identity和输入前缀IDs一致。上述19文件共36,531,983字节已同步至远端K同名目录，逐文件SHA256全部一致；未写C项目或远端code/configs。两模型输入目前仅通过CPU检查，GPU质量/性能尚未运行。


## 可安装交付接续（0.1.3）

`dist/marginal_expert_skip-0.1.3-py3-none-any.whl` 已包含完整rolling输入builder、quality读取分支、BF16/FP8 native tuner和原机制；新增两个CLI，原五个CLI保持。旧wheel不覆盖。干净venv正常安装后7个CLI、配置读取、真实validation/test rolling loader覆盖、分词身份校验和两格式tuning plan均通过；说明在 `runs/package_install_r04/verification.json`。安装验证是CPU范围，不声称新功能已GPU执行。F6新增预热eager循环的主线程CPU消耗 `time.thread_time_ns`，末尾显式同步在CPU区间外，wall和captured-device原指标保留；旧r04不回填新值。

当前完整矩阵继续运行，源文件仍使用提交时版本。0.1.3源码、完整rolling GPU路径、FP8在线转换调优和新增CPU计时将在r07结束后进入下一持锁组，不覆盖运行中的live code。


DeepSeek-V2-Lite已于远端epoch1792100750.357完成四分片下载：最终`runs/deepseek_mirror_20261016_045110_30370/source.json`为WEIGHTS_COMPLETE，4/4权重SHA与原HF固定revision匹配，总31,413,626,576字节；无缺失/大小异常小文件，PID30370已退出。checkpoint位于易失RAM路径`/dev/shm/research_K_expert_skip/deepseek_v2_lite_604d5664`，终态receipt已取回本地且SHA一致。此状态说明输入与权重就绪，模型加载/机制/GPU质量性能尚未运行。


Qwen状态更正（远端epoch1792101049只读核实）：controller21895仍在等待，但child21898实际为T(stopped)，不得再把PID存在或旧status.json RUNNING当作正在下载。C自有`runs/qwen_bf16_download_resume_r02/integration_priority_pause.json`记载1792099084.621发SIGSTOP，以为用户优先的OLMoE完整模型集成保留磁盘余量和I/O，未删文件。当前6/16完整分片、总45,229,031,697字节；K未修改/恢复C进程，Qwen GPU工作等待owner恢复并完成下载，同时继续自身OLMoE/DeepSeek。


r07部分长窗结果（42/63格，六臂已齐；整体运行未结束）：已完成格均通过performance/count输出哈希身份、有效能耗窗>=3秒。marginal_high在B1/2/4/8/16/32/64相对native的TPOT降幅分别为11.03/16.11/18.15/16.14/8.77/3.97/0.68%，J/token降幅为10.19/18.78/24.03/20.41/10.24/5.33/1.02%。B8绝对TPOT为6.90977 vs 8.24013ms，J/token为0.236021 vs 0.296540；各自轨迹上的每token逻辑权重足迹低29.50%，本策略轨迹内直接skip前后减少27.25%，二者口径不同。B8 gate_high对应TPOT低15.96%、J/token低20.97%，与marginal_high的微小性能差异未经重复不能区分。r06同名marginal_high的32篇教师强制KL=.087068、PPL增加.702444，明确存在质量扰动，不能声称同质量或任务无损加速。当前效果是长窗执行出现原生对照改善，下一步完成全部预设策略和真实任务/自然生成，再选必要独立重复；不将局部结果升格为三模型确认结论。


已设置一次性串行接续：远端`deployments/task_quality_pilot_r08_20261016/after_r07.py`，PID38822，当前WAITING_FOR_PREDECESSOR。它只在r07成功终态、两个进程退出、SHA256.json生成后部署暂存的code/configs，再通过现有共享锁提交`task_quality_pilot_r08`；任何前驱失败/其他K活跃run/同名r08已存在均停止，不自动重试。新GPU任务当前尚未注册，运行中的r07代码未变。r08上限3630秒，内部3600秒，十臂各200条自然生成和四任务每leaf16样本pilot；MMLU为每subject16，非总16。当前后续无需重复提交r08，先核对handoff实际进程/状态。

## r07完整矩阵与接续

`runs/olmoe_bf16_matrix_r07`已COMPLETE，63/63格全部完成，原controller34549/worker34802已退出；持卡4118.825秒，连同此前累计5122.172秒。完整588文件原SHA匹配，63格performance/count输出哈希均一致；共2,339,721纯decode输出token，能耗窗5.610–37.143秒，全部>=3秒。`figures/results_r07`已生成F2/F4及63行`bench_points.csv`、7行全策略`bench_by_batch.csv`，图面已核对。

当前效果：marginal_mid在B1/2/4/8/16/32/64的TPOT降幅为7.09/9.51/11.92/11.26/7.18/1.69/−0.15%，J/token降幅为10.70/9.35/14.49/14.87/8.46/2.62/0.03%；更保守marginal_low的TPOT降幅为−1.36%至+0.79%，单次数据不能辨认这一级别的效应。B8 global_j2的TPOT降12.30%、J/token降14.40%，略快于mid；B8 global_j4分别降22.98%/26.43%，但r06相同32篇协议的KL=.290424、PPL增2.215465，大于mid的KL=.045950、PPL增.282419。这里仅比较样本原值。下一步以r08自然生成/任务评价检查实际质量代价，并用匹配tail的简单gate控制及独立重复细化归因；不把当前全部策略的单次点作为确认级Pareto。

串行handoff已于epoch1792102683完成新源码部署并提交r08；其一小时候锁到期未获锁，epoch1792106283终态LOCK_BUSY，controller40103及原交接器38822均已退出，无worker和GPU结果。原状态及SHA完整回传，不将排队超时算作实验负结果。完整rolling、新FP8 tuner及F6 CPU计时源码现已部署，但相应新GPU路径仍未因此获得验证。新增`configs/matched_gate_controls.json`已随r09交接部署，尚未运行；其tail+gate控制用于检查简单尾部限制能否解释已有质量差异，现有rank_gate上界为严格小于，不能宣称逐事件只改count的严格消融。

三模型任务入口已在本地补齐：`run_task_quality.py`接受官方实际`model_type`为`olmoe/qwen3_moe/deepseek_v2`的未量化checkpoint，继续用同份权重执行BF16/online FP8；三份真实config的两格式CPU计划通过，原策略、任务、锁和超时逻辑未变。`code/task_quality_models_commands.txt`记录独立持锁命令。这批修改已随r09交接部署；不能据此声称DeepSeek/Qwen任务或GPU加载已经通过。DeepSeek routed k=6，固定tail_rank=4仅覆盖末两槽，与OLMoE/Qwen的末四槽范围不同。

F5跨模型绘图入口`code/plot_cross_model.py`已在本地完成，接受各模型/格式的真实`bench_points.csv`，按运行及条件分面，保留各自资源与原生调优设置，不将模型差异只归因于E/k。`figures/cross_model_r07/`用现有OLMoE BF16的21个实测点验证入口，明确标为`ONE_MODEL_PARTIAL`，另五个模型×格式缺失；图面已核对。它未加入论文，不充当三模型证据；新脚本已随r09交接部署。

## FP8完整接续r09

`code/run_olmoe_fp8_r09.py`与`configs/olmoe_fp8_matrix_r09.json`已完成本地及远端CPU命令计划检查：七B真实FP8 trace、FP8专用原生调优、dense/marginal graph一致性、F6主线程CPU/GPU计时、rolling九窗路径检查、三B各十臂质量表，以及完整63格graph性能/能耗矩阵。比较沿用r07的同checkpoint、P128/N2048、maxlen2304、显存比例.35和九臂参数；online FP8为MoE W8A8，外层model dtype/auto KV为BF16，不能把结果变化只归因于权重压缩。

远端暂存目录`deployments/olmoe_fp8_matrix_r09_20261016`含9项源码/配置，payload SHA256=`c4406348b5939e73ad10b8a98941d4c01ea13a4f6dd26c38dd47e3278baf085c`。原CPU交接器45387因r08 LOCK_BUSY按约定STOPPED并退出，原记录保留。新`after_r08b.py`/PID52745已在r08b COMPLETE、原进程退出且SHA终态存在后部署这9项文件，并于epoch1792108315提交唯一r09请求；`handoff_r02.json`记录SUCCESSOR_SUBMITTED/controller56295。epoch1792108439实查controller存活且WAITING，无worker，候锁上限21600秒。外层持卡上限7200秒、内部7170秒，计入既有15小时预算；依赖失败不冒跑性能矩阵，独立质量失败保留并继续其他可运行阶段，任何缺失阶段不报整体完成。r09已注册不等同已执行GPU，当前live code不得覆盖。

r08真实任务输入的CPU检查已完成，`datasets/task_eval_r02/context_lengths_olmoe_pilot.json`保留实际构造方式和输入身份。相同seed1701、每leaf16样本、MMLU/GSM8K五样本提示下，MMLU为912篇/3648请求，最长3056 token（阈4095）；GSM8K为16篇，最长1081+输出cap256=1337（阈4096）；HellaSwag/ARC-c各16篇64请求，最长62/84。四项超限均0，因此不改变r08上下文或提前拒绝规则。这里只验证本次pilot输入，不是任务质量结果，也不证明完整任务人口均满足长度限制。

共享资源接续：`run_locked.py`现支持0–21600秒等待，遵循本目录AGENTS的非阻塞取锁约定，每次均`LOCK_EX|LOCK_NB`，忙时有限轮询；GPU运行上限和15小时预算未变。CPU实进程的立即busy、占用超时、释放后重试获锁均通过，全部flags为LOCK_NB。确认r08和两个旧交接进程均退出后，已部署源码SHA256=`78187d2f9c51197b59e5d2d3e751f8f54e3e0024abb4cdc2d690b469a4c09ecf`；阻塞/alarm版本从未部署。唯一新请求`task_quality_pilot_r08b`于epoch1792106455提交，controller52744，候锁21600秒、持卡3630秒、内部3600秒；数据、种子、十臂、显存和任务协议均未改，仅输出目录改名。准确命令与环境在`deployments/task_quality_pilot_r08_20261016/r08b_launch.json`。当前代码仍不得覆盖，新增结果工具只能本地构建或另行暂存。

DeepSeek完整接续入口已补齐：`code/run_deepseek_matrix.py`和`configs/deepseek_matrix.json`分别接受BF16或FP8，执行七B原生trace、layer1对应格式调优、真实padding graph一致性、三B十臂质量和完整63格性能/能耗。保留E64/k6/I1408、26个routed层和两个shared专家不跳过的模型语义；两格式均用.55总显存比例，各自独立调优。外层7200/内部7170秒只是资源上限，首次DeepSeek GPU加载、MLA与实际完成时间仍未测。已暂存于`deployments/deepseek_matrix_20261016`，本地及远端实际config的两格式CPU计划通过，尚未部署live或提交GPU。

新增`code/summarize_task_quality.py`直接将套件原始JSON汇成任务/leaf CSV、自然生成逐窗口与汇总CSV、GSM8K表和独立MC身份表；保留格式、种子、实际人口、缺失NA、终止/失败及同prompt配对差值，不把sample stderr当运行重复。小型解析样例及真实部分快照检查通过后，已用r08b完整终态生成`figures/task_quality_r08b`：70行task metric、1230行leaf原值、10臂free汇总、2000条逐窗口记录和三个LaTeX表。200窗口来自28篇文章，重复指标不等同正确率。该脚本也已暂存于DeepSeek部署目录，未覆盖运行代码。

## r08b已完成的真实生成质量

整轮于epoch1792108303.901 COMPLETE，controller52744/worker52785退出，持卡1827.721秒；原始5152文件共235,814,114字节已回传，逐SHA与远端终态一致，manifest SHA256=`636ce6c9ccf95bee3b4179b3c9cddf0a557835f1162c1a8d8242ff83e9f16f9f`。所有五个family及各自十臂均完成，以下任务分数仍是pilot，不是全量任务表。

自然生成family已COMPLETE，耗时876.985秒，十臂各200条共2000条记录已回传。全部臂fail=0；native/dense的EOS8、cap192、平均输出251.63 token，dense全部200条与native逐token一致。平均重复token四元组比例native=.263055、mid=.300884、high=.247570；按预先代码口径len>=32且比例>.5触发的检查标记分别39/45/27条。mid EOS6/cap194，high EOS5/cap195；全策略94.5%–97.5%输出达到cap256，不能把执行完成等同自然完成，更不能用更低重复率证明正确率更高。gate_high有一条不足4token输出，其重复四元组均值分母199，其余臂200。200窗口来自28篇validation文章，不是200个独立文章或系统运行。

GSM8K官方5-shot pilot也已COMPLETE，耗时134.198秒，每臂同16题、seed1701。strict-match/flexible-extract分别：native和dense为6/16、6/16；global_j1为5/16、5/16；global_j2和j4各1/16、2/16；gate_low为6/16、6/16；gate_high为5/16、5/16；marginal_low为6/16、6/16；mid和high各4/16、4/16。全臂fail=0，j4有1条cap，其余cap=0，EOS与任务stop分别保留。当前效果：mid/high比native少答对2题，mid在续写筛查中也有更多重复；下一步全量GSM和匹配tail gate控制细化质量取舍，并继续FP8/DeepSeek实测。16题pilot不能估计1pp级质量差异、证明显著性或宣称精度无损。

三项MC标准loglikelihood仅覆盖未干预prefill，全十臂对应分数完全相同，最大漂移0pp：MMLU每57个subject各16题、共912题，合计502/912=.550439；HellaSwag16题acc=.5、acc_norm=.625；ARC-c16题acc=.5、acc_norm=.4375。耗时分别630.902/96.827/88.633秒。它们确认本次标准评估中的身份行为，不是decode任务质量无损证据。

为全量任务执行完成了对应CPU输入准备：`datasets/task_eval_r02/context_lengths_olmoe_full.json`覆盖26,575文档、102,342个实际harness请求，全部0超限。MMLU14,042/56,168请求最大3166；GSM1,319/1,319最大1390+cap256=1646；HellaSwag10,042/40,168最大148；ARC-c1,172/4,687最大171。保持既有seed1701、官方few-shot和tokenizer，无需改变4096或过滤长样本。此文件本地/远端SHA一致，仅为完整人口输入检查，未运行全量模型评分。

## r09开始执行与当前论文交付

r09已于epoch1792109065.236获锁，controller56295、worker60387；七B各16步的FP8原生trace和专用调优已COMPLETE，后续graph/质量/完整63格仍在执行，不提前记为全部完成。已结束作业累计持卡6949.893秒，当前r09需在退出后按实际持卡计入。

上一版`paper/main.pdf`为12页，已纳入r08b的200条自由生成、GSM8K小样本和多选身份控制三表，编译与逐页检查通过，SHA256=`2ecb3db11309591198d6221bec316c2eb23ca48109477e0a7cf15a37e0d1d6e3`；与正文、质量表和结果入口一同同步远端并核对哈希。此版本尚未纳入r09结果。

## r09终态与FP8接续

当前效果：r09于epoch1792109524.609退出，外层FAILED、内部INCOMPLETE，持卡459.451秒；native trace/tuning与B1/B8/B32三项十臂质量全部成功。850个归档文件共66,657,598字节已回传、逐SHA匹配；清单SHA256=`6a472ea97ba074687ae4876b04ecac09fb5c04a275d91f5b8fcfa9373c96e192`。controller56295/worker60387均已退出，原r09交接器未重复提交。

两项padded graph父检查未通过：dense在B3→4/B5→8的eager/graph输出不同；marginal在B3时slots_after为1879/1880，B5也有末token差异。五个子执行各自的scheduler、实际batch、padding排除和层计数覆盖均PASS。同为graph时native/dense两组输出精确一致。不能将此直接归因于跳过机制缺陷或自动放宽断言；下一步增加真正无策略的native eager/graph对照，并在完整性能矩阵实际采用的七个无padding batch上验证一致性，通过后继续FP8完整63格。F6、rolling路径和性能此次未运行，成功的采集/调优/质量将复用。

新增`figures/f1_native_fp8_r09`：B1/2/4/8/16/32/64实际独占槽比例100/86.474609/66.638184/34.906006/12.289429/2.980042/0.601959%，每B16decode步×16层。FP8为在线W8A8；拟合q来自同次capture。旧r05 BF16只有8步、prefix32，本次prefix128，跨格式曲线不能当量化的配对因果。

全量GSM入口已在`deployments/task_quality_full_20261016`本地与远端暂存，实际远端CPU计划无警告：原matched_gate_controls八臂加tau=.05/m=.05的gate/tail_gate/marginal三臂，全部1319题、5shot、seed1701、B8、cap256；不含新的global减k全量臂。计划14509次完成，不截样本；7070秒job/7080秒suite/7140秒outer上限，预计70–100分钟仅为小样本外推，超时保留未完成。尚未GPU提交。

## r09b恢复已提交与最新13页论文

唯一恢复请求`olmoe_fp8_matrix_r09b`于epoch1792110034.106提交，controller63659，当前WAITING。准确命令在`deployments/olmoe_fp8_matrix_r09b_20261016/launch.json`；候锁21600秒、外层7200/内层7170秒，累计预算15小时不变。driver SHA256=`89962eb82e939213381b43ceccdd48c86f68b73c6209944c37d6fd20d6626618`，新smoke SHA256=`94290a172a2afca4f11cf75acfd3cc94fa38de02e8baf2f1219d678604164a88`。原生padded输出差异将作为诊断观察保留严格失败状态；七B无padding dense/marginal仍要求严格通过，随后F6、九窗rolling、真实65-token尾窗和63格性能。trace/tune/三个质量阶段按原SHA复用，不重复运行。

`figures/quality_fp8_r09`已完成三B的FP8 F3与CSV，dense/native每B的1024预测完整KL/NLL汇总精确一致。B8 marginal-mid节省21.60425%逻辑字节、KL=.05505614、ΔPPL=.431215；top6为17.71393%、.07771704、.588097。B32 high为8.11159%、.02264569、.130003，top6为7.12776%、.07998617、.629384。这些是各自FP8 native参照下的探索性32文章均值；低预算扰动大小跨格式改变，不保证FP8任务质量或性能。

最新`paper/main.pdf`为13页，已纳入FP8七B刻画及三B F3，保留BF16性能、r08b真实生成与任务质量代价；FP8性能明确待运行。编译及13页渲染检查通过，PDF SHA256=`23915ee4f9116f105be778cbc30ba54e287aa017b287d03e1abca176014c0f3d`。

DeepSeek串行接续已于epoch1792110355.051启动CPU交接器PID64317，`handoff_r09b.json`实查为WAITING_FOR_EXIT，绑定r09b controller63659的boot/starttime身份。仅当前驱七stage、完整bench与终态SHA均成功且进程退出后新增部署DS两文件，再依次提交`deepseek_bf16_matrix_r01`和`deepseek_fp8_matrix_r01`；失败停止、不重试、不终止其他进程。当前DS尚未部署live或申请GPU。两格式最新真实远端CPU计划均无警告、8stage，graph明确使用同七B无padding；每格式外层7200/内层7170秒，仍计15小时总预算。原`after_r09.py`未启动，当前入口为`deployments/deepseek_matrix_20261016/after_r09b.py`（SHA256=`5c8baea5b2bb227dfe60179edcfecb57dddbc9898ac08384840704c688ac7dd7`）。DS新driver/config SHA分别为`277a7ea4e181dca2f7c58c7e8b207793af8a1f0901ca3b0e2f4472b6a8073e1a`、`eed7e50fcc4cb5aaefd822932e3b612827c92f4e329d9803f85e06a477746199`。

## r09b实际图检查、CPU开销与完整test准备

r09b于epoch1792111130.519开始持卡。无策略、无路由计数的原生FP8对照在B3→4/B5→8也出现eager/graph输出差异：严格父产物仍为FAILED，两个执行子进程各自PASS；包装状态DIAGNOSTIC_COMPLETE只表示诊断执行完整，不表示一致性通过。因此这一观察不依赖本课题跳过策略，仍不足以证明动态量化scale是唯一原因。下一步性能只使用原定B1/2/4/8/16/32/64无填充域；该域dense与marginal-high七B的输出和所有计数均严格相同，native/dense同执行模式也通过。没有删除或放宽原有失败断言。

F6实测`runs/olmoe_fp8_matrix_r09b/microbench_policy.json`已PASS并回传（SHA256=`427d8485ff6a87a9db06538b1eb2addad35e8abd15b75a58b6e6197465b5ab90`）。融合Triton在B1/8/32/64的captured GPU时间为2.456576/2.871296/4.714496/6.551552微秒；eager线程CPU时间为30.78495/22.26478/26.76779/23.43211微秒，使用`time.thread_time_ns`直接计量，非墙钟减GPU。它是E64/k8合成路由、含计数器的后处理开销，不是FP8专家GEMM或服务收益。原生padded诊断及两项完整七B图产物已回传，性能尚在进行时不使用整轮终态声明。

完整OLMoE WikiText-2 test执行入口已在`deployments/full_rolling_20261016`本地和远端暂存；真实远端CPU plan通过六个命令。BF16/FP8、B8、七个显式臂（native/dense/global_j2/gate_mid/gate_high/marginal_mid/marginal_high），每格式全部2255窗、288602计分token，非十臂全矩阵。按原283个cohort拆成96/96/91三块，保留B6尾批和最后90-token短窗，不重分batch、不重tokenize、不删数据。完整chunk可按源码/输入/调优/产物SHA续用，失败块保留原attempt后整块重跑；各块是重启边界，不是独立实验重复。完整结果按token合并NLL/KL再计算PPL。两格式总耗时估计3.23 GPU小时、外层硬预算4小时、建议4GiB输出空间，均未实际运行完整test。入口SHA256=`c410ca7d8f22189cb0c5f6634fda80687023abed83f0ae4312aaeaf8504640a2`；优先完成三模型性能后，按剩余GPU预算与空间安排。

r09b的首九窗1152预测及实际最终65-token短尾均已PASS（十臂，真实decode评分；短尾manifest重设本地offset但保留全局出处），只是执行路径验证，不是完整validation/test PPL。完整性能于epoch1792111448.186启动，尚未终态。最新13页论文已更新F6 CPU/GPU双面板及图执行边界，逐页渲染通过，PDF SHA256=`b1b88260cbcb2397fbddea16316e00bf6bee04bbfc1f8901bff761144b573ed3`。

Qwen两格式完整矩阵已在`deployments/qwen_matrix_20261016`暂存并通过远端CPU plan；E128/k8/I768、48层、首MoE0、原生norm_topk_prob=True，.85总显存比例仍待实际加载验证。每格式8阶段含七B trace/tune、严格无填充图检查、三B十臂质量和63格性能，外层7200/内部7170秒；没有提交Qwen GPU。续传入口`deployments/qwen_partial_resume_20261016/resume_qwen.py`及固定三介质放置已通过远端只读plan：K需17,424,291,144B新尾部，同时保留C相同大小的未来append预算，持久盘/shm各20GiB、独立30GiB系统盘10GiB余量，另预留2GiB实验输出。完整6分片和8个最大连续前缀只通过reflink/独立copy复用，C进程与文件不修改；所有16个完整SHA通过才可READY。当前只是计划通过，未下载，后续安排在性能计时之外。

接续已实际调整：epoch1792112499.572启动新CPU桥PID70776，`deployments/qwen_partial_resume_20261016/bridge_launch.json`记录准确命令与来源；旧等待器64317已核对命令和启动身份后SIGTERM并实际退出，其原receipt保留STOPPED（没有提交过DS）。新桥SHA256=`db4391e5f2213f6c562fce81173e1a67029e13b48a87a2ec14919764a3092474`，当前WAITING_FOR_EXIT。仅r09b严格成功并实际退出后，先在无K活动GPU请求的窗口内对r07的189个performance/energy JSON做无损归档、逐原SHA校验后移除重复展开副本，再运行最多3600秒的Qwen CPU续传，随后调用原DS handoff提交BF16→FP8。下载失败/超时保留结果，仍继续独立DS工作；不改其它课题、不持GPU锁进行下载。r07本地完整原数据保持展开，远端压缩只有通过全部成员验证才会进行；本地试建507,001,044B原值压成92,289,166B，约省0.415GB，原SHA256清单与其它结果保留，可按storage.json恢复。当前压缩与下载均未开始。


## r07长轨迹刻画与全量多选评测入口

`figures/f1_long_decode_r07/`直接复用r07已完成native计数回放，覆盖每B的2047个真实decode步×16层=32752层事件，P128/N2048；计数回放与对应性能输出身份已通过。B1/2/4/8/16/32/64独占槽比例为100/86.041310/62.639400/33.409362/13.121694/3.447595/.649298%。各点分母为262016×B，112个逐层原值保留；这些层和token不作为独立运行重复。均匀iid参考在B32/64为1.593124/.022208%，低于本次实测；缺少完整逐专家边际，不能据此区分偏斜与相关性，也不拟合缺失的偏斜曲线。旧r05短轨迹使用不同前缀/长度，不能把两者差异归因为测量时长；未来与相同P128/N2048的FP8比较仍允许生成内容/路由随格式改变。新增图已目视检查，不使用GPU。

完整MC入口`deployments/task_quality_full_mc_20261016/run_full_mc.py`已暂存远端，三个真实CPU计划无警告；MMLU5-shot 14042题、HellaSwag0-shot 10042题、ARC-c0-shot1172题，每个family独立执行，始终无limit。默认十臂，显式选择臂子集时只改变臂集合，记录exact_selected_arms并保留完整人口；不能称作十臂全矩阵。入口复用既有harness与suite，不改运行中的code。全部十臂三family的时间/输出估计为约131分钟/4.30GiB，来自r08b请求循环速率和固定开销分解，既非上界也未分配GPU；完整任务构造、题目长度和共享存储可使估计变化。三项MC只验证prefill身份，真实decode质量继续由全量GSM/rolling补齐。实际GPU尚未提交；优先现有三模型链，再依据实耗安排。

最新论文已将BF16 F1主图换为上述r07长轨迹实测，仍为13页；第3页说明完整decode人口，第4页为新图，FP8性能继续标为待完成。PDF SHA256=`8409174616cfa6ef1f11c73aba8efc33a3816c13795f505e25e6771fbf412853`，latexmk成功，无undefined、LaTeX Warning或Overfull，全部页面渲染，改变页目视通过。


## 任务质量的逐题配对

本地`code/summarize_task_quality.py`已补官方doc_id及doc/prompt/target身份配对；strict-match与flexible-extract分别保存逐题正确性、四格分歧计数及准确率差。缺字段、重复ID、缺行或人口不同整组拒配，不用交集伪造完整差值。`figures/task_quality_r08b_paired/`基于原r08b得到320条文档/filter记录、288条候选配对和18组汇总，全部16题完整身份相同；原派生目录与原数据未改。两filter下mid/high均为native-only2、candidate-only0；global_j1是1/0，gate_high是2/1，两者总分同降6.25pp但分歧人口不同。当前效果仍为16题探索，不支持1pp或显著性结论；下一步直接在全量1319题原始结果上使用同一配对入口。MC保持单列prefill身份口径。新分析脚本SHA256=`ceea2bc4152b0761362b143f4845d7a60feab6c24f5e6e1b5d959b8956fa0e81`仅本地更新，未覆盖运行中live code或DS暂存副本。

## r09b完整FP8矩阵与接续容量修正

`runs/olmoe_fp8_matrix_r09b`于epoch1792114724.516 COMPLETE，controller63659/worker68143实际退出；持卡3594.069秒。686原始文件共582,572,309B已全部回传并逐SHA匹配，manifest SHA256=`17bfb6e35b57346447c993882c699efa0a37924d511f7eef581c1eace9dbf4f4`。七stage完成，bench63/63格；每格性能/独立计数生成token身份一致、FULL且无padding；能耗窗5.389–28.360秒全>=3秒。原padded native严格失败诊断仍保留，不将其改为PASS。`figures/results_fp8_r09b`现有F2/F4、63行原值和7行全策略汇总。

FP8 marginal-mid在B1/2/4/8/16/32/64的TPOT降幅为2.22/5.61/4.71/3.87/2.08/.72/.66%，J/token降幅7.51/11.25/4.61/8.24/4.22/2.23/−.15%；high的TPOT降幅4.14/9.89/8.98/8.25/3.50/2.40/.63%，J/token降幅12.45/16.77/10.06/13.73/6.86/4.25/−.006%。B8 high的各自轨迹逻辑足迹低25.88%，mid低15.99%；global-j2在同B的TPOT/J低4.42%/6.33%，gate-high低7.59%/12.65%。这些策略质量不同，小差异未经独立重复不作胜负判断。当前效果：BF16小到中B执行改善在FP8仍出现但幅度缩小，B64能耗无可辨认改善；下一步保持完整协议扩展两模型，并补任务质量和独立重复，不能称格式无关或纯权重字节的因果效果。保守low在部分B的跨策略逻辑足迹/性能反而增加，所有不利值保留；跨策略轨迹不同，不能把这种足迹差直接当作单次删除产生负字节。

FP8长轨迹F1已在`figures/f1_long_decode_fp8_r09b`生成：七B独占比例100/83.146067/55.011717/29.740025/12.056897/3.318786/.605277%，每B16层×2047decode、262016×B个槽。另有与r07 BF16同P128/N2048外部协议的描述性对比，两格式均单次运行；内容与路由可随量化变化，不是固定route因果。图面及原始输入SHA已核对。

CPU桥已执行r07无损归档：189个performance/energy原文件压为92,290,727B，archive SHA256=`35583e68b0fd346a4267a819204d0e6cf1cbc017771d962d5c325673c04bbd84`，远端`storage.json`为COMPACTED；本地原文件保持完整展开。随后Qwen进程73728实际运行并于epoch1792114807.167 FAILED（source SHA256=`7897df235ba65e04c4ff2082638213eef93de8e070db0e245754be5f2275a386`），动态预算检查发现persistent free从约50.00GB降至约6.34GB。此前plan把FICLONE前缀当零新增容量，但本文件系统可用容量实际扣除了约43.6GB；不能据st_blocks单独判断底层物理共享，却必须按实际df计预算。未发HTTP续传、completed=[]；这不代表没有克隆文件。

root已逐个核对13个K克隆的inode/size/blocks、原C来源dev/ino/size/mtime和下载进程退出，再只unlink这13个可重建K副本，保留small assets、shm52.4MB前缀、原失败与清理记录；C文件和进程未改。`cleanup_failed_clones.json` SHA256=`810e951039cdc8bb976e811914134c423c208950ed10b48c3254a00067259212`，free恢复49,906,106,368B。当前容量不足以同时保留DS RAM权重并独立续完Qwen；下一步在DS两格式及必要复测完成、实际进程退出后，串行复用本课题31.414GB临时RAM存储。6个已完整C分片只读SHA验证后链接，缺失10个分片在K独立存储；不假定reflink省quota、不减少C未来append或既定reserve。未释放DS权重、未重新提交Qwen下载。桥按约定已提交`deepseek_bf16_matrix_r01`，最后观测controller73768真实候锁；不要重复提交或重启旧桥。

## 连接中断与本地接续准备

epoch1792115499.654是当前最后成功的远端观察：DeepSeek BF16 controller73768为WAITING，无worker；桥70776真实存在，等待该任务退出。随后SSH观察被重置，原control socket消失；重新连接westb:50328连续返回Connection refused，2026-10-16T02:04Z复查仍相同。连接失败不能证明机器重启、RAM丢失、任务结束或GPU尚未开始。恢复后先查boot ID、原进程启动身份、各run终态、GPU占用和临时模型文件；不依照旧WAITING记录重复提交。已向用户询问是否暂停或端口变更，尚待回复。此前两格式完整OLMoE原始结果已安全回传，当前没有新增GPU耗时的可靠观测。

新Qwen入口`deployments/qwen_after_deepseek_20261016/resume_qwen.py`已在本地完成（SHA256=`ef5e2c7ad718ee1363f9a9c2913db937fafe65863658bd44cd2547831b61ab09`），未部署、未运行远端plan或下载。它按实际尚缺分片的完整私有容量收费，只读复用经整文件SHA核实的C完整分片，缺失9/16放K自有rootfs，其余缺失分片放K自有shm；普通独立复制前缀，不再使用FICLONE。按最后空间观测，且仅在DS31,413,626,576B临时权重确实可释放的条件下，扣除C未来append、既定空闲余量、实验输出及overhead后，persistent/rootfs/shm仍余8.001/2.609/4.945GiB。此为条件算术，不是实时容量检查；脚本仅使用当前实际free，不执行DS回收。语法、CLI、容量守恒/不足拒绝、真实中断复制后续接且来源不变的CPU检查通过，GPU加载仍未验证。

本地最新14页论文已纳入FP8完整性能和双格式长F1；`paper/plot_format_results.py`从两格式各63行原始汇总重画全策略F2/F4，图例位于坐标轴外。最新论文、这些图、新GSM逐题配对分析和Qwen接续入口尚未同步远端；已部署的DS链及其代码未覆盖。

## 严格独占条件对照的本地实现

`deployments/singleton_ablation_20261016/`新增隔离payload，仅为marginal增加默认`require_singleton=True`的布尔开关。关闭时，参考实现、torch和两条Triton路径都只去掉`count==1`资格条件；仍使用同一rank、含上界gate/mass、弱权重优先预算、rank-zero保护与模型原生重归一化。旧配置默认行为及T1的`rank_gate`半开分桶不变。关闭独占条件会改变哪些槽消耗预算，跳槽数不等于distinct专家减少数，继续用原计数器测量。

新core SHA256=`fb9ed78cf942c21b6e5c160dfece6169e8319326163a19cfa73f7ca862ab5d31`。本地实际CPU检查21项通过、3项CUDA测试跳过，包括旧策略回归及新阈值相等、mass边界、共享专家、padding/零active/空输入和两种renorm；现有quality解析器接受8臂严格对照及14臂完整任务配置。Triton编译尚未执行，96 case/336请求只是离线编译计划及AST签名核对。恢复连接后先做一次受锁GPU数值/graph验证，再进入完整质量测量；不把CPU通过当作GPU验证。

14臂`arms_task_quality.json`（SHA256=`7e81ca78fd1c7932a509c56f84d1733a79d91138d56e29cb332b727c9e2b92f5`）含native/dense、原生global-j1/j2/j4，以及m05/mid/high各自gate、marginal count-on与count-off。三档固定参数为(.05,.05)/(.10,.20)/(.15,.30)，未调新参数网格。当前只完成本地实现与配置，原code/configs、已提交DS源码及所有历史结果未变；完整任务结果仍待实测。

配套`deployments/task_quality_count_controls_20261016/run_full_gsm8k.py`（SHA256=`08e7d0bf30cd36af59cb66a70887b3123bf9438bdd5902c0d2f74b81dd3d3028`）复用原harness，完整1319题×14臂=18466次生成，保持5-shot/seed1701/B8/cap256/.35BF16/max4096，无limit。真实本地CPU plan为一个job，保留远端model config在本机不存在、新core尚未部署两条警告；旧core的执行请求在创建输出和导入GPU前拒绝。成功校验要求每臂两filter的完整指标、原始文档ID、shot/seed/cap与raw completion人口，原16题pilot按完整人口规则被拒绝；从原pilot复制到临时目录、仅调整测试人口声明的schema fixture通过，未改原数据。完整14×1319与缺行拒绝的CPU fixture也通过，但都不是完整GSM8K实测。

此14臂入口替代原未运行的11臂GSM计划，不默认两批都跑。默认suite/job/outer为7080/7070/7140秒，可由root按剩余总预算显式调整；从16题请求循环和一次启动开销粗估约90分钟，输出终止、JIT与全量构造可能改变成本，2小时不保证完成。超时保留原始部分结果，不自动扩预算、删题或少跑策略。部署须等原DS链完成并实际退出，父进程核验live core SHA与实际导入位置，子进程继续使用同一root/code；当前未部署、未GPU运行。
