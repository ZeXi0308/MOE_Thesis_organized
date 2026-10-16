当前主张：收益依赖执行域与指标。OLMoE/ngram compiled五臂留出评价完成40个run；自适应相对所选静态γ0在B64有效TPOT降低4.04%，但人口吞吐降低5.16%、J/token增加8.34%，B1–16尚未获益。保留全部两次原始值，不宣称小差显著。
关键证据：eager/compiled各56格×2矩阵、各32格G7完整成本校准；实际S路由模型MAE1.91专家；kernel留出MAPE 2.35%/1.92%；五臂40run均完整64请求×128输出，0排除且40个NVML累计counter-span有效，F3/F4/F6已生成。
最强替代解释：草稿覆盖、实际执行图、生产与截断边界、排空和host成本共同影响收益。native动态表实际强制PIECEWISE；后截断G7配置不同，不能把策略差归为纯动作选择，也未证明c0主导或将logical bytes当HBM流量。
下一动作：恢复连接后先核实原协调器45296及其当前子进程，取回eager五臂与自然停止；接续神经草稿和其他目标。六份共同128条代码/摘要输入及12个模型/草稿/任务域配置已在本地准备，补充协调器尚未部署或启动；总路线与20小时预算不变。
资源状态：eager AR于09:43:54完成并取回8个run；09:43:59最后核验原协调器已进入static6（74026）。约09:50连接断开，10:11重连仍为Connection refused，后续远端进程、锁与累计用量未知。已取回结束组至少8589.24548秒（2.386小时/20小时）；磁盘46.28GiB为09:29旧快照，不代表当前余量。不因失联重复提交。

## 2026-10-16：建设起点

研究目录独立于历史 paging 课题。继承 `MoE_Scheduling_20261002/NGRAM_RESULTS.md` 的观察，不将 PCIe 专家搬运结果外推为全驻留 HBM 收益。OLMoE 配置为 16 层、64 专家、top-8、BF16；现成权重 revision `7f1c97f440f06ce36705e4f2b843edb5925f4498`。

已读取实际安装源码，vLLM 0.26.0 有 `num_speculative_tokens_per_batch_size`；当前基线应覆盖该原生能力。旧 `disable_by_batch_size` 名称不能直接当成本版本 API。MoESD 已解释小 batch 额外专家工作集和中等 batch 饱和，论文将准确署名并以路由校准、能耗和实际系统行为组织新增证据。

计量修正：以每 token 包含概率 q_e（和为 k）计算 D=Σ[1−(1−q_e)^(B(γ+1))]，独立仅指 token 间。专家工作集字节是逻辑量，实际 HBM 读取需另行硬件证据；同为 SM120 不代表跨设备性能结论可直接搬用。post-proposer 截断保留生成成本，控制 CPU 时间和截断浪费都计入。

代码分工：root 管理 runner、数据、GPU 与结果；runtime_adapter 管理进程内 vLLM patch；cost_model 管理路由模型/分析图；literature_paper 管理论文和近邻。全部在本目录内工作。

## 首个完整矩阵（在途，2026-10-16）

`runs/olmoe_ngram_matrix_r02` 为56格×2重复。全部使用相同64条原序ShareGPT请求，按B拆成同步waves；每条128输出、忽略EOS，完整计入prefill及生成。独立route episode使用32输出、不进入性能热图。实际路径为FlashAttention2、TRITON / TritonExperts / MoEPrepareAndFinalizeNoDPEPModular；权重12.89GiB完全驻留。

首次r01在导入torchcodec时缺少CUDA13的libnvrtc搜索路径，未加载模型。当前效果为初始化问题已通过本进程LD_LIBRARY_PATH修复；下一步继续同一矩阵，r01错误原件保留。没有改共享安装。

r02首格B1两个计数链（host step与scheduler step）每wave均匹配；NVML总能量计数器可用。当前预热仅覆盖B64主形状，首轮其他prefill形状存在冷JIT成本，原始计时全部保留。因此当前矩阵是探索刻画，后续确认使用逐batch/提示形状预热；不能事后只选更快重复。

已建模块：可开关进程内调度/接受率/路由patch，固定预算截断与原生模式；token包含概率模型、经验prefix survival/EWMA、包含草稿成本的查表选择；能耗与完整人口runner；热图/route分析；5页可编译论文骨架。GPU下的自适应臂和第二模型尚未运行，不标为已有收益。

## gamma0–3 初图与下一项构建

`figures/F2_partial_03/` 与 `F4_partial_03/` 保存已完成28格、56个完整运行记录的热图、原始重复值和汇总CSV。B1的gamma2吞吐为AR的1.176倍，J/token由1.498降至约1.34；B16/32的gamma2接近持平。当前效果为小batch获益、较大batch收益弱，下一步补齐gamma4–7并在逐形状预热的compiled执行中测量，随后用相同kernel实测分解工作集与固定开销；暂不将这一局部形态归因为c0主导。

纯AR路由统计逐层合计496个decode观测/B，B1→64的distinct均值为8.000、15.014、24.992、38.097、50.347、59.454、62.651。所有histogram计数满足tokens×top-k。gamma1的B1第二次运行仅约18%的decode请求步存在草稿，条件接受率约0.536，因此不能将名义B(γ+1)或单一α直接代入每一步。控制器已分别记录完整草稿可用率与条件prefix survival。

输出检查：在已取回gamma0–3的B1/8/64中，同配置两次输出64/64完全一致；跨gamma与AR并不总逐token相同（例如gamma2的B1/8/64分别16/56/15条完全一致）。这些是BF16、不同批形状下的观测，尚不能凭此判断质量或归因为运行时错误。下一步保留真实自然终止输出并做任务抽检；固定128输出只控制上限，不宣称生成内容相同。

新增的同kernel微基准读取真实路由直方图/可选token级路由，保留5次重复；`fit_kernel_cost.py`使用预先固定B1/4/16/64校准、B2/8/32留出，拟合非负max(aD,bS)+c。它目前只有CPU自测，GPU测量尚未运行，不用模型预测冒充实测。19项进程内适配测试和31项成本/分析CPU测试已在服务器实际Python环境通过。

## 完整 eager 矩阵交付（2026-10-16）

完整数据入口 `runs/olmoe_ngram_matrix_r02`，全部56格×2重复、每run64请求×128输出，总计112个性能运行；无未完成或排除的性能文件。原始各gamma源码快照、配置、能量计数器和逐请求输出均已取回。F2为 `figures/F2_olmoe_ngram_r02`，F4能耗热图为 `figures/F4_olmoe_ngram_r02`，原始重复和中位数CSV与图同目录。B1全部正gamma吞吐为AR的1.129–1.179倍；B16/32多数接近或低于AR，B64约0.995–1.028。当前效果为eager域未出现预期方向翻转，下一步继续compiled执行和更高覆盖草稿源，不能仅据此断言c0占主导。

F1 `runs/route_prediction_r02/route_prediction.json` / `figures/F1_olmoe_ngram_r02` 以3472个AR层步拟合各层q，再验证23568个gamma1–7层步。使用实际S的MAE=1.9145；使用初始B与名义满草稿长度的MAE=15.2994；保留实际活动B但仍假定满草稿长度时MAE=13.6975，分别揭示drain和草稿可用性影响。fit/test是不同gamma执行轨迹，包含相同prompt子集，不叫独立工作负载验证。所有残差保留，首decode存在明显集中误差，AR中也存在，后续用同kernel成本和新prompt检查相应边界。

完整成本表 `runs/olmoe_ngram_matrix_r02/calibration_complete` 保留每run纯decode完整host-step均值；它是各原生gamma自身proposal budget，不可直接给budget7后截断控制器使用。`runs/controller_selection_eager_r02` 按预先明示的B1/4/16/64等权有效TPOT选择static gamma6、native表[1–3→7,4–15→5,16–64→6]，原生单阈值选择全域启用gamma6。选择分数只是开发数据上的选择依据，政策性能尚未运行；有效TPOT与含prefill吞吐的目标口径分别报告。

后续构建已就绪：25项adapter CPU测试在实际服务器全部通过；compiled诊断分别记录有效验证行与CUDA graph额外物理填充行，避免混淆工作集。新增完整性能人口的草稿可用性/推进分析、自然终止与固定first8输出抽检、同kernel分块配置记录。Granite1B完整既有权重与160条raw continuation已准备，作为E/k较小的辅助边界；Qwen3仍是主第二模型且其BF16下载未完成。OLMo-1B神经草稿下载进行中，EAGLE-3固定权重已完成但尚未原生GPU加载。

## 请求完成时间与有限批次吞吐

`runs/derived/wave_completion_r02_g026_b1_4_16_64/` 对24个性能文件、510个wave做时间恒等式分解。当前固定128输出下，各wave所有请求的首输出来自同一步，因此 `wave wall = 首输出时间 + 127×平均有效TPOT + (最晚完成−平均完成) + 收尾开销`。相对AR，B16/γ6每64请求的post-first平均跨度少0.70788秒，而新增完成时间差合计0.71303秒，首输出合计还少4.36毫秒；B64对应减少0.25573秒、新增0.20864秒，首输出只多0.305毫秒。这里的吞吐差异主要与完成时间不齐对应，prefill增加不足以解释；不把完成时间差直接当成可追回收益。

B16/γ6两个重复中，实际decode B低于标称B的drain占decode host时间38.71%/38.18%；B64为46.28%/47.52%。GPU仍服务剩余请求，这不代表GPU空闲。B64 AR两重复wall为1.853/2.002秒，γ6逐重复吞吐比为0.9817×/1.0685×，仅两次重复不能断言小幅吞吐变化稳定。当前效果为平均请求推进和同步wave排空收益不同，下一步使用相同64请求、并发上限16、γ0/6比较wave准入与完成即补位，并报告统一完整窗口和实际活动batch。相同人口的B64没有waiting请求，不用于验证补位收益。

新增 `figures/F4F5_olmoe_ngram_r02/` 提供按B的绝对J/token、相对AR能耗，以及包含零草稿、drain和终止步的proposal availability / draft slots /实际发出token推进图。56格、112原始重复齐全；图中保留两个运行点，不将请求步当成独立系统重复。绘图完全在本地以PGFPlots完成，复现脚本为 `code/plot_energy_progress.py`。

## 接受统计与控制开销实现

`figures/F5F6_tracking_olmoe_ngram_r02/` 保留旧接受率估计器在固定轨迹上的离线重放与原生观测CPU计时；112 run共239,706 host步骤、813,816 decode请求步，删失7,168终止请求步后为806,648条，其中4,894条旧日志未标的零草稿终止由最后client chunk恢复。没有把重放称为自适应策略实测。r02不含线上prefix/availability快照或adaptive decision计时；新plugin在episode结束记录估计器类型、先验与状态，仍待GPU策略执行。

在r02完整decode范围，已计时的schedule记录段占host step的0.1285–0.3628%，接受更新段占0.0356–0.0707%；control wrapper嵌套于schedule段，不可重复相加。这些段本身不足解释B1约10%级收益，但不能由此证明所有观测扰动为零，也不从墙钟直接相减。原生臂的控制器截断计数均为0，不当作自适应浪费结果。

重放发现原标量alpha是每个eligible step的accepted/tested比率EWMA，测试长度变化时不等于曝光加权接受率。当前效果为统计目标存在区别，下一步已在首次adaptive GPU运行前将默认改为分别平滑accepted与tested计数后取比值，保留 `legacy_step_ratio` 以复现旧曲线。默认先验是1个测试位置、0.7个接受计数；无曝光不更新。prefix survival与proposal availability仍按eligible step更新，未顺带改变算法。新版37项cost/analysis/calibration与26项plugin CPU测试已在服务器通过；这不替代实际政策评价。

## 同kernel实测成本模型

`runs/olmoe_kernel_r03/measurements.jsonl` 包含154个实际路由快照，全部成功，每点5次×20调用；实际S=1–148、D=8–64。按既定B1/4/16/64的88点校准、B2/8/32的66点留出，CUDA event模型 `max(aD,bS)+c` 选到纯D边界：a=0.00947062 ms/expert、b=0、c=0.153261 ms。校准MAE16.86µs/MAPE3.512%，留出MAE12.73µs/MAPE2.345%、最大APE5.763%。纯D+常数与max模型相同；纯S+常数留出MAE102.81µs，常数129.00µs。当前效果为该单层、路由与cache-turnover运行域内D具有额外解释力，下一步使用同kernel图重放检查执行路径变化；不能说三参数物理分解或完整服务模型已被验证。

独立profiler调用的kernel-duration sum中位0.388407 ms，而对应全部case的event中位0.571298 ms、wall中位0.584155 ms。每次主计时前执行256MiB无关缓冲读写并同步，但r03独立profiler调用没有再次执行turnover；因此两者差别含调用边界、潜在缓存状态和launch间隙，不能直接相减成为host成本。resolved tile共5种；原始profile和分块配置均保留。所有D字节仍是selected parameter storage，未测真实HBM流量。后续对照将匹配profile与主计时的turnover协议并保留原结果。

## 编译执行的有效行与物理填充

`runs/olmoe_compiled_routes_r01` 的 γ0/4 × B1/8/32 六份诊断全部完成，每份覆盖同一64请求、每请求32输出、16个目标MoE层。逐步top-k与直方图一致、S/P守恒、有效和物理buffer完整性检查均通过。γ0 decode均为FULL图且S=P；γ4/B1的1747个decode步骤中，1569步为PIECEWISE且S=1、P=5，另外178步为FULL且S=P=5。按decode层步平均，γ4的有效D→物理D分别为B1 9.621→26.730、B8 38.909→46.363、B32 59.271→60.902。P是模型前向行数，不是专家GEMM分块后的tile行数；新增专家存储量不是实测HBM流量。

当前效果为小batch下有效验证行不能代表该native图配置的全部专家工作集，下一步对照未开启路由记录的compiled性能矩阵并核对相同dispatch条件。此诊断启用了原生route-return和物理padding记录，不用其插桩墙钟声称服务收益。单层CUDA graph微基准仍使用原始有效路由快照，用于检查执行路径变化，不等同于这里包含padding的完整模型。

控制器评价尚未执行。分析中发现逐形状预热默认使用测量人口，会预先更新在线接受率；为明确留出边界，新增可选`shape_warmup_offset`，仅将未运行的controller evaluation与natural check设为0，测量保持offset64。历史默认和已完成矩阵不变。评价开始前保存`evaluation_initial_state.json`，EWMA仍在开发预热、后续batch、repeat与wave之间连续更新；它不是每重复冷启动。实际服务器与本地的5项runner协议检查均通过。

后续执行已串行安排：CPU coordinator35541等待`kernel_graph_r01`成功，随后依次启动`olmoe_ngram_compiled_r01`完整56格×2重复和`olmoe_fullbudget_calibration_r01`的B1/4/16/64、γ0–7、固定proposal budget7校准。它自身不申请GPU锁、不增加GPU排队者；每组沿用原有共享锁、20小时累计预算和失败停止规则。完整成本表生成后再进入实际留出政策比较，不根据未运行的预测填F3。

Granite3B 的两个草稿源（ngram、独立Granite1B MoE草稿）配置与自有目录下载器已部署，权重下载尚未开始。固定目标revision为`d4dd87aa3a6c201bc374851d7d7ff4cf39a0b82a`，E40/k8；只读复用B现有8个校验通过文件到S独立副本，剩余约5.0GB需下载，第三目标不再重复运行未测的Granite1B目标。现有plugin以draft phase隔离草稿MoE路由，三个target分析入口均过滤model_phase；同类同名E40/E32层的额外检查在服务器通过，27项plugin检查全部通过。源码还确认Granite的AR默认可能进入V2而推测模式回退V1；后续runner显式设置`VLLM_USE_V2_MODEL_RUNNER=0`并记入环境，确保同一V1路径。此前OLMoE实际已是V1，原结果不改写。

OLMo-1B-0724-hf 神经草稿已于2026-10-16 05:23本地时间完成，两个shard逐文件SHA256匹配，总计5,121,310,602字节。下载manifest与compatibility已取回本地models对应revision目录；普通50280个token映射一致，目标额外pad50280的边界保留，仍需原生GPU加载和接受统计。随后CPU进程35974启动Granite3B自有目录补全，启动时剩余约73GiB；没有加载GPU。论文现为11页，加入真实kernel留出图和compiled有效/物理工作集图，已编译并逐页检查；多模型与政策结果仍明确待测。

## compiled占用模型的最小留出检查

在同一compiled诊断上，以γ0的三个B全部数据拟合逐层AR token包含概率q，留出γ4执行条件、不改prompt人口与分区。B1/8/32的MAE分别为：q+实际S预测有效D：0.566/3.438/1.946；q+实际P预测物理D：3.133/2.898/1.464；仍用S预测物理D：17.675/6.307/1.818。当前效果为P修正显著减少低B漏算，但B1的物理预测仍有正bias约2.562，下一步在完整性能中检验这种成本偏差是否改变动作排序，不将P替换宣称为完整物理模型。

B1每层1569个含padding步骤中有134种padding histogram；同wave相邻且两步均含padding的1370对/层中，padding histogram全部相同，而有效行专家集合很少完全重复。这限定了token独立同分布近似：填充行不是普通AR token的新独立抽样。它不是cache命中或HBM传输测量，也未被用于再拟合一个追求更低误差的新模型。完整派生入口为`runs/compiled_route_prediction_r01`，保持同prompt留出gamma的证据范围。

## 三目标资产与校准后的执行链

Qwen3 W4A16目标于2026-10-16 05:38完成实际哈希校验：14个原生加载必需文件、四个权重分片，以及独立EAGLE3配置/权重均通过，记录为`runs/qwen3_awq_asset_readiness_r01/ready.json`。仓库名称含AWQ，但实际vLLM配置为`compressed-tensors`、对称W4、group128；这是独立的量化目标运行域，不当作BF16的质量或性能复现。当前SM120/group128不符合已安装WNA16 monolithic候选的SM100/group32条件，其余候选可由现有modular hook记录路由；实际选择的后端仍待smoke日志确认，不提前标作Marlin。

Granite3B于05:42完成：自有S目录的6,599,724,398字节全部有固定来源与哈希记录，目标E40/top8、独立Granite1B草稿E32/top8，词表和160条输入ID兼容检查通过。最终manifest和compatibility已取回本地，没有修改B源资产。磁盘剩余约66GiB。OLMoE、Qwen3量化目标、Granite3B的两个草稿源均具备加载资产；这仅关闭下载依赖，不代表草稿已执行、接受率或服务收益已确认。

`prepare_controller_evaluation.py`已在服务器验证可从64个原开发记录重建static6、原生非单调表和阈值表；它只在32个budget7完整成本格实际完成后生成五组留出执行命令。代码和摘要负载的完整配置已取回本地，所有未运行reduced路由诊断统一为相同64提示人口。当前效果为评测输入和成本边界已接通，下一步检查神经草稿smoke的实际调度计数，再由串行协调器进入完整测量。05:51实际进程检查确认34944和35541仍存活；GPU仍由其他已持锁任务使用，S没有增加GPU排队者。

05:58启动CPU协调器38597（`runs/olmoe_evaluation_sequence_r01`），已实际核验处于等待35541的状态。它随后依次执行主域五臂与自然停止、各神经草稿smoke和计数检查、两种草稿源的缩减矩阵、wave/补位及代码/摘要诊断，最后为其余五个域各测12格固定budget7成本并实际执行五种留出策略。校准候选为`{0,3,7}`，基线仅由各自native开发矩阵选择，不跨模型借成本、不用选择分数填F3；每组由原runner独立持共享锁并扣20小时总预算。协调器本身不占GPU或持锁，任一失败会保存现场并停止后续阶段。

新开发矩阵、smoke和主域fullbudget校准开始前已把general/shape warmup显式设为offset0。五域留出配置仍使用64–127，EWMA只在开发预热后开始在线连续更新。历史r02的offset80不改写，主域选择的历史边界继续披露。`check_native_draft_smoke.py`只要求真实调度与验证草稿存在，允许接受数为零；OLMo语义token边界计数同时保留。当前效果为完整评测机制已可自动接续，下一步由真实运行检验成本排序、接受覆盖和策略收益。

## 图重放实测与论文对照表

06:18完成`runs/olmoe_kernel_graph_r01`：154个真实路由快照全部成功，占锁28.0209秒，原始数据与源码已取回本地。当前效果为同kernel的CUDA graph对照已经实测，下一步按原定88个校准、66个留出快照独立拟合，并与eager逐case比较。两种执行域不混合拟合；旧r03 profiler调用没有再次turnover，新图重放profile已匹配turnover，不能将它们的差值当成纯host成本。随后自动启动compiled完整矩阵controller40122，06:19核验为等待共享锁，尚无该矩阵的性能结果。

独立拟合结果为`runs/kernel_cost_fit_graph_r01`，真实图在`figures/kernel_cost_graph_r01`。简单D+常数得到`T[µs]=9.01937D+55.73326`，66个留出快照MAE5.86753µs、MAPE1.92402%；max(D,S)+c的对应值为5.84972µs/1.92048%，只有1/88个校准点走S分支、所有留出点走D分支，额外项尚未给出有意义的普遍计算边界。纯S+常数和常数模型的留出MAPE为28.186%和40.041%。当前效果为D的预测价值在图重放域保持，下一步检验完整步成本及实际动作排序；该层的截距仍不是全模型c0。

154个case的路由hash、源事件、直方图、S/D、权重、kernel源码、tile、seed和主计时turnover配置逐一相同；实际capture全部成功并匹配eager预热输出。逐case graph/eager event比中位数为0.792380（范围0.489871–0.866924），wall比中位0.794309，全部case时间更低。两组在不同时段运行，没有交错运行级重复，因此这是描述性执行域对照，不把差值当作已隔离的CPU发射成本；每模式770个重复均值、15,400个原始调用全部保留。

完整compiled矩阵要区分两个方向相反的因素：同路由图重放可减少单层时间，但模型级图填充可能扩大低B、短实际草稿时的物理专家集合。已有γ4/B1诊断中，很多只有1个有效行的步骤实际执行5行，因此不能从单层graph加速推断推测服务加速。下一步保持该矩阵的完整人口与所有结果，核对实际dispatch，再判断物理填充与草稿覆盖是否改变batch收益形态；诊断的max-seqs32和性能矩阵的64仍是需要披露的条件差异。

论文已采用IEEE会议双栏格式，当前8页；图重放留出验证已进入正文，旧eager数值与原图保留，逐页检查通过。`paper/tables/T2_actions.tex`进入正文，按可用信息、动作位置与草稿成本边界比较MoESD、Cascade、AdaEAGLE、vLLM和本实现；未把动作级对照写成这些论文的性能复现。`T1_assumptions_directions.tex`保留原始理想假设下的方向预测、已测OLMoE结果及待测域，不提前填写未测翻转点。五臂实际评价仍为AR、最佳静态、原生单阈值、原生非单调表和本实现。

本地`plot_policy_comparisons.py`已接通主域和缩减域比较输出：F3保留每次运行及完整重复的中位数，F4使用验证过的完整人口能量计数器窗口，F6分开选择CPU时间与可用草稿截断比例。缺少重复时不补曲线，也不将原生零截断替代自适应结果；图仍等待真实五臂数据。

`analyze_online_acceptance.py`从评价开始时的实际快照，按原batch/repeat/wave顺序恢复连续在线状态并核对日志终态。它导出运行级曝光加权误差和逐步alpha/推进预测，缺失前序不由未来状态补齐。历史r02仍只支持原有冷先验离线重放；新F5等待实际政策执行。`plot_matrix_pgf.py`也已用112条r02记录复现56格吞吐、有效TPOT比与独立窗口能耗图，280项原汇总值差为0；这是绘图复现能力，不计作新增性能证据。

06:31已核实compiled完整矩阵取得共享锁并实际输出γ0数据。引擎记录`enforce_eager=false`、`enable_return_routed_experts=false`、TRITON、显式V1；当前已写出的12个完整性能记录均完成64×128输出，其余按原计划继续。完整矩阵结束前不根据基线或部分γ单独宣布收益形态。另新增`plot_route_prediction_pgf.py`，严格复用现有F1的分组误差而不重新拟合，后续各模型可分别生成本地F1图；原有效行与物理填充行继续分离。

## compiled主比较与真实在线状态

γ0/γ1各14个性能记录已完整取回；每B保留两个运行值。γ1相对同域AR的吞吐比分别为B1/2/4/8/16/32/64的0.7271/0.8641/0.8448/0.8523/0.9008/0.9622/0.9794。当前效果为这两个cap下的小B收益方向与eager不同，下一步补齐γ2–7、实际执行独立compiled校准与控制器，不能把部分矩阵叫作完整翻转，也不能由跨时段、不同预热的eager/compiled差值隔离图执行因果。性能运行未记录逐步dispatch；启动日志证实图捕获完成，具体FULL/PIECEWISE和物理D直接观测仍来自独立诊断，cap32的比例不能移植到cap64性能。

`runs/F5_online_compiled_gamma01_r01`首次用真实初态恢复连续在线接受率：γ1的30,484个alpha_after及254个wave末状态全部匹配，γ0的32,512步及254个末状态也匹配；γ0无接受曝光，误差保持空值。γ1按B的条件accepted/tested约0.511–0.539，非空草稿比例约0.170–0.180；对逐步观测率的误差含请求混合与样本噪声，不把大B误差较小直接解释为估计器更好。原生静态臂没有decision-progress预测，availability状态也没有在线更新，不用其先验伪造方法有效性。两张局部图在`figures/F5_online_compiled_gamma1_partial_r01`，明确campaign仍partial。

compiled AR已经提供更强的原生执行参照，因此新增`prepare_compiled_controller_evaluation.py`和独立32格budget7校准、五臂及自然停止配置。选择仍仅来自开发测量0–63，评价测量为64–127；当前native矩阵沿用general warmup80–143，该历史曝光写入准备材料，不声称全pipeline从未见过评价prompt。新的校准和评价都显式warmup0/shape0。static γ0、全零原生表等合法选择均保留并实际执行；原生表引擎budget7不会冒充无草稿AR。

核实对应run目录全部不存在后，尚未运行的五个额外目标/草稿域及代码/摘要负载改为compiled主配置。每域的smoke、原生矩阵、成本校准和评价保持同域；旧OLMoE eager数据、其五臂及匹配准入诊断保留。原生路由捕获必须从引擎初始化开启，因此原生性能矩阵显式skip-trace；五个独立`*_compiled_routes.json`以repeats0、原生target valid/padding捕获生成F1诊断，不生成性能评估文件。诊断仍有计时/能耗记录，但不能进入F2或成本校准。配置变更记录为`runs/compiled_primary_configuration_change.json`，所有运行继续扣同一20小时预算。

06:49在实际核实38597仍waiting、无子任务且前序GPU矩阵未结束后，仅替换该自有CPU协调器。新`olmoe_evaluation_sequence_r02` PID45296等待35541，顺序为compiled成本/五臂/自然停止、原eager比较及所有额外域；源快照、启动记录和旧supersession均已取回。正在运行的40122及其GPU子进程未受影响。服务器上新准备入口与缩减域对接检查通过；它们只验证执行接线，不替代任何策略收益测量。

论文已转换为标准IEEEtran会议格式，8页含参考文献和六张实测图，编译与逐页检查通过。旧11页自定义版保存在`paper/versions/`；F3、自适应收益和多模型GPU结果仍标为待测，没有用排版完成代替科学验证。

## 完整compiled矩阵与实际在线观察（07:05完成）

`runs/olmoe_ngram_compiled_r01`的γ0–7全部成功，56格、112个性能记录完整取回，未排除任何性能文件；占共享GPU锁2223.78645秒。`runs/olmoe_ngram_compiled_matrix_analysis_r01`保留原始值、汇总、完整人口窗口和NVML counter-span辅助数据；`figures/F2_olmoe_ngram_compiled_r01`生成吞吐比、有效TPOT比及J/token三图，280个重算数值与原分析差为0。

| B | γ1–7吞吐比范围（同B AR=1） | 有效TPOT比范围（AR/推测） |
|---|---:|---:|
| 1 | 0.478835–0.727097 | 0.468768–0.718700 |
| 2 | 0.579847–0.864068 | 0.606794–0.887325 |
| 4 | 0.711088–0.857165 | 0.764339–0.906781 |
| 8 | 0.841169–0.874470 | 0.890535–0.960259 |
| 16 | 0.892255–0.918428 | 0.953292–1.006906 |
| 32 | 0.950430–0.962188 | 1.023254–1.062200 |
| 64 | 0.957835–0.979402 | 1.039752–1.086413 |

49个推测格的sum-of-wave-window吞吐中位数全部低于AR；15格有效TPOT比高于1（B16/γ4及B32、B64的所有正γ），但不能把小幅差异叫作显著或吞吐翻转。49格测量窗口J/token中位数也全部高于AR，最近为B64/γ1的1.002601倍。所有112个完整人口累计能量counter-span均通过原始端点检查；计入批次间隙后，49格人口吞吐仍低于AR、49格counter-span J/token仍高于AR。原始重复值、AR值及窗口定义与图同时保存，不把counter时间边界当runner墙钟边界。

当前效果为该低可用率ngram运行域中，compiled小B变慢但中B尚未转为吞吐加速，下一步继续完整预算校准与五臂实际执行，并沿既定模型/草稿/代码摘要负载检验可用率和完整成本边界。eager小B收益保留为另一执行域的观察；两域预热覆盖与运行时段不同，不能把差值单归CUDA graph或c0。物理路由诊断max-seqs32与性能64的差异仍披露，不直接移植D分布或计时。

完整F5入口`runs/F5_online_compiled_matrix_r01`覆盖8个engine和112个run，240254个host步骤的实际online状态均恢复；7168个终止请求步保留并删失接受率评价，未知终止和`1+accepted`/实际发出量不一致均为0。γ0无曝光，误差为空；原生静态臂没有实际decision-progress预测，availability EWMA未更新，均不伪装成控制器结果。标量α的分母是达到的测试位置数Σmin(accepted+1,scheduled)，不是所有草稿槽，不能自动等同各rank恒定几何接受概率；论文已分别写清几何假设、汇总EWMA和经验prefix/availability主路径。

后续原生策略表由direct-γ开发格选择，引擎最大预算G=7。更正（08:39）：此前将其图尺寸一概描述为G+1整倍数不准确；V1动态表会强制PIECEWISE，FULL分支的整倍数舍入不适用。选择分数仍不是同一捕获域下穷尽优化的表性能。各表将实际执行，AR和真正原生best-static独立保留。07:05已自动接续`olmoe_fullbudget_calibration_r01`；07:09现场核验γ0成功、γ1运行，未重复提交任务。

完整在线观察图在`figures/F5_online_compiled_matrix_r01`，固定展示γ1/4/7的误差、汇总接受率及非空proposal覆盖，保留42个原始run值；例图固定γ4/r0/B64且包含排空。8个engine共240254个逐步alpha检查和2032个wave末快照通过原分析；标量误差下降同时受batch观测噪声影响，不当作latent概率预测精度改善。复现脚本随分析放在`runs/F5_online_compiled_matrix_r01/render_local.py`。

`olmoe_fullbudget_calibration_r01/gamma0`的B1/4/16/64四个实际性能文件已取回，均完整结束。引擎production budget=7、post-proposal cap=0；可用/截断草稿token分别为5953/5970/5890/5857，每格两者相等，scheduled/observed draft均为0。此结果确认零验证长度仍实际支付草稿生产成本，不能与无草稿AR混用；完整γ0–7成本矩阵尚在运行，不提前给出控制器收益。

完整compiled主热图已进入`paper/main.pdf`第4页，正文、T1、证据入口和后续工作随实测更新；保留eager历史、独立诊断、F3与多模型待测边界。最终仍为IEEEtran Letter 8页，编译无未定义引用或排版溢出，全部8页已渲染检查。整合前版本保存在`paper/versions/characterization_pre_compiled_service.*`。本轮为新增实测图表与论文推进，整体三模型/两草稿源和实际政策评价继续，未完成的证据不以本轮交付替代。

## 完整预算成本校准与事后单步参考（07:21）

`runs/olmoe_fullbudget_calibration_r01`完成B1/4/16/64 × γ0–7的32个固定G7成本格，全部原始数据已取回；占锁923.14538秒。`runs/olmoe_fullbudget_cost_analysis_r01`生成32个完整同步host-step均值，无文件排除，包含36669个eligible步骤和680个wave。这些是每格单次校准运行的内部观测，不是数万次独立实验。按cap0–7的完整步成本范围为B1 10.830–11.443ms、B4 11.345–11.948ms、B16 12.139–13.349ms、B64 14.275–14.917ms；非单调小差尚不能解释为稳定物理规律。

`figures/fullbudget_cost_eager_r01/complete_step_costs.pdf`并列保留G7后截断成本与历史G=γ原生成本。旧表64个对应run已从原文件复算，数值差为0，但两组输出长度64/128、逐形状预热、观察代码和时间窗口不同，不能把差值叫作纯草稿或截断开销。当前效果为后截断控制器已有同预算完整成本输入，下一步实际执行既定五臂并检验成本向新请求/动态轨迹的迁移，而不是从校准表直接推断服务收益。

用户要求的事后单步参考已由`runs/derived/stepwise_prefix_eager_r01`落实：使用old r02的γ7固定轨迹和上述G7完整成本，14个run、28306个eligible纯decode步骤中，八个cap均有可识别prefix推进和成本覆盖。未来prefix参考与仅用过去观察的cold-prior重放，在B1/2/4/8/16/32/64的平均score差为0.086092/0.109864/0.068426/0.136073/0.055867/0.042845/0.081560 ms/mean-request-progress。它是固定轨迹上的诊断，不是实际TPOT、可部署oracle、理论上界或端到端加速；小差还依赖单次校准表中的成本细差，不能据此宣称在线信息无价值。两次raw汇总因相同prefix历史、同先验和同表而重合，不等于不确定性消失。

独立图在`figures/stepwise_prefix_eager_r01/stepwise_prefix_diagnostic.pdf`，不混入F3策略实测曲线。全部候选score、识别原因和原始run汇总保留，绘图与成本对照可由分析目录的`compare_and_plot.py`复现。两张新图已视觉检查。

新增`code/analyze_policy_cost_prediction.py`，只比较实际adaptive选中动作记录的`decision.total_step_ms`和同一步完整`host_step_ms`，按run报告误差并保留所有排除原因；缺预测不补0，不评估未选动作或声称动作排序成立。3项事件匹配/缺字段/错预算域CPU检查通过，真实adaptive分析仍待GPU运行。`reproduce.md`补充进程内安装、native关闭截断、disabled=cap0但仍生产及shutdown后卸载的实际接口。compiled独立校准已由52855等待共享锁，保留现有排队，不借用eager成本。

## 已完成引擎的资源口径（07:39）

`runs/engine_resource_summary_r01`汇总eager r02、compiled r01和eager G7校准的24个engine。逐文件原始快照复核一致：初始化后的PyTorch allocated约为19.982833/19.990769/19.982833 GiB，显式KV cap均6 GiB；启动日志均报告model loading 12.89 GiB。compiled γ0–7的graph capture日志依次为0.10/0.15/0.17/0.20/0.20/0.20/0.20/0.21 GiB，eager无该报告并保留空值。

此快照在engine内部加载/捕获完成、runtime attach后且应用warmup前采集；不是NVML驻留量、reserved或峰值，也不是纯权重。KV不能再次加到快照上，不用各口径之差分解graph、元数据或草稿占用。新增本地`code/summarize_engine_resources.py`保留所有匹配日志及上下文，多条model-loading报告不自行分配给target/draft。当前效果为已测域的资源表有真实输入，下一步将神经草稿实际加载结果按相同口径补入，避免将固定6 GiB KV误写成相同实际总驻留量。

## Compiled完整预算校准开始实测（07:55）

`olmoe_compiled_fullbudget_calibration_r01`已于07:51:44获得共享锁；γ0/1成功，继续其余cap。γ0的B1/4/16/64均完整完成64请求×64输出，原始文件已取回；其完整同步host-step成本分别为7.378121/8.139067/9.989018/12.683121 ms，eligible纯decode步骤分别3968/992/248/62。每格为一次校准运行，不把这些步骤作为独立重复。

γ0逐步核对为生产G7、验证cap0：四个B的available=discarded草稿token分别5818/5489/5655/5737，scheduled、observed和accepted草稿均0。这是后截断引擎，不能充作无草稿AR。日志证明48个PIECEWISE和48个FULL图捕获完成、报告0.21 GiB，计时步骤没有dispatch记录，不能据此填写实际FULL/PW占比。当前效果为compiled完整预算路径已实际运行，下一步等全表完成后执行原定五臂，继续保留独立的原生AR和best-static。

08:04全表完成，8个γ引擎均成功，32个性能文件及准备产物已取回。`runs/controller_preparation_compiled_r01/evaluation_commands.json`记录32格成本表SHA `305a7835a12caaa2dc68bb48fbebd07d772eb63e1b85012280c6973220d57271`，五臂命令全部生成；AR评价60478已经自动进入共享锁队列。

原生compiled开发矩阵按既定B1/4/16/64等权有效TPOT选择的固定γ为0，单阈值cutoff为0（全关闭），非单调表为B1–15→0、B16–63→4、B64→7；中间B的规则是已指定的lower-neighbor选择，不是中间B性能实测。AR与固定γ0仍分别执行；全零原生表仍用G_eng7图引擎，不能视为同一个AR执行域。选择来自direct-γ开发格，不冒充同capture-grid穷尽最优。旧native预热80–143与评价64–127有48个prompt身份重叠；新成本校准和评价预热使用0–63。实测开发/评价人口不重叠，但不声称整条流水线prompt-unseen。

## Compiled成本图与单步参考（08:10）

`runs/olmoe_compiled_fullbudget_cost_analysis_r01`从32个原始文件复算完整成本，0排除、36804个eligible host步骤、680个wave；除机器source路径外逐字段与协调器成本表一致，计时差0。B1/4/16/64跨cap范围分别为7.285029–7.521079、8.077651–8.732237、9.989018–11.184456、12.683121–13.796530 ms。单看成本最低cap为7/4/0/0，但这不是cost/progress最优，更不是统计上已确定的动作排序。

独立图`figures/fullbudget_cost_compiled_r01/complete_step_costs.pdf`同时保留同域native的两个原始值。全部112个native run重新分析，图中匹配4个B的64个run。G7成本相对native中位数差为−0.589442至+4.417563 ms；例如B1/cap0为7.378121对2.960558 ms。两campaign的runner/plugin/cost源码SHA相同，但输出长度64/128、general warmup0/80（shape均0）、预算配置与时窗不同；不是纯草拟或截断税。当前效果为大幅成本差已定位到不同完整预算执行域，下一步以相同评价长度实际比较AR、全零native表与后截断控制器，检验这些差异对服务目标的影响。

`runs/derived/stepwise_prefix_compiled_r01`只用compiled γ7轨迹与该域成本表，14个run、28432个eligible步骤均覆盖8个候选；896个终止请求步删失，unknown0，18958个成本查询为实测B、9474个为B间插值。B1/2/4/8/16/32/64的过去信息重放与未来prefix参考score差为0/0.062271/0.045120/0.227528/0.450931/0.277646/0.159022 ms/mean-request-progress。B1零差来自这张单次成本表中的cap7同时最低成本、且前缀推进弱最大，不证明全局在线最优；B32/64重放比分别固定cap7更差的结果也保留。

独立诊断图为`figures/stepwise_prefix_compiled_r01/stepwise_prefix_diagnostic.pdf`。这是固定轨迹、每run冷先验的事后单步参考，不是实际在线控制、F3 TPOT、加速比或理论上界；两个raw汇总因相同prefix轨迹和初始化而重合。两张图均完成视觉检查，未混入eager成本或轨迹。

## 五臂评价的前两臂实测（08:20）

`olmoe_compiled_controller_eval_ar_r01`与`olmoe_compiled_controller_eval_static_r01`均完成B1/4/16/64×2重复。每个perf包含workload records64–127对应的64个唯一source IDs，每请求128输出、共8192 token，全部正常以length结束；16个完整人口能量counter-span均有效。`runs/olmoe_compiled_controller_comparison_r01`已接收16个真实run，原始值和coverage均已保存，状态为pending_or_partial；另外3条记录表示策略输入尚不可用，不是对已有性能数据的排除，也尚未生成F3图。

由于开发选择的静态γ是0，这两臂实际engine参数、plan、五个关键runtime源码完全相同。`runs/olmoe_compiled_baseline_generation_check_r01`保存既有比较器的实际输出：跨臂相同B/repeat/source的512/512个128-token序列相同，各臂重复之间也为512/512；0输入错误、0失败。这是固定长度已观察序列一致性，不是自然停止质量或分布等价证明。

按B1/4/16/64，static/AR中位数比为TPOT 0.998118/0.999103/0.995163/1.012773，完整人口吞吐1.000989/1.000879/1.004521/0.989203，counter-span J/token 0.994698/0.997112/1.002674/0.967094。AR B1两个原始能耗为0.737672/0.678822 J/token，相差约8.67%；B64 static的r0较慢、r1接近AR，但能耗更低。它们描述同配置顺序运行中的局部波动，不合并成4个独立系统重复、不定义通用噪声阈值，也不归因于策略收益。counter-span覆盖跨wave间隙，但端点仍不等同runner人口墙钟。

当前效果为留出人口的真实基线已经取得，且显示小幅能耗/延迟差需要结合原始重复解释；下一步完成原生非单调表、原生阈值和adaptive三臂，再输出完整F3/F4/F6，保持自然停止检查和后续多模型路线。

## Compiled五臂完整策略结果（08:39）

`runs/olmoe_compiled_controller_comparison_r01`现为`measured_comparison_complete`：AR、所选static0、native unrestricted、native threshold、adaptive各8个run，共40个，0排除。每个run均为相同留出records64–127、64请求×128输出、8192 token，全部length结束；完整人口NVML累计counter-span全部有效。开发选择、实际成本表、源码和人口身份均由既有比较器逐项匹配。`figures/F3F4F6_olmoe_compiled_controller_r01`提供实际TPOT、累计counter-span J/token和CPU/截断图；原始两次运行点与中位数同时展示，未将共享引擎下的重复当作独立引擎重复。

按B1/4/16/64等权、使用每格两个run中位数的实际有效TPOT为：AR **7.714910 ms**，static0 **7.740541 ms**，native unrestricted **7.917877 ms**，native threshold **8.141005 ms**，adaptive **8.989071 ms**。这是执行后的服务指标，不是开发选择分数或事后oracle。static0和AR相同配置的局部波动继续保留，见前一节。

| B | adaptive/static0 TPOT比（越低越好） | 完整人口吞吐比（越高越好） | counter-span J/token比（越低越好） |
|---|---:|---:|---:|
| 1 | 2.227642 | 0.460511 | 2.393173 |
| 4 | 1.273748 | 0.745835 | 1.350436 |
| 16 | 1.033728 | 0.916799 | 1.055272 |
| 64 | 0.959581 | 0.948388 | 1.083353 |

当前效果为低覆盖ngram的完整预算自适应尚未改善既定等权TPOT目标；B64的请求TPOT改善没有转成吞吐或能耗改善。下一步继续已经启动的自然停止检查，并按既定链条实测神经草稿、其他模型、代码与摘要负载，检验生产覆盖和完整步成本的边界；不以换阈值替代这项检验。native threshold的B4-r1包含0.458317秒真实跨wave间隙（r0为0.046978秒），已完整计入人口吞吐和累计能量，不剔除、不解释成GPU空闲或控制器开销。

实际F6中，adaptive各run的平均choose调用为39.70–52.14微秒。按B1/4/16/64，discarded/available分别为0/0、0.422666/0.422666、0.731301/0.689627、0.430243/0.430243（两次原始值）；wrapper占host计时约0.729%–0.999%，包含choose与截断且嵌套于scheduler计时，不能相加或从墙钟扣除。弃草稿是token工作量，不是已测GPU浪费时间或能量。available为0时该比例为空，不能将原生全关闭臂的0/0画为0。

## 原生动态表的执行边界修正

实际安装vLLM 0.26的`config/vllm.py::_maybe_override_dynamic_sd_cudagraph_mode`在V1、非空`num_speculative_tokens_per_batch_size`且模式含FULL时改为PIECEWISE；`config/compilation.py`只在V1 decode FULL、uniform query length>1时进行G_eng+1整倍数舍入。两native策略日志均证明51个PIECEWISE捕获完成、无FULL捕获；G7后截断校准则为48个PIECEWISE和48个FULL。因此max G_eng均为7不代表图模式/网格相同。性能run未记录逐步dispatch，不能据启动日志声称实际P1/P8分布或归因warmup/服务差值。

两native表共43,504个记录步骤的`proposal_budget_next`均与实际scheduled requests查表一致，零错配。unrestricted每次repeat在B1/4无草稿，在B16有2843个scheduled/observed draft、接受693，在B64有5269个draft、接受942；controller discarded始终0。threshold所有B均无草稿。表控制下一次生产预算，跨档当步仍可消费上一档已生成草稿，不应误标为漏截断。零预算在ngram匹配主体前返回空，但proposer包装调用仍存在；不称完全卸载或CPU成本为零。

上述比较是各策略实际配置下的完整效果。它同时包含生产边界与图路径差异，不能独立验证选择规则的价值；原生全零表也不能当作AR的同配置重复。此处明确更正此前过宽的G+1图网格描述，历史原始输出、评价选择和不利结果均保留。

## 实际选中动作的成本迁移（08:40）

`runs/olmoe_compiled_policy_cost_prediction_r01`已由真实adaptive运行生成，8个run全部有效；19,759个原始步骤中，19,191个eligible步骤进入同事件预测/完整host-step比较，170个prefill和398个含终止请求步骤单独保留。按B1/4/16/64，两个run的MAE分别为0.3526/0.3647、0.3932/0.4030、0.4234/0.4347、0.3938/0.2698 ms；预测均值比实测低0.0697–0.3422 ms。当前效果为选中动作的完整成本迁移没有出现数量级错误，下一步结合实际online推进估计和新草稿域判断完整cost/progress目标；不由选中动作误差声称未选动作排序正确或服务增益成立。

实际19,589个decode步骤中，γ0/4/7分别为467/4527/14595步；全部19,759步的下一次生产预算仍为7，可用草稿56,791＝截断24,167＋验证32,624，接受6,534，fallback为0。B1两次均选择γ7，符合该张单次成本表中γ7成本最低且有最大可用前缀的组合，不以此证明全域最优。状态从真实开发预热后的α=0.572631开始，连续跨wave/B/repeat，r0升B、r1降B，不是每run冷先验。

修复了本地分析器campaign入口把`gamma7.log`当目录的文件枚举问题，仅增加目录过滤；修复后campaign和engine子目录入口的实际结果一致。runtime源码与已运行数据未更改。

## 自然生成的终止与固定任务抽检（08:42）

`runs/olmoe_compiled_natural_generation_check_r01`保存实际compiled AR/static0/adaptive的三组自然停止结果和tokenizer解码。共12个cell（B1/8×2repeat×3角色），每cell完整16条固定请求，零输入错误、缺失、失败或未结束；允许EOS但最多256输出。B1每次AR/static为4 EOS＋12限长，adaptive为2 EOS＋14限长；B8各臂每次均3 EOS＋13限长。到达token上限不是自然任务成功。

同配置repeat的16/16序列各自一致；跨AR/static相同B/repeat为B1 14/16、B8 15/16，跨AR/adaptive为1/16、3/16。自然停止长度与内容存在差异，不能从无引擎失败或固定长度检查推断内容/分布等价。固定任务样本来自事前的records64–71（source262/264/269/284/288/296/297/305）、B1/r0，不按结果筛选。

八条解码样本围绕对应主题作答，但存在各臂共有的直接可判定问题：288三臂正文均为60个空白分词，超过50词约束，且加入提示未提供、未经核验的个人业绩；305三臂均在未被请求时先给提示；296与297的代码输出都在256上限前未实现要求的完整功能。其他长答案也多被限长截断；未核验产品、医疗、法律或引用数字的事实正确性。当前效果为完成了预定小规模任务抽检，但证据不足以支持“质量不变”或输出分布等价；下一步保留这些样本并在既定新草稿/代码摘要评价中继续报告自然终止与任务结果，而非将token差异直接标成采样错误。

## 实际adaptive的F5状态与推进预测

`runs/F5_online_compiled_adaptive_r01`完成8个raw run的真实在线分析，19,759个逐步alpha与170个wave末完整state均匹配，campaign_partial=false。19,589个实际decision progress记录中19,419步可评分，共58,152个eligible请求步；512个terminal请求步删失，unknown/recovered均0，`1+accepted`与真实client发出量不一致为0。初始预热accepted/tested为1135/1804、最终7378/14136，增量6243/12332一致；这与含terminal的原始草稿接受总数口径不同。

按B1/4/16/64，实际decision-progress加权MAE两次原值为0.231269/0.230631、0.176895/0.177053、0.060344/0.069529、0.066154/0.066131 tokens/request-step；pooled alpha MAE为0.305017/0.306748、0.298563/0.298598、0.225346/0.218632、0.134798/0.137456。前者按请求步、后者按实际tested positions加权，是相对有噪声已实现观测的误差，跨B曝光与动作不同，不能等同latent概率精度提高或策略收益。

`figures/F5_online_compiled_adaptive_r01`新增真实预测误差三面板与固定r0/B64四状态轨迹，保留drain、每run原始值及所有可复现脚本。availability每rank曝光58,152；prefix的rank1–7曝光为6634/6634/6612/6584/1800/1796/1778，截断后高rank少曝光的事实保留，未补造未验证前缀。这些实际记录与此前native只有观察或事后重放的数据分开。

自然AR/static的同配置核实与8条逐项人工记录见`runs/olmoe_compiled_natural_generation_check_r01/manual_inspection.md`。两臂plan/engine参数/seed/采样与五个关键runtime源码相同；B1的source264/320、B8的source297发生token差异，但对应长度及终止一致。源码与日志不足以归因，未标成上游错误。固定样本prompt+256≤348<1024，支持实际输出上限截断而非context耗尽。

## 论文与图表更新

`paper/main.pdf`已将实际五臂F3、F4/F6数值、adaptive F5核验、自然停止计数和固定任务限制纳入正文，并更正native动态表的图模式边界。最终IEEEtran Letter为9页，无未定义引用或overfull；全部页面与5张新增独立图均已渲染检查。F3在第9页，原始两次运行值保留；独立F4/F6及F5实际状态图保持完整分析入口。当前多模型/神经草稿/代码摘要证据仍待接续，论文工作稿不表示整体研究目标已经完成。

F3论文导出随后改为从原PGF中仅删除一行重复脚注，数据及其余绘图命令不变，原始完整图保留；避免PDF裁切后隐藏文字仍参与文本提取。新稿仍为9页，前8页与上一版渲染逐像素一致，第9页视觉与文字检查通过。复现入口为`paper/export_f3.py`，源图/输出哈希记录于`paper/figures/F3_executed_policies_export.json`；这是导出修正，不改变实验结果。

## 多域F2组合入口与eager评价接续

新增`code/plot_matrix_panels_pgf.py`，为后续六个compiled模型/草稿域提供逐域原轴、同指标共同色标的F2组合入口；各面板比值仍使用自己的same-B AR，缺测/重复不足格留空，不插值。以既有OLMoE eager与compiled的224个run、112格检查导出，560项沿用汇总值差为0，吞吐比/TPOT比/测量窗口J-token三张预览图均完成视觉检查。每格保留数值、重复数移至副标题；完整原值和来源随CSV/JSON输出。此项仅增加图表组合能力，不构成新的模型实验或受控跨执行域证据。

实际已完成的`runs/controller_preparation_eager_r01`已取回，选择仍为static γ6、原生三段表7/5/6、阈值全域γ6，并使用独立32格eager G7成本表。`reproduce.md`已记录五个真实run名和分析参数；等待五臂全部完成后再生成比较，不预先创建含缺失结果的结论。

## Eager AR完成与连接中断（10:11）

`runs/olmoe_controller_eval_ar_r01`已完整取回：B1/4/16/64×2重复，共8个run，0排除；每个run覆盖同一留出records64–127的64个唯一source IDs、8192输出token，8个完整人口NVML累计counter-span均有效。此组实际持锁309.947884秒。原协调器在09:43:59已启动static6，之后SSH断开且新连接被端口拒绝。当前仅AR数据已核实；不能将失联解释为实验结束、进程退出或静态臂失败。恢复连接时先检查原协调器状态、进程命令和已有产物，再决定接续动作。

09:45:47.276–09:46:14.281有一次27.005秒的CPU tokenizer扫描，记录于`runs/aux_workload_feasibility_r01/cpu_execution.json`；当时static6 campaign正在进行。扫描设置单CPU affinity `[0]`、nice19、空CUDA_VISIBLE_DEVICES，未加载模型或使用GPU，但这些设置不证明CPU干扰为零。保留其真实时窗和所有原始性能数据；没有足够时钟锚点将它精确对应到性能cell，不减去估计时间，也不据此事后排除运行。

## 三目标共同代码/摘要人口与待部署实验

旧80条辅助输入不足以同时容纳64条开发和64条留出评价。一次CPU扫描复用了已有LongBench源数据的完整500条LCC与200条MultiNews，以及三个目标的实际tokenizer；未重新下载、截断、复制或按答案/性能筛选。三个tokenizer共同可容纳的完整样本数在2048/3072/3584上限下分别为：LCC **81/280/337**，MultiNews **71/123/143**。因此按预先检查的候选长度取最小可行上限：LCC3072、MultiNews3584；每个任务取共同可行集合中原始顺序的前128条，前64开发、后64留出。

`code/build_common_aux_workloads.py`从此次扫描已保存的完整候选构建六份`configs/{lcc,multi_news}_{olmoe,qwen3_awq,granite3b}_common128.json`，768条记录均与候选逐字段相同；来源、tokenizer、配置哈希和分区见`runs/auxiliary_population_preparation_r01/manifest.json`。同一任务跨目标保持原始source身份、完整正文与参考答案；OLMoE/Qwen使用其native chat包装，Granite使用明确的raw continuation，因此不宣称不同模型输入token数或任务行为相同。每目标两种草稿源复用同一文件。旧80条输入、旧计划和历史结果不变。

`runs/auxiliary_resource_plan_r01`依据实际目标/草稿KV几何和各64条人口的完整prompt+128输出+G7预留进行容量核算；OLMoE配置与扫描SHA一致。按所查16/32/64 block候选，B64最大片段载荷估算为OLMoE 37.90625 GiB、Qwen 12.07056 GiB、Granite 15.53809 GiB；固定worker KV候选分别为48/24/24 GiB，同family两种草稿源一致。估算保留目标与草稿attention层的共同占用，不把逻辑载荷叫实测分配或峰值；实际cache分组、block大小、显存峰值与B64行为仍待首次GPU执行。

`code/prepare_auxiliary_campaigns.py`已生成12域×3份配置以及`configs/auxiliary_campaigns.json`。每域先跑B1/4/16/64、γ0/3/7的native开发矩阵（2次、128输出），再跑本域G7完整成本（1次、64输出），最后执行AR、所选static、native非单调表、native阈值、adaptive五臂留出评价（各2次、128输出）。新域各自选择基线和校准，候选caps为0/3/7，不借用ShareGPT或旧80条人口的表。固定长度阶段仍是受控工作量评价，不能替代自然任务质量。

补充CPU协调器`code/run_auxiliary_sequence.py`先等待原`olmoe_evaluation_sequence_r02`完成，随后按12个native矩阵、逐域校准与五臂评价串行执行；每个GPU子阶段继续使用已有共享锁和**同一个累计20小时预算**。总计84个GPU campaign是待运行命令数量，不是已完成结果或能在剩余预算内完成的承诺。输入哈希与`--describe`已检查；新输入、36份配置和补充协调器均只在本地，尚未上传或启动。当前效果为三模型×两草稿源×三负载所缺的共同代码/摘要人口及接续入口已补齐，下一步恢复连接、核实原任务，先取得既有多模型结果，再部署这些剩余工作负载。
