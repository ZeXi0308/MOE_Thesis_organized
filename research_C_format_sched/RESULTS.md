当前主张：采用 SM120 四格式刻画型定位；Qwen首层四格式56点已完整，观测到FP8→AWQ→NVFP4→FP8的最快路径变化；完整质量和服务收益仍未得到证据。
关键证据：两模型形状的 BF16/FP8 共56点已回传；FP8在全部28对延迟比较中更快；校准Qwen NVFP4与AWQ首层各14点完成，实际路径为SM120 CUTLASS FP4 OMMA与Marlin BF16 HMMA，28个v2能耗窗显示延迟与能耗排序可以不同。
最强替代解释：已测排序可由完整投影与辅助项的不同代价平衡解释，不能用辅助开销统一解释两种4-bit路径；默认 kernel 配置、缓存和权重来源差异仍限制外推。
下一动作：遵循用户新优先级，立即交付完整16层模型前向、短自然生成和FP8/AWQ/dual-phase最小serving对照；不再扩展kernel表或微基准。先用真实64×128 AWQ校准形成同母本接入版本，质量未确认。
资源状态：westb:50328，GPU UUID 12433ef7…，使用既有共享锁，总预算约20小时；64×128校准排队r01超时且无GPU工作，已本地核验；r02（PID36199）现已进入有时限内核候锁队列。完整模型与最小serving尚未实跑，自动接续/逐组回传已启动。Qwen BF16下载PID21898为接入预留空间临时SIGSTOP，其余下载继续；接入完成后恢复同一下载进程或核实失败后续传。

# 任务和当前决定

完整用户范围见 [TASK.txt](TASK.txt)：两模型四格式 F1/F2/T1、lm-eval F3、vLLM 双格式方法、七臂 serving F4–F7、LaTeX 初稿及复现入口。此文件的阶段进展不替代完整交付。

限定查新见 [paper/related_work.md](paper/related_work.md)。DynaExq v4 已依据运行时路由热度改变专家精度并评价吞吐，因此触发用户的刻画型定位分支。保留双格式实现与实测评价，不主张首次运行时精度切换。论文骨架已编译，但不是可投稿的完成稿。

## 已核实环境

- RTX PRO 6000 Blackwell Server Edition，97887 MiB，compute capability 12.0；driver 580.95.05。
- vLLM 0.26.0、Torch 2.11.0（CUDA 13.0）、Triton 3.6.0、FlashInfer 0.6.14、nvidia-ml-py 13.615.71。lm-eval 0.4.13 已在独立 tools_venv 安装并通过后端导入；校准用 llmcompressor 0.14.0/CT 0.19 独立 calibration_venv，基础推理环境 CT 0.17 保持不变。
- NVML 累积能量计数器可读取且递增。50 ms 采样仍同时保存。无可见计算进程时 utilization 固定100%，该遥测异常尚未解释，不以它判断占用或瓶颈。共享锁与实际进程另行检查。
- vLLM 首次导入失败：当前 Torch wheel RECORD 之外残留3个旧 `_inductor/kernel` 源文件，与新版子目录重复注册 Triton 模板。已逐文件备份、核对不属于当前 wheel 后移出导入目录；没有改写 wheel 所拥有文件，导入恢复。失败日志及修复备份均在本地 `runs/`。
- 现成 OLMoE checkpoint 的专家形状是 d=2048、intermediate=1024、E=64、k=8、16层。Qwen3-30B-A3B 是 d=2048、expert intermediate=768、E=128、k=8、48层；6144是另一配置字段，不能当专家宽度。

## T1：当前路径证据（Qwen四格式首层完整，非完整模型验证）

| 格式/方式 | 已取得证据 | 状态与范围 |
|---|---|---|
| BF16，显式 Triton | profiler 记录 `fused_moe_kernel`、路由对齐、SiLU、加权归并 | 实际执行；合成权重；JIT PTX 中已核实 BF16 mma.sync 指令 |
| FP8 W8A8，显式 Triton | 以上路径及 `scaled_fp8_quant_kernel_strided_dynamic`、`segmented_max_reduction_strided` | 实际执行；每专家 tensor 权重 scale、动态 per-tensor 激活；不是 auto 后端比较 |
| NVFP4，CUTLASS | 实测SM120 block-scaled profiler符号精确匹配当前主库SASS，含64处静态FP4 OMMA | 首层uniform/Zipf共14点；原trace未保存库SHA，历史二进制同一性不可独立证明；非动态计数 |
| NVFP4，显式 B12X | installed FlashInfer 条件不满足；源码 oracle 默认亦不自动选 B12X | 当前安装不可用，不外推为 SM120 硬件不支持 |
| AWQ W4A16 | 校准checkpoint原生Marlin；B1实测模板与SM120二进制反汇编符号精确匹配，包含HMMA.16816.F32.BF16 | uniform/Zipf共14点完成；3点按显式BF16-roundoff条件接受且非bitwise；非动态计数 |

首个 B=8、OLMoE形状的 eager CUDA-event smoke：BF16 0.5568 ms、FP8 0.3068 ms（各5个进程内重复的中位数）。这两个单点只用于验证工具接通，尚不能支持策略收益或最佳格式结论。默认配置警告明确显示没有本设备专用 MoE 调优表；最终比较需要合理调优。

FP8 首次3秒窗口：累积计数器938.204 J，50 ms 功耗积分814.367 J。两种读数不一致已保留，主指标按用户要求采用计数器；传感器更新窗口与短测量边界仍待检查，不将差值直接解释为算法能耗。

## 数据保存在本地

唯一原始运行入口为 `runs/<run_name>/`，包含命令、状态、提交源码、stdout、JSONL、原始功耗样本及 Chrome trace。终态组的远端 SHA256 与本地文件逐一核验，回执见 `runs/local_backup_receipt.json`。失败组不覆盖。远端位于 `/root/autodl-tmp/research_C_format_sched`；未删除远端数据。

7月三个复用文件经 GitHub connector 以确切 blob 回收，出处见 `ref_july/SOURCES.json`。直接 clone main 因网络超时失败，未声称完整克隆；附件的旧路径在当前 main 读取为404。

## Serving 与真实路由输入已制备（CPU-only）

`datasets/serving_olmoe_r01/manifest.json` 记录已完成的 OLMoE tokenizer 数据制备。来源固定为 ShareGPT `anon8231489123/ShareGPT_Vicuna_unfiltered@192ab2185289094fc556ec8ce5ce1e8e587154ca` 的原始 JSON，以及 LongBench `zai-org/LongBench@5e628be450b7e67fb7ae6e201bd6d8f7056f7672` 的 `data.zip`；完整 SHA256、命令、模板和 tokenizer 快照均保留。LongBench 仅取 qasper/multi_news，使用本研究明确记录的任务提示，不是官方 LongBench 质量评测。

固定 seed=20261016，完整扫描 94,545 条源记录：ShareGPT 94,145 条中 69,656 条合格，另有重复 22,930、缺少 human 轮 1,295、首个人类轮为空或非文本 58、超上下文 206；LongBench 400 条中 212 条合格、短于长输入下限 25、超上下文 163。所有记录及排除原因均在 `population.jsonl`。从两类合格人口各抽 128 条（LongBench 为 qasper 49、multi_news 79），另分别保留 69,528 与 84 条合格但未抽中记录。ShareGPT 只用首个人类轮；两类均按保存的 OLMoE 原生 chat template 渲染，tokenize 使用 `add_special_tokens=True`，不截断。预留 128 个输出 token 后要求总长不超过 4,096；长输入下限为 1,024。选中 prompt 长度为 ShareGPT 17–2,837、LongBench 1,051–3,952 token。

`serving_natural.jsonl` 与 `serving_fixed_workload.jsonl` 各含同顺序的 256 个请求：前者为 greedy、允许自然 EOS、上限 128 token；后者忽略 EOS 并固定生成 128 token，仅供控制生成长度的工作量实验，不是已记录 token 的回放，也不证明质量等价。JSONL 不预埋到达时间，由 `bench_serving.py` 按固定 seed 和指定 λ 生成 Poisson 到达。`routes_natural_32.jsonl` 是两类各 16 条的固定子集，供 32-token 上限的自然生成路由捕获。消费者仍须保留 EOS/cap/error/timeout/unfinished；这些终止结果尚未产生。

两个完整源文件合计 786,770,471 字节，以及 9 个制备 payload 文件合计 36,218,172 字节，已回传本地并逐项核对哈希；manifest、日志和状态元数据亦与远端匹配，回执为 `datasets/serving_olmoe_r01/local_backup_receipt.json`。已有 CPU 检查确认两套 serving 文件的请求与到达顺序一致、32 条路由输入可读取、全部 256 条保存 tokenizer 重算长度一致。Qwen tokenizer 制备也已完成，9文件44,250,424字节已本地核验，显式关闭thinking，详见下文。真实路由、serving和任务质量仍未运行。

## 尚未完成

- OLMoE低bit曲线与两模型合理调优、同权重/真实负载验证；已完成两模型BF16/FP8共56点和Qwen单层四格式完整56点，不能替代这些未测范围。
- 校准量化产物、真实模型路由分布、四格式及混合生成 lm-eval。
- 双格式真实 serving 接线、步级与专家级策略、相位基线、额外 launch 与选择开销。
- ShareGPT/LongBench 七臂两次 serving 以及50/100ms TPOT SLO下完整人口比较。
- F1–F7成图与有实测结果的论文初稿。待数据支持后选投稿格式，不承诺录用。

继续本课题。不能因近邻触发刻画型定位或微基准成功，将完整目标标记为完成。

## 首轮曲线与功耗窗口诊断

`twoformat_survey_r01` 56/56点完成，整组319.001秒，原始5次时延与所有能耗样本已本地校验。图见 `figures/F1_latency.pdf`、`F2_energy.pdf`；数值见 `figures/summary.csv`。两个格式均为合成权重、显式Triton、CUDA graph replay，graph/eager输出逐点相同。FP8相对BF16时延倍率：OLMoE 1.14–1.86×，Qwen3 1.16–1.92×；无已观察到的BF16/FP8交叉点。不外推为四格式最优或服务收益。

56个能量窗口的采样积分均比累计counter低9.30%–17.71%。`power_window_r02` 单工作点诊断显示：空闲后恢复负载的首窗口为−18.915%，连续负载后两窗为+1.376%和+0.191%。这与[NVML官方文档](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)所述Ampere之后powerUsage返回1秒平均功率一致；它支持窗口滞后解释，但不是外部电表校准。首版F2因此标为3秒负载恢复窗口，尚非稳态曲线。后续代码在空闲基线测量后增加3秒持续工作量预热；保留原始点，不静默改写。

`runs/triton_compiled_paths/`保存实际JIT缓存PTX和元数据：BF16变体含`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`，FP8含`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`。与profiler的fused kernel执行记录共同支持Tensor Core路径；不当作硬件指令计数器测量。

NVIDIA校准Qwen NVFP4固定版本为`2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`。首分片下载仍在运行，所需首层1536个tensor全部字节已到达，已提取339,741,696字节供转换。此副本仅含首层专家，不能加载为完整模型；完整分片到达后仍须验证选中tensor一致。发布者未报告量化所用base的确切commit，元数据明确记录未知，不以当前base HEAD冒充来源。

## NVFP4实际执行与资源接续

`qwen3_nvfp4_export_r01`原生转换成功，339,740,672 tensor字节的artifact已回传，本地SHA256 `af0e1e505ba8e03b9d75cf68cfc5dbad36af4203089533633a657c8b429fb8e9`。`qwen3_nvfp4_smoke_r01`实际跑SM120 CUTLASS block-scaled FP4 grouped GEMM（不是Marlin回退），B=8 eager中位0.1978523ms；不得与graph-replay图直接计算收益。

`qwen3_nvfp4_numeric_r01`对15个选中专家的gate/up、down、block scale、global scale和activation inverse均与源tensor一致。小型构造输入对独立W4A4参考relative L2=0.01646、cosine=0.999866；交换gate/up负对照relative L2=0.541。只支持转换/接线合理，不代表lm-eval质量通过，也不是bitwise正确性证明。

截至本次回收，11个终态目录200个文件、343,075,494字节均按远端SHA256本地核验（另有源码/PTX等参考文件）。总持锁时间405.783秒，含初始化与诊断；约20小时预算远未用尽。

`qwen3_nvfp4_survey_r01`取锁失败、未启动GPU。现场确认共享锁由其他课题 `research_A_horizon` controller8450/worker8453持有；不抢占、不修改其任务。本线没有后台GPU候卡，后续先核实该整组终态再启动新的r02目录。原NVFP4首分片下载PID5954仍存活并持续写入，不能重复下载进程。用户同意自己操作磁盘扩容，最近df仍为50GB总容量，尚未认为扩容已完成。

## 来源核验与质量环境

完整 NVFP4 首分片 5,002,180,600 字节已完成，SHA256 `15bb083c92a763a643972134681a65e0953122df749fb0d236ca905e78e709bd`。先前首层1536个tensor逐字节哈希均与完整分片一致，见 `runs/checkpoint_probe/completed_source_verification.json`；原始首层副本也已回传本地 `checkpoints/qwen3_nvfp4_layer0_subset/` 并重新逐tensor校验。原始提取时的 pending receipt 保留，不回写历史。

现有 OLMoE 9个模型与tokenizer文件均与 `allenai/OLMoE-1B-7B-0924-Instruct@7f1c97f440f06ce36705e4f2b843edb5925f4498` 快照同inode；后续四格式以此为同一base。lm-eval后端导入需要显式已安装CUDA13库的 `LD_LIBRARY_PATH`，不是另装CUDA。独立校准环境解决 llmcompressor 新版与推理环境 CT 版本不一致，尚未运行校准或质量任务。

独立校准环境导入成功：llmcompressor0.14.0、CT0.19.0、Transformers5.17.0；尚无校准产物。512条UltraChat train_sft固定样本已本地保存并验证（revision `8049631c405ae6576f93f445c6b8166f76f5505a`，JSONL SHA256 `878fb27f655b87d5da7d6e5eda403c47b4ad5e48d4643a0be5e1c179d8923c5b`）。生产进程写完数据后在解释器finalization报PyGILState_Release并以134退出；此失败保留，独立检查确认512行完整、哈希一致，未重跑或覆盖原始数据。

双格式v2已有FP8 Triton/AWQ Marlin native expert_map代码，未GPU验证。生产安装还必须验证activation basis：AWQ可能同时缩放norm、router和expert，来源相同并不自动说明两专家输入坐标相同；现有公共sidecar不得直接用于同模型质量或serving主张。

## NVFP4完整首层曲线与AWQ路径

`qwen3_nvfp4_survey_r02` 14/14点完成，持锁176.321秒，242个运行文件累计344,240,130字节已本地SHA256核验。全部CUDA graph/eager输出一致、输出有限。与首轮相同shape、routing和graph模式的FP8相比：uniform/Zipf都在B=8/32/128/512/2048观察NVFP4更快（FP8/NVFP4分别1.094–1.461×、1.150–1.414×），B=1与8192为FP8更快。只定位相邻已测点的排序改变区间[1,8]与[2048,8192]，不插值估阈值；BF16/FP8合成权重与NVFP4校准首层的来源差异保留，尚无调优强基线/质量/服务结论。新图 `figures/F1_qwen_three_formats.pdf` 和 `F2_qwen_nvfp4_energy.pdf`。新能耗协议的采样积分相对counter差异为−0.546%至+3.000%，中位+0.098%；不与旧能耗窗口硬比。

AWQ首分片5,001,524,144字节已固定版本完成（SHA256 `435415ffb72d70f56c12e53f9dceafa6be15b92f6365dda19fe58f85e25a9817`）。`qwen3_awq_export_r02` 原生Marlin转换及数值运行完成，产物311,427,072 tensor字节；对独立源signed-int4解码FP32参考relative L2=0.004361、cosine=0.999991，交换gate/up负对照relative L2=0.5656。仅为接线数值诊断，无质量通过主张。产物回传曾在扩容重连期间中断，已续传并完成本地hash核验；当前14个终态目录272文件共655,920,948字节均通过，另有339,741,696字节首层原始tensor与校准数据等独立副本。

完整Qwen下载原controller PID2506，子进程BF16 2509、AWQ 2511、NVFP4 2512；`runs/qwen_full_download_r01/status.json`记录初始剩余需求85,901,655,296字节和20GiB保留量，均仅网络/CPU，不持GPU锁。下载复用已完成的4-bit首分片，结束后按来源LFS SHA256逐片核验再登记manifest。连接断开必须先查原PID/文件，不能重复起下载。

完整五项质量数据已预载：HellaSwag10042、ARC-Easy2376、ARC-Challenge1172、WinoGrande1267、PIQA1838个评测样本。Winogrande初版自定义校验误把共享文本续写当整数gold，r01失败保留；r02按lm-eval真实multiple_input语义（doc_to_text为gold）修复通过，未修改任务/数据。尚无模型准确率结果。

## 既有profiler的组件解释

`runs/profile_component_analysis_r01/`按28个既有Qwen FP8/NVFP4 Chrome trace分解CUDA事件，不新增GPU运行。在B=8192 uniform，NVFP4两个投影合计0.706ms，比FP8的1.163ms短；其他kernel合计1.682ms，比FP8的0.646ms长，插桩总和因此为2.388ms vs 1.810ms。Zipf同向：投影0.804 vs 1.349ms，其他1.670 vs 0.636ms。NVFP4大批次辅助项以显式row shuffle、activation/quantization、output cast/mul/reduce为主。B=1时投影与辅助两部分都较FP8长；B=8/2048投影省时仍覆盖辅助额外耗时。

因此，当前大批次排序改变不能解释为FP4投影计算本身变慢。此结果仅来自每点一次instrumented graph replay的事件分类；CUDA duration总和不是无插桩时延或DRAM计数，也没有证明融合这些操作能获得多少实际收益。全部trace只有一次CPU cudaGraphLaunch，GPU kernel数NVFP4为15（B=1为16），FP8为10（B=1为9）。

论文草稿现为11页，完整四格式网格、能耗口径、T1指令证据及未测生成协议已同步；仍明确缺少完整质量与服务对照。组件图见`figures/F1b_qwen_components.pdf`。真实BF16/原生在线FP8首层导出器已完成CPU来源/shape检查；GPU导出未运行。双格式worker在模型加载后、KV预算确定前安装并计入两套权重常驻字节；新增基于真实调度位置与prompt边界的phase接线，7项CPU边界检查通过，仍待GPU生成验证。

## AWQ部分曲线和非逐位重复性

`qwen3_awq_survey_r01`持锁66.073秒，uniform B=1/8/32/128/512成功，median为0.026637/0.102474/0.232315/0.243316/0.284486ms；B=2048的graph/eager逐位相等断言失败，完整失败记录保留并本地校验。约419万输出中509个不同，最大绝对差0.0009765625。不能将其静默标为通过，剩余Zipf点当时未运行。

随后`qwen3_awq_graph_diagnostic_r01`持锁8.338秒，对uniform/Zipf、B=2048/8192分别执行4次eager和4次graph。eager/eager最大relative L2为2.821e-5，graph/graph为2.786e-5，graph/eager为2.638e-5；Zipf B=2048逐位相同。支持差异并非graph独有；尚未定位底层归约顺序。源码明确传入use_atomic_add=False、use_fp32_reduce=True，不能未经证明归因atomic add。

后续明确采用`--graph-output-check bf16-roundoff`：全部输出有限、relative L2≤BF16 epsilon/8且max absolute error≤epsilon×max|reference|；保存逐位是否相同、误差及判据。这只是捕获/重复数值检查，不替代独立转换参考或模型质量。旧五点均通过更严格的逐位检查，无须重跑。r02/r03因共享锁占用未启动GPU；r04等待600秒后超时，r05等待1800秒后超时，均未取得锁、未启动GPU，终态已回传校验。r06后来完成剩余九点，见完整结果小节；保留所有前序状态。

截至这次回收，19个终态组349文件共656,893,472字节通过本地SHA256；首层原始NVFP4副本和已回传的ShareGPT原始672,837,942字节另计。总持锁约669.365秒。共同输入基保存CPU toy在真实calibration_venv通过，日志`runs/quality_setup/common_basis_cpu_roundtrip.log`；完整OLMoE校准、真实AWQ/CT-NVFP4新模型导出及F3仍未运行。

## AWQ Tensor Core路径证据

`runs/awq_compiled_paths/evidence.json` 将B1 uniform实测Marlin profiler名称与已加载`_moe_C_stable_libtorch.abi3.so`内SM120符号逐字demangle匹配。该唯一函数反汇编含8处`HMMA.16816.F32.BF16`，支持INT4权重解码后以BF16 Tensor Core计算的W4A16路径。保存二进制SHA、原profiler文件SHA、命令、SASS及工具stderr；5个证据文件均回传并按远端SHA核验。静态指令出现次数不是动态执行次数/利用率计数，也不能据此推断归并原子操作。

## 两模型服务数据与真实单格式入口

Qwen固定版本tokenizer元数据已取得，`datasets/serving_qwen3_r01`按同一份原始数据及筛选规则制备256条主请求和32条路由prompt；使用`enable_thinking=false`的官方聊天模板，保留自然EOS与forced128两个口径。9个输出文件44,250,424字节已回传本地并逐项SHA核验，消费者CPU读取通过；尚未生成。两模型的0/.25/.5/.75/1 LongBench请求份额变体也已本地生成，每组128条；份额按来源类别定义，不是GPU prefill时间，ShareGPT中仍有少量长请求。

`SingleFormatMoEMethod`及worker的single配置只持有一套真实expert artifact，直接调用相同原生backend，不计算直方图/相位。FP8-only/AWQ-only可与双格式臂共用相同master非专家参数，且按单套真实占用确定KV容量。真实vLLM0.26环境CPU导入及9种配置接口检查通过；尚未GPU加载或生成。质量与服务入口记录host/sidecar身份、独立安装报告，并区分expert FP8与全模型FP8。默认五项lm-eval均为选择题打分，不能作为decode混合生成质量证据。

## NVFP4静态指令与自动本地回收

`runs/nvfp4_compiled_paths/`已保存实际profiler符号对应的当前主库`_C_stable_libtorch.abi3.so`单函数SM120反汇编，含64处静态`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`，支持FP4 Tensor Core路径分类。5个证据文件均已本地SHA核验。原profiler未保存库SHA，不能独立证明历史二进制同一性；静态指令数量不代表动态执行次数或利用率。

本地`code/sync_terminal_run.py`等待指定运行的终态SHA清单后自动rsync并校验，失败运行同样保留；只复用现有SSH control socket，不保存口令、不启动GPU、不删除远端结果。已用于r04/r05失败终态及r06完成终态回收。每次新的GPU运行前仍须确认上一运行的`VERIFIED_LOCAL`回执。

下载接续时的历史状态：r05超时回收后，20个终态组350文件656,894,160字节本地核验通过，总GPU持锁时间约669.365秒；r06当时开始补同样九点，现已完成。BF16下载原child2509在确认身份后受控停止并保留缓存，改为8 workers的controller21895/child21898；AWQ原child2511因CDN 403退出，保留三个完整分片及缓存后由controller21971/child21976续传；NVFP4原child2512未改动。旧状态与日志均保留，不能将r01整体状态当作后继下载完成状态。

## Qwen完整四格式网格与能耗排序

`qwen3_awq_survey_r06`在既有锁内完成其余9点，持锁118.417秒，50文件954,173字节已本地SHA256核验。与r01前5点合并后，`figures/qwen_fourformat_characterization_summary.json`（schema=2）含完整56/56点、280次进程内时延重复、28个AWQ/NVFP4 v2能耗窗，无缺失。图为`F1_qwen_four_formats.pdf`和`F2_qwen_awq_nvfp4_energy_v2.pdf`。r01原bitwise失败行、r02–r05未测量终态及接受条件仍保留。新增点中uniform B=2048/8192与Zipf B=8192非逐位相同，但满足显式BF16-roundoff条件；最大relative L2=2.30116e-5、maxabs=0.001953125，不能称exact。

| B | 两路由的最低中位延迟格式 | 观测范围 |
|---|---|---|
| 1 | FP8 | 两路由完整排序均FP8<AWQ<BF16<NVFP4 |
| 8、32、128、512 | AWQ | 两路由完整排序均AWQ<NVFP4<FP8<BF16 |
| 2048 | NVFP4 | 两路由完整排序均NVFP4<AWQ<FP8<BF16 |
| 8192 | FP8 | uniform为FP8<BF16<NVFP4<AWQ；Zipf为FP8<NVFP4<AWQ<BF16 |

最快路径改变只能定位至[1,8]、[512,2048]、[2048,8192]，不能拟合精确交叉点或直接当作部署阈值。AWQ在其中四个获胜负载的FP8/AWQ比率为uniform 1.704–2.211、Zipf 1.492–3.449。相邻差距可很小，例如uniform B=2048的AWQ/NVFP4=1.026950、uniform B=8192的NVFP4/BF16=1.011299。五次范围虽分离，仍来自单次驻留内重复，不能当作独立运行显著性。BF16/FP8合成权重与两个不同公共校准首层来源、默认未调优路径的边界不变。

能耗只比较同v2协议的28个单次约3秒counter窗口（空闲基线后先恢复连续负载3秒）。采样积分相对counter：NVFP4为−0.546%至+3.000%，AWQ为−3.994%至+0.631%，全部28窗中位+0.075%；不是外部计量精度或重复不确定性。AWQ/NVFP4总能耗比范围uniform 0.610–1.322、Zipf 0.473–1.385，扣空闲比范围分别0.604–1.398、0.512–1.455。uniform总能耗排序反转在[512,2048]、净能耗在[128,512]；Zipf两口径均在[128,512]。Zipf B=512时AWQ延迟较低（NVFP4/AWQ=1.10827），但总/净能耗分别高14.27%/21.61%；uniform B=512总能耗低3.88%而净能耗高1.08%。这说明当前路径的延迟最优不自动是能耗最优；尤其小差距仍待独立重复，不能画伪误差条或外推到生成token能耗。

## 大负载下AWQ与NVFP4的失利位置不同

`runs/profile_component_analysis_r01/awq_focus_r06.json`对B=2048/8192的两路由、三格式共12条已本地SHA核验trace做CPU分析；八条已有FP8/NVFP4的组件duration和count与旧summary完全一致。下面为B=8192的CUDA kernel duration总和，单位μs；每条trace仅一次instrumented graph replay和一次host `cudaGraphLaunch`。

| 路由 | 格式 | 完整投影kernel | 其他kernel |
|---|---|---:|---:|
| uniform | FP8 | 1163.184 | 646.375 |
| uniform | AWQ | 2078.460 | 429.703 |
| uniform | NVFP4 | 705.981 | 1681.947 |
| Zipf | FP8 | 1348.783 | 635.912 |
| Zipf | AWQ | 2217.250 | 441.639 |
| Zipf | NVFP4 | 803.929 | 1670.065 |

AWQ完整Marlin投影占总kernel时间82.87%/83.39%，相对FP8长1.787×/1.644×；其他kernel反而少216.672/194.273μs。因此，当前AWQ大负载失利位置在整个投影kernel，而NVFP4失利位置在投影之外，不能用“低bit辅助开销大”统一解释。B=2048时AWQ投影比FP8多21.637/37.790μs，其他kernel少48.638/49.633μs，抵消后插桩总和与另行无插桩测量的median仍均低于FP8。

完整projection包含各路径不同的融合工作（scale/addressing/reduction等），不能把差值解释为纯Tensor Core效率、已证实的反量化瓶颈或可以兑现的优化收益。单次trace不提供跨运行不确定性，duration和不等于无插桩延迟，也没有dynamic counters、实际DRAM流量或受控优化干预。权重仍为合成FP8与不同公共校准AWQ/NVFP4；该增量只细化耗时位置。复现入口为`python3 code/analyze_profile_components.py --awq-focus`，不新增GPU实验、不改旧组件summary或图。

## F3多步生成诊断入口与CPU准备

默认五项全量MC仅验证prefill loglikelihood，未覆盖decode格式切换。`eval_quality.py --generation-config configs/quality_generation_gsm8k_128.json`另接入官方`gsm8k_cot_zeroshot`和strict/flexible两种exact-match。固定test revision、seed20261016与升序128题，所有臂同人口、同真实host/sidecar权重身份；batch4、greedy、thinking=false、自然EOS/task stops、cap512、每臂900秒及最多65,536生成token。长度不筛选、不截断；cap/无答案仍评分，错误/超时/未完成保留全人口并使aggregate无效。只有实际worker计数证明两格式参与，才可称混合生成质量。128题是本项目校准/规则选择的留出诊断，不是完整GSM8K分数，也不保证预训练无污染。

CPU准备`runs/quality_generation_data_r01`已完成，4文件277,426字节回传且SHA核验；状态`prepared_no_generation`，128题全`not_started`、输出token0、无质量分数。实际安装lm-eval的CPU接口检查首次因夹具遗漏`truncation_side`失败；补全构造器状态后退出0，真实`generate_until`调用通过、引擎未构造、CUDA未初始化、无任务下载。记录为`runs/quality_setup/generation_cpu_checks.json`；这些只验证接线与人口记账，生成和F3效果仍未测。

## OLMoE 实际校准资源调整

`olmoe_awq_common_r01`未获得GPU锁，没有进行计算；`r02`取得锁后实际进入llmcompressor AWQ搜索。首层shared post-attention norm候选约7秒/步、20步，此外每层还有其他映射；据此预计原1800秒上限不足以完成16层。仅对已核对命令和进程组的本课题任务发送TERM，保留 `early_stop_reason.json`；这是资源试跑，不是质量失败或完整校准结果。 本轮实际持锁236.850秒；40个运行文件648,004字节已本地SHA核验，两个输出目录仅有218字节disk_plan，已标记`VERIFIED_LOCAL_PARTIAL`。

新配置`configs/olmoe_awq_common_r03.json`保留原始模型、512条×512 token校准、seed42和自然top-k路由，仅调整运行上限为7200秒并使用全新r03输出目录。派生AWQ与共同BF16母本分别为`checkpoints/olmoe_awq_common_r03`及`checkpoints/olmoe_awq_common_bf16_r03`；尚未获得完整模型产物。`sync_terminal_run.py`与`sync_derived_checkpoints.py --config configs/olmoe_awq_common_r03.json`等待终态后自动回传、核验。


后续NVFP4共同母本入口已在本地及远端准备：`calibrate_nvfp4_from_common.py`只对专家使用真实NVFP4激活校准，继承r03已做的AWQ平滑、不再改变norm/router/attention、不另存母本。CPU接口/pack检查通过，预计新增checkpoint约4.65GB；完整GPU校准、导出及质量均未运行。当前额外空间问题已询问用户：远端200GiB不足以容纳后续Qwen共同母本和三格式布局全套，建议350–400GiB；本地约118GiB空闲，完整派生模型/布局预计170–190GiB，需要更大本地保存目录或用户调整保存范围。当前OLMoE校准及公共权重下载仍可继续，不因问题待答暂停。

## 当前交付优先级：完整模型接入

用户明确要求尽快实际完成完整前向、短生成、最小serving对照，不以更多kernel表、来源检查或微基准替代。正在进行的512×512校准在第2层时按此优先级受控停止，保留原始日志和partial产物。新的`configs/olmoe_awq_integration_r01.json`使用同一原始OLMoE、seed42和真实AWQ搜索，但先用64条×128 token形成覆盖全16层的接入版本；这是小样本校准的integration prototype，非RTN，尚无质量通过或性能收益主张。三个最小臂为single FP8、single AWQ、dual phase，均加载同一校准后BF16 host；phase的混合步threshold32只是接入默认值，不是已测最优。下一GPU工作集中为全层导出、完整前向/短生成及同输入同预算的实际服务对照。

完整前向/短生成入口`code/run_full_model_integration.py`与三臂分组入口`code/run_full_integration_group.py`已上传。每臂独立进程、两个固定问题，先各生成1 token完成全模型前向，再自然生成最多32 token；dual臂要求全部16层实际出现scheduler prefill-FP8和decode-AWQ。最小服务配置为`configs/minimal_serving_r01/`：固定16请求（ShareGPT/LongBench各8），自然EOS/cap32，λ2/seed42，每臂独立预热后测两次，保留全部失败与终止。以上尚未取得GPU运行结果。

本地`code/advance_integration.py`已经启动，状态见`runs/integration_advance_r01/state.json`：校准成功且运行日志本地核验后，顺序执行`olmoe_integration_export_r01`、`integration_group_r01`、`minimal_serving_group_r01`。每组仍使用共享锁，终态运行目录先回传核验；失败即停、不重复提交。唯一校准作业当前排队等待其他课题释放GPU。大checkpoint与sidecar回传可与后续只读推理并行，交付前必须核验保存完成，不将大文件传输作为完整模型接入的串行前置条件。

接续更正：`olmoe_awq_integration_r01`在1800秒等锁上限到达后FAILED，没有`worker_pid`、`started_utc_s`或`lock_held_s`，没有输出checkpoint目录；唯一942字节终态文件已本地SHA核验，空派生产物回执为`VERIFIED_LOCAL_PARTIAL`。原advance_r01按失败即停退出，没有启动导出或推理。

现场`lslocks`证明其他课题使用内核阻塞等待，本线原NB轮询没有进入该队列。`run_locked.py`改为先NB探测，再使用同一截止时间的内核flock等待；取锁后仍核对NVML进程，忙则解锁；CPU文件锁验证取得与超时/信号恢复路径。仅在原进程消失、终态本地核验且两个输出目录均不存在后，启动`olmoe_awq_integration_r02`，保留原校准命令和原尚未创建的checkpoint路径、GPU上限1200秒；排队上限21600秒不计作GPU工作。现场已确认PID36199为共享锁的WRITE*等待者，不声称严格FIFO或已经开始校准。

当前自动接续为`runs/integration_advance_r02/state.json`，对应`advance_integration.py --calibration-run olmoe_awq_integration_r02 --output runs/integration_advance_r02 --initial-wait-seconds 21600 --lock-wait-seconds 21600`。新校准日志和共同母本同步任务均已启动；`sync_integration_sidecars.py --execute --wait-seconds 21600`等候导出完成后回传32个payload和三份配置。远端仍200GiB，为保证OLMoE母本与sidecar空间及I/O，已确认只对本课题Qwen BF16下载PID21898发SIGSTOP并观察到T状态，断点文件未删；记录见`runs/qwen_bf16_download_resume_r02/integration_priority_pause.json`。后续须恢复该下载，不能遗忘暂停状态。
