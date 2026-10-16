当前主张：OLMoE 六臂和 LLM-jp 24 臂集成通过；尚无完整四模型质量或两卡系统收益结论。
关键证据：六臂OLMoE与24臂LLM-jp集成通过；LLM-jp校准16篇×24臂完整保存，每臂53505个NLL token；40项codec图计时完成。
最强替代解释：小规模集成结果只证明执行路径；图计时显示动态 NVFP4 scale 成本明显，压缩字节不等于服务收益。
下一动作：分析LLM-jp校准Pareto与逐层贡献，推进其余模型；新Triton scale归约等待GPU数值/性能验证。
资源状态：校准driver2111已exit0，无本课题GPU采集在运行；模型下载driver2871仍活跃；用户在比较同机双PRO6000与双5090。

## 2026-10-16 初始化及本地备份

- 本地此前没有 research_B_tiered_combine，远端 /root 深度 4 内亦未找到 tiered 目录；新建独立目录，未把其他课题实验当作本课题数据。
- 环境原始记录和已安装 MoE 源码已从远端回传到 runs/environment_20261016/。
- 两端 SHA256 一致：environment.json `b7fc353b38de63d09a60c496ccfe956f82d65fad60034a11dcb8eb4510e73e10`；vllm_moe_sources.tar.gz `e25bd7000a2a70e1e436ddbb53e2faf6b4adc68c64078f65b40fd1874651ab94`。
- AgRsAll2AllManager.dispatch 在 all2all.py:122 调用 all_gatherv；combine 在 :148 调用 reduce_scatterv。该结论仅限此后端，不泛化为所有非 DeepEP 后端。
- oracle/nvfp4.py:175 的安装源码说明：FLASHINFER_B12X 未纳入自动选择，需要显式选择；实际目标模型 FP8/NVFP4 kernel 路径尚未验证。
- Torch CUDA 构建版本及实际 SM 能力尚待补测；不能只由包版本推出 cu130。
- 全部四模型质量、codec 微基准、两卡系统数据、外推和论文实证部分：未运行。

## 数据回传

运行 `bash code/pull_results.sh`：增量下载远端 runs/，按内容校验，保留被更新的本地旧文件，不删除本地或远端数据。每组实验终态后执行；运行中的日志可能变化，校验非空时待完成后再同步。当前没有后台自动同步进程。

- 回传脚本已实际执行成功，runs/sync.log 记录远端全部现有结果的内容校验一致；不只是 shell 语法检查。
- GitHub 首次 clone 因 HTTP2 framing error 失败。HTTP/1.1、depth=1、blob:none 的重试已启动（本地工具 session 45000）；接续先查询该会话/进程，勿重复提交。既有本地仓库为 sparse/partial clone，能枚举七月路径但所需 blob 不全，不能声称已读完。
- 上轮分类：进展。新增真实远端环境/源码证据并完成本地回传，建立并实测增量同步入口。整体六周课题仍未完成。

## 2026-10-16 codec 初版与资产恢复

- Git CLI 两次失败后，使用 GitHub 连接器取得任务指定的 main 资产。ref_july/SOURCE_FILES.json 记录文件 blob SHA；这是选定文件快照，不是完整 git checkout。事实快照见 FACTS.md。
- 已实现 code/codec.py、codec_reference.py、check_codec.py、compile_codec.py、bench_codec.py，新增 paper/main.tex 骨架（明确所有新实证仍待测）。
- CPU reference 的 15 组零值/随机/离群输入与 ties 检查通过；r01 离线编译失败暴露 FP8 load 的整数 other=0 类型问题，改为浮点 0 后 r02 全部 12 种编译成功。失败和成功日志、代码快照均在本地 runs/。
- Torch 实际版本已补测为 2.11.0+cu130，CUDA build 13.0，可见一张卡。GPU 利用率由 100% 恢复 0%，原因未确定，不再据旧观测判定当前忙碌。
- GPU driver run_codec_gpu.sh 先取得共享锁、查 GPU/计算进程再测；本轮启动 codec_gpu_r01，接续先查 driver/结果，不重复启动。

## GPU 检查与舍入修复（本轮后续）

- codec_gpu_r01 driver PID3442 已退出：d=48/random 的反量化不等于参考；统一为 value × (block_scale × global_scale) 后重测。
- codec_gpu_r02 driver PID3541 已退出：d=48/outlier 的 payload 不一致，未进入微基准。CPU 解释器复现了 3.5/1.75 中点的码值差异。
- 发现保护常量 1e-38 隐式引入 FP64 运算；现在显式使用 tl.full(...,tl.float32) 并使用 tl.div_rn。r03 保留中间修复的编译失败；r04 对最新 codec 全部 12 项离线编译通过。
- CPU 解释器 BF16 store 在观测样例中呈截断（FP32 对照逐值一致，BF16 最大差 0.0078125），故解释器检查限于 payload/scale/FP32 反量化；GPU 检查仍要求 BF16 一致，没有放宽。r04 解释器失败保留，r05 为限定范围后的检查。
- 诊断尝试遇到其他任务持锁后没有启动 GPU；/runs/codec_diag_r01/diagnostic.log 为空，不是通过结果。已现场看到其他课题 PID3648/3651，且用户明确确认存在其他任务。
- 本轮归类为进展：获得资产、实现 codec、运行检查并定位/修复具体舍入问题，建立论文骨架。完整课题未完成，资源等待不是科学结论。

- 最终回传已完成：runs/sync.log 的 20261015T175455Z 记录校验一致；本地确认 r05 interpreter 15/15 PASS、r04 compile 12/12 PASS、GPU r01/r02 失败日志和各版本 code.tar.gz 均存在。当前没有 benchmark 数字。

## codec 真机通过与质量 hook 实现

- codec_gpu_r03（driver 5988）已完成：15 组 GPU/reference 数值检查通过；rows={256,1024,4096,16384} × d={2048,4096} × 5 操作，各 5 次 event 值，全部保留。该版本已回传并 checksum 一致。
- d=2048 时，256 rows 的 FP8 quant / NVFP4 dynamic quant / given-scale pack 中位数为 29.79 / 79.94 / 32.51 us；16384 rows 为 35.94 / 435.84 / 58.24 us。这些是 eager API event span，包含提交间隙和分配相关成本，不作纯 kernel 或系统收益结论。完整汇总 runs/analysis_codec/eager_api.csv。
- 新 bench_codec.py 额外捕获每图 20 次操作，再保留 5 次 graph 回放值并除以 20；两口径分列。r04 尝试时 PID6840 已持锁，driver6900 exit3，GPU 未启动，无图计时数据。
- quality_transform.py 实现已加权输出上的 Mode I 排序分层、Mode II owner partial 和 gate/norm 分层。处理 vLLM 0.26 linear placement 的余数分配，不能套用旧 floor(expert_id*EP/E) 公式。CPU 独立映射/排序/聚合检查通过。
- vllm_quality_hook.py 在当前进程包装 TritonExperts.apply/moe_sum，检查单卡、后置权重、记录层覆盖，可恢复原方法。未改全局安装源码；这是 eager 质量路径，不用于系统计时。
- smoke_vllm_quality.py 使用 LLM.collective_rpc 在 worker 安装 hook、绑定真实层名；预期 OLMoE 16 层。六臂为 native/I-BF16/I-FP8/I-tiered/II-BF16/II-tiered，检查 native 与 I-BF16 prompt logprob 一致。脚本已部署但未运行。现存模型是 OLMoE Instruct，仅作为集成 smoke，不冒充正式 base checkpoint 四模型实验。
- 本轮是进展；不因资源占用重复提交同一 driver，不把未运行图计时或 hook 当完成。

- 最终本地核验：r03 GPU 15 组 PASS、40 项 × 5 原始计时值、quality_cpu_r01 PASS 均在本地。同步校验时间 UTC 20261015T180545Z（北京时间 10 月 16 日 02:05）。末次现场 PID6840 仍活跃且持锁，63817 MiB/88%；本课题无后台候卡或 GPU 任务。下一次接续先核查该进程和锁；图计时使用新目录 r05，不覆盖 r04 的锁忙记录。

## 模型启动路径与质量输入准备

- smoke r01 driver7870：vLLM import 被 torchcodec 依赖的 libnvrtc.so.13 缺失搜索路径中断。库位于 site-packages/nvidia/cu13/lib；仅在本任务 shell 设置 LD_LIBRARY_PATH，未改共享安装。
- smoke r02 driver8152：权重加载和模型 profile 前向已执行，采样器 FlashInfer JIT check_cuda_arch 报 sm75 限制而终止。质量驱动统一设置 VLLM_USE_FLASHINFER_SAMPLER=0 使用支持的替代采样路径；这不改变 combine 精度臂。该设置的完整模型运行待确认。
- smoke r03 driver8556：共享锁忙，exit3，未执行 GPU。当前实际持锁任务已从 PID8450 换为 PID9348，后者现场仍活跃，不能凭旧 PID 消失判断整台空闲。
- logits_capture.py 在 worker 内记录完整 compute_logits 分布，逐文档保存 CPU baseline，输出每位置 KL；包含最终 next-token 分布，与仅 prompt tokens 的 NLL 分开计数。check_logits_capture.py 与独立 categorical KL 对照、identity 与空调用拒绝均 PASS。实际 worker 集成仍待验证。
- 固定 WikiText 数据版本 b08601e04326c79dfdd32d625aee71d232d685c3。官方主机从远端不可达，按同 revision 由 hf-mirror 下载 validation/test；原始 parquet、URL、SHA256 和失败记录在 runs/inputs_wikitext2_r01。
- 原始 validation 3760 行、test 4358 行；按单等号标题分文章且至少80词，分别60/61篇可用；选前16篇 validation 作校准、前32篇 test 作留出，尚未运行候选。全文与行边界、文本 hash 在 runs/quality_articles_r02，不把段落当独立文档。
- 语料准备 r01 因缺 pyarrow 失败；只在课题 .deps 安装 pyarrow20.0.0 后 r02 成功，未改共享 site-packages。
- configs/model_candidates.json 记录四 checkpoint 候选及官方配置来源，未下载或评测；两种 Granite 同架构，不能宣称四架构泛化。
- paper/communication_model.tex 补充逻辑 partial 与跨设备字节的区别、元数据、动态 scale 成本、无重叠诊断公式及其边界。
- 本轮为进展：两项环境失败原因改变了执行配置，完整 KL 数值工具和实际原始/文章输入已准备；整体研究未完成。

- 本地已核验原始 validation 657209 B、test 732610 B 与下载 SHA256 一致，16/32 篇全文 hash 一致；本轮同步 UTC20261015T182032Z 成功。
- 新增唯一有界等待任务 stage0_validation_r01，driver PID11383，最长候锁600秒；获锁后复核空闲、模型 smoke 上限600秒、图计时上限300秒，整组持锁，终态写 exit_code。接续只查该 PID/日志，不另起同组任务。
- LLM-jp 公开模型下载 PID10726，固定 revision 1d5983076dfc67aee4a77ec06a27027f5bab6055、6文件合计1855303826字节，目的地 models/llmjp；配置/来源记录在 runs/model_source_llmjp。下载未完成时不宣称模型可用。仅下载自己的模型，不删除其他会话资产。

## 2026-10-16 集成通过、正式校准与中断恢复

- stage0_validation_r01 的模型已启动，失败来自 callable collective_rpc 被 vLLM 安全序列化限制拒绝。改用官方 worker_extension_cls 和命名 RPC；没有开启 pickle fallback。
- stage0_validation_r02 exit0：OLMoE Instruct 六臂通过，所有干预臂覆盖16层，I-BF16 相对 native 的完整分布 KL=0；LLM-jp 64-token、单文章24臂集成完成。二者仅作集成验证，不作语料质量结论。
- 40项 codec 图计时完成，每项5次回放、每图20次操作。d2048/256 rows：FP8 quant 2.174 us、NVFP4 dynamic quant 15.027 us、given-scale pack 2.627 us；16384 rows 对应17.888/424.630/36.757 us。原始 eager API span 同时保留，不混作同一口径。
- 新 nvfp4_tensor_scale 采用两级 Triton absmax 归约，减少全量 FP32/abs 临时张量；18项离线编译通过，真机数值和性能尚未运行。上述图计时属于优化前版本，code.tar.gz 可追溯。
- evaluate_quality.py 采用单文章、512-token块、1-token重叠；每篇首token以外恰计一次NLL，属于分块上下文 PPL。完整分布KL还包含每块最终预测，人口与NLL不同。Mode II 对照分组BF16；RankProfile 记录 gate 排名下的贡献范数份额，不冒充历史KL损伤比。CPU计数/份额检查已通过。
- LLM-jp 完整校准输入16篇、53521个token、24臂。r01 在连接中断前完成 validation_000—003，validation_004 留有部分窗口；重连后原driver15479已不存在且无exit_code。数据盘可见容量由50GB变200GB，进程ID重置；实例重启为推断，具体原因未确定。原始记录全部保留，不改写原manifest。
- 中断前数据已本地回传并校验一致（UTC 20261015T184311Z）。r02 driver2111 从源文章索引4重新采集，最多候锁600秒，GPU运行上限3600秒；待合并时只复用r01四篇完整文章，部分第5篇不混入完整结果。
- 用户明确先完成单卡阶段、之后提供两卡连接。阶段1需同机双RTX PRO6000，并检查拓扑、P2P和带宽；目前不启动两卡系统实验。
- 本轮为进展；整体六周课题仍未完成，测试集尚未使用。

- 其余三模型下载脚本首次SCP等待过久断开，queue r01因脚本缺失exit127；部署后仅启动queue r02（driver2871），按Granite1b、Granite3b、OLMoE base顺序下载。下载与GPU采集独立，未占用GPU；接续先核查该driver，不重复提交。

- UTC18:48:31现场：r02已处理validation_004第25块，driver2111仍活跃；Granite1b固定revision408b6e90baab8cf24f4aa9f8e19703ffa0a53b29、2671380438字节正在下载。最近一次实时回传因运行中的日志/metrics改变而返回2，已传部分记录，但整次一致性校验未通过；本组终态需再次pull_results.sh。

- UTC18:50:46：llmjp_calibration_r02 exit0。r01四篇完整文章与r02十二篇合并覆盖16篇、24臂，每臂53505个NLL token；本地已检查文档唯一性、输入hash、模型配置、臂定义和版本一致。r01未完成的第5篇窗口保留，不纳入合并。完整校准数据不等于留出确认，测试集仍未运行。

## 双PRO6000收益的当前判断（测前判断，不是结果）

- 对强基线（相同FP8 dispatch、均匀FP8 combine）的增量仍未实测；小batch decode暂不乐观，大消息规模存在待验证空间。不能用相对BF16的约61%载荷压缩替代相对FP8的约22%压缩。
- 条件算例保存在 runs/analysis_break_even/illustration.json：d2048、每方向remote行中一半NVFP4、假设有效带宽40GB/s，256/1024/4096行分别只省2.88/11.52/46.08us的字节传输时间。未计layout元数据及额外分层、scale、pack/unpack成本，不是净收益、端到端预测或上界。remote半数tail并不由EP2逻辑分层自动保证。
- 现有未优化NVFP4 dynamic quant的256×2048图计时15.03us，FP8 quant2.17us；这两项不是完整两lane增量成本，不能直接相减当作最终开销。新两级scale归约尚待GPU验证，不能提前记入收益。
- 资源决策：保持单卡阶段优先；双卡实现准备好后先用短窗口测真实拓扑/NCCL、一个模型的代表性batch和(c)/(d)完整路径，再决定是否扩展原定40GPU小时矩阵。不通过更换硬件追求有利结果。
