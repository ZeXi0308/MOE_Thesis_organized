# 固定全局宽度的原生 MoE companion 探针：实际分组改变，目标输出不变

2026-09-06；`agent/publish-current-moe-code@2a37765`加本轮未提交的独立探针。

**Verdict：`MEASUREMENT_ONLY / NO_TARGET_OUTPUT_DIFFERENCE_IN_TESTED_FIXED_WIDTH_OPERATOR`。**
本轮新增36次真实GPU算子对照，两个预指定自然文本目标、两个独立进程。置换与替换均
实际改变目标的专家内部位置；替换也改变专家负载。但目标router、top-k IDs/weights和
MLP输出全部逐位一致，最大绝对差0。停止当前同宽companion数值外部性探针，
不实现verifier，不把这条局部负结果扩写成所有batch conformance的否定。

**为什么转入这一问题。** 开始时实时检查发现，共享工作区已经完成原计划的
[单次降档及受控重复](../../admission_capacity/20260906_native_single_action_r01/REPORT.md)：
24个episode中，8组单次32→16均未超过同期hold32和static16，该触发formulation停止。
这些是本轮读取的既有结果，本轮没有重复启动或重新认领为新增GPU执行。
按已有[候选排序更正](../../admission_capacity/20260906_research_synthesis_r01/ADDENDUM.md)，
只推进条件备选：固定全局shape是否仍不足以固定MoE内部数值行为。

最弱环节是custom数值现象能否迁移到原生专家后端。历史
[N0d封存结论](../../../N0D_ROUTER_LOGIT_CONFORMANCE_REPORT.md)比较serial与batch4，
未捕获pre-router hidden，也没有同宽companion干预。本轮不重跑其三进程历史审计，
不修改其raw或更正。现有资产没有层输入tensor，所以新捕获真实输入；这不是旧差异事件复现。

**最小实验。** [事前设计](DECISIONS.md)固定以下内容，执行中未调整。

| 项目 | 实际执行 |
|---|---|
| 模型、硬件 | BF16 OLMoE-1B-7B-0924，revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`；单RTX5090 |
| 软件 | vLLM0.26.0、PyTorch2.11.0+cu130、CUDA13.0；原生UnquantizedFusedMoEMethod/Triton |
| 输入来源 | 既有32条冻结文本，各128 prompt tokens；本地source indices0/16，即gate0-060/088、WikiText rows99/142 |
| 捕获 | 原生eager引擎，先提交全部32请求，一次4096-token prefill；用真实request IDs和query offsets核对32段各128token，取零索引layer3 MLP入口的末prompt-token向量 |
| 探针固定项 | 全局M=16、target固定row7、H=2048、E=64、top-k=8、同一目标输入及权重；VLLM_BATCH_INVARIANT=0 |
| 原companions | 目标所属半区其余15个真实source，按源顺序；目标移至row7 |
| 置换 | 反转上述15个companions，目标位置和向量不变 |
| 替换 | 换成另一半区的前15个source，目标位置和向量不变；未按route或差异筛选 |
| 重复 | 每个进程两target×三arm×三repeat=18次；两个新进程，共36次；第二进程加载第一进程保存的完全相同fixture |

各repeat的arm顺序为原/置换/替换、替换/置换/原、置换/原/替换。正式调用前先覆盖
全部arm；每个target/arm还比较原生完整MLP与分离的原生gate/router/专家调用，12个
提取检查均逐位相同。分离调用没有HF权重转换或CPU top-k替代。

输入捕获引擎的token预算4096和eager模式是本次数值提取设置，不是上一轮serving
性能配置。探针不测TTFT/TPOT，不将算子调用数量称为请求完成数。两个target是
prefill末token的真实激活，不能称为两个steady/bursty decode域。

**测量结果。** [完整重算数据](analysis/analysis.json)保留每次比较；下表中的比较数
复用同36次调用，不是额外独立样本或36条独立文本。

| 对照 | 比较数 | 专家token计数改变 | 目标专家块内row改变 | 目标输出改变 | 输出max-abs |
|---|---:|---:|---:|---:|---:|
| 同臂repeat | 36 | 0 | 0 | 0 | 0 |
| 原→置换 | 12 | 0 | 12 | 0 | 0 |
| 原→替换 | 12 | 12 | 12 | 0 | 0 |
| 跨进程同key | 18 | 0 | 0 | 0 | 0 |

上述全部比较中，目标输入、router logits、GPU实际selected IDs、weights和输出均
逐位一致。输入从冻结fixture和每次source indices重建；原始per-call input未另存，
代码始终复制对应fixture行，专家调用也使用独立clone，避免in-place污染下一arm。

这不是只替换标签的无效干预。实际Triton分组函数返回的sorted token IDs、expert IDs、
有效padding长度均已保存，目标assignment的专家身份与native路由一致：

| Target | 置换改变的8条assignment块内row | 替换改变块内row | 替换前→后U / C | 有效padded slots | 专家计数L1差 |
|---|---:|---:|---|---:|---:|
| gate0-060 | 2/8 | 8/8 | 0.75000 / 4.5 → 0.78125 / 4.0 | 768→800 | 108 |
| gate0-088 | 1/8 | 8/8 | 0.78125 / 4.0 → 0.765625 / 4.5 | 800→784 | 114 |

U为被激活专家数/64，C为最大token数/全64专家平均数；这里每调用16token、128条
expert assignments，均值2。计数、padding和位置不是HBM实测或拥塞指标。
只读取有效padding长度内的元数据，没有读取torch.empty尾部。

所有调用记录的kernel参数相同：`BLOCK_SIZE_M=16, N=64, K=128, GROUP_SIZE_M=1,
SPLIT_K=1, num_warps=4, num_stages=4`。本轮确实改变了分组、padding及内部row，
没有触发记录中的kernel配置切换。不能由此证明不同tile、split-K、量化、模型或更大
width下也都一致，更不能声称已证明全局数值不变量。

**能说到哪一层。** 这是`ISOLATED_GPU_RUNTIME_NATIVE_MOE_OPERATOR_NUMERICS`：
两个自然target、一个layer/width、BF16原生Triton后端的局部负结果。它削弱的是
“这些同宽companion变化会改变目标专家输出”的解释，不否定历史serial/batched
pre-top-k差异；后者还包含不同全局shape和上游Attention/KV/Norm路径。
本轮没有检测到可继续定位的差异，也没有用户可见token、质量、SLO、性能或Oracle证据。

已有[vLLM batch-invariance功能](https://docs.vllm.ai/en/stable/features/batch_invariance/)
面向跨batch确定性，并列出多个MoE模型测试。本轮仅测关闭该功能的普通路径，
不能据此认证其完整保证；没有正信号，因此未再运行功能开关对照或创建新verifier。

**保留与复跑。** [配置](config.json)、[精简输入](inputs.json)、[实际上传包](execution.tar.gz)
及[gpu_results](gpu_results/)组成一个紧凑目录。fixture约为32×2048个BF16值；
原始36次router/output和真实分组tensor均保留。原始GPU归档约2MB，无失败GPU尝试。
实际进程为24951、25554，均正常退出；开始及各调用前检查GPU隔离，结束后GPU空闲。
隔离检查不是持续监控。探针源码摘要匹配唯一实际上传包；原生模块摘要另存于environment，
两进程一致。未修改旧原始结果。

实现为[run_native_companion_probe.py](../../../experiments/run_native_companion_probe.py)，
只新增独立探针；定向CPU检查验证固定target位置、置换集合及替换交集，语法检查通过。
未新增全仓测试或审计流程。运行器为[run_campaign.py](run_campaign.py)，两个实际命令
分别留在[process0](gpu_results/process-0/commands.txt)、[process1](gpu_results/process-1/commands.txt)。
从仓库根目录重算到一个不存在的目录：

```bash
.venv/bin/python refine-logs/expert_saturation/outputs/native_companion/20260906_layer3_r01/analyze_probe.py --output-dir /tmp/moe-native-companion-reanalysis
```

**唯一下一步。** 停止当前同宽companion探针；不换target/layer/width寻找正例。
下一轮只资格化已经保留的自然显存压力运行域：在当前OLMoE和5090上，用真实长上下文
请求比较cap16/32，记录实际权重/KV/workspace占用、接纳等待和可服务性，判断专家驻留
与KV竞争是否真实暴露。使用模型允许的上下文和设备可用显存，不人为压小cache或
显存预算制造offload。已有权重+KV字节估算只是下界，不是压力证据；若没有暴露路径，
就停止该运行域，不实现pager/controller。该实验**UNRUN**，本轮未同时推进第二条GPU链。

| 结束字段 | 结论 |
|---|---|
| Verdict / failure | 有效局部负结果；当前固定M16/layer3/两target的companion数值影响未出现 |
| Strongest baseline | 同输入原companions、真实置换负控、同臂和跨进程重复；完整MLP提取对照 |
| Oracle / claim ceiling | 性能Oracle不适用且未运行；仅原生单层数值测量，无方法GO |
| Reopen | 新的独立证据显示代表性后端/真实运行域存在稳定目标差异；不以调参数寻找正例重开 |
| Next | 上述自然长上下文显存容量资格化，UNRUN |

HEAD未改变，未push，未修改`docs/current/README.md`。当前没有形成可承诺CCF B完整论文的方法链。
