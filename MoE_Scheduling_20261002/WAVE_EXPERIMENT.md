# 专家分波后的单次归约：基线修复与执行成本

动机来自真实holdout反转和已定位源码：旧expert-major每波调用原生fused_experts，再BF16相加，改变容量/组序就改变部分结果的舍入。本实验消除这一个实现因素，不能保证attention、dense GEMM或完整生成的batch invariance，也不宣称新论文机制。

动作：固定每个原token/top-k槽位的FC2贡献缓冲；波组互斥且覆盖全部真实专家，执行原FC1/SiLU/FC2，所有波结束后调用一次原生moe_sum。不能让FC1覆盖既有FC2，故两者不再alias；必须排除本波未包含的expert assignment，不能对持久FC2逐波zero。每波路由、专家权重、输入、专家加载策略不变；单波直接走原native路径。

局部成本模型：原G波包含G次assignment/FC1/activation/FC2/zero/sum及G−1次partial add；新路径保持前四者，移除中间zero/sum/add，减为一次最终sum，少做重复workspace分配。代价是额外持久贡献缓冲和内存带宽压力。OLMoE H=2I=2048、k=8、BF16时，核心workspace 48→80 KiB/row，M2048额外64MiB核心存储；实际峰值还受输出/allocator影响。专家字节与KV池容量未增加，但不能声称总峰值相同。

## 已完成的单层GPU实验

`test_deferred_reduction_gpu.py` / `deferred_reduction_gpu_r01.json`，真实RTX5090/vLLM0.26原生Triton、OLMoE尺寸随机BF16权重。4个(M,cap)=(1,24),(16,24),(257,20),(2048,23)，分别以默认与共同固定GEMM配置检查，合计8 cases。

全部PASS：新路径与全权重原生结果逐位相同，反转波组后仍逐位相同；NaN poison检查确认本波只写对应槽、旧槽保持、终态覆盖全部槽。旧路径三种多波case最大误差0.00390625、relative L2约0.0031–0.0033，单波相同。这只是这些固定层状态的因果检查，不是全模型位等价证明。

单层冷加载8次交错重复包含真实pinned H2D，但每波按cap补齐复制，且全GPU权重仅为参考而保留，因此不是受限显存服务实验。新/旧平均wall比分别1.00585、1.00170、1.00015、0.99281；没有显著可用的局部速度正信号。M2048额外allocation峰值由125,945,856增至176,277,504B。不能用M2048的0.72%微收益推导完整加速。

## 完整服务比较

`wave_src`仅从partition_src增加归约模块、执行开关及记录；未改变scheduler或CPU KV。固定expert24/GPU KV1GiB、CPU KV1GiB、cap16、Q2048（engine4096），原模型/source32–47、自然EOS/max512、原warmupQ512固定32。顺序 **standard / deferred / deferred / standard**，四独立engine，共用整组锁。所有冷启动、最终drain与生成变化保留；主指标drained/实际返回tokens、flow、TTFT、hostgap、正确性和完整序列。峰值allocated/reserved及新增buffer单独报告。

该组目的是确认数值修复可在活请求服务中推进及其真实成本，不预设提升，不以旧partial归约为弱对照来命名新方法。若没有完整收益，仍可评估是否将单次归约作为后续更忠实的基线，但旧负结果不能抹去重写。

当前执行状态：四格完整服务组尚未启动。首次服务源码包上传时SSH连接重置，随后41307端口返回Connection refused；已请求确认实例状态或新地址。上传与运行授权已明确，不是当前障碍。单层GPU原始JSON及日志已完整保存在本地。服务源码的显存trace已修正：deferred路径的activation字段记录FC1/activation/FC2实际总量，明确它是包含output的deferred_workspace_bytes子集，不能相加。
