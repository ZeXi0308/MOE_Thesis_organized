# Native offload：固定资源强基线资格

问题：在固定d6 KV下，现成native CPU offload能否实际复用被抢占请求的KV、减少重复计算，并改善完整请求完成？这是强基线资格，不把已有offload算法作为本研究贡献。

范围：原OLMoE BF16/32文档/3072输入/1024固定输出/50ms到达，cap32/token1024/APC off/6656 usable blocks。两臂都不安装headroom/rotation调度适配器，原生FCFS；不改封存旧适配器guard。

off=无host KV；on=16GiB host KV上限，native OffloadingConnector，VLLM_USE_SIMPLE_KV_OFFLOAD=0。16GiB为32×4096×128KiB；配置上限不是实际分配或驻留，记录原生配置和内存。容器现场memory.max=96636764160、current=36029468672，尚有约56GiB余量；仅当时观测，运行前仍需资源检查。

顺序：block0-off/on，block1-on/off；四个新引擎，共128请求，不用旧native wall充当本次对照。所有warmup、失败、未完成和全进程日志保留。warmup后drain pending offload并reset_prefix_cache(reset_connector=True)，返回失败则停止，避免warmup KV污染主测。主测结束继续记录pending transfer drain耗时，不能将它算进已完成请求latency，但需作为后台成本展示。

观测：实际connector类型与配置、lookup matched tokens/async、worker完成load/store bytes/time/sizes；原request/token/step/KV telemetry沿用。worker累积transfer time不能直接当关键路径墙钟，也不能逐项相加成请求saving。旧字段preemption_mode可能仍标native_recompute，必须结合外部加载及computed adjustment解释，不能据标签否定offload动作。

资格：off connector=None/on必须OffloadingConnector；实际KV块数/型号保持，所有请求完成及输出长度正确。on加载字节和matched tokens为0则判动作未暴露，不能宣布offload性能机制成立；有加载仍须核对与重算减少的请求/step对齐。观察器CPU替身只核对透传与数据保存，不证明原生接口和异步行为已通过。

主指标：连续完整请求mean completion及max ITL同时报告，附整批wall、吞吐、TTFT、重算位置、主机和GPU内存、存储/加载及drain成本。双block展示，不以饱和SLO pass或单次几个百分点作GO。新输出token可能因执行batch改变而变化，保留差异并独立报告，不声称质量等价。

运行纪律：整组共享flock、每格GPU空闲检查、busy/查询失败ABORT；第一失败停止后三格，不覆盖、不自动重试。当前B host_cost先行，恢复保护另一会话仍CPU准备，启动前重新读GPU_COORDINATION。

Evidence ceiling: 原生offload基线探索；无新颖性/轮转兼容/方法GO。
唯一后续：若offload稳定恢复并改变请求成本，先将其纳入最强简单基线，再资格化轮转与connector状态；若不生效，先定位缓存存储/查找/加载和资源接口，不扫容量或修改guard掩盖问题。
