# APC 下固定轮转的最小验证

PREPARED_UNRUN；本包仅CPU准备，不上传、不执行GPU。整组排在A d6八格、B jit两格之后。
根会话在bundle外用共同flock /root/autodl-tmp/moe-research-gpu.lock保护整组，串行调用四个run_one_cell入口。
每格原始结果保留，整组终态后统一tar/hash回读；运行中不释放整组物理锁。
主问题仍为固定KV压力下的长恢复等待。已完成APC原生对照减小重算，但长恢复等待仍存在。
唯一问题：默认APC开启后，原most_output轮转能否缩短长暂停，并保持完整请求成本可接受？
四格独立引擎：native_apc / most_apc / most_apc / native_apc；同已观察cohort2，不声称新holdout。
保持OLMoE/BF16/vLLM0.26、32×3072/1024、50ms steady、KV16089350144bytes/7671usable blocks。
cap32、max_num_batched_tokens1024、full-history FCFS、原编译模式及三个公共warmup不变。
APC全部开启；每格warmup后原样排空/reset并保留receipt；捕获与metrics文件逐字复用APC包。
most只用原RotationConfig、most_output，不扫阈值；native不安装策略。保留全部失败/长调用/逐token输出。
动作仍调用原生_preempt_request和worker resumed块表通道；不手动touch、清缓存、截断KV或改变need。
need=ceil(当前完整history/B)-owned_len，含ref0缓存被touch时的free消耗，是保守完整历史预算。
强制受害者释放量为唯一非null/ref_cnt==1物理块数；原生preempt前后实际free增量必须相等。
仅支持单组精确FullAttentionSpec、Unitary、B=hash_block_size=scheduler_block_size=16；禁partial/CoW。
保留同步、无spec/connector/deferred free等旧防护；安装与两臂初始化绑定APC四格共同7项vLLM源码hash。
cohort首16-token块全unique、测量前冷缓存；激活与交换前后检查live块无共享、引用/对象/ID正确。
这限定当前无跨请求共享域，不声称适用于共享system-prompt负载。未知状态直接失败，不能解除防护硬跑。
held请求不得丢KV/推进，protected必须每步得到调度，free仍覆盖remaining need，禁止自然二次抢占。
4格全部合格再按block native→most比较；APC基线旧格只作背景，不替换新native。全部重复保留。
主指标为完整吞吐、平均完成、max-ITL及逐请求收益/受害者损失；报告抢占、hold、重算与调度开销。
wall=scheduler_inclusive+engine_non_schedule+outside_engine_calls，策略检查在scheduler内，不能重复扣除。
capture high-water含APC继承区间；保留原分类，严格重复执行量另由成功执行区间并集分析。
输出沿各自策略独立演化；token一致性不是任务质量等价。参考SLO不证明业务长暂停SLO成立。
无动作记INVALID_NO_ACTION；安全/资源/身份失败保留原始过程并停止。四格结束后只报告MEASUREMENT_ONLY。
若有暂停收益但完整成本退化，报告权衡；若无收益，定位动作/预算/成本，不自动改阈值或更换问题。
Oracle、跨模型/跨负载、质量、最近邻系统、方法GO均未验证。本轮不修改旧raw、共用文件或台账。
