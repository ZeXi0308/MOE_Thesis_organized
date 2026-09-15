# 逐步KV观察器等价低分配消融

研究问题仍是固定KV下恢复/执行成本是否抵消完整请求收益。前序单格cProfile看到state累计107ms/32步，包含get_blocks包装与GC，尚无因果增量。本轮只检验观察成本，不能包装为新调度器。

原生vLLM0.26/OLMoE/BF16/旧d6/6656可用块/APCoff/32固定请求，复用未加profiler的native_offload_baseline包。2×2×反序：offload off/on，block counts original/direct，共8格；完整固定次序在run_group.py。GC保持默认，无Torch/cProfile、无新增每秒CPU采样。original仍经新reader调用，直接法只避免tuple/generator/KVCacheBlocks包装；保留每个快照新counts list、相同字段/频率/前后时点。无共享runtime修改。

源语义依据installed vllm/v1/core/kv_cache_coordinator.py:359–366和kv_cache_manager.py:632–634、700–704；全空组回退empty_kv_cache_blocks，因此实现也保留该语义。读现有req_to_blocks，不调用分配、不修改KV。CPU3000状态通过（1/2/4组、缺失/空/活跃/释放、历史值不变），仍不替代GPU等价。

主要观察：原/直接观察器在各offload状态下的完整请求mean completion与wall，完整报告maxITL/输出/重算/transfer成本。两次同号仅探索，不固定任意3%阈值、不称统计显著。旧baseline只作上下文，不替代本次原观察器对照。

资格：逐步请求/计算量调度签名、memory before/after字段（剔除时钟）、每请求完整输出、实际KV容量、preemption/recompute/transfer字节。若调度/输出不同，定位首分叉而非称等价消融；所有轨迹保留。无收益停止此微优化，不继续参数扫描。若有效，之后重新评估offload本体残余税，仍不能把观察器加速叫方法收益。

GPU按当前新cohort3强基线八格先行，整组flock/逐格现场检查，忙或查询失败ABORT。上传0/GPU0/无driver，运行UNRUN。结果不得回写旧原件。
