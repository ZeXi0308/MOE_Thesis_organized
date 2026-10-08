# 直接优化实验：固定缓存内的组间加载/计算流水

2026-10-02，用户要求减少前置探索，直接针对优化实测；投稿目标更新为 CCF B/C 水平，录用与创新性仍以最终结果判断。

现有新数据足够支持动手：cap24首轮16请求共256输出token，episode 6.588秒，expert ensure CUDA区间总4.372秒（含host提交间隙，并非纯PCIe计时），实际expert H2D219.194GB。纯decode也存在局部窗口：227/1488层行提前完成，中位剩余1.385毫秒。cap64负控0/3840提前完成。cap24/cap64资源不等，不作策略加速比。按完整16-token前缀，13/16请求输出一致；未验证数值/质量等价。

加载顺序的离线分析已足够停止复杂selector搜索：固定冷加载24专家，最简单补齐贪心3771/3777个oracle可完成行；纯decode没有超贪心的残差。这不构成完整请求收益，也不是整个调度方向NO-GO。现在只实现一个能够减少真实暴露路径的机制。

候选：每层保留24专家槽，将本层已经产生的真实top-k按resident-first分成最多12专家的小组。在独立copy stream上提前加载下一组，同时在compute stream执行当前组；不预测未来层、不改路由、不丢贡献、不加权重scratch/KV容量。LRU保证下一组只能使用当前组以外的槽；事件保证覆写前一个组的槽时，其kernel已经结束。每次加载后核对当前计算组的expert→slot没有变化。

三臂：A原24槽大组串行；B相同24槽、12专家小组串行；C相同24槽、相同12专家分组和LRU、copy/compute重叠。唯一预声明顺序 A/B/C/C/B/A，全部保留。所有臂独立fresh engine、相同full-workload warmup、无readiness observer；模型/输入/seed/KV相同，每臂自己生成后续route、KV、output。16条完整GSM8K提示同时到达，固定32输出token，KV1GiB、token budget512。固定长度是首轮净成本对照；有效正信号后才扩展自然EOS/质量、显存压力域与第二模型。

目标：完整episode wall、每请求flow/TTFT/ITL与完成数、输出一致性、实际expert H2D字节、kernel组数、cache峰值。先看C是否胜过A，同时以C/B分离重叠收益与小组税。三臂均计实际排队、加载、kernel、host组织与sampling成本。允许provisional >=3%作为继续信号；若仅C胜B却不胜A，判当前优化没有净收益，不包装为方法成功。符号翻转或噪声仅做必要受控复核，不调参筛选有利结果。

工程规模说明：原生pager只支持先加载整组再计算；要直接回答重叠是否抵消分组成本，必须增加跨stream依赖和slot存活控制。复用原生模型、KV、fused kernel、请求账本、WiSP LRU，仅增加一个约200行的执行模块，不建设跨层异步引擎或更多controller。

已知边界：此机制的独立新颖性尚未成立；WiSP/HybriMoE已有预取与重叠先例。第一步要求出现真实净收益，随后按动作比较查新，决定是可写的方法、工程基线，还是精确负结果。
