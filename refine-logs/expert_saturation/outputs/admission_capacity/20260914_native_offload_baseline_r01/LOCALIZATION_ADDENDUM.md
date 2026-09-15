# 同逻辑执行前缀的成本定位

四格32/32完整输出序列均相等。两off的1869步signature完全相同，两on的1866步signature完全相同；signature含请求身份、执行起点、调度token数和动作前output数，不含物理KV布局、kernel或hidden state，不能声称这些相同。

on/off两block第一signature差异均为step1026，前1026步与至少28592个已生成token前缀相同。共同前缀的engine额外耗时为3.257674/1.593988s，其中scheduler额外1.330824/0.517520s。新结论：额外成本在外部恢复改变实际调度序列之前已经发生，优先定位持续保存、查询和布局/worker成本。不能说外部缓存之前没有动作，后台store本就已执行。

两on全序列相同但engine调用总时长差4.178044s，wall差4.682058s。最大单调用差1.137528s在step1038，不能解释所有差值；第一on原件保留。两off engine总差0.172650s。它们是这次观察，不是总体噪声底。

prefix_timing.json的same_signature可能是不连续子集；post_prefix按可配对索引截至两轨迹较短者，不含off额外3调用，不用于全wall守恒。完整互斥wall账本仍以timing_localization.json为准。

Evidence: 已有raw的CPU定位，新增GPU0。支持下一有限开销观测，不支持新方法或因果物理税结论。

下一四格已在20260914_native_offload_cost_r01准备：同native prompt-offload配置，仅有限clock观测off/on/on/off。保留嵌套互斥、主线程CPU与墙钟，检查观测自身代价；不改恢复策略。
