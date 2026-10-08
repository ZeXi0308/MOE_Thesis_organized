# d6 原生动作前态重建资格

状态：MEASUREMENT_ONLY；两次同配置原生vLLM进程内诊断完成，64/64请求、65536输出token，全量归档回读SHA校验通过。不是策略性能实验。

问题：第329步前，可见token/请求状态相同的独立重建，是否也有相同的有效KV内容？

结果：两次完整记录的请求状态（包括本次物理block编号）、逻辑请求状态均相同。16层×32请求共512分段，其中496非空；每次实际核对13,619,429,376字节（约13.62GB十进制），差异分段0。哈希覆盖当前num_computed_tokens对应有效位置；不含空闲块与最后块未计算尾部。16个空分段来自当前无KV的待恢复请求，不能算作非空验证。

指纹在真实GPU Tensor上逐块读取，按固定FLASH_ATTN逻辑(B,H,N,2D)重排后SHA256。采集时记录的shape/stride相同；每段字节数 independently按tokens×heads×packed_head_size×BF16字节数核对通过。运行源哈希与封包逐文件一致，原生块池6657总/6656可用，初始化GPU均空闲。

同步、GPU到CPU拷贝与哈希分别花12.326/11.754秒，插在step329决策前。这些时间未从raw中删除，受扰动wall/ITL/完成指标只保留原件，不作为优化收益。采集之后两次正常执行至全部请求完成；没有失败、重试或挑选结果。

## 能支持与不能支持

支持：在这两次同模型/同资源/同输入/同既有策略重建中，记录的动作前请求状态和有效KV字节均一致。它把上一轮“只有可见状态一致”升级为实际KV内容资格，为候选动作分支提供已验证入口。

不能支持：完整引擎checkpoint、任意时刻可恢复、所有隐藏/随机/分配器状态一致、其它输入的确定性、候选动作排序正确、吞吐或平均完成改善。greedy temperature=0由采集器要求；仍未验证全部worker/input-batch内部状态。没有宣称模型/方法GO或跨模型泛化。

这是执行者的定向验证，未另做新一轮独立语义审计；已有CPU指纹测试覆盖物理迁移、尾部排除、有效字节变化和跨分片BF16字节一致性。新结果未借用上轮审阅的PASS。

## 下一最小实验

在相同前缀重建且每臂动作前KV指纹一致的条件下，只改变第一次合格交换的动作（least、most、延迟一次），随后都回到同一least策略并独立推进至全部请求完成。先冻结预计恢复调用数和需要的KV，再检查短期预测及完整请求代价；不把旧d2 first_most_then_least的结论当未测，也不把这一个d6事件叫一般Oracle或在线方法。分支尚未封包/未运行。

GPU已释放，本轮无追加组。原始路径readback/results/repeat{0,1}，运行入口pkg/run.sh，包hash见STATUS；主分析E/analyze_recovery_action_state.py，E为refine-logs/expert_saturation/experiments/admission_capacity。归档SHA256：3bc1532d8c95315a658f8c5db6bab65fc0be6280ea74ff0feabf97b6af9b5e42。
